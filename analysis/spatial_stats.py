from __future__ import annotations # PEP 563: Postponed Evaluation of Annotations

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tskit
import pyslim

from numba import njit
from scipy.optimize import curve_fit

from weakref import ref as weakref
from copy import deepcopy


#---------- ----------#
"""
NOTE:
- Clean up main function
    - its a mess! needs subfunctions etc
NOTE: Some more ideas for changes here
- Currently, calling the following functions requires a bunch of consistency
  checks post-call (e.g. checking on `len(s.meta)`):
    - s.set_timeslice() / s.area_slices_center() / s.area_slices_grid()
    - ideally, these checks would be integrated into the functions themselves
    - one way to achieve that would be to not subset `s` itself, but to have
      these functions return a subsetted `s` from some parent `s`. that way one
      could return a subsetted `s` if the checks pass, or `None` if they don't
"""

#---------- CLASS INTERFACES ----------#
class Area:
    def __init__(self, hbounds=[0, 1], vbounds=[0, 1]):
        self._hbounds = hbounds
        self._vbounds = vbounds
        hspan = hbounds[1] - hbounds[0]
        vspan = vbounds[1] - vbounds[0]
        self._set_area_local((hbounds[0], vbounds[0]), hspan, vspan)

    def set_area(self, origin, hspan, vspan):
        self._set_area_local(origin, hspan, vspan)
        self._test_bounds()

    @property
    def area(self):
        return self._area

    @property
    def span(self):
        return self._hspan, self._vspan

    @property
    def min(self):
        return self._hmin, self._vmin

    @property
    def max(self):
        return self._hmax, self._vmax

    @property
    def horiz(self):
        return self._hmin, self._hmax

    @property
    def vert(self):
        return self._vmin, self._vmax

    def _set_area_local(self, origin, hspan, vspan):
        self._horig = origin[0]
        self._vorig = origin[1]
        self._hspan = hspan
        self._vspan = vspan
        self._hmin = self._horig
        self._hmax = self._horig + self._hspan
        self._vmin = self._vorig
        self._vmax = self._vorig + self._vspan
        self._area = self._hspan * self._vspan

    def _test_bounds(self):
        assert self._hmin >= self._hbounds[0]
        assert self._hmax <= self._hbounds[1]
        assert self._vmin >= self._vbounds[0]
        assert self._vmax <= self._vbounds[1]

class BaseTree:
    def __init__(self, ts):
        self._ts = ts
        self._dfe = None
        self._func = None
        self._neut = None
        self._func_ts = None
        self._neut_ts = None

    def annotate_sites(self, func, neut):
        self._func = func
        self._neut = neut

    @property
    def ts(self):
        return self._ts

    @property
    def func(self):
        assert self._func is not None, "Need to annotate sites first!"
        return self._func

    @property
    def neut(self):
        assert self._neut is not None, "Need to annotate sites first!"
        return self._neut

    @property
    def dfe(self):
        if self._dfe is None:
            self._dfe = self._calc_dfe()
        return self._dfe

    def _calc_dfe(self):
        dfe = np.zeros(len(self.func))
        for i, pos in enumerate(self.func):
            site = self.ts.site(pos)
            coef = [
                mut['selection_coeff']
                    for mut in site.mutations[-1].metadata['mutation_list']
                    if mut['mutation_type'] == 1
            ] # `-1` since the last mutation added will have the full `mutation_list`
            dfe[i] = np.mean(coef) # average if multiple functional mutations
        return dfe

    @property
    def func_ts(self):
        if self._func_ts is None:
            self._func_ts = self.ts.delete_sites(self.neut)
        return self._func_ts

    @property
    def neut_ts(self):
        if self._neut_ts is None:
            self._neut_ts = self.ts.delete_sites(self.func)
        return self._neut_ts

class SampleSet:
    def __init__(self, parent: BaseTree):
        self._parent = weakref(parent)
        self._indiv = np.array([ind.id for ind in self.ts.individuals()])
        self._nodes = self._get_nodes()
        self._area = Area()
        self._locs = None
        self._ac = None
        self._pi = None

    @property
    def parent(self) -> BaseTree:
        parent = self._parent()
        assert parent is not None
        return parent

    def _flush_buffers(self):
        self._locs = None
        self._ac = None
        self._pi = None

    def _get_nodes(self):
        if len(self._indiv) > 0:
            return np.array([self.ts.individual(i).nodes for i in self._indiv])
        else:
            return np.array([], dtype=int)

    @property
    def ts(self):
        return self.parent.ts

    @property
    def nodes(self):
        return self._nodes

    @property
    def meta(self):
        meta = pd.DataFrame(
            np.hstack((self.ts.individual_locations[:,:2],
                       self.ts.individual_times.reshape(-1, 1))),
            columns=['Longitude', 'Latitude', 'time']
        )
        return meta.loc[self._indiv]

    @property
    def func(self):
        return self.parent.func

    @property
    def neut(self):
        return self.parent.neut

    @property
    def func_ts(self):
        return self.parent.func_ts

    @property
    def neut_ts(self):
        return self.parent.neut_ts

    @staticmethod
    def subsample_ts(ts, max_sites):
        if ts.num_sites < max_sites:
            return ts
        drop_list = np.random.choice( #type: ignore
            ts.num_sites, ts.num_sites - max_sites, replace=False)
        return ts.delete_sites(drop_list)

    def split_afs(self, max_sites=100000):
        func_ts = self.subsample_ts(self.func_ts, max_sites)
        neut_ts = self.subsample_ts(self.neut_ts, max_sites)

        afs_func = func_ts.allele_frequency_spectrum(
            sample_sets=[self.nodes.ravel()],
            span_normalise=False,
            polarised=True,
            windows="sites"
        )[:, 1:].sum(axis=0)
        afs_neut = neut_ts.allele_frequency_spectrum(
            sample_sets=[self.nodes.ravel()],
            span_normalise=False,
            polarised=True,
            windows="sites"
        )[:, 1:].sum(axis=0)
        afs_func = afs_func / afs_func.sum()
        afs_neut = afs_neut / afs_neut.sum()
        return afs_func, afs_neut

    def split_aage(self, time=None, max_sites=100000):
        func_ts = self.func_ts.simplify(self.nodes.ravel())
        neut_ts = self.neut_ts.simplify(self.nodes.ravel())
        func_ts = self.subsample_ts(func_ts, max_sites)
        neut_ts = self.subsample_ts(neut_ts, max_sites)

        def time_gen(ts, mut_type):
            for site in ts.sites():
                meta_list = site.mutations[-1].metadata['mutation_list']
                mut_list = site.mutations
                for meta, mut in zip(meta_list, mut_list):
                    if meta['mutation_type'] == mut_type:
                        yield mut.time

        if time is None:
            time = 0
        ages_func = np.array(list(time_gen(func_ts, 1))) - time
        ages_neut = np.array(list(time_gen(neut_ts, 0))) - time
        return ages_func, ages_neut

    @property
    def dfe(self):
        return self.parent.dfe

    def calc_Va(self):
        ac = self.allele_counts(remove_neut=True)
        p = ac / self.nodes.size
        a = self.dfe
        assert len(p) == len(a)
        Va = 2 * p * (1-p) * a**2
        return np.sum(Va)

    def set_timeslice(self, time):
        alive = pyslim.individuals_alive_at(self.ts, time)
        alive_intersect = self.meta.index.intersection(alive)
        self._indiv = self.meta.loc[alive_intersect].index.to_numpy()
        self._nodes = self._get_nodes()
        self._flush_buffers()

    def set_area(self, origin, hspan, vspan):
        meta = self.meta
        area = self.area
        area.set_area(origin, hspan, vspan)
        ovl = (
            (meta[['Longitude', 'Latitude']] >= area.min).all(axis=1) &
            (meta[['Longitude', 'Latitude']] <= area.max).all(axis=1)
        )
        self._indiv = meta[ovl].index.to_numpy()
        self._nodes = self._get_nodes()
        self._flush_buffers()

    def set_area_occupied(self):
        meta = self.meta
        area = self.area
        hmin, vmin = meta[['Longitude', 'Latitude']].min()
        hmax, vmax = meta[['Longitude', 'Latitude']].max()
        area.set_area(
            (hmin, vmin), hmax - hmin, vmax - vmin
        )

    def area_slices_center(self, n):
        aslices = []
        for frac in np.linspace(1/n, 1, n):
            hspan, vspan = np.array(self.area.span) * frac
            origin = (
                np.mean(self.area.horiz) - hspan/2,
                np.mean(self.area.vert) - vspan/2
            )
            aslice = deepcopy(self)
            aslice.set_area(origin, hspan, vspan)
            aslices.append(aslice)
        return aslices

    def area_slices_grid(self, step):
        meta = self.meta
        meta = meta.assign(
            lonbin=np.digitize(meta.Longitude,
                               np.arange(0, meta.Longitude.max(), step)),
            latbin=np.digitize(meta.Latitude,
                               np.arange(0, meta.Latitude.max(), step))
        )
        aslices = []
        for i, (cat, group) in enumerate(meta.groupby(["lonbin", "latbin"])):
            origin = group[['Longitude', 'Latitude']].min().to_numpy()
            hspan = group.Longitude.max() - group.Longitude.min()
            vspan = group.Latitude.max() - group.Latitude.min()
            aslice = deepcopy(self)
            aslice.set_area(origin, hspan, vspan)
            aslices.append(aslice)
        return aslices

    @property
    def area(self):
        return self._area

    @property
    def area_size(self):
        return self.area.area

    def get_Geno(self):
        """
        NOTE: Deprecated
        """
        dipl = np.zeros((len(self.meta), self.ts.num_sites), dtype=np.int8) - 1
        for var in self.ts.variants():
            var_dipl = var.genotypes[self.nodes].sum(axis=1)
            dipl[:, var.site.id] = var_dipl
        return dipl

    def allele_counts_old(self, remove_neut=False):
        """
        NOTE: Deprecated
        """
        if remove_neut:
            ts = self.func_ts
        else:
            ts = self.ts

        ac = np.zeros(ts.num_sites, dtype=int)
        for var in ts.variants():
            ac[var.site.id] = var.genotypes[self.nodes].sum()
        return ac

    def allele_counts(self, remove_neut=False):
        if remove_neut:
            ts = self.func_ts
        else:
            ts = self.ts

        ac = ts.sample_count_stat(
            [self.nodes.ravel()],
            lambda x: x,
            1,
            windows="sites",
            mode="site",
            span_normalise=False,
            polarised=True,
            strict=False
        )
        return np.squeeze(ac)

    def set_subset_allele_counts(self, subset_list, remove_neut=False):
        if remove_neut:
            ts = self.func_ts
        else:
            ts = self.ts

        nodes = [s.nodes.ravel() for s in subset_list]
        acm = ts.sample_count_stat(
            nodes,
            lambda x: x,
            len(nodes),
            windows="sites",
            mode="site",
            span_normalise=False,
            polarised=True,
            strict=False
        )
        for i in range(len(subset_list)):
            subset_list[i].an = acm[:,i].copy()

    def seg_sites(self):
        return self.ts.segregating_sites([self.nodes.ravel()], span_normalise=False)

    @property
    def locs(self):
        if self._locs is None:
            self._locs = self.meta[['Longitude', 'Latitude']].to_numpy()
        return self._locs

    @property
    def an(self):
        if self._ac is None:
            self._ac = self.allele_counts()
        return self._ac, self.nodes.size

    @an.setter
    def an(self, allele_counts):
        self._ac = allele_counts

    @staticmethod
    @njit
    def _quickavgdist(l1, l2):
        dists = np.zeros(len(l1)*len(l2))
        n = 0
        for i in range(len(l1)):
            for j in range(len(l2)):
                dists[n] = np.sqrt((l1[i][0] - l2[j][0])**2 +
                                   (l1[i][1] - l2[j][1])**2)
                n += 1
        return np.mean(dists)

    @staticmethod
    @njit
    def _calcFstHudson(a1, n1, a2, n2):
        '''Hudson Fst after Hudson[1992] and Bhatia[2013] (Supplement)'''
        h1 = (a1 * (n1 - a1)) / (n1 * (n1 - 1))
        h2 = (a2 * (n2 - a2)) / (n2 * (n2 - 1))

        N = (a1/n1 - a2/n2)**2 - h1/n1 - h2/n2
        D = N + h1 + h2
        return np.nansum(N) / np.nansum(D), np.nansum(D > 0)

    def dist(self, other: SampleSet):
        dist = self._quickavgdist(self.locs, other.locs)
        return dist

    def fst(self, other: SampleSet):
        fst = self._calcFstHudson(*self.an, *other.an)
        return fst

    @property
    def diversity(self):
        if self._pi is None:
            self._pi = self.ts.diversity([self.nodes.ravel()])
        return self._pi

    def est_popsize(self, mu):
        """
        NOTE: Experimental
        """
        pi = self.diversity
        return pi / (4 * np.squeeze(mu))

    def coal_rate_pair(self, u, v, bins):
        """
        NOTE: Experimental
        """
        twidths = np.zeros(self.ts.num_trees)
        tmrcas = np.zeros(self.ts.num_trees)
        for tree in self.ts.trees():
            twidths[tree.index] = tree.interval[1] - tree.interval[0]
            tmrcas[tree.index] = tree.tmrca(u, v)

        tree_epochs = np.digitize(tmrcas, bins) - 1
        coalr = np.zeros(len(bins)-1)
        for e in range(len(bins)-1):
            epoch_trees = (e == tree_epochs)
            sum1 = np.sum(tmrcas[epoch_trees] - bins[e])

            other_trees = (e < tree_epochs)
            if e > 0:
                sum2 = np.sum(other_trees) * (bins[e] - bins[e-1])
            else:
                sum2 = 0

            ne = np.sum(epoch_trees)
            coalr[e] = ne / (sum1 + sum2)
        return coalr

    def get_bins(self, nbins=40):
        """
        NOTE: Experimental
        """
        tmax = np.array([tree.time(tree.root) for tree in self.ts.trees()]).max()
        bins = 0.1 * (np.exp(np.arange(nbins+1)/nbins *
                             np.log(1 + 10 * tmax)) - 0.1) #type: ignore
        return bins, nbins

    def mean_coalr(self, max_pairs=100):
        """
        NOTE: Experimental
        """
        bins, nbins = self.get_bins()
        nodes = self.nodes.ravel()
        npairs = int((len(nodes) * (len(nodes)-1)) / 2)
        pairs = np.zeros((npairs, 2), dtype=int)
        i = 0
        for u in range(len(nodes)):
            for v in range(u+1, len(nodes)):
                pairs[i] = [u, v]
                i += 1
        pairs = pairs[np.random.choice(len(pairs), #type: ignore
                                       np.fmin(len(pairs), max_pairs),
                                       False)]
        coalr = np.zeros((len(pairs), nbins))
        for i, (u, v) in enumerate(pairs):
            coalr[i] = self.coal_rate_pair(nodes[u], nodes[v], bins)
            if (i % 10 == 0):
                print(f"Done with pair {i}/{len(pairs)}")
        pops = 0.5 * (1/np.nanmean(coalr, axis=0)) #type: ignore
        return pops, bins

    def mean_coalr_across(self, other: SampleSet, max_pairs=100):
        """
        NOTE: Experimental
        """
        bins, nbins = self.get_bins()
        nodesA = self.nodes.ravel()
        nodesB = other.nodes.ravel()
        npairs = int(len(nodesA) * len(nodesB))
        pairs = np.zeros((npairs, 2), dtype=int)
        i = 0
        for u in range(len(nodesA)):
            for v in range(len(nodesB)):
                pairs[i] = [u, v]
                i += 1
        pairs = pairs[np.random.choice(len(pairs), #type: ignore
                                       np.fmin(len(pairs), max_pairs),
                                       False)]
        coalr = np.zeros((len(pairs), nbins))
        for i, (u, v) in enumerate(pairs):
            coalr[i] = self.coal_rate_pair(nodesA[u], nodesB[v], bins)
            if (i % 10 == 0):
                print(f"Done with pair {i}/{len(pairs)}")
        pops = 0.5 * (1/np.nanmean(coalr, axis=0)) #type: ignore
        return pops, bins


#---------- Plotting Helper Functions ----------#
def plot_afs(afs, ages, fig, axnums):
    afs_func, afs_neut = afs
    ages_func, ages_neut = ages
    age_cutoff = ages_neut.max()/2
    assert len(axnums) == 2
    ax = fig.add_subplot(*axnums[0])
    ax.scatter(np.arange(len(afs_func)), afs_func,
               marker="o", edgecolors="None",
               alpha=0.5, label="functional sites")
    ax.scatter(np.arange(len(afs_neut)), afs_neut,
               marker="o", edgecolors="None",
               alpha=0.5, label="neutral sites")
    ax.set_xlabel("Allele Count (singleton, doubleton...)")
    ax.set_ylabel("Fraction of sites - Normalized in Class")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.legend()
    ax = fig.add_subplot(*axnums[1])
    ax.hist((ages_func[ages_func < age_cutoff],
             ages_neut[ages_neut < age_cutoff]),
            bins=50, density=True,
            label=("functional sites", "neutral sites"))
    ax.set_xlabel("Allele Age")
    ax.set_ylabel("Density")
    ax.legend()

def plot_pops(pops, bins, ax, title=""):
    offset = np.where(~np.isfinite(pops))[0].max() #type: ignore
    ax.plot(np.concatenate(([0], np.repeat(bins[offset+2:], 2)[:-1])),
            np.repeat(pops[offset+1:], 2),
            label="Inferred")
    ax.set_xscale("log")
    ax.set_xlabel("Generations Ago")
    ax.set_yscale("log")
    ax.set_ylabel("Population Size")
    ax.set_title(title)

#---------- MAIN ----------#
def main(arglist=None):

    #----- Data Loading
    parser = argparse.ArgumentParser()
    parser.add_argument("basefn")
    parser.add_argument("--pophist", action="store_true", help="Experimental!")
    args = parser.parse_args(arglist)
    basefn = args.basefn
    infile = basefn + ".finished.trees"

    ts = tskit.load(infile)
    print()
    print(ts)

    parent = BaseTree(ts)

    #----- Mutation types
    mut_type = np.zeros(ts.num_sites, dtype=int)
    for site in ts.sites():
        for mut in site.mutations[-1].metadata['mutation_list']:
            # SLiM may introduce recurrent mutations!
            # `-1` since the last mutation added will have the full `mutation_list`
            if mut['mutation_type'] == 1:
                    mut_type[site.id] += 1
    func = np.arange(ts.num_sites)[mut_type.astype(bool)]
    neut = np.arange(ts.num_sites)[~mut_type.astype(bool)]
    print(f"Fraction functional: {len(func) / (len(func) + len(neut))}",
          end="\n\n")

#    # debug snippets for mutation metadata
#    func_gen = (ts.site(i).mutations[-1].metadata['mutation_list'] for i in func)
#    neut_gen = (ts.site(i).mutations[-1].metadata['mutation_list'] for i in neut)
    parent.annotate_sites(func, neut)

    #----- Metadata
    endgen = ts.metadata['SLiM']['tick']
    time_ago = endgen - np.array(ts.metadata['Arguments']['RememberGen'] + [endgen])
    time_ago = np.sort(time_ago)[::-1]

    #----- Set up output structure
    time_dict = {
        "time": [],
        "sample_size": [],
        "area_occupied": [],
        "num_segregating": [],
        "Va": [],
        "zmar": [],
        "fstavg": [],
        "fstmax": [],
    }

    #----- Set up time slices
    set_list = []
    print("Setting up time slices...")
    for time in time_ago:
        s = SampleSet(parent)
        s.set_timeslice(time)
        if len(s.meta) <= 1:
            continue

        time_dict["time"].append(time)
        time_dict["sample_size"].append(len(s.meta))

        s.set_area_occupied()
        time_dict["area_occupied"].append(s.area_size)

        set_list.append((time, s))

    #----- AFS split by site type - pre-extinction
    print("Generating AFS plots...")
    time, s = set_list[0]
    afs_func, afs_neut = s.split_afs()
    ages_func, ages_neut =s.split_aage(time)
    fig = plt.figure(tight_layout=True, figsize=(15, 8))
    plot_afs((afs_func, afs_neut), (ages_func, ages_neut),
             fig, ([1,2,1], [1,2,2]))
    plt.savefig(f"{basefn}_afs_pre-extinct.pdf")
    plt.close()

    #----- AFS split by site type - through time
    fig = plt.figure(tight_layout=True, figsize=(15, 5 * len(set_list)))
    idx = 1
    step = 3
    for time, s in set_list:
        ax = fig.add_subplot(len(set_list), step, idx)
        ax.plot(s.meta.Longitude, s.meta.Latitude, "k.", alpha=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        afs_func, afs_neut = s.split_afs()
        ages_func, ages_neut =s.split_aage(time)
        plot_afs((afs_func, afs_neut), (ages_func, ages_neut),
                 fig, ([len(set_list), step, idx+1],
                       [len(set_list), step, idx+2]))
        idx += step
    plt.savefig(f"{basefn}_afs_timeseries.pdf")
    plt.close()

    #----- number of segregating sites through time
    print("Getting total number of segregating sites...")
    for time, s in set_list:
        time_dict["num_segregating"].append(
            s.seg_sites().squeeze().astype(int))

    #----- Va through time
    print("Calculating relative Va...")
    time, s = set_list[0]
    Va_pre_extinct = s.calc_Va()
    for time, s in set_list:
        time_dict["Va"].append(
        s.calc_Va() / Va_pre_extinct)

    #----- zmar through time
    def powerlaw(x, c, z):
        return c * np.power(x, z)

    zmar_dict = {
        "area": [], "nseg": [], "time": [],
    }
    print("Calculating zMAR...")
    for time, s in set_list:
        area_slices = s.area_slices_center(10)
        area = []
        nseg = []
        for aslice in area_slices:
            if len(aslice.meta) == 0:
                continue
            area.append(aslice.area_size)
            nseg.append(aslice.seg_sites().squeeze())
        try:
            assert len(area) > 1
            fit = curve_fit(
                powerlaw, np.array(area), np.array(nseg),
                p0=[np.mean(nseg), 0.1])
        except (RuntimeError, AssertionError) as error:
            time_dict["zmar"].append(np.nan)
            continue
        zmar_dict["area"].extend(area)
        zmar_dict["nseg"].extend(nseg)
        zmar_dict['time'].extend((np.zeros(len(area)) + time).tolist())
        time_dict["zmar"].append(fit[0][1])
    pd.DataFrame(zmar_dict).to_csv(
        f"{basefn}_zmar.tsv", sep="\t", index=False)

    #----- Fst through time
    fst_dict = {
        "dist": [], "fst": [], "nsites": [], "time": [],
    }
    print("Calculating Fst...")
    for time, s in set_list:
        grid_slices = s.area_slices_grid(0.1)

        # Load all caches at once instead of on-demand:
        s.set_subset_allele_counts(grid_slices)

        fst = []
        dist = []
        for i in range(len(grid_slices)):
            for j in range(i+1, len(grid_slices)):
                g1 = grid_slices[i]
                g2 = grid_slices[j]
                fst.append(g1.fst(g2))
                dist.append(g1.dist(g2))
        if len(fst) > 0:
            time_dict["fstavg"].append(np.mean([f[0] for f in fst]))
            time_dict["fstmax"].append(np.max([f[0] for f in fst]))
        else:
            time_dict["fstavg"].append(np.nan)
            time_dict["fstmax"].append(np.nan)
            continue
        fst_dict["dist"].extend(dist)
        fst_dict["fst"].extend([f[0] for f in fst])
        fst_dict["nsites"].extend([f[1] for f in fst])
        fst_dict['time'].extend((np.zeros(len(dist)) + time).tolist())
    pd.DataFrame(fst_dict).to_csv(
        f"{basefn}_fst.tsv", sep="\t", index=False)

    #----- Generate output table
    pd.DataFrame(time_dict).to_csv(
        f"{basefn}_timeseries.tsv", sep="\t", index=False)

    #----- Population size trajectory (Experimental!)
    if not args.pophist:
        return

    time, s = set_list[0]
    grid_slices = s.area_slices_grid(0.1)
    s1 = grid_slices[0]
    s2 = grid_slices[-1]

    print("Getting population size trajectories...")
    fig = plt.figure(tight_layout=True, figsize=(7, 3))
    ax = fig.add_subplot(121)
    pops, bins = s1.mean_coalr()
    plot_pops(pops, bins, ax, "Within")
    ax = fig.add_subplot(122)
    pops, bins = s1.mean_coalr_across(s2)
    plot_pops(pops, bins, ax, "Across")
    plt.savefig(f"{basefn}_pops.pdf")
    plt.close()


#---------- ENTRY POINT ----------#
if __name__ == "__main__":
    main()

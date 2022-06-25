from dataclasses import dataclass
import numpy as np
import tskit
import msprime
import pyslim
from datargs import parse

from tspipe import *

# ------------------------- ------------------------- #
@dataclass
class DefaultArgsClass(Arguments):
    BaseFn: str = "./output/TMP-FILE"
    SlimSrc: str = "./default_fwd.slim"
    SlimBin: str = "./SLiM/bin/slim"

    N: int = 1000
    L: int = int(1e8)
    FractionFunc: float = 0.1
    MutRate: float = 1e-8
    RecombRate: float = 1e-8
    SigmaA: float = 0.1
    PhenoBurn: int = 100

    GenSwitchK: int = 2000
    GenSwitchEnv: int = 5000
    StopGen: int = 4000

    SampleSize: int = 1000
    ReduceStepK: float = 0.001
    ExtinctMode: str = "step"
    LowDensityBenefit: bool = False
    AbsoluteCompetition: bool = True
    WidthFitnessFunc: float = 5.0

    sDisp: float = 0.01
    sComp: float = 0.05
    sMate: float = 0.05
    MaxAge: int = 10
    FertAge: int = 3
    Fertility: float = 0.5
    EnvOpt1: float = 0.0
    EnvOpt2: float = 10.0
    epsilon: float = 1e-8

@dataclass
class MyArgsClass(DefaultArgsClass):
    BaseFn: str = "./output/N20k_step"
    N: int = 20000
    sDisp: float = 0.005
    PhenoBurn: int = 1000
    GenSwitchK: int = 5000
    GenSwitchEnv: int = 15000
    StopGen: int = 10000


# ------------------------- ------------------------- #
class MyPrecapSim(PreCapSim):
    @staticmethod
    def _assert_required(args):
        required = [
            "BaseFn",
            "N",
            "L",
            "FractionFunc",
            "MutRate",
            "RecombRate",
            "SigmaA",
        ]
        for argument in required:
            assert hasattr(args, argument)

    def _base_sim(self, args):
        Ne = args.N  # Revisit this based on Generation Time!

        demog_model = msprime.Demography()
        demog_model.add_population(initial_size=Ne)
        ts = msprime.sim_ancestry(
            samples=args.N,
            demography=demog_model,
            sequence_length=args.L,
            recombination_rate=args.RecombRate,
        )

        ts = pyslim.annotate_defaults(ts, model_type="nonWF", slim_generation=1)

        ts = msprime.sim_mutations(
            ts,
            rate=(args.MutRate * args.FractionFunc),
            keep=True,
            model=msprime.SLiMMutationModel(type=1),
        )
        return ts

    def _annotate_muts(self, args, ts):
        tables = ts.dump_tables()
        tables.mutations.clear()
        mut_map = {}
        for m in ts.mutations():
            md_list = m.metadata["mutation_list"]
            slim_ids = m.derived_state.split(",")
            assert len(slim_ids) == len(md_list)
            for sid, md in zip(slim_ids, md_list):
                if sid not in mut_map:
                    mut_map[sid] = np.random.normal(scale=args.SigmaA)  # type: ignore
                md["selection_coeff"] = mut_map[sid]
            _ = tables.mutations.add_row(
                site=m.site,
                node=m.node,
                time=m.time,
                derived_state=m.derived_state,
                parent=m.parent,
                metadata={"mutation_list": md_list},
            )
        assert tables.mutations.num_rows == ts.num_mutations
        return tables.tree_sequence()

    def _annotate_inds(self, args, ts):
        tables = ts.dump_tables()
        individual_metadata = [ind.metadata for ind in tables.individuals]
        for md in individual_metadata:
            md["age"] = np.random.choice(2)  # type: ignore # Revisit!
        ims = tables.individuals.metadata_schema
        tables.individuals.packset_metadata(
            [ims.validate_and_encode_row(md) for md in individual_metadata]
        )
        return tables.tree_sequence()

    def _make_ts(self, args):
        ts = self._base_sim(args)
        ts = self._annotate_muts(args, ts)
        ts = self._annotate_inds(args, ts)
        return ts


class MyNeutralMut(AddNeutralMut):
    @staticmethod
    def _assert_required(args):
        required = [
            "FractionFunc",
            "MutRate",
            "InputTreeSeq",
        ]
        for argument in required:
            assert hasattr(args, argument)

    def _make_ts(self, args):
        ts = tskit.load(args.InputTreeSeq)
        ts = pyslim.SlimTreeSequence(
            msprime.sim_mutations(
                ts,
                rate=args.MutRate * (1 - args.FractionFunc),
                model=msprime.SLiMMutationModel(type=0),
                keep=True,
            )
        )
        return ts.simplify()


# ------------------------- ------------------------- #
if __name__ == "__main__":
    args = parse(DefaultArgsClass)
#    args = parse(MyArgsClass)
    print(f"Running with args: {args}")
    sim = Simulation(
        args, precap=MyPrecapSim(), fwdsim=FwdSim(), neutral=MyNeutralMut()
    )
    tsfp = sim.run()

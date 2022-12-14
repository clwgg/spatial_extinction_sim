from typing import Union, List, Optional
from dataclasses import dataclass
from datargs import parse, argsclass
import numpy as np
import tskit
import msprime
import pyslim

from tspipe import *

# ------------------------- ------------------------- #
@dataclass
class BaseArgs(Arguments):
    SlimSrc: str = "./slim/spatial_extinction.slim"
    SlimBin: str = "./SLiM/bin/slim"

    N: int = 1000
    L: int = int(1e8)
    FractionFunc: float = 0.1
    MutRate: float = 1e-8
    RecombRate: float = 1e-8
    SigmaA: float = 0.1
    PhenoBurn: int = 100

    GenStartShrink: int = 2000
    GenStopShrink: int = 5000
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
    GenTime: float = 4.5 # NOTE: Important to adjust - emergent property of fwd sim!

    EnvOpt1: float = 0.0
    EnvOpt2: float = 10.0
    epsilon: float = 1e-8

    RememberGen: Optional[List[int]] = None

    def __post_init__(self):
        if not self.RememberGen:
            self.RememberGen = self._get_RememberGen()

    @staticmethod
    def _seq(start, end, step):
        """ Emulate SLiM `seq()` behaviour """
        return list(range(start, end+1, step))

    def _get_RememberGen(self) -> List[int]:
        """ NOTE:
        this somewhat hackily defines at which generations to take temporal
        samples, relative to `GenStartShrink` and `ReduceStepK`. currently
        hacked together for `ExtinctMode=="step"`.
        """
        one_over_step = int(1/self.ReduceStepK)
        return (
            [self.GenStartShrink -1] +
            [self.GenStartShrink + i for i in range(1, one_over_step, 100)] +
            self._seq(-79+self.GenStartShrink+one_over_step,
                      self.GenStartShrink+one_over_step, 20) +
            self._seq(-14+self.GenStartShrink+one_over_step,
                      self.GenStartShrink+one_over_step, 2)
        )

@dataclass
class Testing(BaseArgs):
    BaseFn: str = "./output/extinct"

@dataclass
class ExtinctHalfTest(BaseArgs):
    BaseFn: str = "./output/extinct_half"
    FractionFunc: float = 0.0
    ReduceStepK: float = 0.5
    GenStartShrink: int = 2000
    GenStopShrink: int = 2001
    GenSwitchEnv: int = 5000
    StopGen: int = 3000

    def _get_RememberGen(self) -> List[int]:
        return (
            [self.GenStartShrink, self.GenStopShrink] +
            self._seq(self.GenStartShrink + 5,
                      self.GenStartShrink + 100, 5)
        )

@dataclass
class Production(BaseArgs):
    BaseFn: str = "./output/extinct_N20k"
    N: int = 20000
    sDisp: float = 0.005
    PhenoBurn: int = 1000

    GenStartShrink: int = 5000
    GenStopShrink: int = 15000
    GenSwitchEnv: int = 15000
    StopGen: int = 10000

@argsclass
class ArgsDriver:
    model: Union[Testing, Production, ExtinctHalfTest]

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
            "GenTime",
        ]
        for argument in required:
            assert hasattr(args, argument)

    def _base_sim(self, args):
        Ne = args.N * args.GenTime
        MutRate = args.MutRate / args.GenTime
        RecombRate = args.RecombRate / args.GenTime

        demog_model = msprime.Demography()
        demog_model.add_population(initial_size=Ne)
        ts = msprime.sim_ancestry(
            samples=args.N,
            demography=demog_model,
            sequence_length=args.L,
            recombination_rate=RecombRate,
        )

        ts = pyslim.annotate(ts, model_type="nonWF", tick=1, annotate_mutations=True)

        ts = msprime.sim_mutations(
            ts,
            rate=(MutRate * args.FractionFunc),
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
            "GenTime",
            "InputTreeSeq",
        ]
        for argument in required:
            assert hasattr(args, argument)

    def _make_ts(self, args):
        MutRate = args.MutRate / args.GenTime

        ts = tskit.load(args.InputTreeSeq)
        ts = msprime.sim_mutations(
            ts,
            rate=MutRate * (1 - args.FractionFunc),
            model=msprime.SLiMMutationModel(type=0),
            keep=True,
        )
        return ts.simplify()


# ------------------------- ------------------------- #
if __name__ == "__main__":
    args = parse(ArgsDriver)
    print(f"Running model with args: {args.model}")
    sim = Simulation(
        args.model, precap=MyPrecapSim(), fwdsim=FwdSim(), neutral=MyNeutralMut()
    )
    tsfp = sim.run()

import pprint
import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess
import numpy as np
import tskit
import msprime
import pyslim
from datargs import parse


# ------------------------- ------------------------- #
"""
NOTE: incomplete TODO list
- help messages for CLI args
- error checking in Simulation.run()
    - check if previous output exists
    - make precap and recap mutually exclusive
- add proper logging capabilities
- live output from SLiM / subprocess ?
- integrate better with analysis pipeline
"""


# ------------------------- ------------------------- #
@dataclass
class Arguments:
    BaseFn: str

    def values(self):
        return dataclasses.asdict(self)

    def run(self):
        tsfp = f"{self.BaseFn}.args.trees"
        ts_metadata = {}
        ts_metadata["Arguments"] = self.values()
        tables = tskit.tables.TableCollection(1)
        tables.build_index()
        tables.metadata_schema = tskit.MetadataSchema.permissive_json()
        tables.metadata = ts_metadata
        tables.dump(tsfp)
        return tsfp


class ClassifyDict:
    def __init__(self, input):
        self.arguments = {}
        for key, value in input.items():
            self.arguments[key] = value

    def __getattr__(self, key):
        return self.arguments[key]

    def __repr__(self):
        return f"\nClassifiedDict:\n{pprint.pformat(self.arguments)}"

    def values(self):
        return self.arguments


# ------------------------- ------------------------- #
class SimComponent(ABC):
    def __init__(self):
        pass

    def _get_args(self, tsfp):
        args_dict = tskit.load(
            tsfp, skip_tables=True, skip_reference_sequence=True
        ).metadata["Arguments"]
        args_dict["InputTreeSeq"] = tsfp
        return ClassifyDict(args_dict)

    @abstractmethod
    def run(self, tsfp):
        pass


class PySimComponent(SimComponent):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def _assert_required(args):
        pass

    @abstractmethod
    def _get_output_tsfp(self, basefn):
        pass

    @abstractmethod
    def _make_ts(self, args):
        pass

    @staticmethod
    def _update_ts_args(ts, args):
        tables = ts.dump_tables()
        ts_metadata = tables.metadata
        ts_metadata["Arguments"] = args.values()
        tables.metadata = ts_metadata
        return tables.tree_sequence()

    def run(self, tsfp):
        args = self._get_args(tsfp)
        self._assert_required(args)
        tsfp = self._get_output_tsfp(args.BaseFn)
        ts = self._make_ts(args)
        ts = self._update_ts_args(ts, args)
        ts.dump(tsfp)
        return tsfp


class PreCapSim(PySimComponent):
    def __init__(self):
        super().__init__()

    def _get_output_tsfp(self, basefn):
        return f"{basefn}.precap.trees"


class Recapitation(PySimComponent):
    def __init__(self):
        super().__init__()

    def _get_output_tsfp(self, basefn):
        return f"{basefn}.recap.trees"


class AddNeutralMut(PySimComponent):
    def __init__(self):
        super().__init__()

    def _get_output_tsfp(self, basefn):
        return f"{basefn}.finished.trees"


# ------------------------- ------------------------- #
class FwdSim(SimComponent):
    def __init__(self):
        super().__init__()

    def _get_output_tsfp(self, basefn):
        return f"{basefn}.fwd.trees"

    @staticmethod
    def _update_ts_args(ts):
        tables = ts.dump_tables()
        ts_metadata = tables.metadata
        args = ts_metadata["SLiM"]["user_metadata"]
        args = {
            key: (value[0] if len(value) == 1 else value) for key, value in args.items()
        }
        ts_metadata["Arguments"] = args
        ts_metadata["SLiM"]["user_metadata"] = ""
        tables.metadata = ts_metadata
        return tables.tree_sequence()

    def run(self, tsfp):
        args = self._get_args(tsfp)
        SlimArgs = (args.SlimBin, "-x", "-d", f"InputTreeSeq='{tsfp}'", args.SlimSrc)
        SlimRun = subprocess.run(SlimArgs, capture_output=True)
        SlimOut = SlimRun.stdout.decode().split("\n")
        SlimErr = SlimRun.stderr.decode().split("\n")
        print("\n\n#----- SLiM Output -----#")
        print("\n".join(SlimOut))
        print("#----- SLiM Errors -----#")
        print("\n".join(SlimErr))
        tsfp = self._get_output_tsfp(args.BaseFn)
        ts = tskit.load(tsfp)
        ts = self._update_ts_args(ts)
        ts.dump(tsfp)
        return tsfp


# ------------------------- ------------------------- #
class Simulation:
    def __init__(
        self,
        args,
        precap=None,
        fwdsim=None,
        recap=None,
        neutral=None,
    ):
        self.args = args
        self.precap = precap
        self.fwdsim = fwdsim
        self.recap = recap
        self.neutral = neutral

    def run(self):
        tsfp = self.args.run()
        if self.precap is not None:
            tsfp = self.precap.run(tsfp)
        if self.fwdsim is not None:
            tsfp = self.fwdsim.run(tsfp)
        if self.recap is not None:
            tsfp = self.recap.run(tsfp)
        if self.neutral is not None:
            tsfp = self.neutral.run(tsfp)
        return tsfp


# --------------------------------------------------- #
# ----------------------       ---------------------- #
# --------------------------------------------------- #


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

import pprint
import dataclasses
from dataclasses import dataclass
from abc import ABC, abstractmethod
import subprocess
import tskit


# ------------------------- ------------------------- #
"""
NOTE: incomplete TODO list
- help messages for CLI args
- error checking in Simulation.run()
    - check if previous output exists
    - make precap and recap mutually exclusive
- add proper logging capabilities
- live output from SLiM / subprocess ?
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


from dataclasses import dataclass
from datargs import parse

from tspipe import Simulation, FwdSim
from test_extinction import TestingArgs, MyPrecapSim, MyNeutralMut

@dataclass
class ProductionArgs(TestingArgs):
    BaseFn: str = "./output/N20k_step"
    N: int = 20000
    sDisp: float = 0.005
    PhenoBurn: int = 1000
    GenSwitchK: int = 5000
    GenSwitchEnv: int = 15000
    StopGen: int = 10000

if __name__ == "__main__":
    args = parse(ProductionArgs)
    print(f"Running with args: {args}")
    sim = Simulation(
        args, precap=MyPrecapSim(), fwdsim=FwdSim(), neutral=MyNeutralMut()
    )
    tsfp = sim.run()

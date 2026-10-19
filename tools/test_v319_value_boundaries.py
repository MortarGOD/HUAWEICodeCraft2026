"""Run existing independent Value math/boundary suite on release source."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from build_v319_candidate import candidate_source
import test_value_metadata as suite


def configured(*unused):
    module=load_solution(ROOT/'solution/solution_v318.py')
    exec(compile(candidate_source(),'v319-boundary','exec'),module.__dict__)
    return module


if __name__=='__main__':
    suite.configured=configured
    suite.main()

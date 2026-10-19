"""Build standalone v362 from v324 plus fast current-Query K translation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = (ROOT / "solution/solution_v324.py").read_text()
    core = (ROOT / "tools/attention_current_query_fast_core.py").read_text()
    source = base + "\n\n" + core.replace(
        '"""Use the current test Query as the deployed K-translation probe at parent cost."""',
        '"""v362: current-test-Query-driven K translation at the v324 candidate count.\n\nLinear is unchanged and calibration calls retain the v324 path.\n"""',
        1,
    )
    compile(source, "solution_v362.py", "exec")
    (ROOT / "solution/solution_v362.py").write_text(source)


if __name__ == "__main__":
    main()

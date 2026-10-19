"""Build standalone v363 from v324 plus current-input Q/K compensation."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = (ROOT / "solution/solution_v324.py").read_text()
    core = (ROOT / "tools/attention_current_qk_comp_core.py").read_text()
    source = base + "\n\n" + core.replace(
        '"""Current-input blockwise Q-error compensation applied to translated K."""',
        '"""v363: current-input blockwise Q-error compensation in translated K.\n\nLinear and calibration retain the v324 implementation.\n"""',
        1,
    )
    compile(source, "solution_v363.py", "exec")
    (ROOT / "solution/solution_v363.py").write_text(source)


if __name__ == "__main__":
    main()

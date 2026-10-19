"""Build the standalone v360 submission from v324 plus sparse order rescue."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = (ROOT / "solution/solution_v324.py").read_text()
    base = base.replace(
        '"""v324: Input-Gram residual updates for bounded up-projection Linear.\n'
        "Attention and the fixed Linear decoder retain v322. Objective is unchanged\n"
        "in exact arithmetic; FP32 update order can alter near-tie decisions.\n"
        '"""',
        '"""v360: sparse calibration-order rescue for outlier Attention groups.\n'
        "Linear and all selected-state dynamic quantizers retain v324 behavior.\n"
        '"""',
        1,
    )
    core = (ROOT / "tools/attention_order_rescue_core.py").read_text()
    source = base.rstrip() + "\n\n\n" + core.rstrip() + "\n"
    compile(source, "solution_v360.py", "exec")
    (ROOT / "solution/solution_v360.py").write_text(source)


if __name__ == "__main__":
    main()

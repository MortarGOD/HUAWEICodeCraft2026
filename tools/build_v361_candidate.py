"""Build standalone v361 from v324 plus current-Query K selection."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    base = (ROOT / "solution/solution_v324.py").read_text()
    core = (ROOT / "tools/attention_current_query_core.py").read_text()
    source = base + "\n\n" + core.replace(
        '"""Test-time current-Query K-origin selection for sequential Attention calls."""',
        '"""v361: current-Query softmax-aware K-origin selection.\n\nLinear and the v324 fallback remain unchanged.\n"""',
        1,
    )
    compile(source, "solution_v361.py", "exec")
    (ROOT / "solution/solution_v361.py").write_text(source)


if __name__ == "__main__":
    main()

"""Full proxy and official-interface validation for v363."""
import ast
import contextlib
import io
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution


BANNED = {"os", "pathlib", "io", "tempfile", "shutil", "subprocess", "socket", "ctypes", "pickle", "marshal", "importlib", "zipimport", "pkgutil", "runpy", "mmap", "resource", "pty", "tty", "sys", "builtins"}
PARENT_SCORE = 81.56157396631853


def main():
    path = ROOT / "solution/solution_v363.py"
    tree = ast.parse(path.read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not imports.intersection(BANNED)
    module = load_solution(path)
    bank = torch.load(ROOT / "local_fitted_models_attention_feedback_v2/attn.pt", weights_only=False)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _ = evaluate_attention(module, bank)
    elapsed = time.perf_counter() - started
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "solution"
        target.mkdir()
        shutil.copy2(path, target / "solution.py")
        result = subprocess.run(["/opt/miniconda3/envs/codecraft/bin/python", str(ROOT / "self_check.py"), "--solution_dir", str(target)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print({"score": sum(scores), "delta": sum(scores) - PARENT_SCORE, "elapsed": elapsed, "debug": module.QKC_DEBUG, "official": result.stdout.strip()}, flush=True)


if __name__ == "__main__":
    main()

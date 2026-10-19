"""Validate and score the standalone fast current-Query v362 probe."""
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
from self_check import validate_attention_calibration_result, validate_hif4_params


BANNED = {"os", "pathlib", "io", "tempfile", "shutil", "subprocess", "socket", "ctypes", "pickle", "marshal", "importlib", "zipimport", "pkgutil", "runpy", "mmap", "resource", "pty", "tty", "sys", "builtins"}
PARENT_SCORE = 81.56157396631853


def main():
    path = ROOT / "solution/solution_v362.py"
    tree = ast.parse(path.read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    assert not imports.intersection(BANNED)
    module = load_solution(path)
    bank = torch.load(ROOT / "local_fitted_models_attention_feedback_v2/attn.pt", weights_only=False)
    started = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _ = evaluate_attention(module, bank)
    elapsed = time.perf_counter() - started
    checks = 0
    module = load_solution(path)
    for group in (bank[0], bank[13], bank[47], bank[49]):
        dims = [group[key] for key in ("q_num_heads", "kv_num_heads", "head_dim")]
        state = module.hif4_calibration_attention(group["calib"], *dims)
        assert not validate_attention_calibration_result(state, "v362")
        checks += 1
        for sample in group["test"]:
            for role, heads in (("q", dims[0]), ("k", dims[1]), ("v", dims[1])):
                params = getattr(module, f"hif4_dynamic_quantize_{role}")(*sample[role], heads, dims[2], state[f"{role}_state"])
                assert not validate_hif4_params(params, sample[role][0].shape, f"v362-{role}")
                checks += 1
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "solution"
        target.mkdir()
        shutil.copy2(path, target / "solution.py")
        result = subprocess.run(["/opt/miniconda3/envs/codecraft/bin/python", str(ROOT / "self_check.py"), "--solution_dir", str(target)], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    print({"score": sum(scores), "delta": sum(scores) - PARENT_SCORE, "elapsed": elapsed, "debug": module.CQF_DEBUG, "checks": checks, "official": result.stdout.strip()}, flush=True)


if __name__ == "__main__":
    main()

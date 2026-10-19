"""Static, interface, and official mini checks for the actual v360 file."""
import ast
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from probe_v319_value import ROOT, load_solution
from self_check import validate_attention_calibration_result, validate_hif4_params


BANNED = {
    "os", "pathlib", "io", "tempfile", "shutil", "subprocess", "socket",
    "ctypes", "pickle", "marshal", "importlib", "zipimport", "pkgutil",
    "runpy", "mmap", "resource", "pty", "tty", "sys", "builtins",
}


def main():
    path = ROOT / "solution/solution_v360.py"
    source = path.read_text()
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not imports.intersection(BANNED)
    module = load_solution(path)
    bank = torch.load(
        ROOT / "local_fitted_models_attention_feedback_v2/attn.pt",
        weights_only=False,
    )
    checks = 0
    for group_index in (0, 1, 47, 49):
        group = bank[group_index]
        dims = [group[k] for k in ("q_num_heads", "kv_num_heads", "head_dim")]
        state = module.hif4_calibration_attention(group["calib"], *dims)
        assert not validate_attention_calibration_result(state, "v360")
        checks += 1
        for sample in group["test"]:
            for role, heads in (("q", dims[0]), ("k", dims[1]), ("v", dims[1])):
                function = getattr(module, f"hif4_dynamic_quantize_{role}")
                params = function(
                    *sample[role], heads, dims[2], state[f"{role}_state"]
                )
                assert not validate_hif4_params(
                    params, sample[role][0].shape, f"v360-{role}"
                )
                checks += 1
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "solution"
        target.mkdir()
        shutil.copy2(path, target / "solution.py")
        result = subprocess.run(
            [
                "/opt/miniconda3/envs/codecraft/bin/python",
                str(ROOT / "self_check.py"),
                "--solution_dir",
                str(target),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
    assert result.returncode == 0, result.stdout + result.stderr
    print({"checks": checks, "official": result.stdout.strip()})


if __name__ == "__main__":
    main()

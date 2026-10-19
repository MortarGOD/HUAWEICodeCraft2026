"""Compare v307 Attention to v302, including forced GPTQ on real inputs.

Forcing the branch is a correctness test, never a score/performance estimate.
"""
import argparse
import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load(version):
    spec = importlib.util.spec_from_file_location(
        f"audit_v{version}", ROOT / f"solution/solution_v{version}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        # v307's constant zero seed is the only extra state metadata.
        lk, rk = set(left), set(right)
        if lk ^ rk == {"rseed"}:
            assert left.get("rseed", 0) == right.get("rseed", 0) == 0
            lk.discard("rseed"); rk.discard("rseed")
        assert lk == rk
        for key in lk:
            equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", required=True)
    parser.add_argument("--reference", type=int, default=302)
    parser.add_argument("--candidate", type=int, default=307)
    args = parser.parse_args()
    data = torch.load(Path(args.datasets_dir) / "attn.pt", weights_only=False)
    groups = data if isinstance(data, list) else data["groups"]
    ref, opt = load(args.reference), load(args.candidate)
    captured = []
    original = ref._upper_cholesky_inv

    def capture(h):
        result = original(h)
        captured.append(result)
        return result

    ref._upper_cholesky_inv = capture
    normal = forced = 0
    for idx, group in enumerate(groups):
        qh, kvh, dh = (group[k] for k in
                      ("q_num_heads", "kv_num_heads", "head_dim"))
        captured.clear()
        a = ref.hif4_calibration_attention(group["calib"], qh, kvh, dh)
        b = opt.hif4_calibration_attention(group["calib"], qh, kvh, dh)
        equal(a, b)
        for sample in group["test"]:
            for role in ("q", "k", "v"):
                heads = qh if role == "q" else kvh
                fa = getattr(ref, f"hif4_dynamic_quantize_{role}")
                fb = getattr(opt, f"hif4_dynamic_quantize_{role}")
                sa, sb = a[f"{role}_state"], b[f"{role}_state"]
                equal(fa(*sample[role], heads, dh, sa),
                      fb(*sample[role], heads, dh, sb))
                normal += 1
                if role in ("q", "k") and len(captured) == 2:
                    u = captured[0 if role == "q" else 1]
                    if u is not None:
                        sa = dict(sa, gq=1, u=u)
                        sb = dict(sb, gq=1, u=u)
                        equal(fa(*sample[role], heads, dh, sa),
                              fb(*sample[role], heads, dh, sb))
                        forced += 1
        if (idx + 1) % 10 == 0:
            print("groups", idx + 1, "normal", normal, "forced", forced, flush=True)
    print("PASS", "normal", normal, "forced GPTQ", forced)


if __name__ == "__main__":
    main()

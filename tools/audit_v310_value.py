"""Verify V surrogate monotonicity and Q/K shortcut on real bank inputs."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from probe_persistent_covariance import load
from audit_v307_attention import equal

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--reference", type=int, default=309)
    p.add_argument("--candidate", type=int, default=310)
    p.add_argument("--prototype-polish", action="store_true")
    args = p.parse_args()
    groups = torch.load(ROOT / "local_fitted_models_covariance_v1/attn.pt", weights_only=False)
    groups = groups[:5] + groups[25:30]
    a, b = load(args.reference), load(args.candidate)
    prototype = None
    if args.prototype_polish:
        from probe_value_solver import configured
        prototype = configured("polish8")
    count = short = 0
    means, mses = [], []
    for g in groups:
        qh, kh, d = (g[k] for k in ("q_num_heads", "kv_num_heads", "head_dim"))
        sa = a.hif4_calibration_attention(g["calib"], qh, kh, d)
        sb = b.hif4_calibration_attention(g["calib"], qh, kh, d)
        equal(sa, sb)
        for sample in g["test"]:
            results = []
            for module, states in ((a,sa),(b,sb)):
                results.append({r:getattr(module,"hif4_dynamic_quantize_"+r)(
                    *sample[r], qh if r=="q" else kh, d, states[r+"_state"]) for r in "qkv"})
            x = b.dequantize_nvfp4(*sample["v"]).float()
            if prototype is not None:
                expected = prototype.hif4_dynamic_quantize_v(*sample["v"],kh,d,sb["v_state"])
                equal(expected,results[1]["v"])
            T = x.shape[0]
            e0 = x-a._deq_params(results[0]["v"])
            e1 = x-b._deq_params(results[1]["v"])
            c0 = e0.mean(0).square() + e0.square().mean(0)/T
            c1 = e1.mean(0).square() + e1.square().mean(0)/T
            assert torch.all(c1 <= c0 + 1e-6*c0.abs().clamp_min(1e-12))
            for key in ("scale_factor","scale_lv2","scale_lv3"):
                equal(results[0]["v"][key],results[1]["v"][key])
            if T == 1:
                equal(results[0]["v"],results[1]["v"])
                value = b._deq_params(results[1]["v"])
                before = b._attention_out(a._deq_params(results[0]["q"]),a._deq_params(results[0]["k"]),value,qh,kh,d)
                after = b._attention_out(b._deq_params(results[1]["q"]),b._deq_params(results[1]["k"]),value,qh,kh,d)
                assert torch.equal(before,after)
                short += 1
            else:
                equal(results[0]["q"],results[1]["q"])
                equal(results[0]["k"],results[1]["k"])
                means.append(float(e1.mean(0).square().sum()/e0.mean(0).square().sum().clamp_min(1e-30)))
                mses.append(float(e1.square().sum()/e0.square().sum().clamp_min(1e-30)))
            count += 1
    out={"reference":args.reference,"candidate":args.candidate,
         "prototype_polish_equal":args.prototype_polish,
         "cases":count,"calibration_groups_equal":len(groups),"t1_attention_equivalence_cases":short,
         "v_scales_unchanged":True,"nontrivial_qk_equal":True,"v_surrogate_nonincreasing":True,
         "mean_error_ratio_median":statistics.median(means),"elementwise_mse_ratio_median":statistics.median(mses)}
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")
    print(out,flush=True)


if __name__ == "__main__":
    main()

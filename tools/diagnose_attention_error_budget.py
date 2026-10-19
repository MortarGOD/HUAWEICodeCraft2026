"""Oracle component diagnostics, not deployable candidates or online estimates."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import gqa_attention, mse, score
from hif4_quant import dequantize_nvfp4, quantize_to_hif4
from evaluate_local import dequantize_hif4
from probe_persistent_covariance import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--groups-per-model", type=int, default=5)
    args = parser.parse_args()
    groups = torch.load(ROOT / "local_fitted_models_covariance_v1/attn.pt", weights_only=False)
    m = load(309)
    counts, records = {}, []
    for g in groups:
        model = g["source_model"]
        count = counts.get(model, 0)
        counts[model] = count + 1
        if count >= args.groups_per_model:
            continue
        qh, kh, dh = (g[k] for k in ("q_num_heads", "kv_num_heads", "head_dim"))
        st = m.hif4_calibration_attention(g["calib"], qh, kh, dh)
        for sample in g["test"]:
            original, base, player = {}, {}, {}
            for role in "qkv":
                x = dequantize_nvfp4(*sample[role]).float()
                original[role] = x
                base[role] = dequantize_hif4(quantize_to_hif4(x), x.shape)
                params = getattr(m,"hif4_dynamic_quantize_"+role)(*sample[role], qh if role=="q" else kh, dh, st[role+"_state"])
                player[role] = dequantize_hif4(params, x.shape)
            def attention(tensors):
                return gqa_attention(*(tensors[r] for r in "qkv"), qh, kh, dh)
            reference = attention(original)
            bm = mse(reference, attention(base))
            qk_oracle = dict(original, v=player["v"])
            v_oracle = dict(player, v=original["v"])
            records.append({"model": model, "rows": original["q"].shape[0],
                            "player": score(bm,mse(reference,attention(player))),
                            "perfect_qk": score(bm,mse(reference,attention(qk_oracle))),
                            "perfect_v": score(bm,mse(reference,attention(v_oracle)))})
    out={"warning":"oracle diagnostics only; component gains are not additive or deployable", "cases":records}
    out["totals"]={k:sum(r[k] for r in records) for k in ("player","perfect_qk","perfect_v")}
    Path(args.output).write_text(json.dumps(out,indent=2)+"\n")
    print(out["totals"],flush=True)


if __name__ == "__main__":
    main()

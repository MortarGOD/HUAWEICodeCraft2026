"""Select one complete v324 Attention state from cyclic calibration holdouts."""
import contextlib
import copy
import io
import json
import time

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution
from self_check import validate_attention_calibration_result


def state_losses(module, samples, state, qh, kh, d):
    losses = []
    for sample in samples:
        rows = sample["q"][0].shape[0]
        index = torch.linspace(0, rows - 1, min(rows, 32)).long()
        raw = [
            module.dequantize_nvfp4(
                sample["q"][0][index], sample["q"][1][index]
            ).float()
        ]
        raw.extend(module.dequantize_nvfp4(*sample[role]).float() for role in "kv")
        target = module._joint_output(*raw, qh, kh, d, index, query_subset=True)
        query = module._deq_params(
            module.hif4_dynamic_quantize_q(
                sample["q"][0][index],
                sample["q"][1][index],
                qh,
                d,
                state["q_state"],
            )
        )
        key = module._deq_params(
            module.hif4_dynamic_quantize_k(*sample["k"], kh, d, state["k_state"])
        )
        value = module._deq_params(
            module.hif4_dynamic_quantize_v(*sample["v"], kh, d, state["v_state"])
        )
        output = module._joint_output(
            query, key, value, qh, kh, d, index, query_subset=True
        )
        losses.append(float((output - target).square().mean()))
    return losses


def main():
    torch.set_num_threads(4)
    bank = torch.load(
        ROOT / "local_fitted_models_attention_feedback_v2/attn.pt",
        weights_only=False,
    )
    module = load_solution(ROOT / "solution/solution_v324.py")
    rows = []
    for group_index in (0, 1, 2, 5, 14, 19, 25, 35, 47, 49):
        group = bank[group_index]
        dims = [group[k] for k in ("q_num_heads", "kv_num_heads", "head_dim")]
        start = time.perf_counter()
        states = []
        for shift in range(5):
            ordered = group["calib"][shift:] + group["calib"][:shift]
            state = module.hif4_calibration_attention(ordered, *dims)
            assert not validate_attention_calibration_result(state, "order")
            states.append(state)
        losses = [state_losses(module, group["calib"], state, *dims) for state in states]
        totals = [sum(values) for values in losses]
        chosen = min(range(5), key=totals.__getitem__)
        calibration_seconds = time.perf_counter() - start
        scores = {}
        for name, state in (("parent", states[0]), ("selected", states[chosen])):
            module.hif4_calibration_attention = (
                lambda *unused, _state=state: copy.deepcopy(_state)
            )
            with contextlib.redirect_stdout(io.StringIO()):
                scores[name], _ = evaluate_attention(module, [group])
        row = dict(
            group=group_index,
            chosen=chosen,
            calibration_losses=losses,
            totals=totals,
            baseline=scores["parent"],
            scores=scores["selected"],
            delta=sum(scores["selected"]) - sum(scores["parent"]),
            calibration_seconds=calibration_seconds,
        )
        rows.append(row)
        print(row, flush=True)
        module = load_solution(ROOT / "solution/solution_v324.py")
    report = dict(
        complete=True,
        rows=rows,
        delta=sum(row["delta"] for row in rows),
        changed=sum(row["chosen"] != 0 for row in rows),
        warning="Modeled activations are not online score evidence.",
    )
    (ROOT / "docs/attention-order-ensemble-screen.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print({"delta": report["delta"], "changed": report["changed"]}, flush=True)


if __name__ == "__main__":
    main()

"""Screen current-Query softmax-KL refinement on held-out modeled activations."""
import contextlib
import copy
import io
import json
import time

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution
from self_check import validate_attention_calibration_result, validate_hif4_params


def main():
    torch.set_num_threads(4)
    bank = torch.load(
        ROOT / "local_fitted_models_attention_feedback_v2/attn.pt",
        weights_only=False,
    )
    base = load_solution(ROOT / "solution/solution_v324.py")
    trial = load_solution(ROOT / "solution/solution_v324.py")
    source = (ROOT / "tools/attention_dynamic_query_softmax_core.py").read_text()
    exec(compile(source, "attention_dynamic_query_softmax_core.py", "exec"), trial.__dict__)
    rows = []
    for group_index in (0, 1, 2, 5, 14, 19, 25, 35, 47, 49):
        group = bank[group_index]
        dims = [group[k] for k in ("q_num_heads", "kv_num_heads", "head_dim")]
        start = time.perf_counter()
        parent = trial._dq_parent_calibration(group["calib"], *dims)
        state = trial._dq_guard(group["calib"], parent, *dims)
        calibration_seconds = time.perf_counter() - start
        debug = copy.deepcopy(trial.DQ_DEBUG)
        assert not validate_attention_calibration_result(state, "dynamic-query")
        base.hif4_calibration_attention = (
            lambda *unused, _state=parent: copy.deepcopy(_state)
        )
        trial.hif4_calibration_attention = (
            lambda *unused, _state=state: copy.deepcopy(_state)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, _ = evaluate_attention(base, [group])
            scores, _ = evaluate_attention(trial, [group])
        for sample in group["test"]:
            params = trial.hif4_dynamic_quantize_q(
                *sample["q"], dims[0], dims[2], state["q_state"]
            )
            assert not validate_hif4_params(
                params, sample["q"][0].shape, "dynamic-query"
            )
        row = dict(
            group=group_index,
            baseline=baseline,
            scores=scores,
            delta=sum(scores) - sum(baseline),
            calibration_seconds=calibration_seconds,
            debug=debug,
        )
        rows.append(row)
        print(row, flush=True)
    report = dict(
        complete=True,
        rows=rows,
        delta=sum(row["delta"] for row in rows),
        warning="Modeled activations are not online score evidence.",
    )
    (ROOT / "docs/dynamic-query-softmax-refine-screen.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        {
            "delta": report["delta"],
            "accepted": sum(row["debug"]["accepted"] for row in rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

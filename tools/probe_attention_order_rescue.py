"""Screen the sparse two-order rescue against the full five-order oracle study."""
import contextlib
import copy
import io
import json
import time

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution
from self_check import validate_attention_calibration_result


def main():
    torch.set_num_threads(4)
    bank = torch.load(
        ROOT / "local_fitted_models_attention_feedback_v2/attn.pt",
        weights_only=False,
    )
    rows = []
    for group_index in range(len(bank)):
        group = bank[group_index]
        dims = [group[k] for k in ("q_num_heads", "kv_num_heads", "head_dim")]
        base = load_solution(ROOT / "solution/solution_v324.py")
        trial = load_solution(ROOT / "solution/solution_v324.py")
        source = (ROOT / "tools/attention_order_rescue_core.py").read_text()
        exec(compile(source, "attention_order_rescue_core.py", "exec"), trial.__dict__)
        start = time.perf_counter()
        state = trial.hif4_calibration_attention(group["calib"], *dims)
        elapsed = time.perf_counter() - start
        debug = copy.deepcopy(trial.OR_DEBUG)
        assert not validate_attention_calibration_result(state, "order-rescue")
        parent = trial.OR_PARENT
        base.hif4_calibration_attention = (
            lambda *unused, _state=parent: copy.deepcopy(_state)
        )
        trial.hif4_calibration_attention = (
            lambda *unused, _state=state: copy.deepcopy(_state)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            baseline, _ = evaluate_attention(base, [group])
            scores, _ = evaluate_attention(trial, [group])
        row = dict(
            group=group_index,
            baseline=baseline,
            scores=scores,
            delta=sum(scores) - sum(baseline),
            calibration_seconds=elapsed,
            debug=debug,
        )
        rows.append(row)
        print(row, flush=True)
    report = dict(
        complete=True,
        rows=rows,
        delta=sum(row["delta"] for row in rows),
        triggered=sum(row["debug"]["triggered"] for row in rows),
        accepted=sum(row["debug"]["accepted"] for row in rows),
        warning="Modeled activations are not online score evidence.",
    )
    (ROOT / "docs/attention-order-rescue-screen.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()

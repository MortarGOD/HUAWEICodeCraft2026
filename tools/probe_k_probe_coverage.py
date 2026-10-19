"""Screen sequence-balanced calibration-query banks for the deployed K translation."""
import contextlib
import copy
import io

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution


def make_probe(module, samples, qh, kh, d, per_sequence, include_holdout=False):
    probes = []
    selected = samples if include_holdout else samples[:-1]
    for sample in selected:
        rows = sample["q"][0].shape[0]
        index = torch.linspace(0, rows - 1, min(rows, per_sequence)).long()
        query = module.dequantize_nvfp4(
            sample["q"][0][index], sample["q"][1][index]
        ).float().reshape(-1, kh, qh // kh, d)
        probes.append(query.permute(1, 0, 2, 3).reshape(kh, -1, d))
    return torch.cat(probes, 1) / d ** 0.5


def evaluate_state(module, group, state):
    dims = [group[key] for key in ("q_num_heads", "kv_num_heads", "head_dim")]
    module.hif4_calibration_attention = (
        lambda *unused, _state=state: copy.deepcopy(_state)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _ = evaluate_attention(module, [group])
    return scores


def main():
    torch.set_num_threads(4)
    bank = torch.load(
        ROOT / "local_fitted_models_attention_feedback_v2/attn.pt",
        weights_only=False,
    )
    totals = {name: 0.0 for name in ("parent", "all4x8", "all4x16", "all5x8")}
    changed = {name: 0 for name in totals if name != "parent"}
    for group_index, group in enumerate(bank):
        dims = [group[key] for key in ("q_num_heads", "kv_num_heads", "head_dim")]
        qh, kh, d = dims
        calibrator = load_solution(ROOT / "solution/solution_v324.py")
        state = calibrator.hif4_calibration_attention(group["calib"], *dims)
        parent_module = load_solution(ROOT / "solution/solution_v324.py")
        parent_scores = evaluate_state(parent_module, group, state)
        totals["parent"] += sum(parent_scores)
        row = {"group": group_index, "parent": sum(parent_scores)}
        for name, count, holdout in (
            ("all4x8", 8, False),
            ("all4x16", 16, False),
            ("all5x8", 8, True),
        ):
            trial_state = copy.deepcopy(state)
            key_state = trial_state["k_state"]
            if "kt_probe" in key_state:
                key_state["kt_probe"] = make_probe(
                    calibrator, group["calib"], qh, kh, d, count, holdout
                )
                changed[name] += 1
            trial_module = load_solution(ROOT / "solution/solution_v324.py")
            scores = evaluate_state(trial_module, group, trial_state)
            totals[name] += sum(scores)
            row[name] = sum(scores) - sum(parent_scores)
        print(row, flush=True)
    print({"totals": totals, "delta": {k: v - totals["parent"] for k, v in totals.items()}, "changed_groups": changed}, flush=True)


if __name__ == "__main__":
    main()

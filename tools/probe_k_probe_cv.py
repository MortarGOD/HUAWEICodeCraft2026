"""Cross-calibrate equal-size Query banks per KV head for K translation."""
import contextlib
import copy
import io

import torch

from evaluate_local import evaluate_attention
from probe_v319_value import ROOT, load_solution


def query_bank(module, samples, qh, kh, d, indices, count):
    banks = []
    for sample_index in indices:
        sample = samples[sample_index]
        rows = sample["q"][0].shape[0]
        pick = torch.linspace(0, rows - 1, min(rows, count)).long()
        query = module.dequantize_nvfp4(
            sample["q"][0][pick], sample["q"][1][pick]
        ).float().reshape(-1, kh, qh // kh, d)
        banks.append(query.permute(1, 0, 2, 3).reshape(kh, -1, d))
    return torch.cat(banks, 1) / d ** 0.5


def head_losses(module, samples, state, qh, kh, d):
    total = torch.zeros(kh)
    for sample in samples:
        rows = sample["q"][0].shape[0]
        pick = torch.linspace(0, rows - 1, min(rows, 32)).long()
        raw = [module.dequantize_nvfp4(sample["q"][0][pick], sample["q"][1][pick]).float()]
        raw.extend(module.dequantize_nvfp4(*sample[role]).float() for role in "kv")
        target = module._joint_output(*raw, qh, kh, d, pick, query_subset=True)
        query = module._deq_params(module.hif4_dynamic_quantize_q(
            sample["q"][0][pick], sample["q"][1][pick], qh, d, state["q_state"]
        ))
        key = module._deq_params(module.hif4_dynamic_quantize_k(
            *sample["k"], kh, d, state["k_state"]
        ))
        value = module._deq_params(module.hif4_dynamic_quantize_v(
            *sample["v"], kh, d, state["v_state"]
        ))
        output = module._joint_output(query, key, value, qh, kh, d, pick, query_subset=True)
        error = (output - target).square().mean((-1, -2)).reshape(kh, -1).mean(1)
        total += torch.log(error.clamp_min(1e-30))
    return total


def evaluate_state(module, group, state):
    module.hif4_calibration_attention = lambda *unused: copy.deepcopy(state)
    with contextlib.redirect_stdout(io.StringIO()):
        scores, _ = evaluate_attention(module, [group])
    return scores


def main():
    torch.set_num_threads(4)
    bank = torch.load(ROOT / "local_fitted_models_attention_feedback_v2/attn.pt", weights_only=False)
    parent_total = 0.0
    trial_total = 0.0
    changed_groups = 0
    changed_heads = 0
    for group_index, group in enumerate(bank):
        qh, kh, d = [group[key] for key in ("q_num_heads", "kv_num_heads", "head_dim")]
        calibrator = load_solution(ROOT / "solution/solution_v324.py")
        parent = calibrator.hif4_calibration_attention(group["calib"], qh, kh, d)
        key_state = parent["k_state"]
        trial = copy.deepcopy(parent)
        choices = torch.zeros(kh, dtype=torch.long)
        if "kt_probe" in key_state and len(group["calib"]) >= 5:
            banks = [
                key_state["kt_probe"].float(),
                query_bank(calibrator, group["calib"], qh, kh, d, (0, 1), 16),
                query_bank(calibrator, group["calib"], qh, kh, d, (0, 1, 2, 3), 8),
                query_bank(calibrator, group["calib"], qh, kh, d, (1, 2), 16),
            ]
            losses = []
            for probe in banks:
                candidate = copy.deepcopy(parent)
                candidate["k_state"]["kt_probe"] = probe
                losses.append(head_losses(calibrator, group["calib"], candidate, qh, kh, d))
            choices = torch.stack(losses).argmin(0)
            selected = torch.stack(banks)
            trial["k_state"]["kt_probe"] = selected[choices, torch.arange(kh)]
            changed = int((choices != 0).sum())
            changed_heads += changed
            changed_groups += int(changed > 0)
        base_scores = evaluate_state(load_solution(ROOT / "solution/solution_v324.py"), group, parent)
        trial_scores = evaluate_state(load_solution(ROOT / "solution/solution_v324.py"), group, trial)
        parent_total += sum(base_scores)
        trial_total += sum(trial_scores)
        print({"group": group_index, "changed": int((choices != 0).sum()), "delta": sum(trial_scores) - sum(base_scores)}, flush=True)
    print({"parent": parent_total, "trial": trial_total, "delta": trial_total - parent_total, "changed_groups": changed_groups, "changed_heads": changed_heads}, flush=True)


if __name__ == "__main__":
    main()

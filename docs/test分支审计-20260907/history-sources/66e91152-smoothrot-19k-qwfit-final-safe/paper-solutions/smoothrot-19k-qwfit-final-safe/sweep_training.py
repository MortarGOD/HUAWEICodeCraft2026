from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("learned_scale", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load solution")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_linear(module, groups: list, baseline_cases: list[dict]) -> tuple[float, int]:
    total = 0.0
    learned_groups = 0
    case_index = 0
    for group in groups:
        weight_ref = module._dequantize_src(*group["weight"])
        calibrated = module.hif4_calibration_and_quantize_weight(
            *group["weight"], group["calib_activation_list"]
        )
        learned_groups += int(
            calibrated["activation_state"].get("learned_scale", False)
        )
        weight_value = module._dequantize_tgt(
            calibrated["weight_params"], tuple(weight_ref.shape)
        )
        for source in group["test_activation_list"]:
            activation_ref = module._dequantize_src(*source)
            activation_value = module._dequantize_tgt(
                module.hif4_dynamic_quantize_activation(
                    *source, calibrated["activation_state"]
                ),
                tuple(activation_ref.shape),
            )
            reference = module._linear_output(activation_ref, weight_ref)
            output = module._linear_output(activation_value, weight_value)
            mse = float(torch.mean((output - reference).square()).item())
            baseline_mse = float(baseline_cases[case_index]["mse_baseline"])
            total += 1.0 - mse / max(baseline_mse, 1.0e-12)
            case_index += 1
    return total, learned_groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=(1,))
    parser.add_argument("--blends", type=float, nargs="+", default=(1.0,))
    parser.add_argument("--ranks", type=int, nargs="+", default=(1,))
    parser.add_argument("--widths", type=int, nargs="+", default=(16,))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    module = load_solution(root / "solution.py")
    groups = torch.load(args.dataset, map_location="cpu", weights_only=True)
    with args.baseline_report.open("r", encoding="utf-8") as handle:
        baseline_cases = json.load(handle)["linear"]["cases"]

    for rank in args.ranks:
        module._PSOT_MAX_RANK = rank
        for blend in args.blends:
            module._SCALE_INITIAL_BLEND = blend
            for steps in args.steps:
                module._SCALE_LEARNING_STEPS = steps
                for width in args.widths:
                    module._SCALE_GROUP_WIDTH = width
                    for rate in args.rates:
                        module._SCALE_LEARNING_RATE = rate
                        score, learned_groups = score_linear(
                            module, groups, baseline_cases
                        )
                        print(
                            f"rank={rank} blend={blend:.3f} steps={steps} "
                            f"width={width} lr={rate:.6f} "
                            f"linear={score:.12f} "
                            f"learned_groups={learned_groups}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()

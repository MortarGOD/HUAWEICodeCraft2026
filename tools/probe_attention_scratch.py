"""Research-only exact-output allocation/packing experiments for v311."""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import time

import torch

from audit_v307_attention import equal
from probe_persistent_covariance import load

ROOT = Path(__file__).resolve().parents[1]


def polish_scratch(x, values, unit, rounds=8):
    tokens, channels = x.shape
    if tokens <= 1:
        return values
    step = (unit * 0.25).T.contiguous()
    code = torch.round(values.T / step).contiguous()
    error = x.T.contiguous() - code * step
    summed = error.sum(1)
    curvature = 2 * step.square()
    twice_step = 2 * step
    rows = torch.arange(channels)
    gradient = torch.empty_like(error)
    gain = torch.empty_like(error)
    positive = torch.empty_like(error, dtype=torch.bool)
    upper = torch.empty_like(positive)
    lower = torch.empty_like(positive)
    blocked = torch.empty_like(positive)
    for _ in range(rounds):
        torch.add(summed.unsqueeze(1), error, out=gradient)
        torch.abs(gradient, out=gain)
        torch.mul(twice_step, gain, out=gain)
        torch.sub(curvature, gain, out=gain)
        torch.gt(gradient, 0, out=positive)
        torch.ge(code, 7, out=upper)
        torch.le(code, -7, out=lower)
        torch.where(positive, upper, lower, out=blocked)
        gain.masked_fill_(blocked, float("inf"))
        idx = gain.argmin(1)
        active = gain[rows, idx] < 0
        if not bool(active.any()):
            break
        direction = torch.where(gradient[rows, idx] > 0, 1.0, -1.0)
        shift = direction * active
        delta = shift * step[rows, idx]
        code[rows, idx] += shift
        error[rows, idx] -= delta
        summed -= delta
    return (code * step).T.contiguous()


def configured(mode):
    m = load(311)
    if mode in ("scratch", "both"):
        m._polish_value_mean = polish_scratch
    if mode in ("defer", "both"):
        source = inspect.getsource(m._dyn_qk)
        source = source.replace("    values = None\n", "    values = None\n    refined = False\n")
        source = source.replace("            p = _values_to_params(values, p)\n", "")
        source = source.replace("                p = _values_to_params(v1.contiguous(), p)\n",
                                "                p = _values_to_params(v1.contiguous(), p)\n                refined = True\n")
        end = source.rfind("    return p\n")
        source = source[:end] + "    return p if refined or values is None else _values_to_params(values, p)\n"
        exec(compile(source, "<research-deferred-qk-pack>", "exec"), m.__dict__)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--groups-per-model", type=int, default=2)
    args = parser.parse_args()
    bank = torch.load(ROOT / "local_fitted_models_covariance_v1/attn.pt", weights_only=False)
    groups = bank[:args.groups_per_model] + bank[25:25+args.groups_per_model]
    base = configured("base")
    cases = []
    for group in groups:
        qh, kh, d = (group[k] for k in ("q_num_heads", "kv_num_heads", "head_dim"))
        state = base.hif4_calibration_attention(group["calib"], qh, kh, d)
        for sample in group["test"]:
            for role in "qkv":
                a = (*sample[role], qh if role == "q" else kh, d, state[role+"_state"])
                expected = getattr(base, "hif4_dynamic_quantize_"+role)(*a)
                cases.append((role, a, expected))
    report = {"parent":311, "threads":torch.get_num_threads(), "groups":len(groups),
              "cases":len(cases), "results":{},
              "warning":"dynamic-only microbenchmark; calibration and scorer excluded"}
    modules = {mode:configured(mode) for mode in ("base","scratch","defer","both")}
    for mode, module in modules.items():
        for role, a, expected in cases:
            equal(expected, getattr(module,"hif4_dynamic_quantize_"+role)(*a))
        report["results"][mode] = {"equal":True,"runs":[]}
    for repeat in range(args.repeats):
        modes = list(modules) if repeat % 2 == 0 else list(reversed(modules))
        for mode in modes:
            module=modules[mode]
            timing={r:0.0 for r in "qkv"}
            for role,a,_ in cases:
                start=time.perf_counter()
                getattr(module,"hif4_dynamic_quantize_"+role)(*a)
                timing[role]+=time.perf_counter()-start
            report["results"][mode]["runs"].append(timing)
            print(repeat,mode,timing,flush=True)
    for result in report["results"].values():
        result["median_seconds"]={r:statistics.median(run[r] for run in result["runs"]) for r in "qkv"}
    Path(args.output).write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()

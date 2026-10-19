"""Fixed-bank GPTQ/refinement ablations; no per-version data fitting."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load

CONFIGS = {
    "baseline": {},
    "damp03": {"GPTQ_DAMP": 0.03},
    "damp01": {"GPTQ_DAMP": 0.01},
    "damp003": {"GPTQ_DAMP": 0.003},
    "ref32": {"_REF_ATTN_ROUNDS": 32},
    "long8": {"ATTN_REFINE_MAX_T": 1024},
    "damp01_long32": {"GPTQ_DAMP": 0.01, "_REF_ATTN_ROUNDS": 32,
                       "ATTN_REFINE_MAX_T": 512},
    "fisher25": {"metric_blend": 0.25},
    "fisher75": {"metric_blend": 0.75},
    "fisher100": {"metric_blend": 1.0},
    "teacherq": {"teacher": "q", "teacher_strength": 1.0},
    "teacherk": {"teacher": "k", "teacher_strength": 1.0},
    "teacherboth": {"teacher": "qk", "teacher_strength": 0.5},
    "tiered32": {"ATTN_REFINE_MAX_T": 1024, "short_rounds": 32},
}


def teacher_module(params):
    source = (ROOT / "solution/solution_v308.py").read_text()
    needle = "    p = _dyn_table(x, None, has_scale=False)"
    assert source.count(needle) == 1
    source = source.replace(needle, "    x = _experiment_center(x, state, num_heads, head_dim)\n" + needle)
    m = types.ModuleType("teacher_experiment")
    exec(compile(source, "teacher_experiment", "exec"), m.__dict__)

    def center(x, st, nh, d):
        if "teacher_p" not in st:
            return x
        p = st["teacher_p"].float()
        pp = p[torch.arange(nh) // (nh // p.shape[0])]
        return (x.reshape(-1, nh, d).transpose(0, 1) @ pp).transpose(0, 1).reshape(x.shape).contiguous()

    def transformed(sample, role, st, nh, kvh, d):
        x = m.dequantize_nvfp4(*sample[role]).float()
        scale = st.get("qs" if role == "q" else "ks")
        if scale is not None:
            if role == "q":
                x = m._qks_apply_q(x, scale.float(), nh, kvh, d, inv=False)
            else:
                x = (x.reshape(-1, nh, d) * scale.float()[None]).reshape(x.shape)
        if st.get("rot"):
            x = (x.reshape(-1, nh, d) @ m._make_R(d, st.get("rseed", 0))).reshape(x.shape)
        return x.reshape(-1, nh, d).transpose(0, 1).contiguous()

    parent = m.hif4_calibration_attention
    m._experiment_center = center

    def calibrate(samples, qh, kvh, d):
        states = parent(samples, qh, kvh, d)
        hold = samples[-1]

        def quantized(sample, role, state):
            nh = qh if role == "q" else kvh
            return m._deq_params(getattr(m, "hif4_dynamic_quantize_" + role)(
                *sample[role], nh, d, state))

        qplay = quantized(hold, "q", states["q_state"])
        kplay = quantized(hold, "k", states["k_state"])
        vplay = quantized(hold, "v", states["v_state"])
        target = m._attention_out(*(m.dequantize_nvfp4(*hold[r]).float() for r in "qkv"), qh, kvh, d)
        loss = lambda q, k: (m._attention_out(q, k, vplay, qh, kvh, d) - target).square().mean().item()
        current_loss = loss(qplay, kplay)
        for role in params["teacher"]:
            opposite = "k" if role == "q" else "q"
            nh = kvh if opposite == "k" else qh
            gram = torch.zeros(kvh, d, d)
            cross = torch.zeros_like(gram)
            for sample in samples[:-1]:
                orig = transformed(sample, opposite, states[opposite + "_state"], nh, kvh, d)
                quant = quantized(sample, opposite, states[opposite + "_state"]).reshape(-1, nh, d).transpose(0, 1)
                g = quant.transpose(1, 2) @ quant
                c = orig.transpose(1, 2) @ quant
                gram += g.reshape(kvh, nh // kvh, d, d).sum(1)
                cross += c.reshape(kvh, nh // kvh, d, d).sum(1)
            eye = torch.eye(d)[None]
            ridge = gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None] * 0.1
            chol = torch.linalg.cholesky(gram + ridge * eye)
            p = torch.cholesky_solve((cross + ridge * eye).transpose(1, 2), chol).transpose(1, 2)
            p = eye + params["teacher_strength"] * (p - eye)
            proposed = dict(states[role + "_state"], teacher_p=p.to(torch.bfloat16).contiguous())
            u = m._upper_cholesky_inv(gram)
            if u is None:
                continue
            proposed.update(gq=1, u=u, H=gram.to(torch.bfloat16).contiguous())
            candidate = quantized(hold, role, proposed)
            new_loss = loss(candidate, kplay) if role == "q" else loss(qplay, candidate)
            if new_loss < current_loss * 0.999:
                states[role + "_state"] = proposed
                current_loss = new_loss
                if role == "q": qplay = candidate
                else: kplay = candidate
        return states

    m.hif4_calibration_attention = calibrate
    return m


def fisher_grams(qb, kb, blend):
    """Average softmax-KL Q Hessian; diagonal-token approximation for K.

    Shared per-KV-head Q metric averages GQA replicas. Original Gram trace
    is preserved so blending and damping have consistent dimensions.
    Sampling is calibration-only and deterministic, never uses test inputs.
    """
    qh, tokens, dim = qb.shape
    kvh = kb.shape[0]
    rep = qh // kvh
    rawk = kb.transpose(1, 2) @ kb
    rawq = qb.transpose(1, 2) @ qb
    q = qb[:, ::max(1, (tokens + 31) // 32)]
    k = kb[:, ::max(1, (tokens + 127) // 128)].repeat_interleave(rep, 0)
    prob = torch.softmax(q @ k.transpose(1, 2) / dim ** 0.5, dim=-1)
    mean = prob @ k
    fk = k.transpose(1, 2) @ (prob.mean(1).unsqueeze(-1) * k)
    fk -= mean.transpose(1, 2) @ mean / q.shape[1]
    fk = fk.reshape(kvh, rep, dim, dim).mean(1)
    fq = q.transpose(1, 2) @ (q * (1 - prob.square().sum(-1)).unsqueeze(-1))

    def normalize(f, raw):
        a = raw.diagonal(dim1=-2, dim2=-1).sum(-1)
        b = f.diagonal(dim1=-2, dim2=-1).sum(-1)
        f = (f + f.transpose(-1, -2)) * 0.5
        f = f * (a / b.clamp_min(1e-20))[:, None, None]
        f = torch.where((b > 1e-12)[:, None, None], f, raw)
        return raw * (1 - blend) + f * blend

    return normalize(fk, rawk), normalize(fq, rawq)


def configured_module(params):
    if "teacher" in params:
        return teacher_module(params)
    if "short_rounds" in params:
        source = (ROOT / "solution/solution_v308.py").read_text()
        old = "total = n_sweeps * _REF_ATTN_ROUNDS"
        assert source.count(old) == 1
        source = source.replace(old, "total = n_sweeps * (_SHORT_ROUNDS if T <= 128 else _REF_ATTN_ROUNDS)")
        m = types.ModuleType("tiered_experiment")
        exec(compile(source, "tiered_experiment", "exec"), m.__dict__)
        m._SHORT_ROUNDS = params["short_rounds"]
        m.ATTN_REFINE_MAX_T = params["ATTN_REFINE_MAX_T"]
        return m
    if "metric_blend" not in params:
        m = load(308)
    else:
        # Research-only source compilation, never emitted in solution.py.
        source = (ROOT / "solution/solution_v308.py").read_text()
        old = ("G_k = torch.bmm(kb.transpose(1, 2), kb)\n"
               "            G_q = torch.bmm(qb.transpose(1, 2), qb)")
        assert source.count(old) == 1
        source = source.replace(old, "G_k, G_q = _experiment_grams(qb, kb)")
        m = types.ModuleType("fisher_experiment")
        exec(compile(source, "fisher_experiment", "exec"), m.__dict__)
        m._experiment_grams = lambda q, k: fisher_grams(q, k, params["metric_blend"])
    for key, val in params.items():
        if key != "metric_blend":
            setattr(m, key, val)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="local_fitted_models_covariance_v1")
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--groups-per-model", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    all_groups = torch.load(ROOT / args.dataset / "attn.pt", weights_only=False)
    chosen, counts = [], {}
    for g in all_groups:
        name = g["source_model"]
        index = counts.get(name, 0)
        counts[name] = index + 1
        if args.offset <= index < args.offset + args.groups_per_model:
            chosen.append(g)
    assert len(chosen) == args.groups_per_model * len(counts)
    report = {"dataset": args.dataset, "groups": len(chosen),
              "offset": args.offset, "provenance": "fixed checkpoint-projection bank; modeled hidden states",
              "units": "raw MSE improvement sums, not online scores", "results": {}}
    for name in args.configs.split(","):
        m = configured_module(CONFIGS[name])
        original = m.hif4_calibration_attention
        flags = {"gq": 0, "rfq": 0, "rfk": 0, "teacherq": 0, "teacherk": 0}

        def capture(*a):
            s = original(*a)
            flags["gq"] += bool(s["q_state"].get("gq"))
            flags["rfq"] += bool(s["q_state"].get("rf"))
            flags["rfk"] += bool(s["k_state"].get("rf"))
            flags["teacherq"] += "teacher_p" in s["q_state"]
            flags["teacherk"] += "teacher_p" in s["k_state"]
            return s

        m.hif4_calibration_attention = capture
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, chosen)
        report["results"][name] = {"params": CONFIGS[name], "score": sum(scores),
                                    "seconds": seconds, "flags": flags, "cases": scores}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(name, sum(scores), seconds, flags, flush=True)


if __name__ == "__main__":
    main()

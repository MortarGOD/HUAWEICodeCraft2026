"""Exact fixed-lattice coordinate jumps on a fixed checkpoint-derived bank.

Research harness; only the numerical kernel may be copied into a submission.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def coordinate_jump(x, values, unit, H, num_heads, n_sweeps, rounds=8):
    """Greedy best-coordinate exact discrete minimization, fixed scales.

    Each coordinate admits integer codes -7..7. Choose its optimal jump,
    then select the most negative quadratic change across all coordinates.
    """
    T, C = x.shape
    B, d = num_heads, H.shape[-1]
    if T == 0 or C == 0 or n_sweeps <= 0:
        return values
    hmap = torch.arange(B) * H.shape[0] // B
    gram = H[hmap].float().contiguous()
    code = torch.round(values / unit * 4).view(T, B, d).permute(1, 0, 2).reshape(B*T, d).contiguous()
    step = (unit * 0.25).view(T, B, d).permute(1, 0, 2).reshape(B*T, d).contiguous()
    target = x.view(T, B, d).permute(1, 0, 2).reshape(B*T, d).contiguous()
    residual = torch.bmm((code * step - target).view(B, T, d), gram).reshape(B*T, d)
    heads = torch.arange(B*T) // T
    diagonal = gram.diagonal(dim1=-2, dim2=-1)[heads]
    curvature = step.square() * diagonal
    inverse = (step * diagonal).clamp_min(1e-30).reciprocal()
    gflat = gram.reshape(B*d, d)
    rows = torch.arange(B*T)
    full = code
    compact = False
    for _ in range(n_sweeps * rounds):
        jump = torch.round(-residual * inverse)
        jump = torch.minimum(torch.maximum(jump, -7 - code), 7 - code)
        gains = jump * (2 * step * residual + jump * curvature)
        idx = gains.argmin(1, keepdim=True)
        active = gains.gather(1, idx)[:, 0] < 0
        chosen = jump.gather(1, idx) * active[:, None]
        code.scatter_add_(1, idx, chosen)
        coeff = chosen * step.gather(1, idx)
        residual.addcmul_(gflat[heads*d + idx[:, 0]], coeff)
        if bool(active.all()):
            continue
        if compact:
            full[rows[~active]] = code[~active]
        rows = rows[active]
        heads = heads[active]
        code = code[active]
        step = step[active]
        residual = residual[active]
        curvature = curvature[active]
        inverse = inverse[active]
        compact = True
        if rows.numel() == 0:
            break
    if compact:
        full[rows] = code
    return full.view(B, T, d).permute(1, 0, 2).reshape(T, C) * (unit * 0.25)


def module_for(config):
    if config.startswith("block"):
        return block_module(int(config.removeprefix("block")))
    m = load(309)
    if config.startswith("vmean"):
        penalty = float(config.removeprefix("vmean"))
        def dynamic_v(q, s, heads, d, state):
            x = m.dequantize_nvfp4(q,s).float()
            p = m._quantize_weighted(x,torch.ones(1,x.shape[1]))
            if x.shape[0] <= 1:
                return p
            unit = m._params_unit_flat(p)
            best = m._deq_params(p)
            def cost(y):
                error = x-y
                return error.mean(0).square() + penalty/x.shape[0] * error.square().mean(0)
            best_cost = cost(best)
            shift = (x-best).mean(0)
            for _ in range(8):
                z = x + shift
                code = torch.round(z / unit * 4).clamp(-7,7)
                y = code * (unit * 0.25)
                value = cost(y)
                accept = value < best_cost
                best = torch.where(accept[None],y,best)
                best_cost = torch.minimum(value,best_cost)
                shift = shift + (x-y).mean(0) * 0.75
            return m._values_to_params(best,p)
        m.hif4_dynamic_quantize_v = dynamic_v
        return m
    if config in ("vwide", "allwide", "vclip", "allclip"):
        extra = tuple((-1, v) for v in (1.0, 1.25, 1.5, 1.75))
        if "wide" in config:
            extra += ((1, 1.5), (1, 1.75))
        grid = extra + m.CAND_GRID
        if config.startswith("all"):
            m._quantize_weighted.__defaults__ = (grid,)
        else:
            def dynamic_v(q, s, heads, d, state):
                x = m.dequantize_nvfp4(q, s).float()
                return m._quantize_weighted(x, torch.ones(1, x.shape[1]), grid)
            m.hif4_dynamic_quantize_v = dynamic_v
        return m
    if config.startswith("np"):
        rounds = None if config == "nptier" else int(config[2:])
        m._refine_attn_heads = lambda *a: coordinate_numpy(*a, rounds=rounds)
        return m
    if config != "baseline":
        rounds = int(config.removeprefix("jump"))
        m._refine_attn_heads = lambda *a: coordinate_jump(*a, rounds=rounds)
    return m


def coordinate_numpy(x, values, unit, H, num_heads, n_sweeps, rounds=None):
    """FP32 CPU active-set lattice search; no compilation or file access."""
    T, C = x.shape
    B, d = num_heads, H.shape[-1]
    if T == 0 or n_sweeps <= 0:
        return values
    hm = torch.arange(B) * H.shape[0] // B
    gt = H[hm].float().contiguous()
    def rows(t):
        return t.view(T,B,d).permute(1,0,2).reshape(B*T,d).contiguous()
    code_t = rows(torch.round(values / unit * 4))
    step_t = rows(unit * 0.25)
    mt = torch.bmm((code_t * step_t - rows(x)).view(B,T,d),gt).reshape(B*T,d)
    code, step, residual, gram = code_t.numpy(), step_t.numpy(), mt.numpy(), gt.numpy()
    heads = np.arange(B*T) // T
    diag = np.diagonal(gram, axis1=-2, axis2=-1)[heads]
    curvature = step * step * diag
    neg2 = -2 * step
    upper, lower = code >= 7, code <= -7
    selected_rows = np.arange(B*T)
    full = code
    compact = False
    gflat = gram.reshape(B*d,d)
    g = np.empty_like(code)
    for _ in range(n_sweeps * (rounds if rounds is not None else (32 if T<=128 else 8))):
        np.abs(residual, out=g)
        np.multiply(neg2, g, out=g)
        np.add(g, curvature, out=g)
        sign = residual < 0
        g[np.where(sign, upper, lower)] = np.inf
        idx = g.argmin(1)
        rr = np.arange(len(idx))
        active = g[rr,idx] < 0
        dr = np.where(residual[rr,idx] < 0, np.float32(1), np.float32(-1)) * active
        code[rr,idx] += dr
        upper[rr,idx] = code[rr,idx] >= 7
        lower[rr,idx] = code[rr,idx] <= -7
        residual += (dr * step[rr,idx])[:,None] * gflat[heads*d+idx]
        if active.all():
            continue
        if compact:
            full[selected_rows[~active]] = code[~active]
        selected_rows = selected_rows[active]
        heads = heads[active]
        code, step = code[active], step[active]
        residual, curvature, neg2 = residual[active], curvature[active], neg2[active]
        upper, lower = upper[active], lower[active]
        compact = True
        if not len(selected_rows):
            break
        g = np.empty_like(code)
    if compact:
        full[selected_rows] = code
    result = torch.from_numpy(full).view(B,T,d).permute(1,0,2).reshape(T,C)
    return result * (unit * 0.25)


def block_module(sweeps):
    source = (ROOT / "solution/solution_v309.py").read_text()
    needle = "    return p\n\n\ndef _dyn_v"
    assert source.count(needle) == 1
    source = source.replace(needle, "    return _experiment_block(x, p, state, num_heads, head_dim)\n\n\ndef _dyn_v")
    m = types.ModuleType("block_experiment")
    exec(compile(source, "block_experiment", "exec"), m.__dict__)
    parent = m.hif4_calibration_attention

    def calibrate(*args):
        states = parent(*args)
        for role in ("q", "k"):
            st = states[role + "_state"]
            if "H" not in st:
                continue
            h = st["H"].float()
            blocks = torch.stack([h[:, i:i+64, i:i+64] for i in range(0,h.shape[-1],64)], 1)
            ridge = blocks.diagonal(dim1=-2, dim2=-1).mean(-1)[...,None,None] * 0.1
            st["block_inverse"] = torch.linalg.inv(blocks + ridge * torch.eye(64)).contiguous()
        return states

    def improve(x, p, st, nh, d):
        T = x.shape[0]
        if T == 1 or T > 1024 or "block_inverse" not in st:
            return p
        nb = d // 64
        hm = torch.arange(nh) * st["H"].shape[0] // nh
        gram = st["H"][hm].float()
        inverse = st["block_inverse"][hm].float()
        v = m._deq_params(p).view(T,nh,d).transpose(0,1).contiguous()
        target = x.view(T,nh,d).transpose(0,1)
        res = (v-target) @ gram
        out = {k:value.clone() for k,value in p.items()}
        for _ in range(sweeps):
            for block in range(nb):
                sl = slice(block*64,(block+1)*64)
                center = v[:,:,sl] - res[:,:,sl] @ inverse[:,block]
                flat = center.transpose(0,1).reshape(T,nh*64).contiguous()
                pp = m._quantize_weighted(flat, torch.ones(1,nh*64))
                vv = m._deq_params(pp).view(T,nh,64).transpose(0,1)
                change = vv - v[:,:,sl]
                gain = (change * (2*res[:,:,sl] + change @ gram[:,sl,sl])).sum(-1)
                accept = gain < 0
                change = change * accept[:,:,None]
                v[:,:,sl] += change
                res += change @ gram[:,sl,:]
                for key, value in out.items():
                    tail = value.shape[2:]
                    view = value.view(T,nh,nb,*tail)
                    mask = accept.T.reshape(T,nh,*([1]*len(tail)))
                    view[:,:,block] = torch.where(mask, pp[key], view[:,:,block])
        return out

    m.hif4_calibration_attention = calibrate
    m._experiment_block = improve
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default="baseline,jump8,jump16,jump32")
    p.add_argument("--groups-per-model", type=int, default=5)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset", default="local_fitted_models_covariance_v1")
    args = p.parse_args()
    all_groups = torch.load(ROOT / args.dataset / "attn.pt", weights_only=False)
    groups, counts = [], {}
    for g in all_groups:
        key = g["source_model"]
        idx = counts.get(key, 0)
        counts[key] = idx + 1
        if args.offset <= idx < args.offset + args.groups_per_model:
            groups.append(g)
    report = {"dataset": args.dataset, "groups": len(groups),
              "offset": args.offset, "results": {}}
    for config in args.configs.split(","):
        m = module_for(config)
        with contextlib.redirect_stdout(io.StringIO()):
            scores, seconds = evaluate_attention(m, groups)
        report["results"][config] = {"score": sum(scores), "seconds": seconds, "cases": scores}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print(config, sum(scores), seconds, flush=True)


if __name__ == "__main__":
    main()

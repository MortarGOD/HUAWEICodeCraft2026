"""Joint held-out portfolio; production uses this code statically, never exec."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import types
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import evaluate_attention
from probe_persistent_covariance import load


def candidate_source():
    source=(ROOT/'solution/solution_v310.py').read_text()
    source=source.replace('GPTQ_BLOCK = 128','GPTQ_BLOCK = 32')
    source=source.replace('n_sweeps: int) -> torch.Tensor:', 'n_sweeps: int, total_rounds=None) -> torch.Tensor:')
    marker='total = n_sweeps * (32 if T <= 128 else _REF_ATTN_ROUNDS)'
    assert source.count(marker)==1
    source=source.replace(marker, marker+' if total_rounds is None else total_rounds')
    start=source.index('def _dyn_qk('); end=source.index('def _dyn_v(',start)
    body=source[start:end]
    marker='    if rot:\n'
    assert body.count(marker)==1
    body=body.replace(marker,'''    if isinstance(state, dict) and "joint_transform" in state:
        full = state["joint_transform"].float().repeat_interleave(num_heads // state["kvh"], 0)
        x = (x.reshape(-1, num_heads, head_dim).transpose(0, 1) @ full).transpose(0, 1).reshape(x.shape).contiguous()
    elif rot:
''')
    body=body.replace('            p = _values_to_params(values, p)\n','')
    body=body.replace('                p = _values_to_params(v1.contiguous(), p)','                values = v1.contiguous()')
    assert body.endswith('    return p\n\n\n')
    body=body[:-len('    return p\n\n\n')]+'    return _values_to_params(values, p) if values is not None else p\n\n\n'
    source=source[:start]+body+source[end:]
    return source+'\n\n'+(ROOT/'tools/attention_joint_core.py').read_text()


def configured():
    m=types.ModuleType('joint_attention_candidate')
    exec(compile(candidate_source(),'joint_attention_candidate','exec'),m.__dict__)
    return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='5,6,7,8,9,30,31,32,33,34')
    p.add_argument('--output',default='docs/v314-joint-holdout.json')
    a=p.parse_args();bank=torch.load(ROOT/'local_fitted_models_covariance_v1/attn.pt',weights_only=False)
    groups=[bank[int(i)] for i in a.groups.split(',')];results={}
    for name in ('v310','joint'):
        m=load(310) if name=='v310' else configured()
        parent=m.hif4_calibration_attention;diagnostics=[]
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'JOINT_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        results[name]=dict(score=sum(scores),seconds=seconds,cases=scores,diagnostics=diagnostics)
        print(name,sum(scores),seconds,diagnostics,flush=True)
        (ROOT/a.output).write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

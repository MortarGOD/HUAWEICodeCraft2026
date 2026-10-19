"""Build and evaluate the static input-gain prototype, never runtime exec in submission."""
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
    source=(ROOT/'solution/solution_v314.py').read_text()
    # Both paths must match exactly, including the original row-count budget.
    start=source.index('def _dyn_qk(');end=source.index('def _dyn_v(',start)
    body=source[start:end]
    marker_x='    x = dequantize_nvfp4(quant, scale).float()'
    assert body.count(marker_x)==1
    body=body.replace(marker_x,marker_x+'''
    if role == "k" and isinstance(state, dict) and "key_center_mask" in state and x.shape[0] > 1:
        head = x.reshape(-1, num_heads, head_dim)
        head = head - head.mean(0, keepdim=True) * state["key_center_mask"][None, :, None]
        x = head.reshape(x.shape).contiguous()
''')
    marker='    p = _dyn_table(x, None, has_scale=False)'
    assert body.count(marker)==1
    body=body.replace(marker,'''    if role == "q" and isinstance(state, dict) and "temperature_gain" in state:
        gain = state["temperature_gain"].float()
        x = (x.reshape(-1, num_heads, head_dim) * gain[None, :, None]).reshape(x.shape).contiguous()
'''+marker)
    source=source[:start]+body+source[end:]
    start=source.index('def _joint_quant_queries(');end=source.index('def _joint_output(',start)
    body=source[start:end]
    assert body.count(marker)==1
    body=body.replace(marker,'''    if "temperature_gain" in state:
        gain = state["temperature_gain"].float()
        x = (x.reshape(-1, qh, d) * gain[None, :, None]).reshape(x.shape).contiguous()
'''+marker)
    source=source[:start]+body+source[end:]
    return source+'\n\n'+(ROOT/'tools/attention_temperature_core.py').read_text()


def configured(center=True):
    m=types.ModuleType('attention_temperature')
    exec(compile(candidate_source(),'attention_temperature','exec'),m.__dict__)
    m._TEMP_CENTER_KEYS=center
    return m


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='local_fitted_models_covariance_v1')
    p.add_argument('--groups',default='2,7,27,32')
    p.add_argument('--output',default='docs/v315-temperature-screen.json')
    p.add_argument('--center',type=int,choices=(0,1),default=1)
    a=p.parse_args();bank=torch.load(ROOT/a.dataset/'attn.pt',weights_only=False)
    groups=bank if a.groups=='all' else [bank[int(i)] for i in a.groups.split(',')]
    result={'dataset':a.dataset,'groups':a.groups,'center':bool(a.center),'results':{}}
    for name in ('v314','temperature'):
        m=load(314) if name=='v314' else configured(bool(a.center))
        parent=m.hif4_calibration_attention;diagnostics=[]
        def capture(*args):
            st=parent(*args);diagnostics.append(dict(getattr(m,'TEMP_DEBUG',{})));return st
        m.hif4_calibration_attention=capture
        with contextlib.redirect_stdout(io.StringIO()):scores,seconds=evaluate_attention(m,groups)
        result['results'][name]={'score':sum(scores),'seconds':seconds,'cases':scores,'diagnostics':diagnostics}
        (ROOT/a.output).write_text(json.dumps(result,indent=2)+'\n')
        print(name,sum(scores),seconds,diagnostics,flush=True)


if __name__=='__main__':main()

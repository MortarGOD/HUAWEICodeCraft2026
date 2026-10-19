"""Static v320 construction; this tool is not a submission dependency."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def candidate_source(reuse=True):
    source=(ROOT/'solution/solution_v319.py').read_text()
    if reuse:
        tree=ast.parse(source)
        center_flags=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                      and any(isinstance(t,ast.Name) and t.id=='_TEMP_CENTER_KEYS' for t in n.targets)]
        assert center_flags[-1] is False,'K cache requires unchanged K after temperature calibration'
        fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_temp_fit')
        old=ast.get_source_segment(source,fit)
        new=old.replace('    gradient = torch.zeros(qh)','    records = []\n    gradient = torch.zeros(qh)')
        new=new.replace('    for sample in samples[:-1][-2:]:',
            '    for sample_index in range(max(0,len(samples)-3),len(samples)-1):\n        sample = samples[sample_index]')
        new=new.replace('        k = k.repeat_interleave(qh//kh, 0)',
            '        records.append((sample_index,k.transpose(0,1).reshape(rows,kh*d).contiguous()))\n        k = k.repeat_interleave(qh//kh, 0)')
        new=new.replace('    return delta.clamp(-0.2, 0.25)','    return delta.clamp(-0.2, 0.25), records')
        source=source.replace(old,new)
        source=source.replace('        delta = _temp_fit(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)',
            '        delta, records = _temp_fit(calib_qkv_list,states,q_num_heads,kv_num_heads,head_dim)')
        source=source.replace('        return _temp_guard(calib_qkv_list[-1],states,delta,q_num_heads,kv_num_heads,head_dim)',
            "        answer = _temp_guard(calib_qkv_list[-1],states,delta,q_num_heads,kv_num_heads,head_dim)\n        return dict(answer,_vr_cache=records)")
    core=(ROOT/'tools/attention_residual_core.py').read_text().replace('_VR_FIT_SEQUENCES = 4','_VR_FIT_SEQUENCES = 2').replace('_VR_ROUTE = False','_VR_ROUTE = True')
    end=source.index('"""',source.index('"""')+3)+3
    return '''"""v320: output-residual learned Value rounding thresholds.

Linear and Q/K decisions frozen at v319. Fit through deployed attention on
calibration only; independent dynamic Value observes V and frozen coefficients.
"""'''+source[end:]+'\n\n'+core


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');a=p.parse_args()
    source=candidate_source();compile(source,'v320','exec')
    if a.write:
        # Mechanical static source generation; editable algorithm remains in core.
        for name in ('solution_v320.py','solution.py'):(ROOT/'solution'/name).write_text(source)
    print('lines',len(source.splitlines()))

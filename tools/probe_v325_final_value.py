"""End-of-pipeline incremental Value correction, not raw-Value replacement."""
import argparse,ast,contextlib,copy,io,json
import torch
from probe_v319_value import ROOT,load_solution
from evaluate_local import evaluate_attention
from self_check import validate_attention_calibration_result,validate_hif4_params


def source():
    parent=(ROOT/'solution/solution_v324.py').read_text()
    nodes=ast.parse(parent).body
    picked=[n for n in nodes if isinstance(n,ast.FunctionDef) and (n.name.startswith('_vr_') or (4107<=n.lineno<=4125))]
    code='\n\n'.join(ast.get_source_segment(parent,n) for n in picked)
    code=code.replace('_vr','_fr').replace('_VR','_FR').replace('VR_DEBUG','FR_DEBUG').replace("'vr'","'fr'").replace('dict(vr=st)','dict(fr=st)')
    code=code.replace('target=v+delta.transpose(0,1).reshape(v.shape)','target=base+delta.transpose(0,1).reshape(v.shape)')
    prefix="_FR_RIDGE=.1\n_FR_TARGET='total'\n_FR_FIT_SEQUENCES=2\n_FR_ROUTE=True\nFR_DEBUG={}\n_fr_parent_calibration=hif4_calibration_attention\n_fr_parent_dynamic=hif4_dynamic_quantize_v\n\n"
    return prefix+code+'\n'


def main():
    p=argparse.ArgumentParser();p.add_argument('--groups',default='0,1,14,19,48,49');p.add_argument('--modes',default='1,2')
    p.add_argument('--bank',default='local_fitted_models_attention_feedback_v2');p.add_argument('--output',default='docs/v325-final-value-screen.json');a=p.parse_args()
    torch.set_num_threads(4);bank=torch.load(ROOT/a.bank/'attn.pt',weights_only=False)
    m=load_solution(ROOT/'solution/solution_v324.py');base=load_solution(ROOT/'solution/solution_v324.py')
    exec(compile(source(),'final-value','exec'),m.__dict__)
    ids=range(len(bank)) if a.groups=='all' else map(int,a.groups.split(','));rows=[]
    for gi in ids:
        g=bank[gi];dims=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=m._fr_parent_calibration(g['calib'],*dims)
        base.hif4_calibration_attention=lambda *unused,_st=parent:copy.deepcopy(_st)
        with contextlib.redirect_stdout(io.StringIO()):old,_=evaluate_attention(base,[g])
        row=dict(group=gi,baseline=old,modes={})
        for mode in map(int,a.modes.split(',')):
            m._FR_ROUTE=mode==1;st=m._fr_guard(g['calib'],parent,*dims)
            assert not validate_attention_calibration_result(st,'attention')
            m.hif4_calibration_attention=lambda *unused,_st=st:copy.deepcopy(_st)
            with contextlib.redirect_stdout(io.StringIO()):scores,_=evaluate_attention(m,[g])
            for sample in g['test']:
                params=m.hif4_dynamic_quantize_v(*sample['v'],dims[1],dims[2],st['v_state'])
                assert not validate_hif4_params(params,sample['v'][0].shape,'value')
            row['modes'][mode]=dict(scores=scores,delta=sum(scores)-sum(old),debug=copy.deepcopy(m.FR_DEBUG))
        rows.append(row);print(gi,{k:round(v['delta'],6) for k,v in row['modes'].items()},flush=True)
    result=dict(complete=True,bank=a.bank,rows=rows,totals={k:sum(r['modes'][k]['delta'] for r in rows) for k in rows[0]['modes']},score_prediction_allowed=False)
    (ROOT/a.output).write_text(json.dumps(result,indent=2)+'\n');print(result['totals'],flush=True)


if __name__=='__main__':main()

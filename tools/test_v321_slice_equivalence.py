"""Compare optimized guard with the first published v321 on shared states."""
import copy
import json
import time
import subprocess
import torch
from probe_v319_value import ROOT,load_solution
from build_v321_candidate import candidate_source
from test_v320_residual import same


def main():
    torch.set_num_threads(4)
    old=load_solution(ROOT/'solution/solution_v320.py')
    original=subprocess.run(['git','show','e7f50ca:solution/solution_v321.py'],cwd=ROOT,check=True,capture_output=True,text=True).stdout
    exec(compile(original,'original-published-v321','exec'),old.__dict__)
    new=load_solution(ROOT/'solution/solution_v320.py')
    exec(compile(candidate_source(),'sliced-v321','exec'),new.__dict__)
    new._KT_BUDGET_ROUTE=False
    new._KT_GUARD_ROWS=64;new._KT_MIN_GAIN=.005
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)
    rows=[];checks=0
    for gi in (0,1,14,19,47,49):
        g=bank[gi];qh,kh,d=[g[k] for k in ('q_num_heads','kv_num_heads','head_dim')]
        parent=old._kt_parent_calibration(g['calib'],qh,kh,d)
        for sample in g['calib']:
            for count in (16,64):
                ix=torch.linspace(0,sample['q'][0].shape[0]-1,min(count,sample['q'][0].shape[0])).long()
                whole=old.dequantize_nvfp4(*sample['q']).float()[ix]
                sliced=new.dequantize_nvfp4(sample['q'][0][ix],sample['q'][1][ix]).float()
                assert torch.equal(whole,sliced);checks+=1
        hold=g['calib'][-1]
        full=old._kt_parent_k(*hold['k'],kh,d,parent['k_state'])
        direct=new._kt_fixed_grid(new.dequantize_nvfp4(*hold['k']).float(),parent['k_state'],kh,d,'k')
        assert same(full,direct);checks+=1
        result={};seconds={};debug={}
        for name,m in (('old',old),('new',new)):
            start=time.perf_counter();result[name]=m._kt_guard(g['calib'],parent,qh,kh,d)
            seconds[name]=time.perf_counter()-start;debug[name]=copy.deepcopy(m.KT_DEBUG)
        assert same(result['old'],result['new']),(gi,debug)
        assert debug['old'].get('chosen')==debug['new'].get('chosen'),(gi,debug)
        assert same(debug['old']['modes'],debug['new']['modes']),(gi,debug)
        rows.append(dict(group=gi,guard_seconds=seconds,chosen=debug['new'].get('chosen',[])))
    report=dict(passed=True,decode_checks=checks,states_and_decisions_exact=True,groups=rows,
                warning='Concurrent screening timing only; not the serial full-bank timing result.')
    (ROOT/'docs/v321-slice-equivalence.json').write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()

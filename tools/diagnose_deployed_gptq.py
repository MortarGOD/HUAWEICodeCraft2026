"""Separate ALS fitting, weight requantization and encoder response (calibration only)."""
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_linear_calibration_result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    bank=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/linear.pt',weights_only=False)
    report={'complete':False,'groups':[],'uses_test_data':False,'online_prediction':None}
    for gi in (0,4,6,7,9):
        m=load_solution(ROOT/'solution/solution_v372.py')
        seen={}
        old_parent=m._dg_parent_calibration
        old_fit=m._dg_fit
        old_quant=m._dg_quantize
        def parent(*args):
            result=old_parent(*args);seen['parent']=copy.deepcopy(result);return result
        def fit(z,y,b,*args):
            result=old_fit(z,y,b,*args)
            seen.update(z=z.clone(),y=y.clone(),b=b.clone(),proposal=result[0].clone())
            return result
        def quant(*args):
            result=old_quant(*args);seen['params']=copy.deepcopy(result);return result
        m._dg_parent_calibration=parent;m._dg_fit=fit;m._dg_quantize=quant
        g=bank[gi]
        m.hif4_calibration_and_quantize_weight(*g['weight'],g['calib_activation_list'])
        assert 'params' in seen,'Unexpected numerical fallback'
        ps=seen['parent']['activation_state']
        packed=g['calib_activation_list'][-1]
        raw=m.dequantize_nvfp4(*packed).float()
        y=raw@m.dequantize_nvfp4(*g['weight']).float().T
        z=m._linear_score_decode(m.hif4_dynamic_quantize_activation(*packed,ps))
        p=seen['proposal']
        params={'parent':seen['parent']['weight_params'],
                'rtn':m._quantize_linear_values(p,offsets=m._LINEAR_WEIGHT_SCALE_OFFSETS),
                'feedback':seen['params']}
        rows={}
        for name,qp in params.items():
            b=m._linear_score_decode(qp)
            row={'fit_fixed_z_mse':float((seen['y']-seen['z']@b.float().T).square().mean()),
                 'hold_fixed_z_mse':float((y-(z@b.T).float()).square().mean()),
                 'weight_delta_mse':float((b.float()-seen['b']).square().mean())}
            if name=='parent':
                row['hold_reencoded_mse']=row['hold_fixed_z_mse']
            else:
                state=dict(ps)
                state.pop('unrestricted_continuous_projection',None)
                state.pop('linear_full_gram',None)
                m._install_unrestricted_exact_target_state(state,ps['unrestricted_target_weight'].float(),qp)
                if b.shape[0]>=2*b.shape[1]:state['linear_full_gram']=b.float().T@b.float()
                assert not validate_linear_calibration_result({'weight_params':qp,'activation_state':state},b.shape,'candidate')
                qz=m._linear_score_decode(m.hif4_dynamic_quantize_activation(*packed,state))
                row['hold_reencoded_mse']=float((y-(qz@b.T).float()).square().mean())
            rows[name]=row
        rows['continuous_unsubmittable']={
            'fit_fixed_z_mse':float((seen['y']-seen['z']@p.T).square().mean()),
            'hold_fixed_z_mse':float((y-z.float()@p.T).square().mean())}
        record={'group':gi,'shape':list(seen['b'].shape),'stages':rows,'guard':dict(m.DG_DEBUG)}
        report['groups'].append(record)
        print(json.dumps(record),flush=True)
        (ROOT/'docs/v372-stage-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True
    (ROOT/'docs/v372-stage-diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

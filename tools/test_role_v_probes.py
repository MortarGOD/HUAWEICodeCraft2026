"""V-only final replacement, with unchanged internal parent calibration."""
import ast,copy,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import validate_attention_calibration_result,validate_hif4_params
from tools.test_dual_operator_isolation import equal


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    group=torch.load(ROOT/'local_fitted_models_attention_feedback_v2/attn.pt',weights_only=False)[10]
    qh,kh,d=(group[k] for k in ('q_num_heads','kv_num_heads','head_dim'))
    for version,mode in ((407,'mean'),(408,'rtn')):
        path=ROOT/f'solution/solution_v{version}.py';source=path.read_text()
        assert source.startswith((ROOT/'solution/solution_v400.py').read_text())
        for node in ast.walk(ast.parse(source)):
            if isinstance(node,ast.Import):assert all(a.name.split('.')[0] in {'torch','numpy','typing','threading'} for a in node.names)
            if isinstance(node,ast.ImportFrom):assert node.module.split('.')[0] in {'__future__','typing','torch','numpy','threading'}
        m=load_solution(path);capture=[];calls=[0]
        cal=m._vp_parent_calibration;dyn=m._vp_parent_v
        def parent(*a):
            result=cal(*a);capture.append(copy.deepcopy(result));return result
        def observed(*a):
            assert not isinstance(a[-1],dict) or 'role_v_probe' not in a[-1]
            calls[0]+=1;return dyn(*a)
        m._vp_parent_calibration=parent;m._vp_parent_v=observed
        states=m.hif4_calibration_attention(group['calib'],qh,kh,d)
        assert calls[0]>0 and states['v_state']=={'role_v_probe':mode}
        assert all(equal(states[r+'_state'],capture[0][r+'_state']) for r in 'qk')
        assert not validate_attention_calibration_result(states,'probe')
        snapshot=copy.deepcopy(states);rng=torch.random.get_rng_state().clone()
        for sample in group['test']:
            fresh=load_solution(path);q,s=sample['v']
            ans=fresh.hif4_dynamic_quantize_v(q,s,kh,d,copy.deepcopy(states['v_state']))
            expected=fresh._vk_parent_v(q,s,None,kh,d) if mode=='mean' else fresh._dyn_table(fresh.dequantize_nvfp4(q,s).float(),None,False)
            assert equal(ans,expected) and not validate_hif4_params(ans,q.shape,'v')
            old=dyn(q,s,kh,d,capture[0]['v_state'])
            assert equal(fresh.hif4_dynamic_quantize_v(q,s,kh,d,capture[0]['v_state']),old)
        assert equal(states,snapshot) and torch.equal(rng,torch.random.get_rng_state())
        print('PASS',version,'parent prefix/imports, original internal V calls',calls[0],'; Q/K state exact, constant V state, fresh dynamic expected outputs, unmarked fallback, frozen state/RNG')


if __name__=='__main__':main()

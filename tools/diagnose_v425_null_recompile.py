"""Fixed-parent-W control for the initial v425 encoder rebuilding assumption."""
from pathlib import Path
import hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from self_check import clone_frozen_state
from tools.verify_v420_v421_linear import assert_tree_equal

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    source=ROOT/'solution/solution_v425.py'
    sha=hashlib.sha256(source.read_bytes()).hexdigest()
    assert sha=='a6e53f37abf2c7c51c99e6439df3e554fa57d75fbdc99f0c12f780a9e5691fcb', 'This diagnostic targets the initial published implementation only.'
    info=json.loads((ROOT/'docs/v425-linear-paired.json').read_text())
    bankpath=Path(info['dataset'])/'linear.pt'
    assert hashlib.sha256(bankpath.read_bytes()).hexdigest()==info['linear_sha256']
    group=torch.load(bankpath,map_location='cpu',weights_only=True)[0]
    module=load_solution(source)
    torch.manual_seed(info['seed'])
    parent=module._v425_parent_calibration(*group['weight'],group['calib_activation_list'])
    frozen=clone_frozen_state(parent)
    state=module._v425_recompile_state(parent['activation_state'],parent['weight_params'])
    assert state is not None
    candidate=dict(parent,activation_state=state)
    raww=module.dequantize_nvfp4(*group['weight']).float()
    guard=module._v425_last_calibration_guard(parent,candidate,group['calib_activation_list'][-1],raww)
    assert_tree_equal(parent,frozen,'parent_unchanged')
    oldp=parent['activation_state']['unrestricted_continuous_projection'];newp=state['unrestricted_continuous_projection']
    changed=[]
    for name in state:
        try: assert_tree_equal(state[name],parent['activation_state'][name],name)
        except AssertionError: changed.append(name)
    result={'source_sha256':sha,'initial_code_publish_commit':info.get('code_publish_commit','d0967129b1de9b8b36320275770797f21e5aa6a5'),
      'dataset':str(bankpath),'dataset_sha256':info['linear_sha256'],'group':0,'parent_weight_bitexact':True,
      'parent_unchanged':True,'projection_bitexact':torch.equal(oldp,newp),'projection_max_abs_delta':float((newp-oldp).abs().max()),
      'changed_state_fields':changed,'guard_before':guard['before'].tolist(),'guard_after':guard['after'].tolist(),
      'guard_ratios':(guard['after']/guard['before']).tolist(),'guard_accept':guard['accept'],
      'scope':'One model group, exact same W; tests whether treating learned encoder P as a stale cache is a neutral operation. No sparse-W candidate applied.'}
    print(json.dumps(result,indent=2),flush=True)
    (ROOT/'docs/v425-initial-null-recompile.json').write_text(json.dumps(result,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==sha
if __name__=='__main__': main()

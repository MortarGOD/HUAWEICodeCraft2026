"""Build v432 with one shared calibration guard and one consistent V risk."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parent = (ROOT/'solution/solution_v422.py').read_text()
temperature = (ROOT/'solution/solution_v430.py').read_text()
tree = (ROOT/'solution/solution_v431.py').read_text()
common = temperature[len(parent):temperature.index('\ndef _a430_power')]
temperature_functions = temperature[temperature.index('\ndef _a430_power'):temperature.rindex('\n@torch.inference_mode()')]
tree_functions = tree[tree.index('\ndef _a431_tree'):tree.rindex('\n@torch.inference_mode()')]
tree_functions = tree_functions.replace('probability=_ac_parent_probability(x,state,kh)',
                                        'probability=_vk_probabilities(x,state,kh)')
calibration = '''

@torch.inference_mode()
def hif4_calibration_attention(samples,qh,kh,d):
    parent=_ac_parent_calibration(samples,qh,kh,d)
    if not _ac_eligible(samples,parent,qh,kh,d):return parent
    try:
        st=parent['v_state'];beta=_a430_fit(samples,st,qh,kh,d)
        if beta is None or not bool(torch.isfinite(beta).all()):return parent
        trial=dict(st,a430_temperature=beta,a430_active=st['active'].clone(),
                   a431_active=st['active'].clone())
        active=_ac_guard(samples[-1],parent,trial,qh,kh,d)&st['active']
        if bool(active.any()):
            return dict(parent,v_state=dict(trial,a430_active=active.contiguous(),
                                            a431_active=active.clone().contiguous()))
    except (RuntimeError,ValueError,KeyError,TypeError):
        pass
    return parent
'''
result = parent + common + temperature_functions + tree_functions + calibration
ast.parse(result)
(ROOT/'solution/solution_v432.py').write_text(result)
print('Built v432 from v422, v430 temperature functions and v431 tree; one final guard.')

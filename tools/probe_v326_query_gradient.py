"""Bounded v325 gradient slicing experiment; does not alter submission files."""
import copy
import hashlib
import json
import time
import torch
from probe_v319_value import ROOT, load_solution
from self_check import validate_hif4_params


def main():
    torch.set_num_threads(4)
    source = (ROOT / 'solution/solution_v325.py').read_text()
    old = "torch.einsum('thd,hde->the',current-reference,h)[:,:,a:b]"
    new = "torch.einsum('thd,hde->the',current-reference,h[:,:,a:b])"
    assert source.count(old) == 1
    baseline = load_solution(ROOT / 'solution/solution_v325.py')
    candidate = load_solution(ROOT / 'solution/solution_v325.py')
    exec(compile(source.replace(old, new), 'gradient-sliced-research', 'exec'), candidate.__dict__)
    bank = torch.load(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    rows = []
    totals = {'parent': 0., 'candidate': 0.}
    for gi in (0, 1, 2, 14, 19, 49):
        group = bank[gi]
        qh, kh, dim = [group[k] for k in ('q_num_heads', 'kv_num_heads', 'head_dim')]
        state = baseline.hif4_calibration_attention(group['calib'], qh, kh, dim)['q_state']
        before = copy.deepcopy(state)
        row = dict(group=gi, dim=dim, active=int(state.get('qg_active', torch.zeros(1)).sum()), exact=True, differing_elements=0)
        for sample in group['test']:
            outputs = {}
            order = [('parent', baseline), ('candidate', candidate)]
            if gi % 2: order.reverse()
            for name, module in order:
                start = time.perf_counter()
                outputs[name] = module.hif4_dynamic_quantize_q(*sample['q'], qh, dim, state)
                totals[name] += time.perf_counter() - start
                assert not validate_hif4_params(outputs[name], sample['q'][0].shape, 'query')
            for key, value in outputs['parent'].items():
                count = int((value != outputs['candidate'][key]).sum())
                row['differing_elements'] += count
                row['exact'] &= count == 0
        for key, value in before.items():
            if isinstance(value, torch.Tensor): assert torch.equal(value, state[key])
        rows.append(row)
        print(row, flush=True)
    report = dict(complete=True, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  change=dict(old=old,new=new), rows=rows, timing=totals,
                  warning='30 modeled activation cases; dynamic Q only, no online score prediction or full-shape equivalence proof.')
    (ROOT / 'docs/v326-query-gradient-screen.json').write_text(json.dumps(report, indent=2)+'\n')
    print(totals, flush=True)


if __name__ == '__main__':
    main()

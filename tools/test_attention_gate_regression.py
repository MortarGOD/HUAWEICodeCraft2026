"""Verify the full diagnostic-bank regression of the fast research path."""
import json
from pathlib import Path


def flatten(value):
    if isinstance(value,list):
        return [x for part in value for x in flatten(part)]
    return [value]


def main():
    root=Path(__file__).resolve().parents[1]
    before=json.loads((root/'docs/attention-gates-frames-full.json').read_text())
    after=json.loads((root/'docs/attention-gates-frames-full-fast.json').read_text())
    assert before['dataset']==after['dataset'] and before['groups']==after['groups']
    assert before['head_cap']==after['head_cap']==0
    old,new=before['results']['frames'],after['results']['frames']
    assert len(old['cases'])==len(new['cases'])==250
    assert old['cases']==new['cases']
    assert len(old['diagnostics'])==len(new['diagnostics'])==50
    max_cost_delta=0.
    for left,right in zip(old['diagnostics'],new['diagnostics']):
        assert left['frame_labels']==right['frame_labels']
        assert left['frame_choice']==right['frame_choice']
        ca,cb=flatten(left['frame_costs']),flatten(right['frame_costs'])
        assert len(ca)==len(cb)
        max_cost_delta=max(max_cost_delta,max(abs(a-b) for a,b in zip(ca,cb)))
    assert new['checks']['states']==50
    base=before['results']['none']['cases']
    diff=[a-b for a,b in zip(new['cases'],base)]
    report=dict(passed=True,cases=250,groups=50,case_scores_exact=True,frame_choices_exact=True,
                calibration_cost_max_abs_delta=max_cost_delta,delta_vs_v314=sum(diff),
                up=sum(x>1e-8 for x in diff),down=sum(x< -1e-8 for x in diff),
                same=sum(abs(x)<=1e-8 for x in diff),fast_checks=new['checks'],
                reference_seconds=old['seconds'],fast_seconds=new['seconds'],
                warning='Equal per-case scores/choices on this bank, not proof that every output tensor is bit-identical on arbitrary inputs. Times include scoring/validation, not online estimates.')
    (root/'docs/attention-gates-fast-regression.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

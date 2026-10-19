"""Summarize completed, pushed-source Attention experiments without score fitting."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def aggregate(record):
    cases = record['cases']
    groups = {c['group']: c for c in record['calibrations']}
    strata = defaultdict(lambda: {'cases': 0, 'delta': 0., 'better': 0, 'worse': 0})
    for case in cases:
        group = groups[case['group']]
        labels = [('length', 'short' if case['tokens'] <= 128 else 'medium' if case['tokens'] <= 1024 else 'long'),
                  ('heads', 'MHA' if group['qh'] == group['kh'] else 'GQA')]
        for dimension, label in labels:
            row = strata[dimension + ':' + label]
            row['cases'] += 1
            row['delta'] += case['delta']
            row['better'] += case['delta'] > 1.e-12
            row['worse'] += case['delta'] < -1.e-12
    counter = record['quotient']
    exceptions = {key: val['exceptions'] for key, val in counter.items()
                  if isinstance(val, dict) and val.get('exceptions')}
    return dict(record['summary'], timing=record['timing'], strata=dict(strata),
                instrumentation=counter, caught_exceptions=exceptions,
                dequantized_value_checks=sum('dequantized_values_same' in x for x in cases),
                changed_value_cases=({r: sum(not x['dequantized_values_same'][r] for x in cases) for r in 'qkv'}
                                     if all('dequantized_values_same' in x for x in cases) else None),
                validated_returns=record['validated_returns'],
                state_checks=record['state_checks'],
                changed_calibration_states={r: sum(not x['state_same'][r] for x in groups.values()) for r in 'qkv'},
                new_v_state_groups=sum(bool(x['new_v_state_keys']) for x in groups.values()),
                largest_regressions=sorted(cases, key=lambda x: x['delta'])[:5],
                largest_improvements=sorted(cases, key=lambda x: x['delta'], reverse=True)[:5])


def main():
    manifest = json.loads((ROOT/'docs/attention-campaign-artifacts.json').read_text())
    output = {'versions': {}, 'warning': 'Raw local quality deltas, no online score or runtime conversion. Instrumented serial timings.'}
    lines = ['质量增量为本地归一化值，不是线上分数。标有校准混杂的分差不能归因于新策略。v434此表保留原定每库30题，追加每库25题及合并结果见attention-campaign-final-summary.json和v434-report.md。', '',
             '| 版本 | 数据 | 题数 | 质量增量 | 优/劣/同 | Q/K/V 参数变化题数 | 归因限制 |',
             '|---|---|---:|---:|---|---|---|']
    versions=[429,430,431,432,433]
    hierarchy_files=[ROOT/f'docs/v434-attention-{tag}.json' for tag in ('paired','covariance','row_permute','block_resample')]
    if all(p.exists() and json.loads(p.read_text()).get('completed') for p in hierarchy_files):versions.append(434)
    for version in versions:
        sha = hashlib.sha256((ROOT/f'solution/solution_v{version}.py').read_bytes()).hexdigest()
        assert sha == manifest[str(version)]['sha256']
        output['versions'][str(version)] = {}
        for label in ('paired', 'covariance', 'row_permute', 'block_resample'):
            record = json.loads((ROOT/f'docs/v{version}-attention-{label}.json').read_text())
            assert record['completed'] and record['source_sha256'] == sha
            summary = aggregate(record)
            output['versions'][str(version)][label] = summary
            changed = '/'.join(str(summary['changed_role_cases'][r]) for r in 'qkv')
            unchanged='qv' if version==429 else 'qk'
            limitation='不变角色校准有差异' if any(summary['changed_calibration_states'][r] for r in unchanged) else '—'
            lines.append(f"| v{version} | {label} | {summary['cases']} | {summary['delta']:+.9f} | {summary['better']}/{summary['worse']}/{summary['equal']} | {changed} | {limitation} |")
    output['joint_comparison'] = {}
    for label in ('paired', 'covariance', 'row_permute', 'block_resample'):
        a,b,c=[output['versions'][str(v)][label] for v in (430,431,432)]
        comparable=max(a['parent_quality'],b['parent_quality'],c['parent_quality'])-min(a['parent_quality'],b['parent_quality'],c['parent_quality'])<1e-8
        output['joint_comparison'][label]={'comparable_parent_quality':comparable,
                                         'v432_minus_v430':c['delta']-a['delta'],
                                         'v432_minus_v431':c['delta']-b['delta'],
                                         'nonadditive_delta':c['delta']-a['delta']-b['delta']}
    if 434 in versions:
        output['hierarchy_matched_comparison']={}
        for tag in ('paired','covariance'):
            h=json.loads((ROOT/f'docs/v434-attention-{tag}.json').read_text())
            ids={(c['group'],c['test']) for c in h['cases']}
            row={'cases':len(ids),'v434_delta':sum(c['delta'] for c in h['cases'])}
            for v in (431,432,433):
                other=json.loads((ROOT/f'docs/v{v}-attention-{tag}.json').read_text())
                delta=sum(c['delta'] for c in other['cases'] if (c['group'],c['test']) in ids)
                parent_delta=sum(c['quality'][0] for c in h['cases'])-sum(c['quality'][0] for c in other['cases'] if (c['group'],c['test']) in ids)
                row[f'v{v}_matched_delta']=delta
                row[f'v434_minus_v{v}']=row['v434_delta']-delta
                row[f'v{v}_parent_quality_difference']=parent_delta
            output['hierarchy_matched_comparison'][tag]=row
    (ROOT/'docs/attention-campaign-summary.json').write_text(json.dumps(output, indent=2)+'\n')
    (ROOT/'docs/attention-campaign-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

"""Linear error-split diagnostic v1. Never changes the submission or its output."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluate_local import load_solution
from hif4_quant import dequantize_nvfp4,dequantize_hif4,quantize_to_hif4
from self_check import validate_linear_calibration_result,validate_hif4_params
from linear_error_split_core import (factor_decoder,decompose,capture_dynamic,
                                     audit_original_transform,legal_search_witnesses)


def aggregate(cases):
    valid=[c for c in cases if c['baseline_mse']>0]
    names=('official_current_mse','geometric_floor_mse','discrete_and_search_mse',
           'teacher_bias_mse','teacher_cross_mse','evaluation_output_increment_mse',
           'mse_reduction_increment_mse')
    normalized={key:sum(c[key]/c['baseline_mse'] for c in valid) for key in names}
    score=sum(1-c['official_current_mse']/c['baseline_mse'] for c in valid)
    return dict(cases=len(cases),scorable_cases=len(valid),raw_score_sum=score,
                normalized_error_sums=normalized,
                geometry_share_of_current_gap=normalized['geometric_floor_mse']/max(normalized['official_current_mse'],1e-300),
                continuous_raw_score_upper=sum(1-c['geometric_floor_mse']/c['baseline_mse'] for c in valid),
                max_orthogonality_relative=max((c['normal_equation_relative'] for c in cases),default=0.),
                max_decomposition_identity_error=max((abs(c['geometric_identity_error']) for c in cases),default=0.),
                max_teacher_identity_error=max((abs(c['teacher_identity_error']) for c in cases),default=0.),
                max_evaluation_identity_error=max((abs(c['evaluation_identity_error']) for c in cases),default=0.))


def markdown(report):
    s=report['summary'];n=s['normalized_error_sums']
    lines=['# Linear误差拆分诊断 v1','',
           f"父版：v{report['version']}；数据：`{report['dataset']}`；完成：{report['complete']}。",
           '', '不修改提交输出；连续解只作离线诊断，不能当作合法HiF4提交。checkpoint权重配建模激活，不能把本地比例直接分摊线上缺口；本报告不推断当前线上分项。',
           '',f"共{s['cases']}题，实际原始分合计{s['raw_score_sum']:.9f}，固定B在精确解码乘法下的数值连续上界{s['continuous_raw_score_upper']:.9f}。后者不含最终dtype舍入，不是最终评分器的严格上界。",
           '', '| 项目 | 各题MSE/基准MSE之和 |', '|---|---:|',
           f"| 当前实际误差 | {n['official_current_mse']:.9f} |",
           f"| 固定B输出空间误差 | {n['geometric_floor_mse']:.9f} |",
           f"| 空间内离散表示＋搜索剩余 | {n['discrete_and_search_mse']:.9f} |",
           f"| 原评分器dtype输出计算增量（有符号） | {n['evaluation_output_increment_mse']:.9f} |",
           f"| FP32 MSE归约增量（有符号） | {n['mse_reduction_increment_mse']:.9f} |",
           '', '教师偏差单独核算，不与上表直接相加：',
           f"- 教师目标偏差归一化之和：{n['teacher_bias_mse']:.9f}。",
           f"- 教师交叉项之和：{n['teacher_cross_mse']:.9f}；它可能抵消也可能放大误差。",
           '', '| 组 | 实际O×I / 路径 | 当前原始分 | 精确乘法连续上界 | 空间误差占比 |', '|---|---|---:|---:|---:|']
    for g in report['groups']:
        a=g['summary'];shape=g['shape']
        lines.append(f"| {g['index']} | {shape[0]}×{shape[1]} / {g['route']} | {a['raw_score_sum']:.6f} | {a['continuous_raw_score_upper']:.6f} | {100*a['geometry_share_of_current_gap']:.2f}% |")
    lines+=['','### 解读边界','',
            '- 空间内剩余包括不可避免的HiF4离散误差和未收敛搜索，不能全部称为可恢复收益。',
            '- 教师误差、教师拟合残差及有符号交叉项构成另一条独立恒等式；不重复计入空间分解。',
            '- FP64 SVD按报告中的阈值判断数值秩；存在数值截断时不能无条件称为严格数学下界。',
            '- 合法搜索例证只使用固定抽样行，不能按全部行/线上时间外推；连续解直接量化也不是离散全局最优。',
            '- 所有计时属于本地诊断开销；没有新增线上耗时预估，生产文件哈希保持。','']
    return '\n'.join(lines)


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--version',type=int,default=316)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--dataset',default='local_fitted_models_attention_feedback_v2')
    p.add_argument('--groups',default='all')
    p.add_argument('--probe-rows',type=int,default=32)
    p.add_argument('--probe-channels',type=int,default=64)
    p.add_argument('--rank-rtol',type=float,default=None)
    p.add_argument('--output',default='docs/linear-error-split-v1.json')
    p.add_argument('--allow-mini',action='store_true',help='Correctness only, never accuracy interpretation')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    if args.probe_rows<0 or args.probe_channels<0:raise ValueError('Probe budgets must be nonnegative')
    dataset=(ROOT/args.dataset).resolve()
    if not args.allow_mini and (not dataset.name.startswith('local_fitted_models') or 'archive' in dataset.parts):
        raise ValueError('Use current checkpoint-derived bank; legacy synthetic datasets are archived')
    source=ROOT/f'solution/solution_v{args.version}.py'
    before={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (source,ROOT/'solution/solution.py')}
    solution=load_solution(source)
    bank=torch.load(dataset/'linear.pt',map_location='cpu',weights_only=False)
    regression_path=ROOT/f'docs/v{args.version}-production-regression.json'
    regression=None
    if regression_path.exists():
        candidate=json.loads(regression_path.read_text())
        if candidate.get('dataset')==args.dataset and candidate.get('sha256')==before[str(source)]:
            regression=candidate['results']['linear']['cases']
    selected=range(len(bank)) if args.groups=='all' else [int(i) for i in args.groups.split(',')]
    report=dict(diagnostic_version=1,version=args.version,dataset=args.dataset,complete=False,
                score_prediction_allowed=False,submission_hashes=before,groups=[],cases=[],
                validated_public_returns=0,validated_probe_returns=0,production_score_matches=0,
                provenance='Checkpoint tensors with modeled activations; reduced dimensions, not online data.',
                warnings=['No current online component score inferred from this diagnostic.',
                          'Geometry + in-space remainder is orthogonal accounting. Teacher terms are a separate identity.',
                          'In-space remainder includes irreducible discrete error, not just search inefficiency.'])
    started=time.perf_counter()
    output=ROOT/args.output
    for gi in selected:
        group=bank[gi]
        weight=dequantize_nvfp4(*group['weight']).float()
        bw=dequantize_hif4(quantize_to_hif4(weight),tuple(weight.shape))
        start=time.perf_counter()
        calibrated=solution.hif4_calibration_and_quantize_weight(*group['weight'],group['calib_activation_list'])
        calibration_seconds=time.perf_counter()-start
        errors=validate_linear_calibration_result(calibrated,weight.shape,'linear')
        if errors:raise AssertionError(errors)
        report['validated_public_returns']+=1
        state=calibrated['activation_state']
        decoded_b=dequantize_hif4(calibrated['weight_params'],tuple(weight.shape))
        b=decoded_b.float()
        start=time.perf_counter()
        geometry=factor_decoder(b,args.rank_rtol)
        factor_seconds=time.perf_counter()-start
        g=dict(index=gi,shape=list(weight.shape),geometry=geometry['metadata'],cases=[],
               calibration_seconds=calibration_seconds,factor_seconds=factor_seconds,
               provenance={k:group.get(k) for k in ('key','profile','source_model','source_revision','weight_source','source_tensor')})
        for ti,packed in enumerate(group['test_activation_list']):
            a=dequantize_nvfp4(*packed).float()
            target=a@weight.T
            # Preserve BOTH baseline and contestant matmul dtypes, matching
            # evaluate_local. FP32 promotion changes the score denominator.
            baseline_a=dequantize_hif4(quantize_to_hif4(a),tuple(a.shape))
            baseline_out=baseline_a@bw.T
            baseline_mse=float((target-baseline_out).square().mean())
            params,teacher,internal_b,route,seconds=capture_dynamic(solution,packed,state)
            errors=validate_hif4_params(params,packed[0].shape,'activation')
            if errors:raise AssertionError(errors)
            report['validated_public_returns']+=1
            decoded_z=dequantize_hif4(params,tuple(a.shape))
            z=decoded_z.float()
            # Match evaluate_local exactly: BF16 parameters yield a BF16
            # matmul output; do not silently upgrade the scorer to FP32.
            actual=decoded_z@decoded_b.T
            split,continuous=decompose(target,z,teacher,geometry,actual)
            # Algebraic tests use energy-scaled tolerance, not an error floor
            # that would hide rank/orthogonality failures on tiny outputs.
            tolerance=1e-8*max(split['total_exact_decoder_mse'],1e-200)
            if abs(split['geometric_identity_error'])>tolerance:raise AssertionError(split)
            if abs(split['teacher_identity_error'])>tolerance:raise AssertionError(split)
            if split['normal_equation_relative']>1e-9:raise AssertionError(split)
            record=dict(group=gi,test=ti,shape=list(a.shape),route=route,baseline_mse=baseline_mse,
                        official_current_mse=float((target-actual).square().mean()),
                        dynamic_seconds=seconds,evaluation_output_dtype=str(actual.dtype),
                        internal_decoder_max_error=float((internal_b-b).abs().max()) if internal_b is not None else None,
                        teacher_audit=audit_original_transform(solution,a,weight,state,teacher),**split)
            record['mse_reduction_increment_mse']=(record['official_current_mse']-
                                                  record['actual_output_mse_double_reduction'])
            record['evaluation_identity_error']=(record['official_current_mse']-
                record['geometric_floor_mse']-record['discrete_and_search_mse']-
                record['orthogonality_cross_mse']-record['evaluation_output_increment_mse']-
                record['mse_reduction_increment_mse'])
            if abs(record['evaluation_identity_error'])>tolerance:raise AssertionError(record)
            if regression is not None and baseline_mse>0:
                flat_index=sum(len(g['test_activation_list']) for g in bank[:gi])+ti
                expected=regression[flat_index]
                observed=(baseline_mse-record['official_current_mse'])/baseline_mse
                if abs(expected-observed)>1e-12:
                    raise AssertionError(('Production score changed',gi,ti,expected,observed))
                report['production_score_matches']+=1
            if args.probe_rows and args.probe_channels:
                ix=torch.linspace(0,len(a)-1,min(len(a),args.probe_rows)).long()
                witness,candidates=legal_search_witnesses(solution,params,target,teacher,b,continuous,ix,args.probe_channels,decoded_b.dtype)
                for name,candidate in candidates.items():
                    errors=validate_hif4_params(candidate,(len(ix),a.shape[-1]),name)
                    if errors:raise AssertionError(errors)
                    report['validated_probe_returns']+=1
                witness['baseline_sample_mse']=float((target[ix]-baseline_out[ix]).square().mean())
                record['search_witness']=witness
            g['route']=route
            g['cases'].append(record)
            report['cases'].append(record)
        g['summary']=aggregate(g['cases'])
        # Avoid duplicating every case twice in the machine-readable report.
        g['case_ids']=[f"{c['group']}:{c['test']}" for c in g.pop('cases')]
        report['groups'].append(g)
        report['summary']=aggregate(report['cases'])
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('group',gi,g['shape'],g['route'],g['summary'],flush=True)
    after={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in before}
    assert before==after,'Production source changed during diagnostics'
    report.update(complete=True,production_unchanged=True,wall_seconds=time.perf_counter()-started)
    report['by_route']={route:aggregate([c for c in report['cases'] if c['route']==route]) for route in sorted({c['route'] for c in report['cases']})}
    report['by_test_index']={str(i):aggregate([c for c in report['cases'] if c['test']==i]) for i in sorted({c['test'] for c in report['cases']})}
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    output.with_suffix('.md').write_text(markdown(report))
    print('complete',report['summary'],'checks',report['validated_public_returns'],report['validated_probe_returns'],flush=True)


if __name__=='__main__':main()

"""Bounded calibration-helper versus deployed-query fidelity diagnostic."""
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.test_v422 import load
from tools.test_v424 import packed_pair
from hif4_quant import dequantize_hif4

torch.set_num_threads(4)
m = load(437, 'forward_audit')
gen = torch.Generator().manual_seed(438)
records = []
with torch.inference_mode():
    for rows in (64, 256, 1152):
        hold = {'q': packed_pair(rows, 128, gen)}
        z = torch.randn(1, 64, 64, generator=gen)
        h = z @ z.transpose(-1, -2) + torch.eye(64)
        u = m._upper_cholesky_inv(h)
        ix = torch.linspace(0, rows - 1, 16).long()
        for gq, rf in ((1, 0), (1, 1), (0, 0), (0, 1)):
            st = {'kvh': 1, 'u': u, 'H': h, 'gq': gq, 'rf': rf, 'rot': 0}
            predicted = m._joint_quant_queries(hold, st, 2, 1, 64, ix)
            params = m.hif4_dynamic_quantize_q(*hold['q'], 2, 64, st)
            deployed = dequantize_hif4(params, hold['q'][0].shape)[ix]
            internal = m._deq_params(params)[ix]
            records.append({'rows': rows, 'gq': gq, 'rf': rf,
                'helper_vs_internal_changed': int((predicted != internal).sum()),
                'helper_vs_deployed_changed': int((predicted != deployed).sum()),
                'helper_max_error': float((predicted - deployed).abs().max()),
                'deploy_dtype': str(deployed.dtype),
                'helper_dtype': str(predicted.dtype)})
    calls = []
    original = m._joint_quant_queries
    def observed(hold, st, qh, kh, d, ix):
        calls.append({'gq': st.get('gq'), 'rf': st.get('rf'),
                      'rows': len(hold['q'][0]), 'qh': qh, 'kh': kh, 'd': d})
        return original(hold, st, qh, kh, d, ix)
    m._joint_quant_queries = observed
    samples = [{r: packed_pair(n, 128 if r == 'q' else 64, gen)
                for r in 'qkv'} for n in (16, 32, 64)]
    m.hif4_calibration_attention(samples, 2, 1, 64)
    synthetic_calls = len(calls)
    bank = torch.load(ROOT / 'local_fitted_models_attention_feedback_v2/attn.pt', weights_only=False)
    for gi in (0, 24):
        group = bank[gi]
        m.hif4_calibration_attention(group['calib'], group['q_num_heads'],
                                     group['kv_num_heads'], group['head_dim'])
report = {'synthetic': records, 'observed_calibration_calls': calls,
          'synthetic_calibration_call_count': synthetic_calls,
          'bank_groups': [0, 24],
          'scope': 'Synthetic forward checks and two paired calibration groups; not exhaustive or online coverage.'}
(ROOT / 'docs/query-forward-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

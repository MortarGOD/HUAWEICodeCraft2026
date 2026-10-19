"""Build deployment-only content residual balance on the pinned online v430."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = '0638a8cbc7948f018770ff782fe23d44b45ccd23279659b6415d3459096d59ee'
parent = (ROOT / 'solution/solution_v430.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
core = (ROOT / 'tools/attention_content_balance_core.py').read_text()
deployment = '''

_v435_parent_calibration = hif4_calibration_attention
_v435_parent_dynamic = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v435_parent_calibration(samples, qh, kh, d)
    return dict(parent, v_state={'v435_content_parent': parent['v_state']})


@torch.inference_mode()
def hif4_dynamic_quantize_v(vq, vs, kh, d, state):
    if not isinstance(state, dict) or 'v435_content_parent' not in state:
        return _v435_parent_dynamic(vq, vs, kh, d, state)
    original = state['v435_content_parent']
    params = _v435_parent_dynamic(vq, vs, kh, d, original)
    enabled = torch.ones(kh, dtype=torch.bool)
    if isinstance(original, dict) and 'kernel' in original:
        active = original.get('active')
        if not isinstance(active, torch.Tensor) or tuple(active.shape) != (kh,):
            return params
        enabled = ~active.bool()
    if not bool(enabled.any()):
        return params
    x = dequantize_nvfp4(vq, vs).float()
    return _v435_refine(x, params, enabled, kh, d)
'''
source = parent + '\n\n' + core + deployment
ast.parse(source)
path = ROOT / 'solution/solution_v435.py'
path.write_text(source)
staging = ROOT / '.v435_self_check'
staging.mkdir(exist_ok=True)
(staging / 'solution.py').write_text(source)
archive = ROOT / 'solution/submission_v435.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    info = zipfile.ZipInfo('solution.py', date_time=(2026, 9, 8, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, source)
manifest = {'version':435, 'parent':430, 'parent_sha256':PARENT_SHA,
            'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_members':['solution.py']}
(ROOT / 'docs/v435-artifacts.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(manifest))

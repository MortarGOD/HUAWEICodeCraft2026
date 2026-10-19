"""Build a pinned, isolated full-content-risk Attention candidate."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = 'cdf094ec4b3de9d5fe835ea085ab25a8f70c049a7f771308ec0a3834262917ba'
parent = (ROOT / 'solution/solution_v435.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA
core = (ROOT / 'tools/attention_content_risk_core.py').read_text()
deployment = '''

_v436_parent_calibration = hif4_calibration_attention
_v436_parent_dynamic = hif4_dynamic_quantize_v


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v436_parent_calibration(samples, qh, kh, d)
    return dict(parent, v_state={'v436_content_parent': parent['v_state']})


@torch.inference_mode()
def hif4_dynamic_quantize_v(vq, vs, kh, d, state):
    if not isinstance(state, dict) or 'v436_content_parent' not in state:
        return _v436_parent_dynamic(vq, vs, kh, d, state)
    params = _v436_parent_dynamic(vq, vs, kh, d, state['v436_content_parent'])
    x = dequantize_nvfp4(vq, vs).float()
    return _v436_reconstruct(x, params, kh, d)
'''
source = parent + '\n\n' + core + deployment
ast.parse(source)
path = ROOT / 'solution/solution_v436.py'
path.write_text(source)
staging = ROOT / '.v436_self_check'
staging.mkdir(exist_ok=True)
(staging / 'solution.py').write_text(source)
archive = ROOT / 'solution/submission_v436.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    info = zipfile.ZipInfo('solution.py', date_time=(2026, 9, 8, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, source)
manifest = {'version': 436, 'parent': 435, 'parent_sha256': PARENT_SHA,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_members': ['solution.py']}
(ROOT / 'docs/v436-artifacts.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest))

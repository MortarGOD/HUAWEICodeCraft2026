import ast
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
parent=(ROOT/'solution/solution_v435.py').read_text()
parent_sha='cdf094ec4b3de9d5fe835ea085ab25a8f70c049a7f771308ec0a3834262917ba'
assert hashlib.sha256(parent.encode()).hexdigest()==parent_sha
source=parent+'\n\n'+(ROOT/'tools/attention_canonical_dual_metric.py').read_text()+'''

_v438_parent_calibration = hif4_calibration_attention


@torch.inference_mode()
def hif4_calibration_attention(samples, qh, kh, d):
    parent = _v438_parent_calibration(samples, qh, kh, d)
    return dict(parent, q_state=_v438_state(parent['q_state'], 'q', kh, d),
                k_state=_v438_state(parent['k_state'], 'k', kh, d))
'''
ast.parse(source)
(ROOT/'solution/solution_v438.py').write_text(source)
staging=ROOT/'.v438_self_check';staging.mkdir(exist_ok=True)
(staging/'solution.py').write_text(source)
archive=ROOT/'solution/submission_v438.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    info=zipfile.ZipInfo('solution.py',date_time=(2026,9,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
    z.writestr(info,source)
manifest={'version':438,'parent':435,'parent_sha256':parent_sha,
    'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
    'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_members':['solution.py']}
(ROOT/'docs/v438-artifacts.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))

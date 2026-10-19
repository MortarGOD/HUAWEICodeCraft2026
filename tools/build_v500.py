"""v500 = v441 + stacked GPTQ mechanisms: deployed-coordinate Grams, Gauss-Newton
output metric, and a learned per-head Q/K gauge rotation.

Splice order matters and is deliberate:

    parent v441
      -> gptq_covariance_core      : fresh deployed-coordinate Grams (Hessian)
      -> pv_full_metric_core       : full-key Gauss-Newton output metric, blended in
      -> learned_rotation_core     : Cayley-learned per-head joint Q/K rotation

* gptq_covariance_core calls _pvfull_guard, which pv_full_metric_core defines later in
  the module; Python resolves that at call time, so both mechanisms stay active.
* learned_rotation_core needs _cov_transform from gptq_covariance_core, hence the order.
* The Q/K joint rotation is a pure gauge freedom (q'k'^T = qk^T for q'=qR, k'=kR), so it
  cannot change the exact attention output; it only changes quantization error.
"""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 500
PARENT = 441
PARENT_SHA = 'b570f46dc7d8b38eedffdc579eb849f647dced2f8c928df731083a9497170fd4'
CORES = [
    'tools/gptq_covariance_core.py',
    'tools/pv_full_metric_core.py',
    'tools/learned_rotation_core.py',
]
parent = (ROOT / f'solution/solution_v{PARENT}.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA, 'parent hash changed'
source = parent
for core in CORES:
    source += '\n\n' + (ROOT / core).read_text()
source += '\n'
ast.parse(source)
dest = ROOT / f'solution/solution_v{VERSION}.py'
dest.write_text(source)
staging = ROOT / f'.v{VERSION}_self_check'
staging.mkdir(exist_ok=True)
(staging / 'solution.py').write_text(source)
archive = ROOT / f'solution/submission_v{VERSION}.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    info = zipfile.ZipInfo('solution.py', date_time=(2026, 9, 10, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    z.writestr(info, source)
manifest = {'version': VERSION, 'parent': PARENT, 'parent_sha256': PARENT_SHA,
            'cores': CORES,
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_members': ['solution.py']}
(ROOT / f'docs/v{VERSION}-artifacts.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))

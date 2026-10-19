"""v501 = v441 with the calibration Hessian carried in fp32 instead of bf16.

`hif4_calibration_attention` stores the logit-space Hessian as

    Hk_bf = Hk.to(torch.bfloat16).contiguous()
    Hq_bf = Hq.to(torch.bfloat16).contiguous()

and `_refine_attn_heads` then uses it (via `gw = H[hv].float()`) to steer a greedy
top-1 lattice refinement, whose accept/reject test is `2*s*d*M + d^2*H[c,c]` with `M = dq @ H`.
bf16 keeps 8 mantissa bits, so every entry of the Hessian is perturbed by ~0.4% before it is
promoted back to fp32; decisions whose margin is below that perturbation are essentially coin
flips. `torch.float32` is in `FROZEN_STATE_ALLOWED_TENSOR_DTYPES`, so carrying fp32 is legal.

This build is a targeted text patch rather than a core append, because the cast lives inside a
250-line function. The patch is exact-match and asserted, so the parent hash still pins the
source.
"""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 501
PARENT = 441
PARENT_SHA = 'b570f46dc7d8b38eedffdc579eb849f647dced2f8c928df731083a9497170fd4'
PATCHES = [
    ('            Hk_bf = Hk.to(torch.bfloat16).contiguous()\n'
     '            Hq_bf = Hq.to(torch.bfloat16).contiguous()\n',
     '            # v501: carry the refinement Hessian at full precision; bf16 perturbs every\n'
     '            # entry by ~0.4% and the greedy flip test compares sub-percent margins.\n'
     '            Hk_bf = Hk.to(torch.float32).contiguous()\n'
     '            Hq_bf = Hq.to(torch.float32).contiguous()\n'),
]
parent = (ROOT / f'solution/solution_v{PARENT}.py').read_text()
assert hashlib.sha256(parent.encode()).hexdigest() == PARENT_SHA, 'parent hash changed'
source = parent
for old, new in PATCHES:
    assert source.count(old) == 1, ('patch anchor not unique', old[:60], source.count(old))
    source = source.replace(old, new)
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
            'patch': 'calibration Hessian H carried fp32 instead of bf16',
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_members': ['solution.py']}
(ROOT / f'docs/v{VERSION}-artifacts.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))

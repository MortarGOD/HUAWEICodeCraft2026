"""v502 = v501 + the K-side balancing factor carried at full precision.

`_qks_fit` rounds the balancing scale s to bf16 once and returns it as fp32, so the Q side is
exact: deployment multiplies Q by the same bf16 numbers calibration used.

The K side is not. The state carries

    k_state["ks"] = (1.0 / qs_s).to(torch.bfloat16)

while calibration applies K through `_qks_apply_q(k, qs_s, inv=True)`, i.e. it divides by the
exact fp32 reciprocal. bf16 keeps 8 mantissa bits, so the deployed K is divided by a
perturbation of the reciprocal of up to ~0.2% per channel, and q*k^T no longer cancels the
gauge exactly -- the logits acquire a per-channel multiplicative perturbation of the same
order as the bf16/fp32 attention difference measured in Round 4.

Carrying the reciprocal in fp32 makes deployment reproduce calibration to within one fp32 ulp.
`torch.float32` is in FROZEN_STATE_ALLOWED_TENSOR_DTYPES.
"""
import ast
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 502
PARENT = 501
PARENT_SHA = '0cae5cf67ff1526d2367e890bc5d0ef62e5bb610b97e88f2a59619ee8b5e137f'
PATCHES = [
    ('        q_state["qs"] = qs_s.to(torch.bfloat16).contiguous()\n'
     '        k_state["ks"] = (1.0 / qs_s).to(torch.bfloat16).contiguous()\n',
     '        # v502: s is already bf16-valued, so the Q side is exact either way; the\n'
     '        # reciprocal is NOT bf16-representable, and calibration divides by the exact\n'
     '        # fp32 reciprocal, so carrying bf16 here breaks the q*k gauge cancellation.\n'
     '        q_state["qs"] = qs_s.to(torch.float32).contiguous()\n'
     '        k_state["ks"] = (1.0 / qs_s).to(torch.float32).contiguous()\n'),
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
            'patch': 'K-side balancing factor (qs reciprocal) carried fp32; restores exact '
                     'q*k gauge cancellation between calibration and deployment',
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'archive_members': ['solution.py']}
(ROOT / f'docs/v{VERSION}-artifacts.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))

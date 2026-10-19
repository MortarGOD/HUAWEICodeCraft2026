"""Rebuild our mirror banks with the OFFICIAL calibration layout.

Round 8/9 found the blind spot: every calibration decision is guarded on the LAST calibration
sample, the official sample puts a degenerate entry there (attention T=1, linear 1 row), and our
mirror puts that entry first (linear) or omits it (attention, 0/50 groups). So a mechanism that
fires at full strength locally can be silently disabled on the judge, which is what makes local
deltas non-predictive.

This writes a diagnostic copy of the mirror banks in which each group's LAST calibration entry is
replaced by a degenerate slice of its own first entry, reproducing the official layout faithfully
(fit on the informative entries, guard on the degenerate one):

  attention  calib[-1] <- calib[0] sliced to a single token
  linear     calib_activation_list[-1] <- first entry sliced to a single row

The test sets are untouched, so scores stay comparable with the original banks. This is an
instrument for mechanism self-checks only: it is a synthetic layout, so no online extrapolation.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def one_token_attn(sample):
    out = {}
    for role in ('q', 'k', 'v'):
        packed = sample[role]
        out[role] = [packed[0][0:1].contiguous(), packed[1][0:1].contiguous()]
    return out


def one_row_lin(pair):
    return (pair[0][0:1].contiguous(), pair[1][0:1].contiguous())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=str(ROOT / 'local_fitted_models_attention_feedback_v2'))
    ap.add_argument('--dst', default=str(ROOT / 'local_fitted_models_official_layout'))
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    dst.mkdir(exist_ok=True)

    attn = torch.load(src / 'attn.pt', map_location='cpu', weights_only=False)
    changed_a = 0
    for g in attn:
        calib = list(g['calib'])
        before = int(calib[-1]['q'][0].shape[0])
        calib[-1] = one_token_attn(calib[0])
        after = int(calib[-1]['q'][0].shape[0])
        assert after == 1, after
        g['calib'] = calib
        changed_a += (before != 1)
    torch.save(attn, dst / 'attn.pt')
    print('attention: %d groups, %d had a non-degenerate last entry replaced by T=1'
          % (len(attn), changed_a))
    print('  after: last-entry T =', sorted({int(g['calib'][-1]['q'][0].shape[0]) for g in attn}), flush=True)

    lin = torch.load(src / 'linear.pt', map_location='cpu', weights_only=False)
    changed_l = 0
    for g in lin:
        cal = list(g['calib_activation_list'])
        before = int(cal[-1][0].shape[0])
        cal[-1] = one_row_lin(cal[0])
        after = int(cal[-1][0].shape[0])
        assert after == 1, after
        g['calib_activation_list'] = cal
        changed_l += (before != 1)
    torch.save(lin, dst / 'linear.pt')
    print('linear: %d groups, %d had a non-degenerate last entry replaced by 1 row'
          % (len(lin), changed_l))
    print('  after: last-entry rows =', sorted({int(g['calib_activation_list'][-1][0].shape[0]) for g in lin}))

    manifest = {
        'schema': 'official-layout-diagnostic-v1',
        'purpose': ('Mirror banks with the official calibration layout (degenerate entry LAST), '
                    'for mechanism self-checks on guard-sample degeneracy.'),
        'score_prediction_allowed': False,
        'source': str(src),
        'attention': {'groups': len(attn), 'last_entry_replaced': changed_a},
        'linear': {'groups': len(lin), 'last_entry_replaced': changed_l},
    }
    (dst / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if (src / 'README.md').exists():
        shutil.copy(src / 'README.md', dst / 'README.md')
    print('manifest written to', dst / 'manifest.json')


if __name__ == '__main__':
    main()

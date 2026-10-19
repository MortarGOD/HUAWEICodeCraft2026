"""Use every eligible pre-holdout calibration sample for factor-risk fitting."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def multical_factor_source():
    source = (ROOT / 'tools/attention_factor_risk_core.py').read_text()
    assert source.count('_AF_SEARCH_RIDGE=.1') == 1
    source = source.replace('_AF_SEARCH_RIDGE=.1', '_AF_SEARCH_RIDGE=1.', 1)
    old = "and s['v'][0].numel()<=4194304][-2:]"
    new = "and s['v'][0].numel()<=4194304][-4:]"
    assert source.count(old) == 1
    return source.replace(old, new, 1)

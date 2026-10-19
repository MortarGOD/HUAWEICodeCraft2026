# Paper-direction solution candidates

All archives contain exactly one root-level `solution.py`. The compact v1 ZIP
judge reports `valid` with 20/20 cases for every candidate.

## Suggested online order

| Order | Archive | Direction | Qwen Linear | Qwen total | SHA256 |
|---:|---|---|---:|---:|---|
| 1 | `zips/baseline-block-hessian.zip` | SmoothQuant + block Hessian baseline | 66.234898 | 73.111759 | `D0AF1A2C672A1B79065FA5F4F56D2D306CB61FA5CF82E1694DB91E6FE2057032` |
| 2 | `zips/rrs.zip` | RRS: SmoothQuant followed by block-64 Hadamard | 75.478392 | 82.355253 | `CCAED3372120F63599C9359779030A582C2D8BDC56DAD757605062921DA88A74` |
| 3 | `zips/rrs-let.zip` | RRS with LET alpha selection | 75.349642 | 82.226503 | `DED550756669C0D16E3CF98E7EBE10529A5E0997C5471499BEC410DF975816B7` |
| 4 | `zips/let.zip` | LET-style alpha selection (0.25, 0.5, 0.75) | 67.201977 | 74.078838 | `CF10B301802FA326D5F0F0B328D2834BD3794DBAFA994FFDA808316B4A525FA1` |
| 5 | `zips/rht.zip` | QuaRot/QuIP#-style signed randomized Hadamard | 66.243003 | 73.119864 | `57B41EA8B96D1F302F234F8E6EFBEB6DCC110BCE2C4DE0518D22C43E260AAA39` |

The Qwen total adds the shared Attention score `6.876861`. These local scores
are diagnostic only; online results should determine the production baseline.

## Candidate isolation

- `rrs/solution.py`: adds only the SmoothQuant + Hadamard Linear family.
- `rrs-let/solution.py`: combines RRS with three calibration-selected LET scales.
- `let/solution.py`: adds only the three calibration-selected LET scales.
- `rht/solution.py`: adds only a fixed signed randomized Hadamard Linear family.

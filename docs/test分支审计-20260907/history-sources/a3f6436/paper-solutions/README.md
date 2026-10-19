# Paper-direction solution candidates

All archives contain exactly one root-level `solution.py`. The compact v1 ZIP
judge reports `valid` with 20/20 cases for every candidate.

## Suggested online order

| Order | Archive | Direction | Qwen Linear | Qwen total | SHA256 |
|---:|---|---|---:|---:|---|
| 1 | `zips/rrs-fast-fp16.zip` | Direct RRS, FP16 online Hessian state | 75.471469 | 82.348330 | `6F29D1769333A00445DC99AA7AD962D76E106E45DBBDADE622D9749019AE8326` |
| 2 | `zips/rrs-fast.zip` | Direct RRS, full FP32 block Hessian | 75.478338 | 82.355199 | `0B5928888A7D8F3FC13473EF044B967EDFF493E86752CF52A2A29114361ACDCD` |
| 3 | `zips/spin-lwc.zip` | Learned Cayley rotation + LWC on top of RRS | 75.542190 | 82.419051 | `C1B2A2F051912A6C64F72E005392E88CDF41FF39E7E0B34A70BA289E941E9C9C` |
| 4 | `zips/rrs.zip` | RRS: SmoothQuant followed by block-64 Hadamard | 75.478392 | 82.355253 | `CCAED3372120F63599C9359779030A582C2D8BDC56DAD757605062921DA88A74` |
| 5 | `zips/rrs-let.zip` | RRS with LET alpha selection | 75.349642 | 82.226503 | `DED550756669C0D16E3CF98E7EBE10529A5E0997C5471499BEC410DF975816B7` |
| 6 | `zips/rrs-lite.zip` | Direct RRS, unweighted radius-1 | 66.773276 | 73.650137 | `B673D8EA0EB73A9A9441C4EFD96FDD19435E07B8F97E0E357C40C6EE9D1B6ACF` |
| 7 | `zips/let.zip` | LET-style alpha selection (0.25, 0.5, 0.75) | 67.201977 | 74.078838 | `CF10B301802FA326D5F0F0B328D2834BD3794DBAFA994FFDA808316B4A525FA1` |
| 8 | `zips/rht.zip` | QuaRot/QuIP#-style signed randomized Hadamard | 66.243003 | 73.119864 | `57B41EA8B96D1F302F234F8E6EFBEB6DCC110BCE2C4DE0518D22C43E260AAA39` |
| 9 | `zips/baseline-block-hessian.zip` | SmoothQuant + block Hessian baseline | 66.234898 | 73.111759 | `D0AF1A2C672A1B79065FA5F4F56D2D306CB61FA5CF82E1694DB91E6FE2057032` |

The Qwen total adds the shared Attention score `6.876861`. These local scores
are diagnostic only; online results should determine the production baseline.

## Candidate isolation

- `rrs/solution.py`: adds only the SmoothQuant + Hadamard Linear family.
- `rrs-let/solution.py`: combines RRS with three calibration-selected LET scales.
- `let/solution.py`: adds only the three calibration-selected LET scales.
- `rht/solution.py`: adds only a fixed signed randomized Hadamard Linear family.
- `spin-lwc/solution.py`: learns a shared Cayley rotation and clipping ratios,
  while retaining the complete RRS candidate pool as a fallback.
- `rrs-fast/solution.py`: directly executes the selected RRS block-Hessian
  route for multi-sample calibration, avoiding the redundant family search.
- `rrs-fast-fp16/solution.py`: reduces online Hessian state copy size by half.
- `rrs-lite/solution.py`: removes the Hessian and radius-3 search for maximum
  timeout margin.

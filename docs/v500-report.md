# v500 三机制叠加 GPTQ（未达标，不交付）

父版 v441（SHA `b570f46dc7d8b38eedffdc579eb849f647dced2f8c928df731083a9497170fd4`）。
本候选 SHA `7b2c3f342059545e122e4e781f05d1ea82ffa3c0c1b691bdfc510bdf080972c6`，
归档 `solution/submission_v500.zip` SHA `a366597a524e50485ca12825d96dba26f23a5328f1f8d4a18455be4abe39d4da`。
self_check 22/22 通过。

## 机制

按下列顺序拼接（顺序是设计的一部分，见 `tools/build_v500.py` 注释）：

1. `tools/gptq_covariance_core.py` —— 部署坐标下重新拟合 Q/K 的 Gram（Hessian），
   可选去掉 Q 的 softmax gauge。三个 core 中从未被任何版本使用过。
2. `tools/pv_full_metric_core.py` —— **全 key、不子采样**的部署 Gauss–Newton 输出度量，
   按 head 守门后与父版阻尼逆混合。从未使用过。
3. `tools/learned_rotation_core.py` —— Cayley SGD 学习的 per-head Q/K 联合旋转
   （纯规范自由度，q'k'ᵀ = qkᵀ），以真实部署输出 MSE 为目标，STE 穿过 GPTQ。从未使用过。

拼接顺序使三者同时生效：`gptq_covariance_core` 调用 `_pvfull_guard`（由第 2 个 core 稍后定义，
Python 在调用时解析）；`learned_rotation_core` 依赖 `_cov_transform`（由第 1 个 core 提供）。

## 实测

| 数据集 | v441（父） | v500 | Δ 本地点 | Δ 线上（×1.294） |
|---|---|---|---|---|
| `public_quantizer`（5 case） | 13991.0 | **13994.6** | **+3.6** | +5 |
| `attention_history_fit_v1`（28 case） | 10796.2 | 10788.8 | −7.4 | −10 |

## 为什么几乎无效（调试证据）

机制**确实运行**，是**守门拒绝**：

- `PVFULL_DEBUG`：`fit_samples=2`，接受 `q_heads=1, k_heads=0`；
  holdout 代价比 `[[1.0, 0.9949], [1.0377, 1.0241]]` —— 提议的 Q 度量好 0.5%（刚过 0.995 门限），
  提议的 K 度量差 3.8%。
- `LR_DEBUG`：`training_history` 从 1.0 降到 0.921（训练损失 −8%），
  `accepted_trials=[3,5]`，`orthogonality_error=1.0e-06`，但
  **`fold_accept=[false,false]`、`adopted=0`** —— 旋转在训练样本上有效，在留出样本上完全不泛化。

根因见 `six-tier-feasibility-20260910.md`：

1. 当前残余误差的 **78% 在 V**，Q/K 侧任何改进的上限只有 22%；
2. Q/K 的旋转规范自由度**已被父版取满**（`_make_R(dh,0)` = Walsh–Hadamard，
   在 12 个 seed 中已是最优，比不旋转好 24.6%）；
3. `public_quantizer` 只有 5 个校准样本，4 个可训练 + 1 个守门，
   高维机制（per-head 256×256 旋转 = 65536 参数）必然过拟合。

## 结论

**拒绝交付**：本地点增量 +3.6，距六档最低要求的 +151 差 40 倍。
本地条件估分约 21309，对应线上第 0 档（基线 21304 之上 +5）。

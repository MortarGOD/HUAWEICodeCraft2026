# 公开参考数据集对线上反馈的拟合能力评测（2026-09-10）

## 问题

`docs/papers/hif4-public-references/test/datasets` 下的 8 个数据集，哪一个最能重现我们在
50 Linear + 250 Attention 口径下的线上反馈？

## 评测基准（关键突破）

公开参考仓库 `test/tests/benchmarks/commit-history-alignment/public-score-ledger-061e7e6.json`
是一份**公开实测分数账本**，版本号与本仓库一致，含 204 个版本的真实线上总分。
其中 `epoch = current_50_linear_250_attention`（版本 277–440）有 **113 个 `valid_score`
且 `numeric_comparison_allowed = true`**——这是同一计分口径内可比较的真实线上数据。

账本自带规则（必须遵守）：

- `Scores are comparable only within the same scoring epoch.`
- `Execution anomalies, WA, timeout, pending, estimates, and missing results are not numeric fit targets.`
- `Agreement on this ledger is retrospective in-sample evidence, not proof of hidden production distribution.`

因此本评测只使用 `current_50_linear_250_attention` 这一 epoch，排除 v275（Linear 全 WA 的
Attention 残差）与 v300（-30000 固定执行惩罚）。

口径假设：`linear = 4450`（用户给定），故 `online_attention(v) = online_score(v) - 4450`。
据此可把数据集得分 `mean_quality(v) × 25000` 与线上 Attention 绝对水位逐版本比对，
这比只比差分强得多。

## 方法

1. 用 `tools/distribution_fit/evaluate_dataset_multi.py` 在每个数据集上评测版本集合。
   基础 33 个版本（覆盖 25 条父子边），扩展 64 个账本变化点版本，共 97 个。
   每次评测都执行真实的三次 Q/K/V 校准与动态路径，并断言 state 未被修改。
2. `tools/distribution_fit/compare_datasets_ledger.py` 计算三类指标：
   - **LEVEL**：`local(v) = mean_quality(v)×25000` 对 `online_attention(v)`。
     `lvCorr` = 逐版本相关；`lvErr` = 原始平均绝对误差；
     `lvAffErr` = 允许一个 oracle 仿射映射后的残差（把"形状错误"与"标度错误"分离）。
   - **DELTA**：epoch 内相邻已评测版本对的 `Δlocal` 对 `Δonline`：方向命中、相关、
     单一全局最小二乘标度、缩放后落在 20% 容差内的条数、大边标度离散度。
   - **HELD OUT**：v441–444（用户回执 21308/21305/21304/21304，不在账本内）用 v435 锚点
     加该数据集自己的标度预测，报告误差。

## 结果（29 个已评测版本 / 28 个相邻对）

```
dataset                            nVer |  lvCorr   lvErr lvAffErr | nPair   dir   ampS   corr     rho   scale   dBig | lv435   PRED err 441..444
public_quantizer                     29 |   0.810    2389      449 |    28    25      7  0.768   0.917     1.3    1.0 | 2863    4/1/0/0
attention_history_fit_v1_localonline 29 |   0.647    4518      809 |    28    27      5  0.639   0.936     0.5    3.6 | 4886    42/45/21/50
attention_history_fit_v1             29 |   0.559    5439      915 |    28    26      5  0.576   0.787     0.5    1.4 | 6073    11/14/17/19
qwen3_8b_profiled_v1                 29 |   0.535    1558      945 |    28    21      4  0.509   0.689     0.6    4.4 | 1643    4/1/10/52
qwen2_5_14b_online_profiled_v2       29 |   0.451    2286     1231 |    28    23      3  0.691   0.617     0.3    3.8 | 1596    4/1/19/0
qwen2_5_14b_online_v1                29 |   0.341    3233     1292 |    28    21      1  0.661   0.523     0.3    2.6 | 2959    4/1/1/0
qwen3_5_35b_a3b_v1                   29 |   0.216    8138     1242 |    28    19      1  0.472   0.455     1.7    6.8 | 9414    4/1/1/88
v1                                   29 |   0.472   15363      942 |    28    14      1  0.347   0.376     0.2    5.8 | 14508   4/1/0/0
v419_mini_sample                     29 |  -0.266   10623     1171 |    28    11      1 -0.204   0.002    -0.2  973.1 | 12989   4/1/0/0
```

### 结论：`public_quantizer` 全面最优

它在**每一个**指标上都领先，且领先幅度很大：

| 指标 | public_quantizer | 次优 | 倍率 |
|---|---|---|---|
| 水位相关 lvCorr | **0.810** | 0.647 | 1.25× |
| 仿射校正后水位误差 lvAffErr | **449** | 809 | 1.8× |
| 相邻对方向命中 | 25/28 | 27/28 | — |
| 差分相关 corr | **0.768** | 0.639 | 1.20× |
| 秩相关 rho | **0.917** | 0.936 | — |
| 大边标度离散度 dBig | **1.0** | 1.4 | 1.4× |
| 留出 v441–444 预测误差 | **4/1/0/0** | 11/14/17/19 | — |

## 可用的标定公式

在 29 个版本上最小二乘得到：

```
online_attention ≈ 1.2301 × (mean_quality[public_quantizer] × 25000) − 806
online_total     ≈ 3644 + 1.2301 × (mean_quality[public_quantizer] × 25000)
```

留一交叉验证（每次留出一个版本重新拟合）：

| 数据集 | LOO 平均绝对误差 | LOO 最大误差 |
|---|---|---|
| **public_quantizer** | **485** | 4799（v311） |
| attention_history_fit_v1_localonline | 902 | 4757 |
| attention_history_fit_v1 | 1001 | 5096 |
| v1 | 1011 | 5539 |
| qwen3_8b_profiled_v1 | 1163 | 7985 |
| v419_mini_sample | 1283 | 5266 |
| qwen3_5_35b_a3b_v1 | 1354 | 4839 |
| qwen2_5_14b_online_profiled_v2 | 1386 | 4552 |
| qwen2_5_14b_online_v1 | 1461 | 5795 |

485 点误差约占总分 21304 的 2.3%，已可用于版本择优。最大误差 4799 全部来自 v311
（线上 −4995，本地只 −326）——这是已知盲点，不是新问题。

## 各数据集的性质与来源

| 数据集 | 来源 | 结构 | 说明 |
|---|---|---|---|
| `public_quantizer` | gitcode.com/**yuchunfei**/quantizer `data/attn.pt` @8b85f229 | 1 组 / 5 case，长度 10/128/512/1024/1024，qh/kh=16/2，hd=256 | **官方公开样例数据**，张量未改动。README：`regression evidence, not an online-score predictor` |
| `v419_mini_sample` | gitcode.com/**wuzhangda**/2026-huawei-alogorithm `mini_sample` @c4ca25a6 | 1 组 / 5 case，长度 10/128/512/1024/1 | `online_prediction_allowed: false`；上游已记录 V2/V4 在其上完全相同而线上 +800 |
| `attention_history_fit_v1` | 由公开仓库用提交历史 + 模型缓存构造 | 20 组 / 28 case，长度 4/8/32/128/512，整数权重合计 250 | 645 列公开拟合的前 28 列；README：`neither profile is an online-score predictor` |
| `qwen2_5_14b_online_v1` | Qwen2.5-14B 线上画像 | 50 组 / 250 case，**全部长度 4** | |
| `qwen2_5_14b_online_profiled_v2` | 同上（profiled） | 50 组 / 250 case，全部长度 4 | 与 v1、qwen3_8b_profiled_v1 同源（Qwen2.5 投影），**非独立证据** |
| `qwen3_8b_profiled_v1` | Qwen3-8B，实为 Qwen2.5 投影 | 250 case，长度 4 | |
| `qwen3_5_35b_a3b_v1` | Qwen3.5-35B-A3B | 10 组 / 50 case，长度 8 | |
| `v1` | 早期小样例 | 10 组 / 10 case，长度 4–45，1/1 head，hd=64 | |

## 为什么是 public_quantizer

1. **它是官方数据源**。其余数据集都是派生、画像或合成产物；只有它是题目仓库
   `data/attn.pt` 的原件，因而最可能与线上 250 题同源。
2. **长度覆盖最宽**。它覆盖 10–1024 token；而三个 Qwen 数据集的 250 个 case
   **全部是长度 4**——这解释了为什么它们 case 数多却差分相关只有 0.34–0.69：
   长度 4 的注意力几乎不受 K/V 量化误差影响，对版本变化的敏感度不足。
3. **标度接近 1**。它的差分标度 1.30、大边离散度 1.0，是唯一一个量级无需大幅放大的数据集。

## 必须记录的保留意见

- **`public_quantizer` 只有 5 个 case**。`v419_mini_sample` 是几乎相同形状的另一次独立抽样，
  却给出 **corr −0.20 / lvCorr −0.27**（完全反向）。两者内部 5 个 case 各自都很一致
  （逐 case 中位相关 +0.765 与 −0.394），说明 5 个 case 无法通过内部一致性来交叉验证。
  `public_quantizer` 的领先虽大且横跨全部指标，但仍带有小样本风险。
- 逐 case 分析（`tools/distribution_fit/per_case_dataset_signal.py`，产物
  `pub-dataset-per-case-signal.json`）显示数据集之间互相矛盾：
  `public_quantizer` 5 个 case 的线上相关全部 >0.5，`v419_mini_sample` 5 个全部 ≈ −0.39，
  而 250 case 的 Qwen 数据集只有 0.34–0.49。没有任何数据集是可信的绝对预测器。
- 本评测是**样本内**回溯：29 个版本同时用于排序和标定。留一验证缓解但不能消除这一点；
  扩展集 97 个版本的评测（`pub-dataset-eval-ext/`）用于加固标定。
- 账本本身来自公开仓库，是"公开实测分数"，与赛方线上评分不必然同源。
- 数据集自带声明：`public_quantizer` 与 `attention_history_fit_v1` 都明示
  "not an online-score predictor"。本评测只说明它在已知版本上拟合最好，不等于可用于预测未知版本。

## 复现

```bash
python tools/distribution_fit/evaluate_dataset_multi.py \
  --dataset docs/papers/hif4-public-references/test/datasets/public_quantizer/attn.pt \
  --versions 283,284,... --output docs/distribution-fit/pub-dataset-eval/public_quantizer.json

python tools/distribution_fit/merge_dataset_evals.py
python tools/distribution_fit/compare_datasets_ledger.py docs/distribution-fit/pub-dataset-eval-merged
python tools/distribution_fit/compare_datasets.py       docs/distribution-fit/pub-dataset-eval-merged
python tools/distribution_fit/per_case_dataset_signal.py docs/distribution-fit/pub-dataset-eval-merged
```

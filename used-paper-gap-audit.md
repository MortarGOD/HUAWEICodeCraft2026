# 已使用论文的未挖掘点审计

## 统计口径

“已使用”不等于论文机制已经完整落地。本文件逐项对照当前 v7，区分：

- 已落地：代码存在且进入当前提交路径；
- 未挖掘：论文中的另一个独立机制尚未实现；
- 接口边界：是否能在固定 HiF4 返回格式和 300 秒限制内合法实现。

## 审计矩阵

| 论文 | 当前已经使用 | 尚未挖掘 | 可落地候选 | 优先级 |
|---|---|---|---|---|
| HiFloat4 | E6M2/E1_8/E1_16 层级格式；E1 子树精确 SSE 选择 | mant 只用 half-to-even RTN；没有 half-away 或任务感知上下舍入；所有角色共享同一 E6 offset 集 | tie-aware rounding；敏感 block mant 上/下邻级；Q/K/V role-specific offsets | 高 |
| Four Over Six | 同一 64-block 生成多个 E6 scale 并按 SSE 选 | 论文的多候选思想尚未扩展到 rounding mode、role 和输出目标 | 保持表示合法，联合搜索 E6 offset + 少量 rounding policy | 高 |
| MXFP4 OAS/MBS | 允许 base 附近较小/较大 scale，包含轻微 clipping 候选 | offset 范围固定，未根据峰均比、zero-collapse 或 role 自适应 | 由校准分布选择保守/激进 offset 集 | 中 |
| SmoothQuant | v1 Linear 固定 alpha=0.875；Attention 用解析平方根平衡 Q/K | 当前最佳没有搜索 rotation 后 alpha/gain；没有跨 calibration fold 的稳健选择 | 在最终 transform 内只搜索少量 alpha/gain，并保留解析平方根回退 | 高 |
| RPTQ | 依据通道峰值生成 sorted permutation | 只有 identity/全排序；没有稳定聚类、block assignment 和校准漂移保护 | K 个范围桶；候选 permutation 在 holdout 上选择 | 高 |
| QuaRot | 每个 head 内 Q/K 成对正交 FWHT，严格保持 logits | 没有 full-head/多级 Kronecker 排列消融；没有 rotation ordering 比较 | 保持同一 Q/K 变换，仅扩展合法 block decomposition | 中 |
| QuIP# | deterministic random sign + Hadamard | sign 只有 seed 0/1；没有小型 incoherence ensemble | seed 0..3 或低差异 sign 集，校准选择一个，在线 state 不变 | 高 |
| RRS | 重排、随机符号、block Hadamard、旋转后动态 HiF4 scale | runtime smooth 无法原样实现；当前仍未单独归因 permutation/sign/block/smooth | 逐项消融；用 HiF4 动态 E6 作为 runtime block statistic 的合法代理 | 高 |
| BRECQ | holdout Attention 最终输出 MSE 选择 transform | 只有最后一个 calibration sample；没有多 fold、混合 logit/output 目标和局部重构稳定项 | 小样本 leave-one-out；logit MSE + output MSE 混合评分 | 高 |
| OmniQuant | v2/v3 搜索等价平滑参数和输出误差；v7 保留等价变换思想 | 没有把 alpha/gain 搜索与最佳 RRS transform 合并；没有轻量坐标更新 | 固定 transform 后搜索少量 alpha/gain，不学习 dense affine | 高 |
| NVFP4 | 输入反量化与 block16 scale 解析 | 输入格式由题目固定，无法改变其 pre-scale/global-scale | 仅用于构造更精确的校准统计，不单独形成版本 | 低 |

## 最值得回收的五个点

### 1. RRS/QuIP# sign ensemble

当前只比较全正 sign 和一个伪随机 sign。扩展到 4 个 deterministic sign 不增加动态计算或 state 大小，只增加校准候选。它与 DuQuant permutation 是正交变量，必须分版本测试。

v9 已实现该候选：保留 seed 0/1，并加入跨 head_dim 64/128/256 最大化最小归一化海明距离的 seed 7/163；等待线上验证。

### 2. BRECQ 稳健目标

v7 虽然修正了“平滑前选择、平滑后部署”的错配，但仍用最后一个 calibration sample 做 holdout。下一步可做低成本 leave-one-out 或把目标改为：

`loss = output_mse + lambda * normalized_logit_mse`

其中 lambda 只能从少量固定候选中选择，避免把 holdout 本身过拟合。

v10 先实现最小风险版本：保持 v7 的 4 个 fit 样本不变，仅将 transform 的最终 Attention 输出评分从最后 1 个样本扩展为最后 2 个样本。logit/output 混合目标留待该版本线上有效后再单独测试。

### 3. SmoothQuant/OmniQuant 的 rotation 后参数搜索

当前 reciprocal smooth 固定为 `sqrt(q_peak/k_peak)`。该公式平衡峰值，却不保证 HiF4 Attention 输出最优。可以在最终 rotation 固定后搜索少量 alpha/gain；这不改变严格等价性，因为 Q 除以 s、K 乘以同一 s。

### 4. HiF4/Four Over Six 的任务感知舍入

当前已经把 E6/E1 metadata 搜得较充分，但 mant 仍是 RTN。对于固定 Weight，可在校准 A@W 上选择少量 mant 上/下邻级；对于动态 Q/K，只能学习 role/block 级 rounding threshold，不能拟合测试样本输出。

v13 已实现第一阶段受限 AdaRound：固定 v11 的 E6/E1 metadata 和 Activation 路径，只在原始 RTN、0.375、0.625 三个全局 Weight mant 阈值中按 sampled-row 校准 A@W 选择；RTN 始终保留为逐组回退候选。

### 5. RPTQ 的稳健分桶

全排序会让相邻通道高度同质，但对 calibration peak 漂移敏感。可把通道划为少量 log-range bucket，在 bucket 内保持原顺序或用确定性交错，降低 permutation 对单个峰值的依赖。

## 与第二轮新论文队列的关系

- v8 仍优先测试 DuQuant zigzag permutation，因为它直接扩展当前已证实有效的 Attention RRS；
- 若 v8 无收益，优先回收 QuIP#/RRS sign ensemble；
- 若 v8 有收益，先单独测试 DuQuant 第二 rotation，再进入 BRECQ 稳健目标；
- SmoothQuant/OmniQuant alpha/gain 搜索必须在 transform 固定后单独测试；
- HiF4 mant adaptive rounding 和 GPTQ/AdaRound 合并到固定 Weight 分支，不与 Attention 实验混跑。

## Acceptance criteria

所有回收候选统一满足：

1. 只改变一个主要变量；
2. Q/K 变换保持 FP32 logits 等价误差 <2e-4；
3. 单文件隔离导入成功，self-check 22/22；
4. state 只包含有限 CPU 基础对象/张量；
5. 线上耗时 <300 秒；
6. 得分不高于父版本则回退，并记录该机制为修改或放弃。

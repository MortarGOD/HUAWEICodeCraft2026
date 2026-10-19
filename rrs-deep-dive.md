# RRS 深度研究与 HiF4 适配

## 核心结论

RRS 将异常值分为长期集中于固定通道的 channel-wise outlier，以及少数 token 的单点 spike outlier。离线 SmoothQuant 会受校准/在线 scale mismatch 影响；spike 还会拉大同组 scale，使普通值成为量化到零的 victim。

论文方案依次为：按通道峰值重排；运行时用 block 最大值平滑；Hadamard rotation 将 spike 扩散到 token 内多个维度。关键消融显示纯 Runtime Smooth 随 group size 从 1 增大到 128 明显退化，而 RRS 在 group size 1–128 基本稳定。HiF4 的 block size 64 正处于论文验证的稳定区间。

## 接口适配边界

原版 Runtime Smooth 需要在每个 reduction block 的 GEMM 部分和上乘回当前输入的动态 scale。判题器只执行一次 `A_HiF4 @ W_HiF4^T`，没有分块 epilogue，在线 Weight 又已固定，因此不能原样实现。

可严格保持输出等价的部分：

1. Linear：`A' = A P D H`，`W' = W P D H`，因此 `A'W'^T = AW^T`；
2. Attention：每个 head 内对 Q/K 使用同一正交变换，保持 `QK^T`；V 不旋转；
3. calibration channel peak 生成冻结 permutation；
4. HiF4 在线 Activation 本身逐 64 block 动态生成 E6/E1，可承担一部分 runtime block smoothing。

## v4 设计

- rotation block size 候选：64、256、以及不超过 2048 的最大二次幂因子；
- permutation 候选：identity、按 calibration activation peak 升序聚类；
- randomized Hadamard 候选：全正 sign 和确定性随机 sign；
- calibration holdout 用廉价单 E6 候选量化选择变换；
- 最终 Weight 只执行一次完整 HiF4 层级搜索；
- rotation 后重新计算 v1 的 A/W 或 Q/K 可逆平滑；
- state 只保存 `permutation:int32`、`sign:int8`、`block_size:int`、`smooth:float32`。

## Acceptance criteria

- FP32 等价性测试中 Linear/Attention 旋转前后误差处于浮点舍入量级；
- `self_check.py` 22/22 通过；
- solution-only 按 100 组外推小于 270 秒；
- 首次线上只比较新增 RRS transform；若分数下降或超时则回退 v1；
- 若 v4 获益，再分别消融 permutation、block size、random sign 和 post-rotation smooth。

当前验证：FP32 Linear/logit 等价误差约 `1e-4`；self-check 22/22；solution-only smoke 约 1.48 秒/组，按 100 组粗略外推约 148 秒。mini_sample 的 score 不作为线上收益证据。

## 线上消融版本

- v4：Linear 与 Attention 都启用 RRS-compatible transform；
- v5：只在 Linear 启用 RRS，Attention 回退 v1；
- v6：Linear 回退 v1，只在 Attention Q/K 启用 RRS。

三者不依据 mini_sample 收益排序；它们用于线上拆分 RRS 对两个题型的真实贡献。

## v7：平滑感知的候选选择

线上消融显示 v5=8951/137秒、v6=10314/144秒；相对 v1 的 Linear 贡献为 -499，Attention 贡献为 +864，且两者之和精确复现 v4=9815。因此后续固定 v1 Linear，只优化 Attention。

v6 在 holdout 上按未平滑的 Q/K proxy 误差选择 permutation/sign/block，选择结束后才重新计算旋转后的互逆平滑。v7 修正这一目标错配：每个候选先从 fit Q/K 计算部署时使用的平滑，再用 `Q/s`、`K*s` 的实际路径评估 holdout Attention MSE。候选数量、动态量化计算和 state 均不增加。

Acceptance criteria：成对旋转 FP32 logits 最大误差 <2e-4；单文件隔离导入成功；self-check 22/22；线上高于 v6=10314 且耗时 <300 秒。

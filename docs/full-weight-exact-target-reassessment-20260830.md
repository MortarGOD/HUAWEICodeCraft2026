# 完整 W 测试教师的再审视（2026-08-30）

## 1. 精确问题

记原始权重为 `W ∈ R[M,K]`，最终返回的 HiF4 权重为
`B=Q(W)`，测试激活为 `A ∈ R[R,K]`。测试阶段已能计算完整教师

`Y = A Wᵀ`。

真正需要求解的是离散最小二乘：

`min_{Z ∈ HiF4} ||Y - Z Bᵀ||²`，其中最终 `Z=Q(A)`。

当前 v264/v266 已使用 Y 做精确接受，但从普通 `Q(A)` 开始，只执行候选树、
相邻 128 通道 Newton、64 通道 Newton 和 4 通道 micro lookup。它是局部坐标下降，
并没有求出或近似求出全局连续最优点。

## 2. 尚未利用的全局连续解

带回归到原始 A 的 ridge 问题有闭式解：

`Z*(λ) = (YB + λA)(BᵀB + λI)⁻¹`。

等价的输出空间形式为：

`Z*(λ) = A + (Y - ABᵀ)(BBᵀ + λI)⁻¹B`。

- `M < K` 的 down projection 优先使用输出空间形式；
- `M >= K` 的 up/square 优先使用输入空间形式；
- λ 不只用于数值稳定，还控制“精确拟合 Y”与“保持 Z 接近易量化的 A”的权衡。

不能显式求 4K 逆矩阵，应该用 block-Jacobi 预条件 CG、低秩 Woodbury 或二者结合。
v242 只做了两步无预条件输出空间 CG，线上仍获得 +125 分，证明全局连续方向有效。

## 3. 论文对应关系与当前缺口

1. GPTQ/QuIP/BlockLDLQ：应用 `BᵀB` 的全局误差反馈，而不是反复使用相邻块的
   局部逆。QuIP 指出 Hessian 近似低秩，可用低秩全局项加 block-diagonal 项近似。
2. AQLM：离散码之间存在 pairwise interaction，单轨迹贪心容易停在局部最优；
   在少量高残差块上保留 beam=2/4 或 randomized ICM。
3. AdaRound/BRECQ：HiF4 E6/E1/S1P2 都应在最终输出二次型下更新，而不是先按
   proposal 的元素 MSE 固定一个编码再只做 old/new 接受。
4. RILQ/LoftQ：4-bit 权重量化误差存在可利用的低秩成分，但应优化 Linear output
   discrepancy，而不是单纯 `||W-B||`。v240 已做 rank-32 原型，却只对 square 开启；
   线上没有 square，因此该思路没有得到有效验证。
5. MR-GPTQ：先优化 FP4 层级网格，再执行全局误差反馈；静态 act-order 应在网格固定
   后用于求解顺序，不能破坏官方连续 64 通道格式。

## 4. 优先实现路线

### P0：全局 ridge/PCG warm-start（下一主版本）

以 v262 的 249 秒高速版本为父版：

1. 计算真实 `Y=A@Wᵀ` 和初始残差 `R=Y-Q(A)Bᵀ`；
2. 使用现有 64 块逆作为 block-Jacobi 预条件器，对全局正规方程执行 2/4 步 PCG；
3. 生成 `λ × relaxation × HiF4 grid` 的少量全矩阵初值；
4. 按每行真实输出误差选择初值，再做一轮 64/4 通道离散收口；
5. 用全局 warm-start 替换 v264 的四轮 128 块扫描，而不是叠加，目标低于 390 秒。

验收门槛：4K up/down 代理必须超过 v264 的 9.131968，且同批耗时不高于 v264；
否则仍保留为线上分布探针，但不宣称高收益。

### P1：低秩 Woodbury 全局预条件器

校准阶段对 `BᵀB` 的 block-diagonal 之外部分做 randomized rank-64/128 近似，存储
`O(Kr)` 的因子。测试阶段用一次 Woodbury 预条件得到接近多步 PCG 的全局 correction，
再量化和精确接受。该方案利用 QuIP 的低秩 Hessian观察，目标是把全局求解压到两次
`R×K×r` GEMM。

### P2：ADMM/交替投影

交替执行：

1. 连续 ridge 解 `Z`；
2. HiF4 投影 `Q=Π_HiF4(Z+U)`；
3. 对偶更新 `U←U+Z-Q`。

只做 2–3 轮，并保留每轮 Q 作为精确 Y 下的候选。它比“Newton 后立即量化并丢弃
连续信息”更能跨越量化单元边界。

### P3：小 beam 离散搜索

在 P0/P1 最优初值上，仅对真实残差贡献最高的 8–16 个块使用 beam=2/4；每个扩展
只尝试相邻 E6 与上下 mantissa。用 AQLM 的增量二次型更新 beam loss，避免复制完整
`R×M` residual。它用于解决贪心路径依赖，不再扩大所有块的候选网格。

## 5. 不再优先的方向

- 继续增加相邻 128 块扫描轮数：v264 已证明 +153 秒只换 +45 分；
- 单独调 E6、mantissa threshold、Newton damping：v266 前消融均已收口；
- 只对 square 做低秩 warm-start：v240 线上同分已证明线上基本无 square；
- 低秩只拟合 `W-B` 的元素 Frobenius loss：应拟合 correction operator 或最终 Y；
- 在当前局部解后叠加更多候选：强求解器会吸收它们，必须改变初值和全局求解结构。

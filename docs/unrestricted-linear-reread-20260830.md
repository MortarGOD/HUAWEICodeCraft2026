# 无限制 Linear 论文重审与后续路线（2026-08-30）

## 线上事实与目标

- v239：32602 / 293 秒；v240：32602 / 292 秒。v240 只增强 square，完全等分说明线上 Linear 基本由矩形 up/down 权重构成。
- v241 在 v239 上只增强 down 第二轮 Newton；线上32912/328秒，比v239提升310分/35秒。本地 4K down 五测试总分 4.648783→4.728983 的方向成功迁移。
- 第一名 35715 / 389 秒；当前差距 3113 分，仍有约 127 秒线上预算。继续微调 offset、阈值或 square 路径不足以弥合差距。
- 本文按 8 月 31 日前的无限制假设：测试阶段可由完整 W 构造真实目标 `Y=A@W.T`，并允许用该目标拟合最终 `Q(A)`。

## 当前求解器真正做到了什么

令部署权重为 `B=Q(W)`，测试阶段每一行实际求解：

`min_x ||Y-x B.T||², x in HiF4`，其中 `Y=A W.T`，Hessian 为 `H=B.T B`。

v239–v241 已实现：

1. 宽 RTN/clipping/0.375/0.625 初始候选；
2. 真实输出残差驱动的 64 通道坐标下降；
3. 每个 64 块用 `B_b.T B_b` 的正则逆产生连续 Newton 方向；
4. 把方向重新投影到合法 HiF4，并按精确输出二次型接受；
5. square 两轮、up 一轮、v241 down 两轮（第二轮压缩为三个候选）。

核心近似是只在方向中使用 `H` 的 64×64 对角块。虽然残差更新看到了完整 B，但跨块 `B_b.T B_c` 只能靠多轮 Gauss-Seidel 间接消化；这正是剩余误差和迭代耗时的主要来源。

## 重新精读后仍未落地的高价值机制

### P0：QuIP#/GPTQ/OBC 的跨块 LDL/Schur 误差传播

论文核心不是“多做几轮”，而是由 Hessian 的 LDL 分解构造线性反馈，使前面块的离散舍入误差一次传播到后续块。BlockLDLQ 进一步把标量反馈扩为向量块反馈。

本题落点：

- Hessian 直接取 `H=B.T B`；目标线性项取 `Y B`。
- 不保存完整 4K/8K Hessian；保存 64 块的带状非对角项，或 128/256 superblock 的阻尼逆/Schur 系数。
- 先用两到三步 PCG 或 banded-LDL 得到包含跨块耦合的连续 teacher，再投影到 HiF4；最后保留一轮 v239 精确残差坐标修正。
- down 的 H 奇异，采用输出空间 `B B.T` 的 PCG 或阻尼最小范数解；up 可直接用输入空间 PCG。

预期：这是唯一有合理机会带来数百至低千分收益、接近 35715 的单一结构升级。主要风险是额外 GEMM 使耗时逼近 420 秒，因此应替换部分现有 sweep，而不是简单叠加。

### P0：QuIP# Vector Quantization + AdaRound 的 HiF4 microgroup 离散优化

当前 `_best_linear_for_base` 先按元素重构 SSE 选择 E1_8/E1_16 与 mantissa，外层精确目标只在少量“整 64 块候选”之间选择。因此四元素 microgroup 的 metadata 和逐元素上下舍入仍没有直接最小化输出损失。

本题落点：

- 固定一个 E6M2 base 后，对每个 4 元 microgroup 枚举合法 E1_16 与邻近 mantissa 码字；
- 用 `B_g.T B_g`、当前输出残差和精确二次型选择 4D 码字；
- 按 BlockLDLQ/残差顺序传播到下一个 microgroup；
- 只对 Newton 后仍有高残差的块启用，避免 16 microgroup×全部块的成本。

它保留官方 HiF4 格式，不需要自定义 E8 codebook，却吸收了 QuIP# 的“向量码字 + adaptive rounding”核心。预期收益中高，且比继续增加完整 block sweep 更省 GEMM。

### P1：连续最小二乘 teacher + 量化投影（PCG/迭代细化）

当前 block-Jacobi 全局方向实验只有一步，up 几乎零收益、down +0.0013，说明简单预条件梯度不足，但不能否定 Krylov 子空间。

本题落点：

- 从当前 Q(A) 出发，对正规方程做 2–3 步 batched PCG；
- 每一步通过 `p@B.T@B` 或 down 的输出空间算子完成，不显式构造/求逆 4K² 矩阵；
- 用精确线搜索与残差停止条件，随后一次性 HiF4 投影，再执行一轮局部修正；
- 按 stride/行数门控，只把高成本迭代用于贡献大且行数可控的布局。

它是 P0 LDL 路线的低开发成本探针。若 PCG 连续 teacher 明显优于 block-Jacobi，但量化后收益丢失，说明下一步必须转 microgroup adaptive rounding，而不是继续加 PCG 步数。

### P1：Q(W) 与动态 Q(A) 求解器交替优化

v239 使用的是 v226 独立校准出的 B；它未必是“最接近 W”的同时也是“最容易让 HiF4 A 拟合 Y”的 B。

本题落点：校准阶段在五个 A 上模拟动态精确目标求解，交替更新合法 B 与 Q(A_calib)，直接优化 `sum ||A_i W.T-Q(A_i)B.T||²`。B 更新使用 MR-GPTQ/OBC 的 FP4 grid、输出残差与 Hessian 补偿，最后固化为唯一返回权重。

风险是校准激活到测试激活的泛化，以及校准成本。必须使用五布局等权、训练/验证行拆分和最大布局回退门控。

### P2：RILQ/LoftQ 残差子空间作为矩形 warm-start

v240 只对 square 使用 rank-32 `W-B` SVD，而且本地低秩本身仅有约 0.002 的增益；不能据此证明矩形完全无效，但优先级下降。

可改进点是不用 Weight-SVD 直接截断，而在校准 A 的输出残差上学习 rank-32/64 子空间，并将其作为 PCG/LDL 的 warm-start。RILQ 的论文结论也提示单层 Weight-SVD 不如输出差异训练。因此它应融合进全局求解器，不应再单独提交。

### P2：SmoothRot/ButterflyQuant/RRS 的求解器条件数目标

已有大量旋转、平滑、置换版本证明，仅优化分布峰值或元素 MSE 在线上不稳定。仍未测试的角度是：选择变换时直接最小化 `cond(B.T B)`、block off-diagonal energy 和精确目标求解后的误差，使 B 更适合有限步 LDL/PCG。

这是二阶融合项，不再作为独立旋转版本。只有 P0 求解器确认受条件数限制时才启用。

## 已基本饱和或不应重做

- RTN scale offset、0.375/0.625 阈值、clipping 候选库：v224–v233 已充分验证，继续调参预期个位到十位分。
- 单纯增加 block-Newton sweep：有收益但 v238 已超过 420 秒；v241 是压缩后的最后一次高性价比 sweep 探针。
- square 专属低秩或深迭代：v240 线上完全不命中。
- 只按 activation 统计路由校准标签：v217–v221 线上无效或负向。
- 单独替换为 RRS/SpinQuant/FlatQuant/SmoothRot：历史上收益不足、退化或超时；只保留与 P0 求解器融合的条件数优化。
- 自定义 lattice/codebook、稀疏旁路和额外输出残差：无法由官方 HiF4 参数表达，不作为正式接口方案。

## 收到 v241 反馈后的执行顺序

1. v242：两步down输出空间PCG teacher。替换第二轮时离散投影后仅4.682168，低于v241的4.728983；融合五松弛投影和三轮Newton后为4.766661，作为“连续teacher必须配合离散修正”的线上探针。
2. v243：对高残差 down 块加入 4D microgroup 输出感知码字选择；比较单位耗时收益。
3. v244：128/256 superblock 或带状 BlockLDLQ，把 PCG 探针升级为结构化一次反馈。
4. v245：校准阶段交替优化 B 与模拟动态 Q(A)，并使用 holdout 门控。
5. 仅当全局求解受条件数限制，再融合 SmoothRot/Butterfly 的低成本条件数目标。

## Acceptance criteria

- `self_check.py` 22/22，状态有限、无梯度、节点/深度合法；active solution 与版本文件逐字一致。
- 4K up/down 分开验证；不得用 square 的本地提升外推线上。
- 相对 v239/v241，候选必须在真实输出 MSE 上逐测试非退化，或作为本地不能可靠否定的单变量线上探针保留。
- 线上预估耗时中心不超过 380 秒，上界不超过 415 秒；超过时优先减少 GEMM/候选，不牺牲结果文件。
- 每次只提交一个主结构变量，并记录连续解收益、离散投影损失和最终收益，以判断瓶颈属于求解器还是 HiF4 码字。

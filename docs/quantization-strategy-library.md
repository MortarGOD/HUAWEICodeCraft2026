# HiF4 算法优化策略库

## 审计范围与结论

本轮以 `docs/paper_list.md` 的 30 个编号条目为入口，逐项核验一手来源。
截至 2026-08-25，清单中的 29 篇论文和 1 篇 NVIDIA 官方技术文章均已解析、
下载并可离线阅读；本地档案另含 14 篇 GPTQ、AWQ、QuIP 等对照论文，共 44 份
PDF、171.88 MiB。RRS 交叉阅读官方 LaTeX 源码与 PDF，其余来源均已抽取全文。
论文 PDF、官方网页快照和提取文本位于
`docs/papers/quantization/{pdf,source,text}`。这些原文资料仅作为本地研究档案，
不提交到远程仓库；本策略库和由其导出的决策证据提交到仓库。

`paper_list.md` 中 NVIDIA 的 `nvfp4-pretraining` 链接当前返回 404，也没有可
验证的原生 PDF。为避免拿二手文章补位，本地保存了 NVIDIA 当前官方文章
“Using NVFP4 Low-Precision Model Training for Higher Throughput Without Losing
Accuracy”的 HTML，并由 Chrome 打印为 PDF。该文章发表于 2026-02-23，不能
被误写成原清单所称的 2025-08 原生论文。

## 赛题约束映射

本库只把能在当前六函数接口中表达的机制列为实现候选：

- 校准阶段不限制行为，可以用精确 `A @ W` 同时评价或拟合固定 `Q(W)` 与
  `Q(A)` 策略，也可以使用高阶统计、搜索、交替优化和教师输出。
- 提交状态仍必须编码为合法 HiF4；测试阶段只能依赖校准状态与当前输入，不能
  增加第二路残差 GEMM、LoRA、FP16 通道或改变返回格式。
- 线上 300 秒是硬约束。离线搜索容量可以大，但在线变换、状态体积和动态候选数
  必须受控。
- 本地和线上分布不同。本地负收益只否决数值不稳定、接口不合法或灾难性退化的
  方向；理论合理且风险可控的机制仍可进入一次线上受控消融。

## 逐项学习记录

| # | 一手来源 | 关键证据，不照搬的迁移结论 | 当前决策 |
|---:|---|---|---|
| 1/8 | HiFloat4 / HiGPTQ, 2602.11287 | Algorithm 1 用 4→8→1 的峰值树生成 E6M2、E1_8、E1_16；直接转换的归一化 MSE 为 HiF4:NVFP4:MXFP4 = 1:1.32:1.89。HiGPTQ 证明二阶误差传播能与层次格式结合。当前代码已实现合法树、E6M2 多候选和输出反馈，但仍可把层次位选择从局部误差提升为块间 `A@W` 目标。 | **继续，核心基线** |
| 2 | HiFloat4 RL, 2607.26515 | Rollout-ResQ 对 `X-Q(X)` 再量化，并用 dense、50% channel、2:4 或 32×64 稀疏残差做第二次 GEMM；2:4 在多项实验中最稳。赛题无第二路 GEMM，不能直接实现，但“激活残差的结构化重要性”可转为校准时的 Q(A) 位/尺度选择权重。 | **修改迁移** |
| 3 | NVIDIA NVFP4 官方技术文章 | NVFP4 是 4-bit 值加两级尺度；官方大规模训练仍需选择性保留 BF16 层，说明格式本身不是无损。原链接失效，当前文章偏训练而非完整格式规范。 | **背景证据** |
| 4 | Four Over Six, 2512.02010 | 消融显示尺度 E4M3 误差影响小，主要误差来自 E2M1 在 4↔6、2↔3、3↔4 之间的大间隔；逐块比较 max/6 与 max/4，直接以 MSE 选优。它要求尺度能表达 1.5 倍变化，E8M0 做不到。HiF4 不应机械套 4/6，而应针对 S1P2 与层次倍率的决策边界生成候选。 | **继续，定向尺度候选** |
| 5 | TetraJet-v2, 2510.27527 | 1×128 外层 + 1×16 内层尺度优于整张量全局尺度；前向 RHT 可能因新增舍入误差而有害，反向才稳定受益；OutControl 依赖跨输入稳定的结构化异常通道；OsciReset 是训练期主权重重置。说明旋转必须由前向输出证据选择，不能把“削峰”当充分条件。 | **部分继续** |
| 6 | Quartet II, 2601.22813 | MS-EDEN 用 128 维 RHT，再以 `||x||²/<x,Q(x)>` 修正共线偏差，只对 FP8 尺度随机舍入，误差约比元素级 SR 低 2 倍；post-hoc range alignment 省一次全张量载入。它是无偏梯度估计，依赖随机性和额外尺度表达。赛题推理不采用随机输出，但其确定性共线修正可作为校准诊断。 | **修改后观察** |
| 7 | MR-GPTQ, 2509.23202 | 三个实质组件是层次尺度交替 MSE 优化、**固定网格后的 static act-order**、与量化组对齐的微旋转。论文证明旋转对 MXFP4 有利、对 absmax NVFP4 可能有害；static act-order 保留 GPTQ 收益且避免 10–20% 动态重排开销。当前已有微旋转和固定 Q(W) 反馈，但还没有严格的 static act-order 反馈顺序。 | **最高优先级缺口** |
| 9 | MXFP4 OAS/MBS, 2603.08713 | OAS 对约 15% 块故意把最大值映射到 (3.5,7] 并允许有限饱和，换取尾部动态范围，提升约 0.5 dB QSNR；MBS 用 1×128 高精度宏尺度隔离异常值。说明 absmax 无饱和不是最优，尺度搜索必须显式包含“牺牲峰值、保护主体”候选。 | **高优先级** |
| 10 | ARCQuant, 2601.07475 | 校准后按阈值 `2^-3 M` 识别结构化异常通道；在线量化 `X-Q(X)` 的异常子通道并追加到 K 维，权重侧复制对应列，一次扩维 GEMM 完成残差补偿。跨校准域较稳，但需要扩展 K 和返回格式，不能直接提交。阈值化通道重要性可用于布局、尺度和反馈正则。 | **原形态拒绝，统计迁移** |
| 11 | DecDEC, 2412.20185 | 静态异常通道对真实动态 top 通道召回率仅约 20%；DecDEC 在线 Top-K 激活通道，从 CPU 取 4-bit 权重残差并并行 GEMV。它证明固定异常表泛化有限，也证明激活幅值适合做动态风险信号。第二路残差和 CPU 存储不符合接口。 | **原形态拒绝，动态门控迁移** |
| 12 | NVFP4 QAD, 2601.20088 | 用原模型教师和 KL，而不是标签交叉熵；QAT 可匹配标签损失却偏离教师分布，QAD 把 KL 从 0.311 降到 0.004。数据源鲁棒，甚至随机 token 不会破坏模型；原尺寸教师优于更大教师。赛题已有精确 `A@W` 教师，启示是直接对齐目标输出而非只优化张量 MSE，但无模型 logits，不能照搬 KL。 | **原则已采用** |
| 13 | M²XFP, 2601.19213 | 关键消融揭示 W/A 非对称：固定尺度时激活适合子组 top-1 元素额外尾数；允许离线自适应尺度时，权重适合子组 2-bit 尾数尺度。Top-1 与 top-2 几乎等效。当前 W/A 已分策略，但 Q(A) 尚无显式 top-1 风险特征，Q(W) 层次位可做更完整离线搜索。 | **高优先级结构启示** |
| 14 | MX+, 2510.14557 | 块最大值指数已知，可把指数位复用为尾数，并存 5-bit 最大值位置；MX++ 再用 3-bit 调整非最大值尺度。平均只增 0.25 bit/值。HiF4 返回格式不能增加元数据，但可把“最大值与主体分别计价”写入尺度目标。 | **修改迁移** |
| 15 | FPE2M2, ACL Findings 2025 | 5-bit E2M2 在多模型上是精度/实现折中点，解码可用寄存器位操作；论文明确依赖近高斯权重，显著异常值会破坏假设。NVFP4 反量化已能直接到 FP32，增加 E2M2 中间层只会再量化一次，不会恢复已丢信息。 | **放弃中间格式** |
| 16 | Microscaling Formats, 2310.10537 | MX 标准把数据定义为共享尺度 X 与 k 个窄值；量化轴决定尺度共享方式，转置与 MX 转换不交换。该结论支持始终沿 GEMM reduction 维组织 64 通道组，也解释了布局优化的合法性和风险边界。 | **基础约束** |
| 17 | FP4 All the Way, 2505.19115 | block 8/16/32/64/128 消融显示小于 16 收益递减；E4M3 尺度和 block16 最稳；随机舍入只用于 backward/update 的选择性位置，滥用会增损。训练末期梯度低于约 3 倍量化噪声后需高精度 QAF。 | **训练专用，确定性结论保留** |
| 18 | Training LLMs with MXFP4, 2502.20586 | 先乘 3/4 防止 6–8 截断，再 SR，可令量化值无偏；乘积累加器乘 16/9 恢复无偏输出。更大 RHT 块降低 SR 方差，但即使无偏仍会把小梯度随机归零。对赛题的价值在于“偏差与方差分开评价”，不适合随机推理。 | **诊断原则** |
| 19 | QeRL, 2510.11696 | 量化噪声在 RL 中提高策略熵；AQN 用指数调度的通道噪声，并把噪声合并入 RMSNorm 权重。这个收益依赖探索目标，不代表推理重建精度提升；线上判题应避免随机噪声导致方差。 | **放弃实现** |
| 20 | Why Do Some Inputs Break Quantization?, 2506.12044 | 不同方法的逐样本错误相关均值 0.82；大误差样本在上层残差幅值更小，经 RMSNorm 后发生幅值反转。激活 patch 显示 MLP gate 输出/输入最关键，恢复 gate 权重本身无效；累计激活误差比末层权重精度更关键。说明 Q(A) 风险门控比继续堆 Q(W) 容量更值得探索。 | **最高优先级 Q(A) 启示** |
| 21 | RILQ, 2412.01129 | 局部 Linear/Layer loss 对低秩补偿高度敏感；最后 Transformer 输出的 Model-Loss 允许跨层误差互相抵消，rank16 可胜局部 SVD rank256。赛题只见单层 `A@W`，没有模型末端或 LoRA 返回通道，不能复制；它也警告单层校准最优不等于线上任务最优。 | **原形态不兼容** |
| 22 | RRS, 2409.20361 | 把异常值分成跨 token 稳定的 channel-wise outlier 与单 token spike。RS 用当前输入 `s_j=max_i|X_ij|`，避免校准尺度失配；Hadamard 先摊平 spike，显著降低粗粒度 smoothing 的 victim 效应。原实现需要 GEMM 对每个 reduction block 的局部输出乘回动态 `s_j`，本题普通 MatMul 接口无法原样表达。迁移为旋转后 runtime channel profile、victim-aware HiF4 候选代价，并用校准 `A@W` 门控，禁止无补偿地返回 `A/s`。 | **最高优先级 Q(A) 动态分支** |

### 2026-08-25 新补齐原文的增量结论

| 来源 | 关键证据，不照搬的迁移结论 | 当前决策 |
|---|---|---|
| DuQuant, 2406.01721 | `G=ΛR1PR2` 成对作用于 A/W；先针对最坏通道构造块旋转，再以 Zigzag 使各块最大值均衡，第二次旋转收尾。W4A4 消融中 smoothing 单独出现 NaN，完整链路才从 8.48/6.79 改善到 6.28，证明顺序和互补性不能拆开。 | **P1，成对离散双旋转** |
| BASE-Q, 2506.15689 | 旋转保持通道均值向量范数，无法消除 mean drift；bias correction 在 Qwen2.5-3B 是主收益，非对称 activation scale 也稳定增益。但两者都依赖输出偏置补偿或非成对尺度，当前接口无法精确表达。 | **只迁移均值/能量诊断** |
| ButterflyQuant, 2509.09679 | `log2(n)` 层 Givens butterfly 以 `n log2(n)/2` 个角度表达可学习正交变换，Hadamard 是其特例；identity 初始化最好，约 86% 收益在前 200/500 步获得，分箱均匀正则可改善泛化。 | **P2，稀疏离散角原型** |
| SmoothRot, 2506.05413 | 只平滑 FFN down-projection，`α` 约 0.45–0.6；随机 token 校准仍有效。但与 GPTQ 或 SpinQuant 叠加时优势消失或轻微转负。当前已有平衡+旋转+输出反馈，不能再无条件堆叠。 | **核心已覆盖，改为参数审计** |
| AFPQ, 2311.01792 | 小组浮点分布常有正负不对称，正负独立尺度能改善 RTN/GPTQ/AWQ；直接实现需要每组两套尺度，HiF4 状态不支持。 | **只迁移符号不对称特征** |
| TetraJet, 2502.20853 | truncation-free scale 用 `ceil(log2(2M/(Qp-Qn)))` 保证峰值不截断；随机舍入、Q-EMA、Q-Ramping解决训练梯度与权重振荡，不适用于确定性 PTQ。 | **迁移跨布局候选稳定性惩罚** |
| Post-Training Model Expansion, 2503.17513 | `N×M (M>N)` 行正交 Hadamard 保持 A@W、扩大激活零空间，理想 GPTQ 误差上界可缩小 `sqrt(N/M)`；实验随扩维递减。它必须改变 A/W reduction shape，违反当前接口。 | **原形态拒绝，零空间目标继续** |

## 从论文导出的可执行优化方向

### P2：MR-GPTQ static act-order 的剩余 HiF4 化

代码审计纠正了最初判断：v32-v42 已按量化激活 Hessian 能量排序 64 通道组，
因此 MR-GPTQ 的 group-level static act-order 已经覆盖。真正缺少的是固定层次网格
后的组内顺序；它需要新的顺序量化器，不能被描述为一个简单零成本开关。

不要照搬列级 GPTQ：HiF4 的不可分原子是 64 通道尺度组，候选应比较三种粒度：

1. 组间按累计敏感度排序，组内保持 v42；
2. 组内仅改变四元素叶节点的量化/反馈顺序，不破坏 E1_16；
3. 以 8 元素中层为单位排序，再恢复原位置。

该方向降为 P2。验收条件仍是固定 Q(A) 后校准 `A@W` 改善、近/远 donor 不出现
灾难退化、在线图不增加操作；但在原型前必须先估算 16 个叶单元顺序传播的校准
成本，不得挤占 300 秒预算。

### P0：输出感知的层次尺度位联合选择

v42 已在每个 E6M2 base 下寻找合法 E1_8/E1_16，但局部候选仍主要由重建项和
压缩输出状态近似驱动。论文共同指出尺度元数据和异常值处理不能分开。候选应在
每个 64 组内以小型 beam search 联合搜索：

- E6M2 base 的邻域与 OAS 式更小 base（允许峰值有限饱和）；
- 8 个 E1_8 和 16 个 E1_16 的合法层次位；
- `A@W` 输出残差、峰值饱和损失、主体能量损失三项；
- 对 top-1、次大值、其余主体分别记账，而不是只看总 MSE。

搜索只在校准发生，最终仍输出标准 HiF4 参数。先在 Q(W) 使用完整搜索；Q(A)
只保留从校准总结出的少量确定性 profile，避免在线组合爆炸。

### P1：S1P2 边界感知的尺度候选

Four Over Six 的本质不是常数 4 和 6，而是围绕 codebook 最大间隔构造少数候选。
对 HiF4 应枚举 S1P2 经 E1_8/E1_16 放大后的实际重建格点，找到峰值、次峰值和
主体分位数跨越最大 Voronoi 边界的位置，反推出 E6M2 候选。与当前固定 code
offset 并集后去重，再由输出目标选择。若它只复制现有 offset 输出，则直接删去。

### P1：激活风险路由，而不是更多固定 Q(W) 容量

输入失效论文和 DecDEC 都说明动态激活形态比静态异常表更能预测误差。利用当前
输入本身计算低成本、尺度无关特征：

- 行 RMS 的变异系数及低分位/中位比，用于捕捉 RMSNorm 前后的反转风险；
- 每 64 组 top-1 能量占比、近零率和峰均比；
- 通道 RMS 的稳定集中度，区分结构化异常和单次尖峰；
- shape/aspect ratio，继续保留现有线上反馈证实的分支。

校准时可用 `A@W` 为这些特征学习一个小决策树或离散阈值；测试时只算特征并选择
最多两个固定 Q(A) profile。任何阈值必须有 v42 精确 fallback，且额外时间预算不
超过线上 v42 锚点的 5%。

RRS 将这个方向具体化：profile 必须在既有 Hadamard 后从当前矩阵逐通道 absmax
生成。由于不能在判题 GEMM 的局部累加后乘回动态尺度，profile 不直接改变 FP32
张量坐标，而是改变合法 HiF4 层次候选对正常值/异常值的计价。对通道尺度大的
元素降低绝对误差支配度，相当于在可表达接口内逼近 `Q(X/s)*s` 的相对误差目标；
是否启用及平滑强度仍由相同行数布局的校准精确输出误差决定；未知布局精确回退。

### P1：确定性 EDEN 共线诊断

对每组记录 `c = ||x||² / <x,Q(x)>`。不直接引入随机 MS-EDEN，而是检查当前
候选是否存在稳定的系统性收缩/膨胀：

- 若 c 在校准 shape 间稳定，可把它转成相互抵消的 A/W 乘子或 E6M2 候选偏置；
- 若 c 方差大，只把它作为候选拒绝特征；
- 任何 A/W 成对变换必须先验证 FP32 `A@W` 等价，不能用测试目标选择。

### P2：结构化激活残差的重要性代理

不能增加 ARCQuant/Rollout-ResQ 的第二路 GEMM，但可在校准时计算
`R_A=A-Q(A)` 的 2:4、top-channel 和 32×64 block 能量，作为 Q(W) Hessian/反馈
权重。它关注“Q(A) 最难表达且会放大到输出”的方向，比单纯 A RMS 更接近真实
误差。如果压缩代理不稳定，保留原 rank-4 输出状态。

## 明确不进入提交代码的方向

- ARCQuant、Rollout-ResQ、DecDEC：需要额外残差通道、GEMM 或 CPU 状态。
- QAD、RILQ、OsciReset、QeRL：需要完整模型、教师 logits、训练或 LoRA。
- FPE2M2 中间转换：不能恢复 NVFP4 已丢失的信息，反而增加舍入。
- 随机舍入、MS-EDEN、AQN：训练期的无偏或探索收益不等于确定性推理 MSE。
- MX+/M²XFP 的新增元数据位：返回格式固定；只迁移其分层计价与 W/A 非对称思想。
- 无门控的大型旋转：TetraJet-v2 和 MR-GPTQ 都证明旋转对前向 NVFP4 不保证正
  收益，且 v41 的 16-base 远 donor 已负。

## Candidate DAG

```text
v42 (Linear bit-reverse hierarchy + gated Attention QK hierarchy)
  ├─ v43: rollback Attention to v41 + ordinary-wide Q(W) saturation
  │    ├─ keep: one E6M2 code below peak/7 for input<=2*output
  │    └─ reject: unconditional and -2 candidates
  ├─ research-a / P2: inner-group static act-order residual feedback
  │    ├─ covered: group order
  │    ├─ held: E1_8 unit order
  │    └─ held: four-value leaf order
  ├─ research-b / P0: output-aware hierarchical beam
  │    ├─ ordinary legal E6M2 neighbours
  │    ├─ OAS-like controlled saturation
  │    └─ top-1 / body split objective
  ├─ research-c / P1: S1P2 boundary-derived candidate set
  ├─ research-d / P1: input-risk Q(A) controller
  │    ├─ row-RMS reversal proxy
  │    ├─ top-1 energy / near-zero / peak-RMS
  │    └─ exact v42 fallback
  ├─ research-e / P1: deterministic EDEN shrinkage diagnostic
  ├─ research-f / P2: structured Q(A) residual importance
  ├─ research-g / P1: DuQuant discrete R-P-R paired transform
  ├─ research-h / P1: signed-mean and energy-drift candidate penalty
  ├─ research-i / P2: sparse butterfly/Givens calibration search
  └─ research-j / P1: cross-layout candidate-flip stability penalty

v43 (controlled Q(W) saturation)
  └─ v44 / RRS: rotated runtime channel profile
      ├─ reject: direct unpaired runtime division
      └─ retain candidate: victim-aware HiF4 scale cost + A@W gate

v44 (submitted; online result pending)
  └─ WAIT: do not create v45 until online feedback arrives
```

执行顺序不是把所有机制一次堆入新版本。v43 先验证受控饱和；下一主分支做
`research-b/c` 的层次尺度联合搜索，Q(A) 风险控制器再单独做线上消融。组内
static act-order 因校准成本降为 P2。每个分支保留稳定版本 fallback 和单变量对照。

## 原文档案校验表

下列 SHA-256 取前 12 位，用于发现下载损坏或来源被替换。页数按 PDF 解析器的
实际页树统计，可能高于阅读器首页显示的正文页码。

| 本地 stem | 页数 | MiB | 提取词数 | SHA-256/12 |
|---|---:|---:|---:|---|
| `HiFloat4_2602.11287` | 8 | 2.06 | 5,993 | `846021fab9f5` |
| `HiFloat4_RL_2607.26515` | 15 | 0.69 | 8,132 | `34fc9773288f` |
| `NVIDIA_NVFP4_technical_blog` | 15 | 0.98 | 3,274 | `59ed45fb6a0c` |
| `FourOverSix_2512.02010` | 19 | 1.04 | 9,921 | `f6f898f0dce0` |
| `TetraJetV2_2510.27527` | 20 | 0.86 | 11,518 | `fb877bd41f97` |
| `QuartetII_2601.22813` | 16 | 0.87 | 9,579 | `0c3ed313fdd1` |
| `MR_GPTQ_2509.23202` | 35 | 1.16 | 14,292 | `46366a1f76d6` |
| `MXFP4_OAS_MBS_2603.08713` | 15 | 0.97 | 9,812 | `bc9df153931b` |
| `ARCQuant_2601.07475` | 15 | 4.99 | 7,206 | `4d98102573ba` |
| `DecDEC_2412.20185` | 17 | 3.80 | 12,380 | `b509754ef8a8` |
| `NVFP4_QAD_2601.20088` | 17 | 0.66 | 7,045 | `ed5da30e5b07` |
| `M2XFP_2601.19213` | 17 | 3.56 | 14,718 | `e51c195a3e84` |
| `MXPlus_2510.14557` | 15 | 1.02 | 14,861 | `22a2a00b4269` |
| `FPE2M2_ACL2025` | 12 | 0.56 | 6,445 | `0721ef2caf9a` |
| `MXFormats_2310.10537` | 9 | 0.39 | 4,173 | `3c22864ff532` |
| `FP4AllTheWay_2505.19115` | 16 | 0.86 | 7,115 | `4c2a418cdfbf` |
| `TrainingMXFP4_2502.20586` | 19 | 1.20 | 9,283 | `0c514da58b8a` |
| `QeRL_2510.11696` | 21 | 1.89 | 10,673 | `3bbcf6a7bd75` |
| `InputBreakQuant_2506.12044` | 20 | 4.01 | 10,457 | `77ffc08beaa5` |
| `RILQ_2412.01129` | 14 | 1.72 | 8,941 | `c1d1c9f73148` |
| `RRS_2409.20361` | 16 | 29.20 | LaTeX source reviewed | `426edb29aaeb` |
| `DuQuant_2406.01721` | 29 | 22.86 | 14,840 | `9ca78e7cb577` |
| `BASEQ_2506.15689` | 19 | 30.10 | 7,313 | `5a1fd16252dc` |
| `ButterflyQuant_2509.09679` | 16 | 0.76 | 8,436 | `ace8809ace99` |
| `SmoothRot_2506.05413` | 6 | 1.24 | 3,888 | `62e8b0703479` |
| `AFPQ_2311.01792` | 8 | 0.45 | 4,375 | `1f2a8454ff04` |
| `TetraJet_2502.20853` | 15 | 1.16 | 8,155 | `1d08c54daa71` |
| `ModelExpansion_2503.17513` | 9 | 0.29 | 5,215 | `a6dc23317c85` |

## 一手来源

- HiFloat4: https://arxiv.org/abs/2602.11287
- HiFloat4 RL: https://arxiv.org/abs/2607.26515
- NVIDIA NVFP4 technical article: https://developer.nvidia.com/blog/using-nvfp4-low-precision-model-training-for-higher-throughput-without-losing-accuracy/
- Four Over Six: https://arxiv.org/abs/2512.02010
- TetraJet-v2: https://arxiv.org/abs/2510.27527
- Quartet II: https://arxiv.org/abs/2601.22813
- MR-GPTQ: https://arxiv.org/abs/2509.23202
- OAS/MBS: https://arxiv.org/abs/2603.08713
- ARCQuant: https://arxiv.org/abs/2601.07475
- DecDEC: https://arxiv.org/abs/2412.20185
- NVFP4 QAD: https://arxiv.org/abs/2601.20088
- M²XFP: https://arxiv.org/abs/2601.19213
- MX+: https://arxiv.org/abs/2510.14557
- FPE2M2: https://aclanthology.org/2025.findings-acl.943/
- Microscaling Formats: https://arxiv.org/abs/2310.10537
- FP4 All the Way: https://arxiv.org/abs/2505.19115
- Training LLMs with MXFP4: https://arxiv.org/abs/2502.20586
- QeRL: https://arxiv.org/abs/2510.11696
- Why Do Some Inputs Break Quantization?: https://arxiv.org/abs/2506.12044
- RILQ: https://arxiv.org/abs/2412.01129
- RRS: https://arxiv.org/abs/2409.20361
- DuQuant: https://arxiv.org/abs/2406.01721
- BASE-Q: https://arxiv.org/abs/2506.15689
- ButterflyQuant: https://arxiv.org/abs/2509.09679
- SmoothRot: https://arxiv.org/abs/2506.05413
- AFPQ: https://arxiv.org/abs/2311.01792
- TetraJet: https://arxiv.org/abs/2502.20853
- Improving Quantization with Post-Training Model Expansion: https://arxiv.org/abs/2503.17513

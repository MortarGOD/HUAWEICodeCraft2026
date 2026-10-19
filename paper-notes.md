# 论文审阅与本题映射（47/47）

已覆盖 `docs/papers/quantization/pdf` 下全部 44 篇论文：43 篇对应文本版，加上只有 PDF 的 RRS。根目录的 `HiFP4.pdf`、`NVFP4.pdf` 与量化论文目录中的对应资料重复，不另计。结论先行：本题的返回格式严格固定，最有价值的自由度是合法 HiF4 层级 scale/舍入搜索、校准输出重构、A/W 与 Q/K 的可逆缩放，以及低成本二阶代理。需要额外码本、残差通道、混精度旁路、模型结构修改或训练的方案不能直接进入提交格式。

## 格式与 block-scale 设计

- **HiFloat4**：三级 E6M2/E1_8/E1_16 + S1P2；官方算法以 absmax/7 和阈值贪心构造元数据。直接启发 v1 的“固定格式内重搜元数据”。
- **HiFloat4 RL**：指出 activation outlier 引发 zero collapse，activation 比 weight 更敏感；残差 GEMM 不兼容本接口，但支持优先改善在线激活分辨率。
- **MX Formats**：微缩放的核心是 block scale 与元素格式的联合折中；支持以 block 为最小优化单元。
- **NVIDIA NVFP4 technical blog / NVFP4**：确认 NVFP4 的 16 元素 E4M3 scale 与 E2M1 carrier、forward RTN 和全局/局部 scale 约束，是输入反量化与二次量化分析基础。
- **Four Over Six**：对同一块生成多个 scale 候选并按 MSE 选择，通常优于固定 absmax；这是 v1 E6M2 邻域搜索最直接的证据。
- **M2XFP**：少量灵活 metadata 可显著恢复 microscaling 精度；与 HiF4 已有三级 metadata 的“编码自由度充分利用”一致。
- **MX+**：针对块内 outlier 重新分配表示精度；本题不能改格式，但可通过 E6/E1 选择改变精度落点。
- **MXFP4 OAS/MBS**：允许受控 overflow、提高粗粒度 scale 精度并用 SSE 搜索；启发较大 E6 候选和裁剪/主体分辨率权衡。
- **FPE2M2**：分析低比特 FP 指数/尾数组合；格式不可改，作为 S1P2 非均匀误差分析参考。
- **AFPQ**：正负独立 scale 可处理非对称分布；HiF4 只有共享无符号 scale，无法直接实现，后续只能研究符号相关舍入偏置。

## 输出重构、二阶信息与舍入

- **GPTQ**：用近似 Hessian 和误差补偿做一次性权重量化；v2/v3 在校准阶段采用输出重构，后续可实现 64 维块内 GPTQ 代理。
- **OBC/OBQ**：用逆 Hessian 指导逐权重/逐列量化顺序，并把量化误差传播给尚未量化参数；v21 将其限制在连续64维 HiF4 block，补偿结果完全固化进最终 Weight mant。
- **MR-GPTQ**：指出传统 outlier 方法在细粒度 microscaling 上可能失效，并用 block Hadamard/FP4 特定优化；支持“格式专用而非照搬 INT4 GPTQ”。
- **AdaRound**：nearest rounding 不是任务最优，局部二阶目标可选择上下舍入；HiF4 mant 与层级 scale 都可纳入类似离散选择。
- **BRECQ**：block reconstruction 在跨层依赖和泛化间折中；本题只有单层数据，校准 holdout 是更适合的防过拟合措施。
- **OmniQuant**：LWC + LET，以可学习裁剪和等价变换做 block-wise 重构；对应 v2/v3 的候选平滑与输出误差选择。
- **HQQ/AdaDim**：量化分组方向应匹配 activation outlier 与 weight sensitivity；格式固定连续 64，但 importance/重排思想可用于代理目标。
- **AWQ**：activation 统计识别重要输入通道，用等价缩放保护显著权重；直接对应 Linear 逐通道可逆缩放。
- **AQLM**：输入感知加性码本与跨 block 联合优化精度高；自定义码本不兼容固定 HiF4。
- **QuIP**：Hessian incoherence + 自适应舍入；二阶目标有用，但完整正交预处理需要评分端逆变换。
- **QuIP#**：随机 Hadamard + E8 lattice vector quantization；lattice 码本不兼容，Hadamard 只在可完整抵消时可用。
- **SpQR**：隔离高敏感 outlier 并高精度存储；本题禁止额外稀疏高精度表示。
- **SqueezeLLM**：Fisher 加权非均匀量化与稀疏异常值保护；自定义码本/FP16稀疏旁路不兼容，但敏感度候选选择和异常值优先量化可迁移到固定 HiF4 网格。
- **RILQ**：用 LoRA 协同补偿量化误差；没有训练/adapter 返回接口，不能直接使用。
- **InputBreakQuant**：不同方法的坏样本高度相关，误差与 residual magnitude 和后层精度敏感性相关；支持线上反馈后构造“困难分组”而不是相信单一 mini。

## 平滑、旋转、重排与异常值

- **SmoothQuant**：用逐通道可逆缩放在 activation/weight 间迁移难度；Linear v1 的核心变换。
- **Outlier Suppression+**：先 shift 处理非对称、再 scale 平衡量化负担；scale 可直接借鉴，shift 因缺少 bias 补偿接口风险较高。
- **RPTQ**：按通道动态范围聚类/重排；评分端不会重排，除非 A/W 同时以同一 permutation 输出，Linear 可作为后续候选，Attention 更受 head 结构约束。
- **QuaRot**：成对正交旋转消除 outlier；本题 Linear 可以同时旋转 A/W，但在线矩阵乘成本与 state 大小需严格评估。
- **SpinQuant**：学习旋转显著优于随机旋转；校准可学习，但 300 秒限制和在线旋转成本较高。
- **DuQuant**：outlier-aware block rotation + zigzag permutation + 二次旋转；对连续 HiF4 block 很相关，但需要在线变换，列为高成本分支。
- **SmoothRot**：缩放后 Hadamard 兼顾 massive/normal outlier；可作为 Linear 后续 vN，但先验证纯缩放线上收益。
- **RRS（Rotated Runtime Smooth）**：区分 channel-wise outlier 与 spike outlier；运行时分组最大值解决校准 scale mismatch，Hadamard rotation 缓解 spike 导致的 victim。题目不能在 GEMM 后附加运行时 scale，但可迁移其“动态块统计 + 成对 A/W 旋转/重排”思想。
- **AffineQuant**：把对角缩放推广为可逆 affine 变换；优化空间大，但在线矩阵变换不符合低时延优先级。
- **ButterflyQuant**：可学习正交 butterfly，以 O(n log n) 降低旋转成本；若线上证明 outlier 主导，可作为比 dense rotation 更现实的方向。
- **BASE-Q**：rotation 后均值未对齐且高斯化会加重 clipping，提出 bias correction + asymmetric scaling；提醒旋转不是无条件收益。

## FP4 训练、残差和格式外补偿

- **ARCQuant**：增加量化残差通道，用第二阶段 GEMM 补偿；固定逻辑 shape 不允许扩展 reduction 维度。
- **DecDEC**：动态取回显著通道 residual；需要 CPU residual 和额外计算，不兼容返回契约。
- **Model Expansion**：用少量扩维换取低比特精度；逻辑 shape 固定，不能使用。
- **TetraJet**：MXFP4 训练的 weight oscillation 与 EMA/ramping；本题是静态转换，不存在训练振荡。
- **TetraJet-v2**：unbiased double-block、OsciReset、OutControl；outlier 结论有参考价值，训练/混精度机制不可直接用。
- **Training MXFP4**：stochastic rounding + 随机 Hadamard 获得低方差无偏梯度；训练期方案，推理转换仍以确定性 RTN/MSE 为主。
- **FP4 All The Way**：NVFP4 block16、E4M3 scale、forward RTN 最稳定；确认输入格式设计合理，不改变 HiF4 输出约束。
- **Quartet II**：MS-EDEN 改善无偏梯度估计；本题无 backward，不采用随机量化。
- **QeRL**：量化噪声可增加 RL 探索；目标不是训练探索，不适用。
- **NVFP4 QAD**：蒸馏可恢复 NVFP4 模型精度且对数据质量较稳健；无 teacher/student 训练入口，不适用。
- **BitDistiller**：QAT+CAKLD 无训练入口；其正负非对称 clipping 初始化可近似为 HiF4 合法网格内的符号相关舍入候选，进入 v21 校准消融。

## 当前实现决策

- v1：HiF4 合法 E6 邻域 + E1 子树 SSE 精确选择；Linear A/W 与 Attention Q/K 可逆平滑。
- v2：允许的 calibration `A@W` / Attention 输出重构搜索；作为完整方法原型保留，但实测 smoke 耗时超预算。
- v3：fit/holdout 划分 + 低成本 proxy 编码选择变换，最终只运行一次完整 HiF4 搜索。
- v4/v5/v6：完成 RRS Linear/Attention 拆分；线上证明 Linear 分支 -499 分、Attention 分支 +864 分，当前只保留 Attention。
- v7：每个 RRS transform 候选先拟合旋转后 Q/K 平滑，再按实际部署路径计算 holdout Attention MSE。
- 下一主线：DuQuant zigzag permutation；随后依据线上证据选择第二 rotation 或 deterministic sign 扩展。
- 暂不实现：额外残差/混精度/码本/扩维/训练；BASE-Q bias/asymmetric scale 因无法严格抵消，只先用于诊断和候选排序。

已使用论文不视为“开发完毕”。HiFloat4 mant 舍入、RRS/QuIP# sign ensemble、RPTQ 稳健分桶、BRECQ 多样本目标、SmoothQuant/OmniQuant rotation 后参数搜索仍有独立候选空间，详见 [`used-paper-gap-audit.md`](used-paper-gap-audit.md)。

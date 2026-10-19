# 校准阶段用完整 W 拟合测试阶段 Q(A)：论文参考指南（v2.0）

> **场景形式化**：
>
> $$\underbrace{A_1 W \xrightarrow{\text{calibrate}} Q(W),\ \Theta}_{\text{校准阶段}} \quad \longrightarrow \quad \underbrace{A_2 \cdot Q(W) \xrightarrow{\text{infer}} Q(A)}_{\text{测试阶段}}$$
>
> 其中 $W$ 是完整精度权重，$Q(W)$ 是量化后权重，$\Theta$ 是"其他信息"（缩放因子、零点、旋转矩阵、平滑因子、残差矩阵等），$A_1$ 是校准激活，$A_2$ 是测试激活。优化目标本质上是：
>
> $$\min_{Q(W),\ \Theta} \|A_2 W - A_2 \cdot Q(W)\| \approx \min_{Q(W),\ \Theta} \|A_1 W - A_1 \cdot Q(W)\|$$
>
> 即用校准数据上的输出重建误差来近似测试数据上的泛化误差。

---

## 一、Hessian-based 逐层重建（优化 Q(W) 的核心方法）

> 这条路线直接求解 $\min_{Q(W)} \|WX - Q(W)X\|_F^2$，用 Hessian 逆来指导逐列/逐行的最优量化顺序和误差补偿。

### 1. GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- **作者**：Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh（IST Austria）
- **发表**：ICLR 2023 (arXiv:2210.17323)
- **核心思想**：基于 Optimal Brain Quantization (OBQ) 的高效近似。用逆 Hessian $H^{-1} \approx (XX^T)^{-1}$ 指导逐列贪心量化——每量化一列权重后，用 $H^{-1}$ 更新剩余列以补偿误差。175B 参数模型约 4 GPU 小时完成 3-4 bit 量化。
- **与你场景的关系**：GPTQ 是最经典的"校准阶段用完整 W 拟合测试阶段输出"的方法。$A_1$ 用于估计 Hessian，$Q(W)$ 通过 Hessian-guided 补偿得到，$\Theta$ 为空（纯权重量化）。
- **链接**：https://arxiv.org/abs/2210.17323

### 2. OBQ: Optimal Brain Quantization
- **作者**：Elias Frantar, Dan Alistarh
- **发表**：NeurIPS 2022 前身工作
- **核心思想**：GPTQ 的理论前身。逐权重量化后用二阶信息更新剩余权重，计算复杂度高但精度极好。GPTQ 通过 Cholesky 分解和 lazy batch updates 将其加速数个数量级。
- **与你场景的关系**：OBQ 是"用完整 W 拟合 Q(A)"这一思想的理论源头——其目标函数就是 $\|WX - Q(W)X\|_F^2$。

---

## 二、激活感知缩放（优化 Θ = 缩放因子）

> 这条路线不直接修改量化过程，而是在量化前对权重做等价变换 $W' = \text{diag}(s) \cdot W$，$X' = X \cdot \text{diag}(s)^{-1}$，使 $W'$ 更容易被量化。$\Theta = \{s\}$。

### 3. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
- **作者**：Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Song Han 等（MIT / NVIDIA）
- **发表**：MLSys 2024 Best Paper (arXiv:2306.00978)
- **核心思想**：发现仅 1% 的"显著权重通道"对量化误差贡献极大。通过校准数据 $A_1$ 的激活幅度识别显著通道，将其权重放大 $s > 1$ 倍（同时激活缩小 $1/s$），使量化后相对误差更小。通过网格搜索确定最优 $s$。
- **与你场景的关系**：$\Theta = \{s\}$（per-channel 缩放因子），$Q(W)$ 是对缩放后权重的标准量化。校准阶段用 $A_1$ 的统计信息确定 $s$，测试阶段 $A_2$ 先乘以 $\text{diag}(s)^{-1}$ 再与 $Q(W)$ 相乘。
- **链接**：https://arxiv.org/abs/2306.00978

### 4. SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
- **作者**：Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han（MIT / NVIDIA）
- **发表**：ICML 2023 (arXiv:2211.10438)
- **核心思想**：激活中的异常值使激活难以量化。通过等价变换 $s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$ 将量化难度从激活迁移到权重。$\alpha$ 控制迁移程度，$\alpha=0.5$ 时难度均分。
- **与你场景的关系**：$\Theta = \{s, \alpha\}$。校准阶段用 $A_1$ 计算通道级最大值确定 $s$，测试阶段 $A_2$ 先除以 $s$ 再量化为 $Q(A)$，$Q(W)$ 已经吸收了 $s$ 的缩放。
- **链接**：https://arxiv.org/abs/2211.10438

### 5. SmoothQuant+（待核实条目）
- **核验状态**：截至 2026-08-27，没有找到与本条标题、算法组合和“W4A4 提升 2-5%”完全对应的独立论文或 Intel Neural Compressor 官方算法说明。
- **处理原则**：逐层 $\alpha$ 搜索、分组量化和通道重排应分别作为已有方法的工程组合做消融，不能把尚未核实的二手摘要当作论文结论或预期收益。

---

## 三、自适应舍入（优化 Q(W) 的舍入决策）

> 标准量化用 round-to-nearest，但最优舍入方向（向上 or 向下）取决于数据和任务损失。

### 6. AdaRound: Up or Down? Adaptive Rounding for Post-Training Quantization
- **作者**：Markus Nagel, Rana Ali Amjad, Mart van Baalen, Christos Louizos, Tijmen Blankevoort（Qualcomm AI Research）
- **发表**：ICML 2020 (arXiv:2004.10568)
- **核心思想**：将舍入决策建模为二次无约束二值优化（QUBO）问题。通过 Taylor 展开近似任务损失，引入可学习参数 $V_{ij}$ 控制每个权重的舍入方向（$\sigma(V_{ij})$ 决定向上/向下），用退火策略使 $V_{ij}$ 趋向 0/1 二值解。
- **与你场景的关系**：$Q(W)$ 不仅包含量化值，还包含最优舍入方向信息。$\Theta$ 为空（舍入方向已固化到 $Q(W)$ 中）。校准阶段用 $A_1$ 学习每个权重的最优舍入方向。
- **链接**：https://arxiv.org/abs/2004.10568

### 7. BRECQ: Pushing the Limit of Post-Training Quantization by Block Reconstruction
- **作者**：Yuhang Li, Ruihao Gong, Xu Tan 等（上海 AI Lab / 商汤）
- **发表**：ICLR 2021 (arXiv:2102.05426)
- **核心思想**：发现 block 级（而非 layer 级或 network 级）重建是二阶误差分析中跨层依赖与泛化误差的最佳平衡点。用对角 Fisher 信息矩阵近似预激活 Hessian 来加权重建损失。首次将 PTQ 推到 INT2。
- **与你场景的关系**：将"用完整 W 拟合 Q(A)"的重建粒度从单层扩展到 Transformer block，更好地捕获层间依赖。
- **链接**：https://arxiv.org/abs/2102.05426

---

## 四、旋转与不相关化处理（优化 Θ = 旋转矩阵）

> 通过正交旋转使权重/激活分布更均匀（高斯化），消除异常值，使后续量化误差大幅降低。

### 8. QuIP: 2-Bit Quantization of Large Language Models With Guarantees
- **作者**：Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa（Cornell）
- **发表**：NeurIPS 2023 (arXiv:2307.13304)
- **核心思想**：提出"不相关化"（incoherence）概念——量化误差在权重矩阵元素幅度均匀、Hessian 特征向量与坐标轴不对齐时最小。通过随机正交矩阵 $Q_1, Q_2$ 对权重做共轭变换 $W' = Q_1 W Q_2^T$，使 $W'$ 满足不相关条件，再量化。
- **与你场景的关系**：$\Theta = \{Q_1, Q_2\}$（旋转矩阵）。校准阶段确定旋转矩阵，测试阶段 $A_2$ 先乘以 $Q_2$，再与 $Q(W')$ 相乘，最后乘以 $Q_1^T$。
- **链接**：https://arxiv.org/abs/2307.13304

### 9. QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks
- **作者**：Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa（Cornell）
- **发表**：ICML 2024 (arXiv:2402.04396)
- **核心思想**：用 Randomized Hadamard Transform (RHT) 替代 QuIP 的 Kronecker 分解做不相关化，复杂度从 $O(n^2)$ 降到 $O(n \log n)$。引入基于 $E_8$ 格的向量量化码本，实现最优 8 维球填充。加入层间微调进一步提升精度。
- **与你场景的关系**：$\Theta = \{H, s_1, s_2\}$（Hadamard 矩阵 + 符号向量）。校准阶段确定 RHT 参数，测试阶段用快速 Walsh-Hadamard 变换处理 $A_2$。
- **链接**：https://arxiv.org/abs/2402.04396

### 10. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **作者**：Saleh Ashkboos 等（ETH Zurich / IST Austria）
- **发表**：NeurIPS 2024 (arXiv:2404.00456)
- **核心思想**：利用计算不变性 $Wx = (WQ^T)(Qx)$，将随机 Hadamard 矩阵 $Q$ 融入权重矩阵，使隐藏状态中的异常值被分散到所有通道，实现端到端 4-bit 量化（权重+激活+KV cache）。
- **与你场景的关系**：$\Theta = \{Q\}$（Hadamard 矩阵）。校准阶段确定 $Q$，测试阶段 $A_2$ 先乘以 $Q$，再与 $Q(WQ^T)$ 相乘。
- **链接**：https://arxiv.org/abs/2404.00456

### 11. SpinQuant: LLM Quantization with Learned Rotations
- **作者**：Zechun Liu, Changsheng Zhao 等（Meta / UIUC）
- **发表**：ICLR 2025 (arXiv:2405.16406)
- **核心思想**：发现不同随机旋转矩阵的量化效果差异巨大（零样本性能差距可达 13 个百分点）。用 Cayley 优化在小验证集上学习最优旋转矩阵，替代固定 Hadamard 旋转。
- **与你场景的关系**：$\Theta = \{Q_{\text{learned}}\}$（学习到的旋转矩阵）。校准阶段通过优化确定 $Q$，比固定 Hadamard 更适配模型特定的异常值结构。
- **链接**：https://arxiv.org/abs/2405.16406

---

## 五、可学习仿射变换（优化 Θ = 仿射参数）⭐ NEW

> 将 SmoothQuant 的手工缩放推广为端到端可学习的仿射变换，是"手工设计 → 可学习"范式跃迁的关键节点。

### 12. FlatQuant: Flatness Matters for LLM Quantization ⭐ NEW
- **作者**：Yuxuan Sun、Ruikang Liu、Haoli Bai 等（华为诺亚方舟实验室 / 清华大学 / 香港中文大学）
- **发表**：ICML 2025（arXiv:2410.09426，2024-10 首发）
- **核心思想**：提出 **Fast and Learnable Affine Transformation**。核心不是“对角缩放 + 偏移”，而是为每个线性层学习可逆矩阵 $P$，并用 $P=P_1\otimes P_2$ 的 Kronecker 分解降低存储与计算量；此外联合学习通道缩放 $c$、权重/激活裁剪阈值 $\alpha_w,\alpha_a$。校准目标是 Transformer block 输出重建 MSE。
- **与你场景的关系**：严格等价形式为 $Q(A P)Q(P^{-1}W^T)$。本题没有论文所用的融合 GPU kernel，直接部署完整因子有超时风险，因此先用 H8 初始化加稀疏 Givens 更新近似 $P_1,P_2$，并由五类校准布局交叉验证是否替换固定 H64。
- **链接**：https://arxiv.org/abs/2410.09426

---

## 六、平滑 + 旋转联合可训练优化 ⭐

> 将通道缩放（Smooth）与旋转（Rotation）统一为端到端可训练的等价变换，解决 max-based 策略的过度迁移问题。

### 13. Trainable Smooth-Rotation Transforms with Learned Channel Scales for LLM Quantization
- **作者**：Patrik Czakó, Gábor Kertész, Sándor Szénási
- **发表**：IEEE INES 2026 (arXiv:2606.09927, 2026-06)
- **核心思想**：指出基于 max 统计量的 SmoothRot 风格等价变换存在**过度迁移（over-migration）** 问题——max 对异常值过于敏感，导致缩放因子过大，正常值被过度压缩。提出两项改进：(1) **分位数鲁棒缩放策略**——用高分位数替代 max 作为激活统计量，减少异常值对缩放因子的主导；(2) **约束梯度优化通道缩放**——在等价变换框架内通过轻量级梯度训练学习最优通道缩放参数。在 LLaMA-3.2-1B W4A4 量化上，仅分位数策略搜索比 SmoothRot 基线降低选定层误差 11.1%，联合 $(\alpha, q)$ 搜索降低 12%，训练达到 18.5%。将最优策略重放到所有 decoder-block down-projection 层，全层平均误差从 97.51 降至 78.08（↓19.9%）。
- **与你场景的关系**：**直接适用于校准阶段**。$\Theta = \{s_{\text{learned}}, q, \alpha\}$（学习到的通道缩放 + 分位数 + 迁移系数）。校准阶段用 $A_1$ 通过梯度优化确定这些参数，测试阶段 $A_2$ 先做分位数鲁棒的等价变换再与 $Q(W)$ 相乘。相比 SmoothQuant/SmoothRot 的固定 max-based 策略，该方法在保持等价变换框架的同时获得了显著精度提升。
- **链接**：https://arxiv.org/abs/2606.09927

---

## 七、异常值分离（优化 Θ = 异常值索引 + 高精度残差）

> 将少量对量化误差贡献极大的异常值权重保留在高精度，其余正常量化。

### 14. SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression
- **作者**：Tim Dettmers, Rodion Svirschevski 等（IST Austria / NVIDIA）
- **发表**：arXiv:2306.03078, 2023
- **核心思想**：发现 ~1% 的敏感权重占总量化误差的 75% 以上。用 Hessian-based 敏感度检测隔离异常值，保留为 FP16，其余用 3-4 bit 量化。引入双级量化压缩量化统计信息本身。使用极小分组（8-32）提高精度。
- **与你场景的关系**：$\Theta = \{\text{outlier indices},\ W_{\text{outlier}}^{\text{FP16}}\}$。校准阶段确定哪些权重是异常值，测试阶段对异常值通道用高精度计算，其余用 $Q(W)$。
- **链接**：https://arxiv.org/abs/2306.03078

### 15. SqueezeLLM: Dense-and-Sparse Quantization
- **作者**：Sehoon Kim 等（UC Berkeley）
- **发表**：ICML 2024
- **核心思想**：结合密集量化和稀疏异常值保留。用 Hessian 敏感度识别关键权重，将其分配到更高精度的码本中。
- **与你场景的关系**：$\Theta = \{\text{sensitivity map},\ \text{high-precision codebook}\}$。

---

## 八、蒸馏校准（用教师模型指导 Q(W) 的学习）

### 16. QAD: Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery
- **作者**：Xin Yu 等（NVIDIA）
- **发表**：arXiv:2601.20088, 2026
- **核心思想**：以 BF16 原始模型为教师，量化模型为学生，用 KL 散度蒸馏恢复精度。对经过 SFT+RL+模型合并等多阶段后训练的模型特别有效。
- **与你场景的关系**：校准阶段不仅用重建误差 $\|A_1 W - A_1 Q(W)\|$，还用教师模型的输出分布 $P(A_1 W)$ 来指导 $Q(W)$ 的学习。$\Theta$ 可能包含蒸馏温度等超参数。
- **链接**：https://arxiv.org/abs/2601.20088

### 17. BitDistiller: Unleashing the Potential of Sub-4-Bit LLMs via Self-Distillation
- **作者**：Dayong Yang 等
- **发表**：ECCV 2024
- **核心思想**：用渐进式自蒸馏训练超低比特（2-3 bit）模型。从高比特教师逐步蒸馏到低比特学生。
- **与你场景的关系**：如果 $Q(W)$ 的位宽极低，蒸馏可以提供比纯重建更丰富的监督信号。

---

## 九、浮点量化专用校准（针对 FP4 格式）

### 18. MR-GPTQ: Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization
- **作者**：Vage Egiazarian 等（IST Austria / NVIDIA）
- **发表**：arXiv:2509.23202, 2025
- **核心思想**：首个全面研究 MXFP4/NVFP4 PTQ 的工作。揭示 NVFP4 小群大小使传统异常值缓解技术失效。提出 MR-GPTQ——融合 Hadamard 旋转 + 分块网格搜索缩放优化。
- **与你场景的关系**：当 $Q(W)$ 是 NVFP4 格式时，$\Theta$ 包含缩放因子和旋转矩阵，MR-GPTQ 提供了联合优化这些参数的方法。
- **链接**：https://arxiv.org/abs/2509.23202

### 19. HiGPTQ（HiFloat4 论文中提出）
- **来源**：arXiv:2602.11287 中的实验部分
- **核心思想**：基于 GPTQ 框架为 HiF4 定制的 PTQ 方法。利用 HiF4 的细粒度三级结构进行 Hessian-aware 的逐层量化。
- **与你场景的关系**：当目标格式是 HiF4 时，HiGPTQ 是最直接的校准方法——用 $A_1$ 估计 Hessian，优化 HiF4 三级缩放因子的分配。
- **链接**：https://arxiv.org/abs/2602.11287

---

## 十、KV Cache 量化（优化 Θ = 极坐标参数 + 校正器）⭐ NEW

> 针对推理时 KV Cache 的在线量化，与权重量化互补。

### 20. TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate ⭐ NEW
- **作者**：Google Research
- **发表**：arXiv:2504.19874, 2025-04 首发
- **核心思想**：针对 **KV Cache**（而非权重）的量化。提出两阶段框架：(1) **PolarQuant**（极坐标量化）——将 KV 向量通过随机旋转变换到极坐标系后量化，消除 per-channel 缩放因子的元数据开销；(2) **QJL**（Quantized Johnson-Lindenstrauss 变换）——用 1-bit 校正器消除系统性偏差。实现 3-bit KV Cache，内存降 6 倍，注意力计算提速 8 倍（H100），精度零损失。免训练，无需校准阶段。
- **与你场景的关系**：如果你的系统不仅需要量化权重 $W$，还需要量化推理时的 KV Cache，TurboQuant 是当前最优解。它的"旋转+量化"思路与 QuaRot/Czakó et al. 一脉相承，但专门针对 KV Cache 的在线量化场景设计。注意：该论文存在与 RaBitQ（ETH Zurich 高健扬团队）的技术相似性争议。
- **链接**：https://arxiv.org/abs/2504.19874

---

## 十一、向量量化与码本方法

### 21. AQLM: Extreme Compression of Large Language Models via Additive Quantization
- **作者**：Valerii Egiazarian 等（IST Austria）
- **发表**：ICML 2024
- **核心思想**：用加法量化（Additive Quantization）将权重表示为多个码本的加权和。每个码本通过残差迭代学习。支持 2-bit 极端压缩。
- **与你场景的关系**：$\Theta = \{\text{codebooks}\}$。校准阶段学习码本，测试阶段 $Q(W)$ 是码本索引，$A_2 \cdot Q(W)$ 通过查表+加法实现。

---

## 技术路线与场景对应关系总结

| 技术路线 | 优化的变量 | Θ（其他信息） | 代表论文 |
|----------|-----------|--------------|----------|
| **Hessian 重建** | $Q(W)$ 的量化值 | 空（或 Hessian 逆） | GPTQ, OBQ |
| **激活感知缩放** | $Q(W)$ 的缩放因子 | per-channel scale $s$ | AWQ, SmoothQuant, SmoothQuant+ |
| **自适应舍入** | $Q(W)$ 的舍入方向 | 空（方向已固化） | AdaRound, BRECQ |
| **旋转/不相关化** | $Q(W)$ 的旋转后值 | 旋转矩阵 $Q$ | QuIP, QuIP#, QuaRot, SpinQuant |
| **可学习仿射变换** ⭐ | $Q(W)$ 的缩放+偏移 | learnable affine params | **FlatQuant** |
| **平滑+旋转联合可训练** ⭐ | $Q(W)$ 的缩放+旋转 | learned scale, quantile, $\alpha$ | **Czakó et al. (2026)** |
| **异常值分离** | $Q(W)$ + 异常值掩码 | outlier indices + FP16 残差 | SpQR, SqueezeLLM |
| **蒸馏校准** | $Q(W)$ 的分布 | 教师模型输出 | QAD, BitDistiller |
| **FP4 专用** | $Q(W)$ 的缩放+旋转 | scale + rotation | MR-GPTQ, HiGPTQ |
| **KV Cache 量化** ⭐ | $Q(KV)$ 的极坐标表示 | polar params + QJL corrector | **TurboQuant** |
| **向量量化** | $Q(W)$ 的码本索引 | codebooks | AQLM, QuIP# |

---

## 等价变换范式演进脉络

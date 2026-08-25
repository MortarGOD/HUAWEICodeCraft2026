# NVFP4 → HiF4 转换精度优化：全技术路线论文参考指南

> **课题本质**：跨格式重量化（re-quantization）——将 NVFP4 编码的模型（E2M1 + 16元素块 + E4M3缩放因子）转换为 HiF4 格式（S1P2 + 64元素组 + 三级层次缩放），同时尽可能恢复精度。
>
> **技术路线全景**：4-bit 量化领域存在多条并行且正交互补的技术路线——格式设计、旋转/平滑、GPTQ校准、蒸馏、残差补偿。本指南全部覆盖。

---

## 一、核心格式定义论文（必读）

### 1. HiFloat4 Format for Language Model Inference
- **作者**：Yuanyong Luo, Jing Huang, Yu Cheng 等（华为）
- **发表**：arXiv:2602.11287, 2026-02
- **核心内容**：HiF4 格式的完整定义——64元素组、三级缩放层次（E6M2全局基缩放 + 8-way E1微指数 + 16-way E1微指数）、S1P2元素编码。论文给出了从 BF16 到 HiF4 的完整转换算法（Algorithm 1），包括三级树归约、层次缩放元数据推导、组内元素缩放与量化。还提出了 HiGPTQ——基于 GPTQ 框架为 HiF4 定制的 PTQ 方法，在 Qwen2.5-14B 上甚至超越 BF16 基线。
- **与课题的关系**：**目标格式的定义论文**。Algorithm 1 是转换的"终点"。HiF4 vs NVFP4 的 MSE 对比（HiF4 低 24%）是精度上界参考。
- **链接**：https://arxiv.org/abs/2602.11287

### 2. HiFloat4 Format for End-To-End Reinforcement Learning Post-Training of LLMs
- **作者**：华为 / Global Computing Consortium
- **发表**：arXiv:2607.26515, 2026-07
- **核心内容**：HiF4 在 RL 后训练场景的扩展。系统诊断了 rollout 激活下溢问题，提出 Rollout-ResQ 稀疏残差量化机制。证明 HiF4 三级缩放在 RL 动态分布场景下优于 MXFP4（基线 82% vs 73%）。
- **与课题的关系**：如果转换场景涉及训练，Rollout-ResQ 残差机制可借鉴——将转换误差以稀疏残差形式补偿。
- **链接**：https://arxiv.org/abs/2607.26515

### 3. NVIDIA FP4 Quantization Algorithm（NVFP4 技术博客 + 多篇论文综合）
- **来源**：NVIDIA Developer Blog, 2025-08; 综合 TetraJet-v2、Four Over Six 等
- **核心内容**：NVFP4 的完整定义——E2M1 元素（{±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}）、16元素块 + E4M3缩放因子 + 可选 FP32 全局缩放。量化/反量化算法、双块缩放、4/6 自适应缩放等。
- **与课题的关系**：**源格式的定义**。NVFP4 的量化误差特征（近最大值处误差最大）直接决定转换时哪些信息已丢失、哪些可恢复。
- **链接**：https://developer.nvidia.com/blog/nvfp4-pretraining/

### 4. Microscaling Data Formats for Deep Learning
- **作者**：Bita Darvish Rouhani 等（OCP 联盟）
- **发表**：arXiv:2310.10537, 2023-10
- **核心内容**：OCP MX 格式的定义论文，系统评估了 MX 数据格式在推理和训练中的表现，首次展示 sub-8-bit 权重/激活/梯度下训练生成式语言模型的可行性。
- **与课题的关系**：理解 MX 格式的设计哲学有助于理解 HiF4 为何选择三级缩放而非 MXFP4 的单级缩放。
- **链接**：https://arxiv.org/abs/2310.10537

---

## 二、旋转与平滑路线（Outlier Suppression via Rotation & Smoothing）

> 这条路线的核心思想是：在量化**之前**，通过正交旋转或通道级平滑操作消除激活中的异常值，使数据分布更均匀，从而让后续的 4-bit 量化误差大幅降低。与格式设计路线**正交互补**——旋转处理的是"数据分布问题"，格式设计处理的是"编码精度问题"。

### 5. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
- **作者**：Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li 等（ETH Zurich / IST Austria）
- **发表**：NeurIPS 2024 (arXiv:2404.00456, 2024-04)
- **核心内容**：提出基于 Hadamard 旋转的量化方案。利用计算不变性（computational invariance）：Wx = (WQᵀ)(Qx)，将随机 Hadamard 矩阵 Q 融入权重矩阵，使隐藏状态中的异常值被分散到所有通道，从而无需保留任何高精度通道即可实现端到端 4-bit 量化（权重+激活+KV cache）。LLaMA2-70B 上 WikiText-2 困惑度损失 ≤0.47，零样本性能保留 99%。
- **与课题的关系**：**旋转路线的奠基论文**。在 NVFP4→HiF4 转换中，可在反量化后对激活做 Hadamard 旋转，使数据分布更接近高斯，让 HiF4 的三级缩放因子更精准地覆盖数据范围。旋转操作是 training-free 的，可无缝嵌入转换 pipeline。
- **链接**：https://arxiv.org/abs/2404.00456

### 6. SpinQuant: LLM Quantization with Learned Rotations
- **作者**：Zechun Liu, Changsheng Zhao, Igor Fedorov 等（Meta / UIUC）
- **发表**：ICLR 2025 (arXiv:2405.16406, 2024-05)
- **核心内容**：发现不同随机旋转矩阵的量化效果差异巨大（零样本推理性能差距可达 13 个百分点）。提出用 Cayley 优化在小验证集上**学习**最优旋转矩阵，替代 QuaRot 的固定 Hadamard 旋转。在 LLaMA-2 7B 上 W4A4K4 量化，与全精度差距仅 2.9 个百分点。对难量化的 LLaMA-3 8B，比 QuaRot 缩小精度差距 45.1%。
- **与课题的关系**：**可学习旋转**的思想可用于 NVFP4→HiF4 转换——针对特定模型的激活分布，学习一组最优旋转矩阵，使转换后的 HiF4 量化误差最小化。相比固定 Hadamard，能更好地适配模型特定的异常值结构。
- **链接**：https://arxiv.org/abs/2405.16406

### 7. DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
- **作者**：Haokun Lin, Haobo Xu, Yichen Wu 等（中科院自动化所 / 深圳先进院）
- **发表**：NeurIPS 2024 Oral (arXiv:2406.01721, 2024-06)
- **核心内容**：区分两类异常值——Normal Outliers（跨 token 幅值较大的通道级异常值）和 Massive Outliers（个别 token 中幅值极大的尖峰异常值）。提出双重变换：(1) 以异常值维度为先验，构造旋转矩阵将 Massive Outliers 重分配到相邻通道；(2) Zigzag Permutation 跨块均衡异常值分布，降低块级方差。后续再施加一次旋转进一步平滑。
- **与课题的关系**：**Massive Outliers 的处理对你的课题至关重要**。NVFP4 的 16 元素块极易被单个 Massive Outlier 主导缩放因子，导致块内其他元素精度严重下降。DuQuant 的"先定位异常值维度→旋转分散→排列均衡"三步策略，可在 NVFP4 反量化后、HiF4 量化前使用，显著改善 HiF4 三级缩放的精度。
- **链接**：https://arxiv.org/abs/2406.01721

### 8. Rotated Runtime Smooth (RRS): Training-Free Activation Smoother for Accurate INT4 Inference
- **作者**：Ke Yi, Zengke Liu, Jianwei Zhang, Chengyuan Li, Tong Zhang, Junyang Lin, Jingren Zhou（阿里达摩院 / 通义实验室）
- **发表**：ICLR 2025 (arXiv:2409.20361, 2024-09)
- **核心内容**：提出即插即用的激活平滑器 RRS，由两部分组成：(1) **Runtime Smooth**——推理时动态计算每个通道的 absmax，用通道级最大值实时缩放激活，消除 channel-wise 异常值；(2) **Rotation**——基于 QuaRot 的 Hadamard 旋转，将尖峰异常值的能量扩散到多通道，缩小异常值与正常值的差距，缓解 Runtime Smooth 对正常值的"误伤"（victim 问题）。在 LLaMA/Qwen 家族 INT4 推理上将 WikiText-2 困惑度从 57.33 降至 6.66，超越所有 SOTA 方法。
- **与课题的关系**：
  - **Runtime Smooth** 可在 NVFP4 反量化后、HiF4 量化前使用，消除通道级异常值，让 HiF4 三级缩放因子更精准。
  - **Rotation** 将尖峰异常值能量扩散到多通道，使数据分布更接近高斯，提升 HiF4 层次缩放的覆盖率。
  - **Training-free** 特性使其可无缝嵌入转换 pipeline，零额外训练成本。
  - 与 ARCQuant（残差补偿）互补——RRS 事前预防，ARCQuant 事后补偿。
- **链接**：https://arxiv.org/abs/2409.20361

### 9. BASE-Q: Bias and Asymmetric Scaling Enhanced Rotational Quantization for Large Language Models
- **作者**：Liulu He, Shenli Zheng, Karwei Sun 等
- **发表**：arXiv:2506.15689, 2025-05
- **核心内容**：识别了旋转量化的两个根本局限：(i) 旋转无法对齐通道均值，导致量化边界变宽、舍入误差增大；(ii) 旋转使激活分布更接近高斯，增加了裁剪误差引起的能量损失。提出 **bias correction**（偏置校正，对齐通道均值）+ **asymmetric scaling**（非对称缩放，减少裁剪误差）。支持逐块优化，无需全模型反向传播。相比 QuaRot 缩小精度差距 50.5%，相比 SpinQuant 缩小 42.9%。
- **与课题的关系**：**直接适用于 HiF4 转换流程**。NVFP4 反量化后的数据经旋转变换后，同样面临通道均值不对齐和裁剪误差问题。BASE-Q 的 bias correction 可在 HiF4 量化前对每个子组做均值对齐，asymmetric scaling 可优化 HiF4 S1P2 的量化边界。其逐块优化特性也适合大模型的逐层转换。
- **链接**：https://arxiv.org/abs/2506.15689

### 10. ButterflyQuant: Ultra-low-bit LLM Quantization through Learnable Orthogonal Butterfly Transforms
- **作者**：Bingxin Xu, Zhen Dong, Oussama Elachqar, Yuzhang Shang
- **发表**：arXiv:2509.09679, 2025-09
- **核心内容**：用可学习的 **Butterfly 变换**替代固定 Hadamard 旋转。Butterfly 变换由连续 Givens 旋转角度参数化，可微且保证正交性，支持梯度优化。仅需 128 个校准样本、单 GPU 数分钟即可收敛。不同 Transformer 层展现出不同的异常值模式（注意力层正尾、早期 MLP 负区、晚期 MLP 双峰），Butterfly 变换可为每层学习自适应旋转。LLaMA-2-7B 2-bit 量化困惑度 15.4（QuaRot 为 22.1，QuIP 为 37.3）。
- **与课题的关系**：层自适应旋转的思想对 NVFP4→HiF4 转换有重要启发——不同层对转换误差的敏感度不同，可为敏感层学习更精细的旋转矩阵。Butterfly 变换的 O(n log n) 复杂度也保证了实际可用性。
- **链接**：https://arxiv.org/abs/2509.09679

### 11. SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs
- **作者**：Patrik Czakó, Gábor Kertész, Sándor Szénási（Óbuda University）
- **发表**：arXiv:2506.05413, 2025-06
- **核心内容**：只针对 FFN down-projection，在 Hadamard 旋转前加入离线通道平滑，`s_j=max|X_j|^α/max|W_j|^(1-α)`；缩放被吸收到相邻权重中，因此不增加在线延迟。LLaMA2 7B 上 W4A4KV4 RTN 的 WikiText-2 困惑度相对 QuaRot 改善 0.82。
- **与课题的关系**：当前方案已有成对通道平衡与 Hadamard，其核心机制基本覆盖。论文消融还显示，叠加 GPTQ 或 SpinQuant 后 SmoothRot 的额外收益消失或轻微转负，因此后续只审计平滑强度和变换顺序，不把它作为无条件新增模块。
- **链接**：https://arxiv.org/abs/2506.05413

---

## 三、精度优化：缩放因子与量化策略

### 12. Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling
- **作者**：Jack Cook, Junxian Guo, Guangxuan Xiao, Yujun Lin, Song Han（MIT）
- **发表**：arXiv:2512.02010, 2025-12
- **核心内容**：发现 FP4 格式在近最大值处量化误差最大（如 4 和 6 之间步长为 2），提出对每个块评估两个候选缩放因子（max/6 和 max/4），选 MSE 更低的。可在 Blackwell GPU 上高效实现。
- **与课题的关系**：**直接适用于转换流程**。在将 NVFP4 反量化后重新量化到 HiF4 时，可以对 HiF4 的每个子组做类似的自适应缩放选择。此外，如果源 NVFP4 模型本身就用了 4/6 缩放，反量化精度会更高。
- **链接**：https://arxiv.org/abs/2512.02010

### 13. TetraJet-v2: Accurate NVFP4 Training for LLMs with Oscillation Suppression and Outlier Control
- **作者**：Yuxiang Chen 等（清华大学 / Bosch AI）
- **发表**：arXiv:2510.27527, 2025-10
- **核心内容**：提出三项关键技术——无偏双块量化（double-block scaling）、OsciReset 抑制权重振荡、OutControl 保留异常值精度。双块缩放用外层大块（128元素）+ 内层微块（16元素）的层次化缩放。
- **与课题的关系**：**双块缩放的思想与 HiF4 的三级缩放天然契合**。NVFP4 模型如果使用了 double-block scaling，反量化时可以利用两层缩放信息更精确地还原原始值，再映射到 HiF4 的三级结构。
- **链接**：https://arxiv.org/abs/2510.27527

### 14. Quartet II: Accurate LLM Pre-Training in NVFP4 by Improved Unbiased Gradient Estimation
- **作者**：Panferov 等
- **发表**：arXiv, 2026-01
- **核心内容**：提出 MS-EDEN——一种新型无偏量化原语，MSE 约为元素级随机舍入的一半。结合方形分组缩放 + 4/6 技术优化前向量化。在 NVFP4 全量化训练中比 NVIDIA SOT 方案损失降低 ≥20%。
- **与课题的关系**：MS-EDEN 的无偏量化思想可用于转换过程中的随机舍入策略，减少转换引入的系统性偏差。其"事后范围对齐"（post-hoc range alignment）技巧也可用于 NVFP4→HiF4 的缩放因子映射。
- **链接**：https://arxiv.org/abs/2601.22813

---

## 四、后训练量化（PTQ）方法——可迁移到转换流程

### 15. Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization
- **作者**：Vage Egiazarian 等（IST Austria / NVIDIA）
- **发表**：arXiv:2509.23202, 2025-09
- **核心内容**：首个全面研究 MXFP4/NVFP4 PTQ 的工作。揭示 NVFP4 小群大小使传统异常值缓解技术失效，MXFP4 幂次缩放严重降低精度。提出 **MR-GPTQ** 算法——融合 Hadamard 旋转 + 分块网格搜索缩放优化。在 NVIDIA B200 上实现 2.2× 端到端加速。
- **与课题的关系**：MR-GPTQ 的 Hessian-based 缩放优化可以直接用于 HiF4 转换——在反量化后，用 Hessian 信息指导 HiF4 三级缩放的最优分配。
- **链接**：https://arxiv.org/abs/2509.23202

### 16. Unveiling the Potential of Quantization with MXFP4: Strategies for Quantization Error Reduction
- **作者**：Jatin Chhugani 等（Intel）
- **发表**：arXiv:2603.08713, 2026-03
- **核心内容**：提出 Overflow-Aware Scaling (OAS) 和 Macro Block Scaling (MBS) 两种纯软件技术。OAS 通过检测溢出风险调整缩放因子；MBS 在大块级别引入额外缩放层。将 MXFP4 与 NVFP4 的精度差距从 ~10% 缩小到 <1%。
- **与课题的关系**：OAS 思想可用于转换时的溢出检测——NVFP4 反量化值映射到 HiF4 时，需要确保不超出 S1P2 的表示范围。MBS 的宏块缩放思想与 HiF4 的三级结构高度一致。
- **链接**：https://arxiv.org/abs/2603.08713

### 17. AFPQ: Asymmetric Floating Point Quantization for LLMs
- **作者**：Yijia Zhang 等
- **发表**：arXiv:2311.01792, 2023-11
- **核心内容**：发现浮点量化在小分组或 sub-4-bit 下表现差的原因是缺乏非对称性。提出 AFPQ——对正值和负值分别设置独立的缩放因子，无需额外存储开销。可即插即用地嵌入 GPTQ 和 AWQ。
- **与课题的关系**：HiF4 的 S1P2 格式天然有符号位，但缩放因子是对称的。AFPQ 的非对称缩放思想可用于 HiF4 量化时——对正负值分别计算最优缩放，减少非对称分布数据的量化误差。
- **链接**：https://arxiv.org/abs/2311.01792

---

## 五、误差补偿与残差技术

### 18. ARCQuant: Boosting NVFP4 Quantization with Augmented Residual Channels for LLMs
- **作者**：Haoqian Meng 等
- **发表**：arXiv:2601.07475, 2026-01
- **核心内容**：提出增强残差通道框架——将量化残差编码到额外的"残差通道"中（也用 NVFP4），在 GEMM 时通过标准矩阵乘法实现误差补偿。理论证明双阶段 NVFP4 量化的最坏情况误差界与 MXFP8 相当。RTX 5090 上实现 3× 加速。
- **与课题的关系**：**极其适合你的场景**。NVFP4→HiF4 转换必然引入额外误差，可以用 ARCQuant 的思路：转换后的 HiF4 主通道 + 一个低精度残差通道来补偿转换误差。
- **链接**：https://arxiv.org/abs/2601.07475

### 19. QDEC/DecDEC: Pushing the Envelope of Low-Bit LLM via Dynamic Error Compensation
- **作者**：Yeonhong Park, Jake Hyun 等（首尔大学）
- **发表**：arXiv:2412.20185, 2025-12
- **核心内容**：在 CPU 中存储全精度权重与量化权重的残差矩阵，推理时动态获取与显著通道（由激活异常值标记）对应的残差来补偿量化误差。采用 4-bit 对称均匀量化残差，零拷贝传输。
- **与课题的关系**：可以将 NVFP4 与 HiF4 反量化结果的差值作为残差存储，在关键层动态补偿。
- **链接**：https://arxiv.org/abs/2412.20185

### 20. RILQ: Rank-Insensitive LoRA-based Quantization Error Compensation
- **作者**：Geonho Lee 等
- **发表**：arXiv:2412.01129, 2024-12
- **核心内容**：用低秩适配器（LoRA）补偿量化误差，发现激活差异损失的秩不敏感特性，跨层协同调整适配器。
- **与课题的关系**：如果转换后精度不够，可以用 LoRA 微调来补偿 NVFP4→HiF4 的转换误差。
- **链接**：https://arxiv.org/abs/2412.01129

---

## 六、量化感知蒸馏（QAD）——精度恢复的"终极武器"

### 21. Quantization-Aware Distillation for NVFP4 Inference Accuracy Recovery
- **作者**：NVIDIA（Xin Yu 等）
- **发表**：arXiv:2601.20088, 2026-01
- **核心内容**：以 BF16 原始模型为教师，NVFP4 量化模型为学生，用 KL 散度蒸馏恢复精度。关键优势：对经过 SFT+RL+模型合并等多阶段后训练的模型特别有效，传统 QAT 在这些场景不稳定，而 QAD 稳定恢复至接近 BF16 精度。对数据质量和覆盖度鲁棒。
- **与课题的关系**：**如果精度要求极高，这是最终方案**。流程：NVFP4 反量化 → 初步 HiF4 量化 → QAD 蒸馏微调。用原始 BF16 模型（如果有）做教师，HiF4 模型做学生，通过 KL 散度损失微调 HiF4 权重使其输出分布对齐教师。
- **链接**：https://arxiv.org/abs/2601.20088

---

## 七、格式设计与混合精度参考

### 22. M²XFP: Metadata-Augmented Microscaling Data Format
- **作者**：Hu 等
- **发表**：arXiv, 2026-01
- **核心内容**：在标准 MX 块上增加元数据增强——子组级共享缩放（2位/子组）+ 元素级精度修正（Top-1 元素扩展尾数）。额外仅 0.25 bit/元素开销，即可闭合 4-bit 到 8-bit MXFP 精度差距的 70-95%。
- **与课题的关系**：M²XFP 的"子组级微调缩放"思想与 HiF4 的二级/三级微指数设计异曲同工。理解 M²XFP 有助于理解 HiF4 三级缩放的设计动机。
- **链接**：https://arxiv.org/abs/2601.19213

### 23. MX+: Pushing the Limits of Microscaling Formats for Efficient LLM Serving
- **作者**：Jungi Lee 等
- **发表**：arXiv:2510.14557, 2025-10 (MICRO 2025)
- **核心内容**：发现异常值不需要使用元素数据类型的指数域，因此将指数域重新用作扩展尾数来提高异常值元素的精度。存储开销可忽略，性能显著优于 MXFP4。
- **与课题的关系**：MX+ 的异常值处理思路可借鉴——在 NVFP4→HiF4 转换时，对块内异常值元素做特殊处理（如保留高精度尾数信息）。
- **链接**：https://arxiv.org/abs/2510.14557

### 24. FPE2M2: Approaching Lossless and Efficient Quantization with Native Floating Point
- **作者**：Ke Yi 等（阿里达摩院 / 通义实验室）
- **发表**：ACL 2025 Findings
- **核心内容**：将 IEEE 754 浮点标准扩展到 ExMy 量化模式，发现 E2M2 在精度和效率间达到最优平衡。ExMy 到 FP16 的转换可通过寄存器级操作实现。
- **与课题的关系**：E2M2 格式（5-bit）可以作为 NVFP4→HiF4 转换的中间表示——先将 NVFP4 反量化到 E2M2 获得更高精度中间值，再量化到 HiF4 的 S1P2。
- **链接**：https://aclanthology.org/2025.findings-acl.943/

---

## 八、训练量化方法（如果转换后需要微调）

### 25. FP4 All the Way: Fully Quantized Training of LLMs
- **作者**：Brian Chmiel 等（Intel / Technion）
- **发表**：NeurIPS 2025 (arXiv:2505.19115)
- **核心内容**：首次在大规模 LLM 上实现全 FP4 量化训练。系统研究了块大小、缩放格式、舍入方法等设计选择。
- **链接**：https://arxiv.org/abs/2505.19115

### 26. Training LLMs with MXFP4
- **作者**：Albert Tseng, Tao Yu, Youngsuk Park
- **发表**：AISTATS 2025 (arXiv:2502.20586)
- **核心内容**：首个近无损 MXFP4 训练方案，使用随机舍入计算无偏梯度估计，结合 Random Hadamard Transform 控制方差。
- **链接**：https://arxiv.org/abs/2502.20586

### 27. TetraJet: Oscillation-Reduced MXFP4 Training for Vision Transformers
- **作者**：Yuxiang Chen 等
- **发表**：ICML 2025 (arXiv:2502.20853)
- **核心内容**：识别了 MXFP4 训练中权重振荡问题，提出 EMA Quantizer (Q-EMA) 和 Adaptive Ramping Optimizer (Q-Ramping)。
- **链接**：https://arxiv.org/abs/2502.20853

### 28. QeRL: Quantization-enhanced Reinforcement Learning for LLMs
- **作者**：Wei Huang 等（NVIDIA / MIT）
- **发表**：ICLR 2026 (arXiv:2510.11696)
- **核心内容**：将 NVFP4 量化与 LoRA 结合加速 RL rollout。发现量化噪声可增加策略熵、增强探索。
- **链接**：https://arxiv.org/abs/2510.11696

---

## 九、误差分析与理论参考

### 29. Why Do Some Inputs Break Low-Bit LLM Quantization?
- **作者**：Ting-Yun Chang 等
- **发表**：EMNLP 2025 (arXiv:2506.12044)
- **核心内容**：分析 3-4 bit 量化误差在不同输入上的分布规律。发现量化误差高度相关（平均 0.82），残差流幅度可预测未来量化误差，MLP 门控输出对精度保持至关重要。
- **与课题的关系**：帮助识别转换后哪些层/通道对精度最敏感，指导混合精度策略（敏感层保留高精度或跳过转换）。
- **链接**：https://arxiv.org/abs/2506.12044

### 30. Improving Quantization with Post-Training Model Expansion
- **作者**：Giuseppe Franco, Pablo Monteagudo-Lago, Ian Colbert, Nicholas Fraser, Michaela Blott
- **发表**：arXiv:2503.17513, 2025-03
- **核心内容**：把在线 `N×N` Hadamard 换成截取自更大矩阵的 `N×M` 行正交投影（`M>N`），同时把激活和权重扩到 M 维。扩维保持原始乘积不变、增加校准激活的零空间，并把理想 GPTQ 重建误差上界缩小 `sqrt(N/M)`；实验收益随扩维率递减。
- **与课题的关系**：这个机制必须同时把 A 的列数和 W 的行数扩为 M，会改变当前六函数接口的张量 shape，因此不能原样提交。可迁移的结论是：校准时应优先把 Q(W) 误差推入 A 的低能量/近零空间，而不是单纯降低权重 MSE。
- **链接**：https://arxiv.org/abs/2503.17513

---

## 十、转换流程建议与论文对应关系

| 阶段 | 技术动作 | 核心参考论文 |
|------|----------|-------------|
| **① 理解源/目标格式** | 精读 NVFP4 和 HiF4 定义 | #1, #2, #3 |
| **② 反量化** | NVFP4 → BF16/FP32 | #3, #12（4/6 缩放还原） |
| **③ 异常值预处理** | 旋转 + 平滑消除异常值 | #5(QuaRot), #8(RRS), #11(SmoothRot) |
| **④ 初步转换** | BF16 → HiF4（Algorithm 1） | #1（HiF4 论文 Algorithm 1） |
| **⑤ 缩放优化** | 优化 HiF4 三级缩放因子 | #9(BASE-Q), #12, #13, #16(OAS/MBS) |
| **⑥ Hessian 优化** | 用 GPTQ 风格方法精调 | #15(MR-GPTQ), #1(HiGPTQ) |
| **⑦ 误差补偿** | 残差通道补偿转换误差 | #18(ARCQuant), #19(QDEC) |
| **⑧ 蒸馏微调** | 用 BF16 教师蒸馏到 HiF4 | #21(QAD) |
| **⑨ 敏感性分析** | 识别关键层/通道 | #29, #20(RILQ) |

---

## 十一、技术路线对比总览

| 技术路线 | 核心思想 | 代表论文 | 与 NVFP4→HiF4 的关系 |
|----------|----------|----------|---------------------|
| **格式设计** | 优化 4-bit 浮点编码与缩放结构 | HiF4, NVFP4, MXFP4, AFPQ | 定义源/目标格式的编码规则 |
| **旋转/平滑** | 量化前消除异常值，使分布均匀 | QuaRot, SpinQuant, DuQuant, RRS, BASE-Q, ButterflyQuant, SmoothRot | 在反量化后、HiF4量化前预处理数据 |
| **GPTQ校准** | 用 Hessian 信息优化量化参数 | MR-GPTQ, HiGPTQ | 优化 HiF4 三级缩放因子的分配 |
| **蒸馏** | 用全精度教师指导量化模型 | QAD | 最终精度恢复手段 |
| **残差补偿** | 用额外通道/LoRA 补偿量化误差 | ARCQuant, QDEC, RILQ | 补偿转换引入的不可逆误差 |

> **关键洞察**：NVFP4 和 HiF4 的 4-bit 元素编码不同（E2M1 vs S1P2），转换不是简单的位操作，而是必须经过**反量化→重新量化**的过程。精度损失主要来自两步：(1) NVFP4 本身的量化误差（已丢失的信息无法恢复）；(2) 重新量化到 HiF4 时引入的新误差。课题核心在于**最小化第 (2) 步的误差**——旋转/平滑路线（#5-#11）用于事前预防，GPTQ 校准（#15, #1）用于事中优化，残差补偿（#18-#20）和蒸馏（#21）用于事后恢复。五条路线层层递进、正交互补，组合使用可达到最优精度。

---

## 十二、2026-08-25 原文补齐与深读审计

本清单 30 个编号来源现已全部解析到一手原文。本地档案共 44 份 PDF（含清单外的
GPTQ、AWQ、QuIP 等对照资料）；本轮新补齐 7 篇，PDF 与抽取文本存放在
`docs/papers/quantization/{pdf,text}`，仅作本地研究档案，不提交远程仓库。

| 本轮补齐原文 | 本地 stem | 深读后的赛题迁移结论 |
|---|---|---|
| DuQuant, 2406.01721 | `DuQuant_2406.01721` | 保留“最坏通道优先旋转→Zigzag 跨块均衡→第二次微旋转”；必须成对保持 `A@W` 等价，并由校准输出门控。 |
| BASE-Q, 2506.15689 | `BASEQ_2506.15689` | 偏置校正和正负非对称尺度需要额外加法补偿，当前接口不能原样表达；迁移为候选的均值误差/能量漂移惩罚。 |
| ButterflyQuant, 2509.09679 | `ButterflyQuant_2509.09679` | 可学习稀疏 Givens 角度是固定符号 Hadamard 之外的新容量；完整在线 butterfly 有超时风险，后续只做少量离散角、校准融合版本。 |
| SmoothRot, 2506.05413 | `SmoothRot_2506.05413` | 核心已被现有平衡+旋转覆盖；与 GPTQ/SpinQuant 的负交互是禁止无条件叠加的新证据。 |
| AFPQ, 2311.01792 | `AFPQ_2311.01792` | 单组正负两套尺度与 HiF4 返回格式不兼容；保留符号非对称度作为尺度候选和均值偏差诊断特征。 |
| TetraJet, 2502.20853 | `TetraJet_2502.20853` | Q-EMA/Q-Ramping 属于训练期；可迁移为跨校准 shape 的候选翻转率/不稳定性惩罚，避免选择只在单一校准布局获胜的尺度。 |
| Post-Training Model Expansion, 2503.17513 | `ModelExpansion_2503.17513` | 扩维改变矩阵 shape，不能提交；其“把误差藏入 A 的零空间”理论直接支持更强的 `A@W` 输出反馈。 |

审计结论不是把七种机制一次堆入下一版本。用户要求等待 v44 线上反馈，因此这里只
更新证据与候选优先级；v44 结果返回前不建立 v45 代码分支。

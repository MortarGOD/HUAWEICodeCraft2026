# attention-qwen-3.8：数学与外部原文核验

核验日期：2026-09-08。范围为原文 §1、§3–5、§6 的数学、§7–8；SageAttention/USTC 具体实现、赛题 HiF4 编码和本仓版本绑定另行核验。本文件只记录分析，没有修改候选、运行大评测或发布算法版本。

本次按问题阅读了下列原文章节与官方实现，**不能据此声称此前或本次把清单里所有论文、仓库都完整学过**。数学推导与文献结论在下文分开标明。

## 1. 结论摘要

输出感知目标、合法网格上的离散搜索、协调量化残差和逐项消融都值得保留，但原文混有可直接导致错误实现的公式、错引论文，以及把有条件效果写成必然收益的问题。尤其不能用其“最大单项增益”“零风险”“15% / 20%”作为本题的已证实预期。

| 原文位置 | 判定 | 更正 |
|---|---|---|
| 第 68 行 HiFloat8 arXiv ID | 确定错误 | `2312.11978` 是数学论文 *The mystery of Carleson frames*；HiF8 正确来源是 *Ascend HiFloat8 Format for Deep Learning*, `2409.16626`。 |
| 第 202 行 MR-GPTQ“必须靠 Hadamard” | 反向概括、过度推广 | 论文给出小组 NVFP4 中 Hadamard 可负收益的分析和实验，同时研究无旋转 GPTQ。不能推出本题 HiF4 Attention 必须旋转。 |
| 第 210 / 385 行 Q/K 旋转等式 | 中间乘法次序错误 | 应为 `(QR)(KR)^T = Q R R^T K^T = QK^T`；`QRK^TR^T` 一般甚至维度不合法。 |
| 第 226 行 Smooth 后旋转必然互补 | 等价性可成立，收益不成立 | 两种顺序均可构造连续等价变换，但量化误差不同，需匹配 Q/K 的实际变换及 GQA 头。 |
| 第 164 行“高 logit 误差远大于低 logit” | 非普遍规律 | 局部影响为 `p_j (V_j−O)`；高概率趋近 1 时反而可饱和，所有 V 相同则 logit 误差不影响 O。 |
| 第 186 行 Q/K/V 最优超参“必然不同” | 推理过强 | 应独立评估，但可以恰好同参最优；双方匹配变换参数更不能各自随意独立选择。 |
| 第 252 / 389 行历史缩放优于即时缩放 | 概念迁移不成立 | TE delayed scaling 为迭代 FP8 性能而保存并更新历史；固定校准先验不是同一机制，也不保证更准。 |
| 第 286 行 RNE“系统性对称偏置” | 无普遍依据 | ties-to-even 只决定精确中点，不能把“中点附近”统称偏置；任务损失有耦合才可能偏离最近舍入更优。 |
| 第 293 行 AdaRound 阈值 | 上下写反 | 令 `t=(v−low)/(high−low)`，`t<0.5+δ` 应取 low，否则取 high（等号另定）；原式在 `v=low` 也会取 high。 |
| 第 309–311 行 RaBitQ/TurboQuant 推出低秩 PCA `r≤4` | 不是被引算法或定理 | 它们使用随机旋转/投影及量化编码或内积估计；TurboQuant 残差 QJL 是每坐标一 bit 的附加表示，不是 rank-4 PCA。 |

## 2. 论文核验

### MR-GPTQ

正确标题是 [*Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*, arXiv:2509.23202v3](https://arxiv.org/html/2509.23202v3)。本次读摘要、§3.1–3.2 和 §4.1 的相关分析/结论。分析在独立分组、Laplace/Normal 分布及 RTN 等条件下展开；§3.1–3.2 指出小组 NVFP4 的旋转可能增加误差，§4.1 同时提出无旋转 NVFP4 GPTQ 和两种 MR-GPTQ 路线。研究对象主要是 LLM 层的权重/激活量化，并非本题角色分开校准的 Q/K/V 转码。可借鉴按格式分析 scale 与舍入的关系，不能从论文推出旋转的必然收益。

### HiFloat8 与 NVFP4 引文

已核对 [2312.11978](https://arxiv.org/abs/2312.11978) 标题页，其题目是 *The mystery of Carleson frames*。正确 HiF8 原文为 [*Ascend HiFloat8 Format for Deep Learning*, 2409.16626v2](https://arxiv.org/abs/2409.16626v2)，本次仅核对元数据/摘要的 tapered precision 说明，没有把它算作全文精读。NVFP4 `2509.25149` 的准确题目为 [*Pretraining Large Language Models with NVFP4*](https://arxiv.org/abs/2509.25149)；原文简写标题可辨认，但不能混淆训练配方与本题静态推理转码。

### AdaRound 与随机舍入

本次读 [*Up or Down? Adaptive Rounding for Post-Training Quantization*, 2004.10568v2](https://arxiv.org/html/2004.10568v2) §2、§3.1–3.3，以及 §5 的相关 grid/bias 消融。该方法给固定权重的每个元素学习连续舍入变量，再正则化至 floor/ceil；原因是输出损失中的误差交叉项，不是 E2M1 恰好有一位尾数，也不是只学习三个张量级阈值。其损失设定不能直接提供动态 Attention 的真实 Hessian。

RNE 规定取最近值，只在等距时选偶数，参见 [NVIDIA CUDA 浮点文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)。固定码本网格上，最近舍入逐点最小化标量平方误差。为了输出误差改用其他舍入可有价值，但需要传播权重或残差相关性的证据。固定种子只保证扰动可重现，不能证明单次残差是零均值白噪声，更不能保证 MSE 稳定下降。

### RaBitQ 与 TurboQuant

本次读 [*RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search*, 2405.12497](https://arxiv.org/html/2405.12497) §3.1–3.2 的码本/估计器段落：随机旋转二值码本与归一化因子的内积估计用于 ANN，并非 PCA 低秩残差校正。

本次读 [*TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*, 2504.19874v1](https://arxiv.org/html/2504.19874) §1.3、§2.2、§3.1–3.2 和 Algorithm 1/2。内积版本以 `b−1` bit MSE 量化后，对残差再存 `d` 个符号 bit 与范数，解码时把 QJL 校正项加回。不能把该附加表示直接交给固定 HiF4 解码器。内部借鉴残差目标后重新投影到合法 HiF4 参数可以研究，但它是另一个算法，也不继承原论文保证。文中不存在由 4096 state 节点推出 `r≤4` 的依据；确切 state 容量还应按本题验证器定义计算。本审计不判断其学术争议，争议标签也不能替代技术核验。

## 3. 需要修正的数学表达

以下是本次直接推导，不作为任何论文的实证结果。

设单头 `Q∈R^(Tq×d)`、`K∈R^(Tk×d)`。同一归一化正交矩阵 R 满足 `RR^T=I`，则 `(QR)(KR)^T=QRR^TK^T=QK^T`。未归一化 Hadamard 满足 `HH^T=dI`，必须处理归一化。连续等价不等于量化后无误差。

设 D 为可逆对角矩阵。先 Smooth 后旋转可取 `Q'=QDR, K'=KD^(-1)R`；先旋转后 Smooth 可取 `Q'=QRD, K'=KRD^(-1)`。两者的 QK 连续乘积都不变，但不同顺序生成不同分布；“互补不冲突”没有量化收益保证。更一般的必要匹配条件是 Q 侧矩阵 A 与 K 侧矩阵 B 满足 `AB^T=I`。独立校准、分别 gate、GQA 一对多对应都可能破坏该条件。任意把 V 的输出特征基旋转后直接交给评测器，会产生 `OR` 而非 O，确实需要逆变换；这不等于禁止量化器内部用变换做计算后再回到合法输出坐标。

令 `p=softmax(z), O=Σ p_j V_j`，一阶扰动为：

`δO = Σ_j p_j (V_j−O) δz_j + Σ_j p_j δV_j`。

由此可见：共同 logit 平移完全无效；所有 V 相同时 logit 的影响为零；概率趋于单点时局部 Jacobian 会饱和。Q/K 量化的 logit 扰动还包含 `ΔQ K^T + Q ΔK^T + ΔQ ΔK^T`，不只是“乘性叠加”。正确方向是优化最终输出，但真实配对信息不足时，应明确所用近似目标和失效条件。

固定 p 下 V 的输出误差为 `pE_V`。跨 token 残差相关项决定它会相干累加还是抵消；即使残差均值为零，方差和相关性仍会影响最终平方误差。固定种子抖动不是免费消偏操作。

## 4. 官方工具链的适用边界

[Transformer Engine FP8/FP4 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html) 的 FP8 delayed scaling 段落明确以历史 amax 生成下一迭代 scale，并更新历史；动机之一是避免当次输出多次访存。本题动态接口已收到当前自身输入，且 state 需保持纯数据/不可变使用。冻结校准统计可作为先验候选，但不能引用该机制证明它优于当次合法 scale 搜索。

[TE NVFP4 官方文档](https://nvidia.github.io/TransformerEngine/features/low_precision_training/nvfp4/nvfp4.html) 的 Data Format、Stochastic Rounding、RHT 段落本次已核对：NVFP4 使用 block amax 与 global amax 的分级缩放；随机舍入启用于梯度，RHT 用于 wgrad 的输入/梯度。它们的训练收益不是动态 Attention V 抖动的验证。

[Model Optimizer 官方 NVFP4 文档](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.kernels.quantization.common.nvfp4_quant.html) 可用于 E2M1 网格与假量化参考，但本次只读 API 模块说明，没有审计其全部量化/解码源码。必须与赛题的 scale 含义、数据布局、BF16 舍入顺序逐项对齐后，才能充当数值 oracle。

[microsoft/microxcaling README](https://github.com/microsoft/microxcaling) 与 [`mx/mx_ops.py`](https://raw.githubusercontent.com/microsoft/microxcaling/main/mx/mx_ops.py) 的 `_shared_exponents` / `_quantize_mx` 本次已读：主路径用共享整数指数及 `2**shared_exp` 缩放。它是 OCP MX 仿真参考，不能直接等同于本题的 NVFP4 E4M3 block scale × FP32 global scale 解码，更不能替代 HiF4 官方基线。

## 5. 实施含义

值得保留的是：以合法反量化结果衡量候选、scale 与码值联合离散优化、关注误差相关性、把连续等价性与量化后最终输出分开验证。需要撤回的是：保证无负例、保证 15%/20% 改善、旋转/percentile/历史缩放总优于其他方法、少量校准即可准确代表未知对端。

本文件没有穷举所有可能优化；也没有用错误引文反推其他正确思路无效。是否已覆盖某个方法，仍须检查本仓实际运行绑定、触发统计、线上角色协议与已有消融。

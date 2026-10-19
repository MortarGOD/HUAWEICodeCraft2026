# attention-qwen-3.8 引用来源审计

核对日期：2026-09-08。仅审计外部来源与迁移论证；没有修改待审原文或算法。本文的“本次阅读”不表示之前已经完整学习过这些资料。来源存在、曾经下载、读过摘要、精读方法、实现验证，应分别记录。

## 1. 格式论文与参考编号

- **HiFloat4 原始依据与文档的核心前提直接冲突。** [原论文](https://arxiv.org/abs/2602.11287)及[社区原文的 Format Definition / Table 1–2](https://hifloat.gccorg.com/docs/en/hifloat4/white_paper/hifloat4_format_for_language_model_inference.html)明确：HiF4 元素为 S1P2（概念上 E1M2），不是 NVFP4 的 E2M1。未经三级缩放的正值为 0、0.25、0.5、0.75、1、1.25、1.5、1.75。其 64→8→4 的三级树也不能化约为整个张量的几个档位。由“值集完全相同”推出“误差几乎全部来自 scale 错配”没有成立的前提。
- **SageAttention2 的编号写错。** [2501.09747](https://arxiv.org/abs/2501.09747)是机器人动作分词论文 FAST；正确是 [2411.10958](https://arxiv.org/abs/2411.10958)，标题为 *SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization*。
- **HiFloat8 的编号和标题写错。** [2312.11978](https://arxiv.org/abs/2312.11978)是 *The mystery of Carleson frames*；HiFloat 社区引用的对应论文是 [2409.16626，Ascend HiFloat8 Format for Deep Learning](https://arxiv.org/abs/2409.16626)。本次只核对此论文元数据和摘要，没有宣称精读全部 HiF8 方法。
- NVFP4 的 [2509.25149](https://arxiv.org/abs/2509.25149)编号正确，实际题名为 *Pretraining Large Language Models with NVFP4*；本次核对元数据，并读了 NVIDIA 官方 Transformer Engine 的对应配方说明。
- MR-GPTQ 的 [2509.23202](https://arxiv.org/html/2509.23202v1)编号正确，实际题名为 *Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization*。

## 2. SageAttention 的引用哪里准确、哪里混淆

| 资料 | 原文内容 | 对待审文档的判断 |
|---|---|---|
| [SageAttention，§4.4、算法1、表6](https://arxiv.org/html/2410.02367v9) | 主要实现用 INT8 Q/K，P/V 保持 FP16；另列全 INT8 变体。K smoothing 是沿 token 维减去共同均值。 | 不能把 1/2/3 统称为同一种“FP4 全量化”。 |
| [SageAttention2，§2.3、§3.1–3.2](https://arxiv.org/html/2411.10958v7) | Q/K 为 INT4，P/V 为 FP8；Smooth Q/K 是均值平移。Q 平移需要额外 GEMV 修正，K 平移利用 softmax 行平移不变性。 | 文档把 Smooth K 写成 Q/K 互逆通道缩放，**归因错误**。原文反而指出 Q/K 都有 outlier，直接套 SmoothQuant 在其实验中不有效；不是“K outlier 一定可迁给 Q”的依据。 |
| [SageAttention3，§3.1–3.2、算法1](https://arxiv.org/html/2505.11594v3) | Q/K/P/V 的两个矩阵乘都使用 NVFP4。P 的在线 softmax 分块值落在 [0,1]，E4M3 block scale 的利用率不足，因此引入额外行 scale。 | 描述 P 的数值问题基本有据；但本题固定评测端计算 P，选手仅返回 Q/K/V HiF4 参数，不能直接加 P 量化层或替换 softmax。只能迁移“按最终误差找瓶颈”的思想。 |

本次没有逐行审完 SageAttention CUDA 全仓，也没有验证其 kernel 性能；上述判定基于作者论文指定章节。Q/K 对角互逆缩放本身具有连续数学等价性，但其收益、校准时双方能否得到相容参数，属于另一组问题。

## 3. USTC 百分位方案：真实，但结论被放大

[HiFloat 社区](https://hifloat.gccorg.com/en/)明确列出中科大贡献的获奖方案；[官方发布仓](https://github.com/GCC-HiFloat/Percentile_Wan2.2)同样标注获奖，并链接作者仓与模型。[比赛官网](https://challenge.gccorg.com/)列出 USTC-zhaslab。不能把该资源判为虚构。

本次阅读了作者报告 [Tail-Aware HiFloat4: W4A4 Post-Training Quantization for Wan2.2，2605.26628v1](https://arxiv.org/html/2605.26628v1) 的摘要、III–VI 节及全部三张表：百分位统计沿**校准 token/sample 维逐通道**汇总，用于 Linear 激活/固定权重的互逆通道 mask；它没有改写运行时 HiF4 算术。表 III 比较 BF16 与整套 W4A4，未给 percentile 对 amax 的受控消融，也未给 Attention Q/K/V 输出 MSE 的实验。因此“block-64 百分位显著优于 amax”“可直接迁移”未获这份原文支持。

“唯一在 HiF4 上实证过的校准策略”也不成立：HiFloat4 原论文已有 HiGPTQ，社区另列 LLMC、HiSQRot4 等实践。官方仓 README 还说明发布范围以最终推理复现为主，公开配置 `viditq.enabled=false`，并不保证完整私有校准流程可直接复现。应继续核对训练校准和恢复后的 mask，不能仅凭获奖标签断言机制和收益。

## 4. LLMC 与 CANN：需要区分具体分支、对象和接口

- **LLMC 实践确实存在。** [GCC-HiFloat/LLMC_Wan2.2 README](https://github.com/GCC-HiFloat/LLMC_Wan2.2)给出 RTN + hif4、Wan2.2、OpenS2V、VBench 的流程；[社区说明](https://hifloat.gccorg.com/zh/news/20260605)明确列为独立实践。本次没有验证 OpenGVLab/LLMC 上游当前每一分支均原生支持这套 HiF4；引用应落到实际发布仓与配置，不能用项目总名代替具体实现。
- **HiGPTQ、HiAutoRound、HiFlatQuant 也有真实原始说明。** [昇腾 CANN 2026-09-04 文章，§3–4](https://cann.csdn.net/6a9a3e0548977663a5dcfc9b.html)分别描述 HiF4 的 64/8/4 组阶码固定策略、三级阶码与逐元素尾数的可学习补偿、可学习等价矩阵和通道 scale。HiGPTQ/HiAutoRound 的对象主要是固定 Linear 权重；HiFlatQuant 依赖激活与权重配对变换。这不等于动态 Attention 已具备可靠 Hessian 或共享变换。
- [CANN/amct 的目录与合并记录](https://gitcode.com/cann/amct/tree/master/amct_pytorch)搜索索引显示新增 HiAutoRound/HiFlatQuant 支持及 `auto_round.py` / `hifp.py` 适配；直接网页打开发生 Internal Error。本次**没有完成该源码的逐行审计**，不能把索引里的通过测试记录当成本地复验。

## 5. 其他迁移过度

- [MR-GPTQ §1、§4.1](https://arxiv.org/html/2509.23202v1)并非证明“必须旋转且必然最大增益”。原文指出标准 RTN 下旋转可损害 NVFP4，随后将分组旋转、MSE scale 优化和 GPTQ 补偿组合；研究对象是可控制权重与激活的 Linear PTQ。不能据此承诺 HiF4 Attention 单独旋转后输出误差再降 20%。
- [Transformer Engine 官方文档，Mixed precision training with FP8 / NVFP4 training recipe](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)把 delayed scaling 描述为利用前几次训练迭代 amax 历史减少开销；并没有给出“HiF4 Attention 历史 scale 必然优于即时 scale”的结论。NVFP4 配方的随机舍入主要针对梯度，不能直接转成对固定输入 V 保证更优的策略。
- [TurboQuant 作者摘要](https://arxiv.org/abs/2504.19874)和[Google Research 原始说明](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)描述的是旋转量化后对残差做 1-bit QJL 校正；这不是文档所述 r≤4 PCA 低秩校正器的实证。额外残差信息的存储和评分端解释能力必须另行审查。本次没有裁定相关学术优先权争议。

## 阅读边界

本次完整阅读待审文档；精读上面明确列出的原文章节与官方 README/文章。SageAttention 1/2/3 未全仓执行，CANN 仓直接读取受限，HiF8/NVFP4 预训练/TurboQuant 论文只核对指定范围。上述记录不覆盖过去各轮的实际阅读历史，也不声称文档列举的所有基础论文和仓库已在本次精读。

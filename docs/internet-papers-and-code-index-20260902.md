# NVFP4 → HiF4 赛题：互联网论文与代码索引（2026-09-02）

> 2026-09-07 更新：镜像现位于本项目 `docs/papers/hif4-public-references/`，已扩充到 24 个仓库、75 条远端分支。最新来源、固定提交与抓取范围见[搜索与全分支更新报告](赛题公开代码仓搜索与全分支更新-20260907.md)。以下 2026-09-02 的路径、HEAD 和数量保留为历史快照。

## 1. 调研范围与结论

本索引针对《2026 年华为算法大赛——高精度 4Bit 数值转换算法》初赛任务书，检索目标为：

- NVFP4（E2M1、每 16 值一个 E4M3 scale）到 HiF4（S1P2、64 值、E6M2 + E1_8 + E1_16）转换；
- 以最终 `Linear` 输出 MSE 为目标的 Weight/Activation 联合量化；
- 以 `softmax(QK^T)V` 输出 MSE 为目标的 Q/K/V 动态量化；
- calibration、scale search、二阶/输出感知误差补偿、异常值平滑/旋转，以及 300 秒级运行约束；
- 赛题六个 API 的公开实现与可迁移开源代码。

检索渠道包括 arXiv、论文引用链、GitHub 仓库/代码搜索、官方 HiFloat/NVIDIA 文档与公开竞赛仓库。所谓“所有”是截至 2026-09-02 的公开互联网快照，不包括未公开仓库、私有提交、尚未索引的网页或之后新增材料。

当前本地已经形成：

- 66 份论文/技术资料 PDF：`docs/papers/quantization/pdf/`；
- 既有论文纯文本：`docs/papers/quantization/text/`；
- 10 个高相关公开仓库的 Git 镜像：`/Users/ee/Code/hif4-public-references/`；
- 公开赛题方案的详细审计：`docs/github-public-solution-audit-20260902.md`。

最重要的新增发现是：

1. [HiFA4](https://arxiv.org/abs/2607.04302) 几乎直接命中本题 Attention：用 calibration-static Smooth-QK 迁移 K outlier，并用最终 Attention 一致性修正误差。
2. [zhaow22/NVFP4-HiFloat4](https://github.com/zhaow22/NVFP4-HiFloat4) 直接研究 NVFP4 → HiF4，并对 E6M2 邻域与两级 E1 做精确离散优化；但它是 CUDA/packed-format 实验，不能直接满足赛题 Python API/鲲鹏环境。
3. [Four Over Six](https://arxiv.org/abs/2512.02010)、[SOAR](https://arxiv.org/abs/2605.12245) 与 [RaZeR](https://arxiv.org/abs/2501.04052) 共同说明：max-based scale 不是误差最优，离散 scale 候选搜索、量化/反量化 scale 解耦和冗余码值利用是高收益方向。
4. GitHub 精确搜索六个函数名后，公开可直接对照的赛题仓库集中在 `openaddr`、`suibianll`、`youxilee`、`analytistic`、`jansenin`；它们均已完整镜像。多数没有许可证，只能阅读和独立重实现，不能直接复制提交。

## 2. 最高优先级论文

| 优先级 | 论文 | 与本题的直接关系 | 本地 PDF |
|---|---|---|---|
| S | [HiFloat4 Format for Language Model Inference](https://arxiv.org/abs/2602.11287) | HiF4 格式、合法值、基准转换算法、HiGPTQ | `HiFloat4_2602.11287.pdf` |
| S | [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149) | NVFP4 定义、RHT、二维缩放、stochastic rounding | `NVFP4_Pretraining_2509.25149.pdf` |
| S | [HiFA4: Training-Free 4-bit FlashAttention on Ascend HIF4 NPUs](https://arxiv.org/abs/2607.04302) | Q/K 等价平滑、Attention 最终输出一致性、Ascend HIF4 | `HiFA4_2607.04302.pdf` |
| S | [Unleashing Low-Bit Inference on Ascend NPUs](https://arxiv.org/abs/2602.12635) | HiF4 在 W/A、KV-cache 与 PTQ 中的系统评估 | `HiFloat4_Evaluation_2602.12635.pdf` |
| S | [Four Over Six](https://arxiv.org/abs/2512.02010) | 每 block 比较不同有效动态范围，避免近最大值误差 | `FourOverSix_2512.02010.pdf` |
| S | [SOAR](https://arxiv.org/abs/2605.12245) | scale 联合闭式优化、离散 scale 搜索、量化/反量化解耦 | `SOAR_2605.12245.pdf` |
| S | [ARCQuant](https://arxiv.org/abs/2601.07475) | 用 residual channels 补偿细粒度 FP4 误差；仓库支持 HiF4 模拟 | `ARCQuant_2601.07475.pdf` |
| S | [RaZeR](https://arxiv.org/abs/2501.04052) | 利用冗余零码与 scale 元数据；代码含 HiF4 fake quantizer | `RaZeR_2501.04052.pdf` |
| A | [Adaptive Block-Scaled Data Types](https://arxiv.org/abs/2603.28765) | block 内自适应值集/格式选择，启发 HiF4 候选族设计 | `AdaptiveBlockScaled_2603.28765.pdf` |
| A | [Quartet II](https://arxiv.org/abs/2601.22813) | microscaling 下的低误差估计与 RHT，适合异常值/旋转消融 | `QuartetII_2601.22813.pdf` |
| A | [HiFloat4 Format for LLM Pre-training on Ascend NPUs](https://arxiv.org/abs/2604.08826) | HiF4 与 MXFP4、RHT、训练稳定性；数值格式侧证据 | `HiFloat4_Pretraining_2604.08826.pdf` |
| A | [Full-Stack FP4](https://arxiv.org/abs/2607.04422) | 分析 Linear 误差累积和 Attention 的 PV 敏感路径 | `FullStackFP4_2607.04422.pdf` |
| A | [HiFloat4 for End-to-End RL Post-Training](https://arxiv.org/abs/2607.26515) | 激活 outlier、underflow 与 residual 补偿 | `HiFloat4_RL_2607.26515.pdf` |
| A | [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) | MX 格式与 block scaling 理论/实验基线 | `MXFormats_2310.10537.pdf` |

## 3. Linear：输出感知量化论文

这些论文不一定使用 HiF4，但优化目标和赛题 Linear MSE 高度相似。

| 方向 | 论文/代码 | 可迁移点 |
|---|---|---|
| 二阶误差补偿 | [GPTQ](https://arxiv.org/abs/2210.17323), [OBC](https://arxiv.org/abs/2208.11580), [MR-GPTQ](https://arxiv.org/abs/2509.23202) | 用 calibration Hessian/Gram 选择权重量化误差与残差更新；HiF4 论文明确给出 HiGPTQ |
| W/A 等价平滑 | [SmoothQuant](https://arxiv.org/abs/2211.10438), [AWQ](https://arxiv.org/abs/2306.00978), [OmniQuant](https://arxiv.org/abs/2308.13137) | 通道尺度迁移、activation-aware 权重敏感度、校准参数搜索 |
| 旋转去异常值 | [QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406), `SmoothRot`, `DuQuant`, `ButterflyQuant` | Hadamard/学习旋转、异常值均化；必须遵守本题状态大小与在线耗时 |
| rounding/clipping | [AdaRound](https://arxiv.org/abs/2004.10568), [BRECQ](https://arxiv.org/abs/2102.05426), `RPTQ`, `AffineQuant`, `AFPQ` | 从 max/RTN 转向输出误差驱动的 clipping、码值和舍入选择 |
| 极低比特重建 | [AQLM](https://arxiv.org/abs/2401.06118), [QuIP](https://arxiv.org/abs/2307.13304), [QuIP#](https://arxiv.org/abs/2402.04396), [HQQ](https://arxiv.org/abs/2309.15531) | block/codebook/坐标下降、最坏情况误差与可计算近似 |
| outlier/稀疏残差 | [SpQR](https://arxiv.org/abs/2306.03078), [SqueezeLLM](https://arxiv.org/abs/2306.07629), `OutlierSuppression+`, `InputBreakQuant` | outlier 定位与补偿；赛题输出必须仍是统一合法 HiF4，不能直接保留高精度旁路 |
| W/A 系统协同 | [Atom](https://arxiv.org/abs/2310.19102), [ZeroQuant-V2](https://arxiv.org/abs/2303.08302), `SliM-LLM`, `TetraJet`, `TetraJetV2` | group 布局、动态量化、误差传播与高吞吐实现 |

## 4. Attention / QKV 论文

| 论文 | 对本题的价值 | 本地 PDF |
|---|---|---|
| [HiFA4](https://arxiv.org/abs/2607.04302) | 最直接；Smooth-QK 使用校准统计构造 Q/K 可逆通道缩放，P-Reordering揭示 softmax/PV 的一致性误差 | `HiFA4_2607.04302.pdf` |
| [QServe](https://arxiv.org/abs/2405.04532) | SmoothAttention、W4A8KV4、Attention 精度/系统协同 | `QServe_2405.04532.pdf` |
| [KIVI](https://arxiv.org/abs/2402.02750) | K 适合 per-channel、V 适合 per-token 的分布结论 | `KIVI_2402.02750.pdf` |
| [KVQuant](https://arxiv.org/abs/2401.18079) | pre-RoPE K、非均匀值集、逐向量 dense+sparse outlier | `KVQuant_2401.18079.pdf` |
| [ZipCache](https://arxiv.org/abs/2405.14256) | 用 Attention 显著度加权 token；可迁移为 V 候选评分权重 | `ZipCache_2405.14256.pdf` |
| [SKVQ](https://arxiv.org/abs/2405.06219) | channel reorder、clipped dynamic quantization、滑窗高精度思想 | `SKVQ_2405.06219.pdf` |
| [APTQ](https://arxiv.org/abs/2402.14866) | Attention 非线性输出敏感度/Hessian；虽为 mixed precision，目标函数可借鉴 | `APTQ_2402.14866.pdf` |
| [Attention-aware PTQ without Backpropagation](https://arxiv.org/abs/2406.13474) | attention-aware Hessian，适合校准阶段构造 Q/K 误差度量 | `AttentionAwarePTQ_2406.13474.pdf` |

注意：本题只允许分别返回 Q/K/V 的 HiF4 参数，不能修改 softmax、量化 P、保留高精度窗口或增加运行时 Q-K 交互。因此这些论文中的系统机制只能转化成 calibration state、合法的等价变换、候选评分或 role-specific scale search。

## 5. 赛题直接相关代码镜像

本地根目录：`/Users/ee/Code/hif4-public-references/`

| 本地目录 | GitHub | 调研时 HEAD | 作用/风险 |
|---|---|---:|---|
| `openaddr-hif4-contest` | [openaddr/hif4-contest](https://github.com/openaddr/hif4-contest) | `f40932101ac2` | 六 API、多版本、输出监督/Hessian/Hadamard；无明确许可证 |
| `suibianll-AI-Contest` | [suibianll/AI-Contest](https://github.com/suibianll/AI-Contest) | `f7af6d1...` | 六 API、评测器、官方回传记录和大规模消融；无明确许可证 |
| `youxilee-hif4` | [youxilee/hif4](https://github.com/youxilee/hif4) | `dd5ee6515323` | 联合 W/A 残差、Attention 端到端选择；无明确许可证 |
| `analytistic-T4bit` | [analytistic/T4bit](https://github.com/analytistic/T4bit) | `4a2c4c744f38` | 完整 `A@W` 目标、back-projector 与离散坐标下降；无明确许可证 |
| `jansenin-quantization-contest-work` | [jansenin/quantization-contest-work](https://github.com/jansenin/quantization-contest-work) | `a81b34c2f0b6` | 六 API 基线、真实模型 capture/evaluator；无明确许可证 |
| `zhaow22-NVFP4-HiFloat4` | [zhaow22/NVFP4-HiFloat4](https://github.com/zhaow22/NVFP4-HiFloat4) | `582309ea189f` | packed NVFP4→HiF4；E6M2 邻域 + E1 精确树优化；CUDA 专用、无许可证 |
| `global-computing-consortium-HiFloat4` | [官方 HiFloat4](https://github.com/global-computing-consortium/HiFloat4) | `6d937b6fcf34` | CUDA/NPU HiF4、NVFP4 模拟与官方转换参考；Apache-2.0；采用 sparse checkout 排除巨型数据集 |
| `mit-han-lab-fouroversix` | [mit-han-lab/fouroversix](https://github.com/mit-han-lab/fouroversix) | `dadfad6901d4` | NVFP4 4/6、IF4、RHT、GPTQ/AWQ/SmoothQuant/SpinQuant evaluator；MIT |
| `actypedef-ARCQuant` | [actypedef/ARCQuant](https://github.com/actypedef/ARCQuant) | `dbf841e12320` | NVFP4 residual channels、HiF4 模拟、CUDA kernels；未发现许可证 |
| `abdelfattah-NVFP4-RaZeR` | [abdelfattah-lab/NVFP4-RaZeR](https://github.com/abdelfattah-lab/NVFP4-RaZeR) | `78be7fc78857` | NVFP4/RaZeR/4over6/HiF4 quantizer；论文声明专利申请，使用前审查许可 |

新镜像均使用 sparse checkout，仅获取源码/文档，避免官方 HiF4 仓库中无关的大型数据文件。

## 6. 重要但未本地完整镜像的代码

大型通用框架只保留链接，避免重复占用磁盘；需要实现时再按具体文件 sparse checkout。

| 仓库 | 相关内容 |
|---|---|
| [NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | NVFP4 PTQ、MSE/max/GPTQ calibration、Four-over-Six、local-Hessian scale sweep、MXFP4→NVFP4 cast |
| [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | NVFP4 encode/decode、Blackwell FP4 kernels、RHT/quantization primitives |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | ModelOpt NVFP4 pre/post-training 集成 |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | NVFP4 模型加载、量化与推理路径 |
| [IST-DASLab/gptq](https://github.com/IST-DASLab/gptq) | GPTQ 原始二阶权重补偿 |
| [mit-han-lab/smoothquant](https://github.com/mit-han-lab/smoothquant) | W/A 等价通道平滑 |
| [mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq) | activation-aware channel importance/scale search |
| [spcl/QuaRot](https://github.com/spcl/QuaRot) | Hadamard/rotation 端到端 W/A/KV 4-bit |
| [facebookresearch/SpinQuant](https://github.com/facebookresearch/SpinQuant) | 学习旋转与 PTQ |
| [jy-yuan/KIVI](https://github.com/jy-yuan/KIVI) | K/V 不同量化轴和 kernels |
| [SqueezeAILab/KVQuant](https://github.com/SqueezeAILab/KVQuant) | KV 非均匀/离群值量化 |
| [mit-han-lab/omniserve](https://github.com/mit-han-lab/omniserve) | QServe/QoQ/SmoothAttention |
| [IST-DASLab/Quartet-II](https://github.com/IST-DASLab/Quartet-II) | NVFP4 microscaling、MS-EDEN、Blackwell kernels |
| [steven-bao1/SOAR](https://github.com/steven-bao1/SOAR) | 当前仓库仍标记 TODO，尚无完整 PTQ 源码；只能依据论文独立实现 |

## 7. 完整本地 PDF 清单（66）

```text
AFPQ_2311.01792.pdf
APTQ_2402.14866.pdf
AQLM_2401.06118.pdf
ARCQuant_2601.07475.pdf
AWQ_2306.00978.pdf
AdaRound_2004.10568.pdf
AdaptiveBlockScaled_2603.28765.pdf
AffineQuant_2403.12544.pdf
Atom_2310.19102.pdf
AttentionAwarePTQ_2406.13474.pdf
BASEQ_2506.15689.pdf
BRECQ_2102.05426.pdf
BitDistiller_2402.10631.pdf
ButterflyQuant_2509.09679.pdf
DecDEC_2412.20185.pdf
DuQuant_2406.01721.pdf
FP4AllTheWay_2505.19115.pdf
FPE2M2_ACL2025.pdf
FourOverSix_2512.02010.pdf
FullStackFP4_2607.04422.pdf
GPTQ_2210.17323.pdf
HQQ_2309.15531.pdf
HiFA4_2607.04302.pdf
HiFloat4_2602.11287.pdf
HiFloat4_Evaluation_2602.12635.pdf
HiFloat4_Pretraining_2604.08826.pdf
HiFloat4_RL_2607.26515.pdf
InputBreakQuant_2506.12044.pdf
KIVI_2402.02750.pdf
KVQuant_2401.18079.pdf
LLM_int8_2208.07339.pdf
M2XFP_2601.19213.pdf
MR_GPTQ_2509.23202.pdf
MXFP4_OAS_MBS_2603.08713.pdf
MXFormats_2310.10537.pdf
MXPlus_2510.14557.pdf
ModelExpansion_2503.17513.pdf
NVFP4_Pretraining_2509.25149.pdf
NVFP4_QAD_2601.20088.pdf
NVIDIA_NVFP4_technical_blog.pdf
OBC_2208.11580.pdf
OmniQuant_2308.13137.pdf
OutlierSuppressionPlus_2304.09145.pdf
QServe_2405.04532.pdf
QeRL_2510.11696.pdf
QuIPSharp_2402.04396.pdf
QuIP_2307.13304.pdf
QuaRot_2404.00456.pdf
QuartetII_2601.22813.pdf
RILQ_2412.01129.pdf
RPTQ_2304.01089.pdf
RRS_2409.20361.pdf
RaZeR_2501.04052.pdf
SKVQ_2405.06219.pdf
SOAR_2605.12245.pdf
SliM_LLM_2405.14917.pdf
SmoothQuant_2211.10438.pdf
SmoothRot_2506.05413.pdf
SpQR_2306.03078.pdf
SpinQuant_2405.16406.pdf
SqueezeLLM_2306.07629.pdf
TetraJetV2_2510.27527.pdf
TetraJet_2502.20853.pdf
TrainingMXFP4_2502.20586.pdf
ZeroQuantV2_2303.08302.pdf
ZipCache_2405.14256.pdf
```

## 8. 对当前算法最值得落地的路线

1. **HiF4 层级精确求解作为内核**：固定 E6M2 时，16 个四值子组与 8 个父节点可做小型动态规划/枚举；E6M2 只搜相邻合法码或由二次目标导出的候选。
2. **Linear 直接优化最终输出**：用 calibration 构造 `X^T X` 或 block Gram，使目标从元素 MSE 变成 `||(W-QW)X||²`；权重与激活误差必须联合评估，避免单边改善、交互项恶化。
3. **Q/K 用等价变换 + holdout 准入**：优先验证 HiFA4 Smooth-QK、per-head/per-channel reciprocal scale；候选在不同 calibration fold 上生成和验收，最终用 causal-softmax 输出 MSE 选。
4. **V 用 token/概率加权而不是纯通道 MSE**：借鉴 ZipCache/QServe，以校准 Attention 概率或 `P^T P` 近似 V 误差敏感度；在线函数仍只返回合法 HiF4。
5. **严格按耗时定预算**：Weight calibration 可使用较深的二阶/坐标下降；动态 A/Q/K/V 只允许矢量化的少量候选。短序列和高敏感 role 分配更多 sweep，长序列减少候选。
6. **许可证隔离**：公开赛题仓库多数无许可证。正式 `solution.py` 应保留算法来源记录并独立实现；Apache-2.0/MIT 仓库也要保留许可和归属，不复制带专利风险的 RaZeR 实现。

## 9. 更新方法

复查新增公开赛题实现：

```bash
gh search code 'hif4_calibration_and_quantize_weight' --limit 100
gh search code 'hif4_calibration_attention' --limit 100
gh search repos 'NVFP4 HiF4' --sort updated --limit 50
```

更新已镜像仓库时，仅在工作树干净的前提下运行 `git pull --ff-only`。论文建议按 arXiv ID 命名，放入 `docs/papers/quantization/pdf/`；索引日期和仓库 HEAD 同步更新。

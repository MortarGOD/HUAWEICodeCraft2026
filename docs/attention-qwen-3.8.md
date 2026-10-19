# NVFP4 → HiF4 高精度转换算法 · 优化实施方案

> 赛题：2026 华为算法大赛初赛（NVFP4 → HiF4 转换）
> 聚焦：**Attention 场景**（Q/K/V 校准 + 在线动态量化）
> 目标：最大化 `(MSE_std − MSE_you) / MSE_std`，规避劣化负分与合法性无效

---

## 0. 赛题本质分析

### 0.1 评分链路

```text
基线： O_ref = Attention(dequant_NVFP4(Q), dequant_NVFP4(K), dequant_NVFP4(V))   # FP32 基准
选手： O_sub = Attention(dequant_HiF4(Q̂), dequant_HiF4(K̂), dequant_HiF4(V̂))
得分： score = Σ (MSE_std − MSE_you) / MSE_std      # MSE_std = 标准 HiF4 基线的 MSE
约束： MSE_you > MSE_std 时产生负分；单次提交总耗时 < 5 min

```

### 0.2 格式对比与核心洞察

| 维度 | NVFP4 | HiFloat4 (HiF4) | 差异性质 |
|---|---|---|---|
| 尾数 | E2M1 | E2M1 | **完全相同** |
| 离散值集 | {0,±0.5,±1,±1.5,±2,±3,±4,±6} | 同左 | **完全相同** |
| 块大小 | 16 | 64 | **4 倍膨胀** |
| Scale 层级 | FP32 global + E4M3 block（2 级） | E6M2(lv1) + E1_8(lv2) + E1_16(lv3)（3 级） | **层级断裂** |
| 量化单位 | 浮点 | 定点 GEMM 友好 | 表达能力错位 |

**核心洞察：**
尾数与值集完全一致 ⇒ **误差几乎 100% 来自 scale 体系错配**（块粒度膨胀 + 三级定点 scale 的离散网格损失）。
⇒ 优化重心是 **scale 重构 / 等价变换 / 输出感知调参**，而非尾数编码。

### 0.3 三大技术难点归因

1. **Scale 层级断裂**：NVFP4 的连续 FP32 global × E4M3 无法直接映射到 HiF4 的三级离散定点网格。
2. **块粒度膨胀 16→64**：单个 outlier 稀释整组 64 个元素的 scale，有效动态范围骤降。
3. **二次截断误差**：`NVFP4 → FP32 → HiF4` 路径中，反量化再重量化引入额外舍入。

---

## 1. 资源清单（可复用代码仓 + 论文）

### 1.1 代码仓

| 仓库 / 资源 | 用途 | 优先级 |
|---|---|---|
| **HiFloat 社区开源仓**（hifloat.gccorg.com，GCC 全球计算联盟） | HiF4 三级 scale 精确语义、官方 MSE/SQNR 实验数据 | ★★★★★ |
| **thu-ml/SageAttention**（1/2/3） | FP4 Attention 全量化算子；Smooth K、两级量化、softmax(P) 处理 | ★★★★★ |
| **NVIDIA/TensorRT-Model-Optimizer** | NVFP4 quantizer/dequantizer 官方 PyTorch 仿真，复现基准 | ★★★★★ |
| **NVIDIA/TransformerEngine** | FP4/FP8 microscaling kernel；amax→scale 映射、delayed scaling | ★★★★☆ |
| **microsoft/microxcaling** | OCP MX 规范轻量纯 PyTorch 仿真，快速搭本地评测器 | ★★★★☆ |
| **OpenGVLab/LLMC** | HiFloat4 W4A4 PTQ 工具链（ICME 2026 HiF4 赛道获奖方案所用） | ★★★☆☆ |
| **昇腾 CANN 社区**：HiGPTQ / HiAutoRound / HiFlatQuant | HiF4 上的 GPTQ 式二阶舍入、可微舍入、旋转变换适配实践 | ★★★★☆ |
| **USTC「Wan2.2 HiFloat4 百分位校准量化实践」**（ICME 2026 获奖，代码仓挂于 HiFloat 社区） | 证明 **percentile 校准对 block-64 显著优于 amax**，可直接迁移 | ★★★★★ |
| IST-DASLab/gptq · cornell-relaxlab/quip-sharp · facebookresearch/SpinQuant | 舍入优化 / Hadamard 旋转参考实现 | ★★★☆☆ |

### 1.2 论文

**FP4 格式与 Attention 量化（核心必读）**

- *HiFloat4 Format for Language Model Inference*（昇腾）—— HiF4 三级缩放动机、64 元素分组、定点 GEMM 化原始论文
- *SageAttention3: Microscaling FP4 Attention for Inference*（arXiv 2505.11594）—— FP4 Q/K/V/P 全量化；**注意力图小值集中于 [0,1] 导致 scale 动态范围窄**的分析直接适用本题
- *SageAttention2*（arXiv 2501.09747）/ *SageAttention*（arXiv 2410.02367）—— outlier smoothing、per-thread INT4、Smooth K
- *NVFP4: FP4 Pretraining*（NVIDIA, arXiv 2509.25149）—— NVFP4 双级 scale、Hadamard 变换、随机舍入配方
- *The Promise and Performance Gap of Microscaling FP4*（MR-GPTQ, arXiv 2509.23202）—— 小分组下 outlier 缓解失效、块级 Hadamard + GPTQ 改进
- *HiFloat8: An Efficient 8-Bit Floating-Point Format*（arXiv 2312.11978）—— HiFloat 家族"锥度精度"设计思想

**量化数学基础（舍入与变换）**

- *GPTQ*（arXiv 2210.17323）、*Optimal Brain Quantization / Compression* —— 二阶信息驱动逐元素舍入
- *AdaRound*（arXiv 2004.10568）、*BitSplit* —— 学习舍入方向而非最近舍入
- *SmoothQuant*（arXiv 2211.10438）—— 激活 outlier 在相邻张量间等价迁移
- *QuaRot*（arXiv 2404.00456）、*SpinQuant*（arXiv 2405.16406）、*QuIP#*（arXiv 2402.04396）—— 正交/Hadamard 旋转消除 outlier
- *RaBitQ*（SIGMOD'24）、Google *TurboQuant*（ICLR'26）—— 旋转 + 1bit 误差校正器（注：TurboQuant 存在学术争议，仅参考技术思路）

---

## 2. 实施顺序总览

```text
Phase 0  环境搭建与基线复现
   ↓
Phase 1  精确转换层（保底正分）
   ↓
Phase 2  输出感知调参层（性价比最高）
   ↓
Phase 3  等价变换层（最大单项增益）
   ↓
Phase 4  统计校准层（稳健性打磨）
   ↓
Phase 5  舍入与补偿层（精细榨取）
   ↓
Phase 6  工程合规与风险控制

```

| Phase | 名称 | 预期增益 | 实施成本 | 风险 | 依赖 |
|---|---|---|---|---|---|
| 0 | 环境 & 基线复现 | 0（但为一切前提） | 低 | 无 | — |
| 1 | 精确转换层 | 中 | 低 | 极低 | P0 |
| 2 | 输出感知调参 | **高** | 中 | 低 | P0/P1 |
| 3 | 等价变换（旋转+Smooth） | **最高** | 中高 | 中 | P1/P2 |
| 4 | 统计校准 | 中 | 中 | 低 | P1 |
| 5 | 舍入与补偿 | 中低 | 高 | 中 | P2/P4 |
| 6 | 工程合规 | 决定生死 | 低 | — | 全部 |

---

## 3. Phase 0 · 环境搭建与基线复现

### 原因
- 评分含**负分机制**：无对照就无法判断策略是正收益还是劣化。
- 必须先在本地复现 `标准 HiF4 基线`（per-64 amax + RNE），才能量化每个组件的边际贡献。
- 校准集的意义就是**让你拟合评测指标**，前提是本地 pipeline 与评测端一致。

### 思路
1. 用 `TensorRT-Model-Optimizer` / `microxcaling` 复现 NVFP4 反量化 → FP32 基准路径。
2. 实现最朴素 HiF4 量化：块内 `amax / 6.0` → 编码 E6M2，lv2/lv3 取默认档位，尾数 RNE。
3. 搭建本地 Attention 评测器：与题目一致的 `O_ref` / `O_sub` 计算与 MSE 输出。
4. 产出 **消融表格模板**：每个 Phase 新增组件都记录 `MSE_std / MSE_you / score`。
5. 校验 state 序列化合法性（节点数 ≤ 4096、嵌套 ≤ 8、无 NaN/Inf/grad）。

### 验收标准
- [ ] 基线 score ≈ 0（与标准 HiF4 对齐）
- [ ] 单组 Attention 端到端耗时可测，250 组总耗时有预估
- [ ] 消融框架可一键跑通

---

## 4. Phase 1 · 精确转换层（保底分，必做）

### 原因
零风险、低成本、直接收益。既然值集相同，**只要 scale 选得比 amax 更好，误差立刻下降**。这是所有后续优化的地基。

### 思路

**1.1 块内 scale 离散穷举搜索（替代 amax）**
- E6M2 是**离散网格**，不是连续变量 ⇒ 每块遍历相邻 2~4 个候选编码，直接最小化块内 64 个反量化值的 MSE。
- 实现要点：候选编码张量 + `gather`，全向量化，禁止 Python 逐块循环。
- 这是小规模穷举，开销极低而收益直接。

**1.2 三级 scale 协同（档位选择器思路）**
- `scale_lv2 (E1_8)`、`scale_lv3 (E1_16)` 自由度极少 ⇒ 视为**动态范围档位选择器**而非精细调节器。
- 校准阶段统计各张量（Q/K/V）全局量级分布 → 为每个在线张量选择使 E6M2 落在**最优表达区间**的档位组合 → 再在 lv1 上做精细搜索。

**1.3 对齐 NVFP4 二级结构，识别"可无损块"**
- 1 个 HiF4 块（64）= 4 个 NVFP4 子块（16）。
- 若 4 个子块的 E4M3 scale 比值可被 HiF4 层级精确吸收（比值为 2 的幂 / E6M2 可表达值），则该块可**零误差转换**。
- 策略：优先检测无损块并快速通道处理，把误差预算集中投给不可对齐的块（可分配更大搜索半径）。

### 验收标准
- [ ] score 稳定为正，无任一组劣化
- [ ] 无损块占比统计完成
- [ ] 单组耗时 < 预算的 30%（为后续 Phase 留空间）

---

## 5. Phase 2 · 输出感知调参层（性价比最高）

### 原因
**最终 MSE 是 Attention 输出的 MSE，不是张量重构 MSE。**
softmax 的指数放大意味着：高分值位置的 Q/K 误差对输出的影响，远大于低分值位置。
因此"最小化张量重构误差"与"最小化输出误差"**不是同一个目标**——直接用输出 MSE 当目标函数调参，是本题投入产出比最高的一步。

### 思路

**2.1 统一超参搜索框架**
- 将所有超参收拢为一个低维向量（约 10~20 个标量）：

| 超参 | 含义 | 搜索范围建议 |
|---|---|---|
| `p` | 百分位裁剪比例 | 0.99 ~ 1.0 |
| `r` | E6M2 搜索半径 | 1 ~ 4 |
| `α` | Smooth 迁移强度 | 0 ~ 1 |
| `R_on` | 是否启用 Hadamard 旋转 | {0, 1} |
| `δ_q/δ_k/δ_v` | 舍入偏置阈值 | −0.1 ~ +0.1 |
| `γ` | 档位安全裕度 | 0.9 ~ 1.1 |
| `w` | 贝叶斯收缩权重 | 0 ~ 1 |

- 目标函数 = **校准集上的输出 MSE**（1 calib : 1 test 的数据配比意味着校准集就是该组的"预览版"）。
- 优化器：CMA-ES / 贝叶斯优化 / 坐标下降（标量维度低，坐标下降即可快速收敛）。

**2.2 Q/K/V 分别独立调参**
- 三者误差传播路径不同：**Q/K 经 softmax 非线性放大，V 线性传递到输出** ⇒ 最优超参必然不同，不可共用一套。

**2.3 分组自适应**
- 250 组数据的分布未必同质。可先对校准集做**聚类/特征分箱**（按 amax 分布、outlier 密度、块间 scale 方差），为每类拟合一套超参，state 中存"分箱规则 + 各箱超参"。
- 注意 state 预算：4096 节点 ⇒ 特征 + 分箱 + 超参需精简编码。

### 验收标准
- [ ] 校准集输出 MSE 相对 Phase 1 下降 ≥ 15%
- [ ] 超参在 5 个校准集上泛化（无过拟合单组）
- [ ] 全部最优超参可序列化进 state

---

## 6. Phase 3 · 等价变换层（最大单项增益）

### 原因
Phase 1/2 是"在既定分布下选更好的 scale"，Phase 3 是**改变分布本身**——从根源上消除 outlier，让 block-64 的动态范围压力自然缓解。这是 MR-GPTQ 论文的核心结论：**小分组场景下常规 outlier 缓解手段失效，必须靠变换（Hadamard/旋转）重建分布**。

### 思路

**3.1 Q/K 同步正交旋转（QuaRot / SpinQuant）**
- 对 head_dim 维施加同一 Hadamard 矩阵 `R`：
  ```text
  Q̂ = quant_HiF4(Q · R),   K̂ = quant_HiF4(K · R)
  ⇒ Q̂K̂ᵀ ≈ (QR)(KR)ᵀ = QRKᵀRᵀ = QKᵀ     # 数学严格不变

  ```
- 收益机制：旋转后 outlier 能量被摊平到所有通道（CLT 效应）⇒ 块内动态范围骤降 ⇒ 64 元素块的 scale 错配误差大幅减小。
- `R` 在校准阶段确定并存入 state（CPU tensor，合法）。
- ⚠️ **V 绝对不能旋转**：输出 `O = P·V`，评测端无法施加逆变换 `R⁻¹`，旋转会直接破坏结果正确性。

**3.2 Q/K 通道级 Smooth（SmoothQuant / SageAttention 的 Smooth K）**
- SageAttention 论文核心发现：**K 的通道间幅值差异远大于 Q**。
- 校准阶段搜索通道缩放向量 `s`：
  ```text
  Q' = Q · diag(s),   K' = K · diag(s⁻¹)  ⇒  Q'K'ᵀ = QKᵀ（严格等价）

  ```
- 作用：把 K 的通道 outlier **迁移**给 Q，使两者块内分布同时趋于均匀。
- `α`（迁移强度，`s = (K_range/K_mean)^α / (Q_range/Q_mean)^(1−α)`）由校准集输出 MSE 搜索，存入 state。
- 与旋转的先后：**先 Smooth（通道级）后 Hadamard（混合级）**，二者互补不冲突。

**3.3 Q/K 误差补偿对偶**
- Q 与 K 的量化误差在 `QKᵀ` 中是**乘性叠加**。
- 若 K 某通道量化误差系统性偏大，可微调对应 Q 通道的 scale 或舍入偏置 `δ` 做部分抵消。
- 校准阶段学一组低维补偿系数（维度 = head_dim 或更低），存入 state。

### 风险控制
- 旋转/Smooth 属激进变换，**必须逐项消融**：`基线 → +Smooth → +Hadamard → +补偿`，每步确认校准集输出 MSE 为正收益才保留。
- 对收益不稳定的组（如 V 极端 outlier 组）提供**回退开关**（state 中存 per-group flag），回退到 Phase 1/2 路径。

### 验收标准
- [ ] Q/K 旋转后块内 amax/median 比值显著下降（分布均匀化可量化验证）
- [ ] `Q̂K̂ᵀ` 与 `QKᵀ` 的相对误差可控
- [ ] 输出 MSE 相对 Phase 2 再降 ≥ 20%
- [ ] V 路径未被误旋转（断言校验）

---

## 7. Phase 4 · 统计校准层（稳健性打磨）

### 原因
Attention 张量的 amax **逐 token 波动剧烈**（SageAttention3 / TransformerEngine 的一致结论）。在线阶段若依赖即时 amax，极易被单个 outlier 拉爆 scale。需要把校准阶段的**分布先验**注入在线决策。

### 思路

**4.1 延迟缩放（Delayed Scaling）**
- 校准阶段统计 Q/K/V 的 amax 分布（均值、分位数、EMA），写入 state。
- 在线阶段：`scale_lv2/lv3 = state统计量 × γ(安全裕度)`，`γ` 由 Phase 2 搜索得到。
- 效果：全局档位由**历史分布**决定而非当次瞬时值，抗 outlier 抖动。

**4.2 百分位裁剪（Percentile Clipping）**
- ICME 2026 HiF4 获奖方案已验证：**percentile 校准对 block-64 结构显著优于 amax**。
- 对 outlier 密集的块，主动裁剪最高 0.1%~1% 幅值以换取更优 scale。
- 本质是**"牺牲个别大值、保全 63 个小值"的正期望交易**。
- 裁剪比例 `p` 在校准集上按输出 MSE 搜索，存入 state（Q/K/V 分别一套）。

**4.3 鲁棒 scale 估计（贝叶斯收缩）**
- 用校准数据拟合每种张量的分布模型（attention 激活近似 **高斯/拉普拉斯 + 重尾**）。
- 在线阶段对每块做收缩：
  ```text
  scale = w · block_amax + (1 − w) · state先验

  ```
- `w` 由块内样本可靠性决定（如块内有效元素数、是否含 NaN/极值）。

**4.4 逐块自适应裁剪**
- 不做全局统一 `p`，而是按块内 outlier 密度分档：低密度块用 amax，高密度块用强裁剪。
- 分档阈值由校准集搜索。

### 验收标准
- [ ] 在线 amax 方差相对即时统计显著下降
- [ ] 极端 outlier 组不再出现负分
- [ ] 裁剪策略在 250 组上无一劣化

---

## 8. Phase 5 · 舍入与补偿层（精细榨取）

### 原因
E2M1 尾数只有 1 bit ⇒ 舍入决策是**二值的**，且 round-half-even 在中点附近存在**系统性对称偏置**。打破该偏置可小幅但稳定地降低 MSE。属高成本低增益项，放在最后。

### 思路

**5.1 学习舍入（AdaRound 轻量版）**
- 校准一个"中点偏移阈值"`δ`：
  ```text
  若 |v − grid_low| / (grid_high − grid_low) < 0.5 + δ  →  向上舍入
  否则                                                  →  向下舍入

  ```
- `δ_q / δ_k / δ_v` 分别校准，由 Phase 2 的搜索框架统一优化。

**5.2 GPTQ 式二阶舍入（若时间预算允许）**
- 在块内做**逐列误差补偿舍入**：前一个元素的舍入误差，通过调整同块后续元素的 scale/舍入方向吸收。
- 需二阶信息（Hessian 近似），校准阶段可从 Q/K 校准数据估计并存入 state。
- ⚠️ 计算量大，须先测单组耗时。

**5.3 确定性抖动（针对 V）**
- V 的误差**线性传递**到输出、无 softmax 放大 ⇒ 系统性偏差会在 `P·V` 累加中**相干叠加**。
- 方案：按校准误差统计给块内元素加**受控扰动**再舍入，使量化误差近似零均值白噪声，破坏相干性。
- 扰动幅度由校准集输出 MSE 决定，种子固定（保证可复现、可校验）。

**5.4 低秩误差校正器（RaBitQ / TurboQuant 思想）**
- 校准阶段估计量化误差的主成分，在线阶段对残差做低秩补偿。
- 受 state 节点数（4096）约束，秩必须很低（如 r ≤ 4）。

### 验收标准
- [ ] 舍入偏置 `δ` 使 MSE 有稳定小幅下降
- [ ] 抖动方案不引入 NaN/Inf，结果可复现
- [ ] 总耗时仍在 5 min 预算内

---

## 9. Phase 6 · 工程合规与风险控制（决定生死）

### 9.1 时限合规（5 分钟硬约束）

| 要求 | 具体措施 |
|---|---|
| 全向量化 | 所有在线函数禁止 Python 逐块循环，统一用 torch 批量块操作 |
| 搜索实现 | E6M2 穷举用"候选编码张量 + `gather`"，不用 for 循环 |
| 预算前置 | 先测**单组**耗时 → 推算 250 组总耗时 → 反推可用搜索半径 |
| 降级策略 | 若耗时超预算，优先砍 Phase 5（GPTQ/低秩），保 Phase 1~3 |
| 缓存复用 | 校准结果全量写入 state，在线阶段**零重复计算** |

### 9.2 负分防线

1. 上线任何激进策略前，先用**标准 HiF4 基线复现**（per-64 amax + RNE）做对照。
2. 逐项消融，**每个组件在校准集输出 MSE 上必须为正收益**才保留。
3. state 中存 per-group 回退 flag：收益不稳定的组自动回退到基线路径。
4. 最终提交前，跑一遍"全组 score 分布"，确认**无任一组显著劣化**。

### 9.3 合法性自检（提交前必做断言）

```python
# 输出格式合法性
assert scale_factor.dtype/取值域 ∈ E6M2 合法编码，且无 NaN/Inf
assert scale_lv2 ∈ E1_8 合法编码
assert scale_lv3 ∈ E1_16 合法编码
assert sign ∈ {0, 1}
assert mant  ∈ {0, 1}                      # E2M1 尾数位
assert 张量 shape 与题目定义严格一致

# state 合法性
assert 仅含 None / bool / int / 有限 float / str / CPU Tensor
assert 嵌套深度 ≤ 8
assert 节点总数 ≤ 4096
assert 无 NaN / Inf / requires_grad / grad_fn
assert 所有 tensor 均在 CPU（非 CUDA）

# 代码合规
assert solution.py 中无任何文件读写操作

```

### 9.4 提交物

- `solution.py` → 打包为 `solution.zip`
- 必须实现的 6 个 API：

| 函数 | 阶段 | 作用 |
|---|---|---|
| `hif4_calibration_and_quantize_weight` | 离线 | 权重 HiF4 量化 + 生成激活量化 state |
| `hif4_dynamic_quantize_activation` | 在线 | 激活数据动态 HiF4 量化 |
| `hif4_calibration_attention` | 校准 | Attention 校准 + 生成 Q/K/V state |
| `hif4_dynamic_quantize_q` | 在线 | Q 动态量化 |
| `hif4_dynamic_quantize_k` | 在线 | K 动态量化 |
| `hif4_dynamic_quantize_v` | 在线 | V 动态量化 |

---

## 10. 关键决策速查表

| 决策点 | 结论 | 依据 |
|---|---|---|
| 优化重心放尾数还是 scale？ | **scale** | 两者值集完全相同，误差全来自 scale 错配 |
| E6M2 用 amax 还是穷举？ | **穷举（半径 2~4）** | 离散网格，向量化开销极低、收益直接 |
| lv2/lv3 如何定位？ | **动态范围档位选择器** | 自由度极少，不宜当精细调节器 |
| Q/K 能否旋转？ | **能**（同一 Hadamard R） | `QRKᵀRᵀ = QKᵀ` 严格等价 |
| V 能否旋转？ | **不能** | `O = PV`，评测端无法逆变换 |
| Smooth 迁移方向？ | **K → Q** | K 通道幅值差异远大于 Q（SageAttention） |
| 校准用 amax 还是 percentile？ | **percentile** | ICME 2026 HiF4 获奖方案实证优于 amax |
| 在线 scale 用即时还是历史？ | **历史先验 + 收缩** | amax 逐 token 波动剧烈（delayed scaling） |
| 调参目标函数？ | **输出 MSE**（非重构 MSE） | softmax 指数放大，两者非同一目标 |
| Q/K/V 共用超参？ | **不共用，各自独立** | 误差传播路径不同（非线性 vs 线性） |
| 抖动优先给谁？ | **V** | 线性传递、无 softmax 放大、偏差相干叠加 |
| 优先级最高的两个 Phase？ | **P2（调参）+ P3（变换）** | 增益最大、风险可控 |

---

## 11. 推荐执行时间线

```text
Day 1-2   Phase 0：环境搭建、基线复现、消融框架、本地评测器
Day 3-4   Phase 1：E6M2 穷举 + 三级 scale 协同 + 无损块识别   → 提交，锁定正分
Day 5-7   Phase 2：超参搜索框架 + Q/K/V 独立调参              → 提交，验证泛化
Day 8-11  Phase 3：Smooth K/Q → Hadamard 旋转 → 误差补偿      → 提交，逐项消融
Day 12-13 Phase 4：delayed scaling + percentile + 贝叶斯收缩  → 提交，检查负分组
Day 14-15 Phase 5：学习舍入 δ + V 抖动（视耗时预算取舍）      → 最终调优
Day 16    Phase 6：合规自检、耗时压测、per-group 回退开关、定稿提交

```

---

## 12. 三份必读材料（优先级排序）

1. **SageAttention3**（arXiv 2505.11594）—— 覆盖 FP4 Attention 量化的全部工程要点（Smooth K、两级量化、scale 动态范围分析）
2. **QuaRot**（arXiv 2404.00456）—— Q/K 旋转的数学等价性证明与实现细节
3. **ICME 2026 HiF4 赛道获奖方案（USTC 百分位校准）**—— 唯一在 **HiF4 格式上**实证过的校准策略，与本题格式完全一致

> 这三份材料几乎覆盖本题 Attention 场景的全部技术要点，建议精读后再动手实现。
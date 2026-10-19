# 量化论文研读记录（2026-09-10）

用途：为"基于 public_quantizer、以 GPTQ 叠加优化 Attention"的版本开发提供依据。
按 AGENTS.md 要求记录**路径、章节、适用条件、迁移限制**。已下载 ≠ 读过原文。

---

## 1. GPTQ — `docs/papers/quantization/pdf/GPTQ_2210.17323.pdf`
文本：`text/GPTQ_2210.17323.txt`

**读的章节**：§3 背景（OBQ，行 181–226）；§4 "The GPTQ Algorithm" Step 1 Arbitrary Order Insight
与 Step 2 Lazy Batch-Updates（行 228–299）。

**原文结论**
- Step 1（行 229–237）：OBQ 的贪心顺序相对"任意固定顺序"的改进**通常很小**，在**大型、参数化充分的层**上尤其如此。
  理由：单个误差较大的权重数量略少，与被推到流程末尾、可补偿的未量化权重变少，两者相互抵消。
- Step 2（行 289–298）：列 i 的舍入决策只受**该列自身**已发生的更新影响，后续列更新此刻无关 →
  可以把 B=128 列成组惰性批量更新，只在该 B×B 块内维护 H⁻¹。

**迁移到本题的条件差异（必须记录）**
- 原文的 d_col 是 LLM 权重矩阵维度（数千），本题注意力按 KV head 分组后 d_col = head_dim = 256（public_quantizer），
  d_row = token 数（≤1024）。**矩阵显著更小**，因此"任意顺序都可以"的结论不能直接照搬；
  但 v445 双向顺序搜索在参考库上得 −315，经验上与"顺序不是主要矛盾"一致。
- 原文的 Hessian 来自层输入 X（层间激活），本题 Q/K 的 Hessian 需自行构造（见 §3）。
- 本题 B=128 的批量更新在实现上是 `GPTQ_BLOCK = 32`（见 `solution_v445.py:2523`）。

---

## 2. SpinQuant — `docs/papers/quantization/pdf/SpinQuant_2405.16406.pdf`
文本：`text/SpinQuant_2405.16406.txt`

**读的章节**：§3.2 Cayley-Optimized Rotation（行 283–335）；§4.1 实验设置（行 344–358）。

**原文结论**
- 旋转在 Stiefel 流形上用 **Cayley SGD** 优化（式 3、4），保证正交性（行 297–325）。
- **初始化用随机 Hadamard 矩阵**（行 334、345）；lr 从 **1.5** 线性衰减到 0，**100 次迭代**，800 个校准样本（行 348–349）。
- R2 是**逐头旋转**，形状 (D_head, D_head)，**每层单独学习**（行 347–348）；R1 是残差旋转 (D_token, D_token)。
- Table 2 结论（行 416）：**学习旋转一致且显著优于 Hadamard**；对随机初始化鲁棒（行 334–335）。
- R3/R4 是**在线旋转**（无法并入权重，保留为 Hadamard），因其可用 Hadamard 变换高效实现（行 286–288）。

**迁移到本题的条件差异**
- 本题 Q/K 的**联合**旋转 R：q'=qR, k'=kR ⇒ q'k'ᵀ = qkᵀ，logits 严格不变 → 这是一条**纯规范自由度**，
  只要 Q 与 K 用同一个 R，就是无损等价变换。对应原文的"在线旋转"情形：**必须在运行时施加**，
  因此**有算力成本**，受 300 秒硬限约束（本题内部目标 280 秒）。
- 本题已经有 per-head 旋转搜索（`_make_R(dh, seed)`，`ATTN_ROTATION_CANDIDATES = 1`）与
  `tools/learned_rotation_core.py`，但后者只有 `_LR_STEPS = 4`、`_LR_RATE = .05`、单位阵初始化，
  与原文的 100 次迭代 / lr 1.5 / Hadamard 初始化差距很大。
- 原文优化的是任务损失（交叉熵）；本题直接可微的替代是**部署后 Attention 输出 MSE**，
  `learned_rotation_core._lr_loss` 已实现该目标（输出 MSE 或中心化 logit MSE）。

---

## 3. QuaRot — `docs/papers/quantization/pdf/QuaRot_2404.00456.pdf`
文本：`text/QuaRot_2404.00456.txt`

**读的章节**：§3.1 Orthogonal/Rotation/Hadamard 矩阵（行 121–140）；§3.2 Incoherence Processing（行 143–152）。

**原文结论**
- 计算不变性（computational invariance）：把 Hadamard 变换并入权重，得到等价网络（行 63–64、110–116）。
- 随机 Hadamard 变换可同时作用于激活，使激活也被 incoherence 处理（行 116）。
- µ-incoherence：最大元素相对于幅值范数过大的矩阵难以量化；旋转降低 incoherence（行 143–152）。
- 对 d ≠ 2ⁿ 的尺寸，把 d = 2ⁿm 分解，用已知 Hadamard 矩阵构造（行 135–137）。

**迁移到本题的条件差异**
- 本题 head_dim = 256 = 2⁸（public_quantizer），**存在 Walsh–Hadamard 矩阵**，可精确构造；
  但 v1 数据集的 head_dim = 64、qwen 数据集为 128，也都是 2 的幂。
- QuaRot 的 incoherence 论证针对**每通道/每张量**尺度下的离群值；本题 HiF4 是**分层块浮点**
  （见 §4），微指数已吸收组间离群值，因此旋转收益预计**小于**原文在 int4 per-channel 下的收益。
  这必须由本地实测判定，不能由原文推断。

---

## 4. HiFloat4 — `docs/papers/quantization/pdf/HiFloat4_Evaluation_2602.12635.pdf`
文本：`text/HiFloat4_2602.11287.txt`

**读的章节**：Appendix A Format Definition（行 141–214）。

**原文结论（本题的量化格式本身）**
- 一个 HiF4 单元 = **32 bit 共享缩放元数据 + 64 个 4-bit 元素**，平均 4.5 bit/值（行 142–144）。
- 三级缩放：level-1 为无符号 FP8 **E6M2**（全局基数，bias 48，unbiased exp ∈ [−48,15]）；
  level-2 为 **8 路 E1 micro-exponent**；level-3 为 **16 路 E1**。三级合计 32 bit（行 147–186）。
- 元素编码为 **S1P2**（符号 + 1 位整数 + 2 位小数），等价于 E1M2；
  可表示幅值 {0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75}，**非均匀格点**（行 187–195、Table I）。
- 取值：V_i = E6M2 × 2^({E1_8}_⌈i/8⌉ + {E1_16}_⌈i/4⌉) × {S1P2}_i（式 2，行 211）。

**对建模的直接影响**
- **格点非均匀**（靠近 1.0 更粗、靠近 0.25 更细），所以"均匀 int4 的舍入误差模型"不适用；
  代码用 `(x/unit*4).round().clamp(-7,7)*(unit*0.25)` 复现 S1P2 舍入（见 `tools/gptq_scratch_core.py`）。
- **每 4 个元素共享一个 level-3 微指数**，即量化误差在**组内 4 元之间相关**；
  GPTQ 逐列独立传播误差的假设在此只是近似。
- 微指数是**共享决策**：某个元素需要更大的指数，会同组其他元素一起改变有效格点。
  这给"组内联合优化"留出空间，但也意味着朴素 GPTQ 的增益上界低于均匀 int4。

---

## 5. 本次研读直接产生的可证伪假设

**H1（旋转规范自由度）**：Q/K 的 per-head 联合旋转是严格无损的自由度，HiF4 下的量化误差随 R 变化。
当前只搜索 1 个候选 seed（`ATTN_ROTATION_CANDIDATES = 1`）且 `learned_rotation_core` 只走 4 步 Cayley。
按 SpinQuant §3.2/Table 2，增强旋转优化应带来可测的部署输出 MSE 下降。

**替代解释**：HiF4 的分层微指数已吸收离群值（§4），旋转在本题格式下的收益可能远小于 int4 per-channel；
因此 H1 可能被本地实测否证——这正是要做的实验。

**H2（GPTQ 度量）**：Q/K 的 GPTQ Hessian 由"logit 空间"（Σ Q_hᵀQ_h）换成**部署输出的 Gauss–Newton 度量**，
应更贴合最终 MSE 目标。已有未使用实现：`tools/pv_full_metric_core.py`（全 key、无 key 子采样、逐头守门）。

**H3（V 的作用）**：注意力输出 out = P·V，V 的量化误差经 softmax 权重线性传导；
V 的度量应为 H_V = PᵀP 而非单位阵。需核对当前 V 路径用的是哪一种（见 `_quantize_weighted(v_rot, ones_k)`）。

**每一项都必须用 public_quantizer 实测判定**，不能由论文推断。

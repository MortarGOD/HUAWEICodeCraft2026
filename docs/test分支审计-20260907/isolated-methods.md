# 独立动态接口候选：四分支源码与历史高分审阅

本笔记审阅 `experiment/legal-inputisolated-v4`、`experiment/legal-inputisolated-v5`、`experiment/legal-tensor-recon-v2`、`experiment/new-paper-implementation` 的非重复算法，以及历史回执 25062 / 23040 对应的保存源码。只做源码、AST、文档和回执阅读，没有导入或执行候选、评测、提交。`runtime170-*` 由 implementation_map 单独审阅。

证据范围：沿用父任务的 Verify Tier 2 图谱上下文（项目 `Users-ee-Code-2026-huawei-alogorithm`，generation `2026-09-07T02:35:31Z`），它覆盖当前参考入口和根 v419；`.git` 被排除，新分支快照没有图谱覆盖证明。因此以下分支结论来自精确源码与 AST，不能称图谱完整验证。AST 组号采用 [algorithm-files.json](algorithm-files.json) 的零基下标；完整来源见 [manifest.json](manifest.json)。

## 1. 实际文件与入口

以下短名明确指向快照中的完整文件；后文 `短名:行号` 均引用此表，不把分支名当作实际部署实现。

| 短名 | AST 组 / 前 12 位 | 源文件与实际绑定 |
|---|---|---|
| B0 | 1 / `f13c814558c8` | [v5 的 tensor-recon-baseline-v1][b0]；四分支主 `solution/solution.py` 均属于这个旧基线，不能从主入口推断已运行下列方法。公开别名最终绑定 405–410。 |
| B4 | 4 / `64dd674ccf4f` | [v4 calhessian 当前源码][b4]，等于 v4/v5 的 `submission-aonly-v4.zip/solution.py`；Q 2260–2269、K 2387–2396 直接走 `_dynamic_attention`，V 2749–2818 才走 token Hessian。 |
| C25 | 5 / `e0c791252b09` | [new-paper calhessian 当前源码][c25]，也等于 v4/v5 的普通 `submission.zip/solution.py`，并等于历史 25062 源码。它与 B4/B5 不同。Q 2314–2323、K 2441–2450 直接走通用动态量化；V 2803–2872。 |
| B5 | 6 / `c700ce9c0faa` | [v5 calhessian 当前源码][b5]，等于 `submission-v5.zip/solution.py`；Q 2640–2649、K 2767–2776 调 `_dynamic_attention` 2478–2554，此函数执行块 Hessian 与离散修正；V 3343–3434。 |
| QG5 | 7 / `0e3cd9aac270` | [v5 quotient-Givens 当前源码][qg5]，等于其本目录 `submission.zip/solution.py`；校准 2948–4002，公开 Q 4739–4801、K 5198–5258、V 5616–5745 为最终实现。 |
| L2 | 11 / `287209de5c37` | [legal-tensor-recon-v2 当前源码][l2]，等于 `legal-aonly-smooth-rht-v2-1024.zip/solution.py`，也等于历史 23040 源码；最终别名 481–486。 |
| QG20 | 12 / `af89c4cc7095` | [new-paper 中 target20 ZIP 成员][qg20]。与 QG5 的 Attention 函数逐函数 AST 相同，QG20 对应行号为 QG5 减 47；差异集中在 Linear。不能把两包的总分混在一起。 |

这里的“等于”指 AST 相同；源码与 ZIP 成员可因 CRLF/LF 而字节 SHA 不同。AST 不含注释与排版，因此算法去重不代替 ZIP 字节身份核对。B4/B5 目录中普通 `submission.zip` 仍保存 C25，是最容易误读的版本边界。

## 2. 信息可得性：动态独立与分别校准是两项检查

这些候选的 live Q/K/V 动态路径使用当前本角色张量及其冻结 state，没有 Compensated V2 那条依赖当前 Q→K→V 的全局缓存链。这是可借鉴的部署边界，但不自动证明适配本项目三角色分别校准的协议。

Q/K 采用等价变换必须满足同一次推理的配对关系，例如 `Q' = Q D R`、`K' = K D^-1 R`，要求两端的 `D` 和 `R` 一致。L2:350–398 从一份配对 Q/K 校准统计返回同一个 balance；B4/B5/QG5 还从配对 Q/K/V 选择 route、balance、反射和旋转。若 Q 和 K 的校准分别收到真实自身与不同假对端，则“两个 state 在同一次 calibration 中相同”不能推出“最终取用的两个 state 相同”。B0 的 Q/K 共享旋转布尔值也有相同条件（B0:324–337）。该问题同样适用于根项目任何依赖共享校准结果的变换，不能只检查外部候选。

QG5 的 [audit_attention_isolation.py][isolation] 检查每数据集首末组（54 行）、同一次配对校准（63 行）、调用顺序/插入其他样本（66–83 行）与新模块重复校准（86–93 行）。它没有实现三个角色分别用不同假对端校准的测试。其通过只能支持“调用顺序及当前样本隔离”，不能支持未知假对端分布下的变换一致性或冻结 Hessian 有效性。

Linear 需另分两级：

- L2 是严格 A-only state：`_linear_activation_balance` 275–316 仅接收 A 校准，319–331 先算 balance 再解码 W，state 恰为 `{"balance": ...}`；动态 334–347 只读这个字段。
- B4/B5 的平滑统计也来自 A（B4:112–139、B5:135–164）。QG5 的平滑函数 216–245 明确忽略 `weight`，四个 RHT 符号候选也由 A 选取（1663–1665）。虽校准中仍计算 W 残差与修正，最终多校准返回 1840–1856 把 `quadratic/affine` 置为 `None`，importance 来自 A；单校准返回 2046–2058 同样不返回 W 的二次型/仿射修正。不能仅凭未清理的中间计算误判最终 state 携带这些值。
- C25 不属于严格 A-only：108–140 的 `_linear_smooth_balance` 使用 W RMS，校准 1284–1584 用真实输出选择平滑指数等，最终把 balance 存入 state（1455、1578 行）。它的动态 A 确实没有当前 W，也不携带 W 原张量/残差投影；“冻结变换允许依赖校准 W”与“任何 W 信息不得进 A state”是不同规则。QG20 亦保留较强的 W 派生 affine state，其 [SOURCE-TUNING.md:80][qgsource] 已明确此点。

## 3. K quotient：相对 v419 最值得分离验证的变化

QG5 在通道变换后先把 K 跨 token 去均值（4005–4082），基础量化再做四轮 recenter（4085–4129），随后在 **去除了 token 常量误差的目标** 上做离散修正。常量平移 `K -> K - 1c^T` 对精确 softmax 恒等，因此这里的结构依据不依赖知道当前 Q；但这个恒等式不意味着任意再量化后的误差必下降。

核心目标见 QG5:4804–4962：令 `C = I - 11^T/T`、误差 `E = K_hat - K_target`，使用

`L(E) = sum_t (CE)_t^T H_t (CE)_t + gamma ||sum_t F_t^T (CE)_t||²`。

实际实现包括 `.25 H_cal + .75 I`（4820 行后）、factor 跨 token 中心化（4836–4838）、误差中心化（4854–4857）、投影后的坐标曲率（4860–4864）、中心化梯度（4874–4885），以及同时更新候选的整头目标重新计算/拒绝（4916–4944）。后者很关键：投影使不同 token 的更新耦合，不能只把误差减均值后继续使用独立 token 的旧接受式。

根 v419 已有 K 的当前均值/平移搜索（[v419:4169][rootkt]、4262–4273），已有全头 Gram 离散修正（2634–2651）；所以“去均值”本身不是新增。具体差异是对量化误差采用 quotient 目标，并连同梯度、曲率、候选接受一起投影。可先在根已有冻结的每头 H 上实现较小的可证伪版本，不需要复制参考实现的 token 专属 `H_t/F_t`。

对 **所有 token 使用同一 H** 的简化目标，以下为本次数学推导，不是已测结果：单 token 增量 `delta` 的精确变化是

`Delta L = 2 delta^T H (E_t - mean(E)) + (1 - 1/T) delta^T H delta`。

多个 token 同时更新时，还必须保留 `-(sum delta_t)^T H (sum delta_t)/T` 这一交叉项。`T=1` 时 quotient 项为零，应保持稳定回退或独立的正则约束。建议第一步只替换根 K 的风险/接受式，保持编码格点、H、迭代预算和其他模块不变；同时检验常量平移不变性、一次/多次更新的解析增量与直接目标一致、代理目标不增、真实输出留出变化。当前 K 与冻结 H 足以计算此代理目标；代理下降仍不等于真实 Attention 输出下降。

参考版本更重的 `H_t/F_t` 由配对校准输出灵敏度建立（QG5:3611–3735），K state 为 `[kv_heads,T,d,d]` 与 `[kv_heads,T,d,16]`（3847–3855），动态必须匹配 T（4809–4821）。这还假设 token 位置的风险跨输入可迁移；维度相同不能验证该假设。状态成本分别为 `O(kv_heads*T*d²)` 和 `O(kv_heads*T*d*16)`，不能直接照搬到长序列。K 的 outer recenter（5198–5258）也没有给整个最终链提供真实输出单调保证。

## 4. Givens：低自由度旋转与有限的留出证据

QG5:2321–2358 在每个 64 通道块内应用一层互不重叠的二维 Givens 对，配对索引形状 `[layers,kv_heads,d/64,32,2]`，角度形状 `[layers,kv_heads,d/64,32]`，按 GQA 重复广播给 Q heads。有效的互斥配对加同一个角度矩阵构成正交 R。相同 R 应用于 Q/K 可保持未量化 logits；不同角色各自学出的 R 不享有此保证。

学习过程（2505–2703）用变换后 Q/K 的组合能量做每 64 维高低配对（2530–2561），角度从零开始，用 HiF4 fake quant 与校准 Attention 输出误差做 8 步 STE/手写 Adam，学习率 `.02`，角度截断 `±pi/4`（2639–2667）。当样本数至少 5 时，角度更新只用前 `n-2`，后两份作平均改善和最差相对退化不低于 `-0.05` 的门控（2567–2570、2669–2703）；不足 5 时训练/门控使用同一批样本。

**不能把完整流程写成严格的 3/2 独立留出。** 配对拓扑的 Q/K 能量用全部校准样本，且上游 route、统计变换亦已经接触这些样本；隔离的是角度梯度拟合，不是整个选择流程。文档将其称 3/2 split，应带这个范围说明。

根 v419 已有校准变换、Hadamard、矩阵平衡与 Gram/输出灵敏度，因此“论文旋转”不是全新方向；本候选有区别的是 **每块 32 个角度的一层互斥配对**。迁移前必须先证明分别校准仍得到相同 R/D，或将旋转限定为代码固定、按维度确定的共同变换。最小实验应把配对、上游变换和角度全部只由训练份确定，以独立模型/层和剩余样本比较：原变换、一层固定 Givens、一层拟合 Givens。先验证等价性和开销，再判断预测价值。

[SOURCE-TUNING.md:48–68][qgsource] 记录的是本地消融：quotient+STE 的 Qwen3 Attention `158.631377`，加一层 8 步 Givens `159.080264`；12 步、2/4 层、5 份全训练更差。这支持“小自由度值得实验”，不证明独立外部验证或线上提分；多轮在同一验证集选步数仍有选择偏差。QG20 与 QG5 的 Attention AST 相同，这组 Attention 消融可以解释相同算法，但总分不能跨 Linear 包复制。

## 5. 冻结 V token Hessian 与离散逃逸

B4:2029–2054 从校准概率构造 `H = mean(P^T P)`，只有各校准 T 相同时才保存；V 2749–2818 要求 H 形状正好等于当前 `[kv_heads,T,T]`，用 `.7 H + .3 I` 收缩后执行 token 坐标下降、双 token 逃逸、lv3 子树翻转，再 CD/pair 恢复。B5 的 live 路径相同但预算更大（3343–3434；顶部常量 1–67）；V scale 与 lv2 函数虽被调用，轮数为 0，不能计作有效优化。B4/C25 留着 Q/K output-Hessian helper，却没有从最终 Q/K 入口调用它们，不能从函数名声称已启用。

B5 真正新增的是 Q/K **64×64 块二阶矩**：对变换后的校准 K/Q 求 second moment、按对角均值归一后与 I 按 `.625` 混合（1883–1932），最终 state 2390–2415 保存 quadratic，动态 2478–2554 消费它并执行块内离散修正。它不是 live 的 Q/K 输出 Hessian 路径。根 v419 已有全 head Gram/GPTQ 与输出灵敏度，不能把更小块的二阶矩笼统称作超过根版本的升级。

QG5 的 V 默认采用基础量化 Q/K 得到的 `P_quant^T P_quant`（3540–3606）；`V_USE_QUANTIZED_HESSIAN=True`，但 `V_FINAL_QK_HESSIAN=False`（63–64 行），所以它没有在最终 Q/K quotient/CD 完成后重新拟合。V affine gamma 和 prototype weight 均为 0（47–48 行），不能将相应 helper / state 字段当作有效部署增益。最终仍然要求相同 T（5616–5745）。

可移植的具体离散方法是：

- **双 token 逃逸**：B4:2492–2639 为每个 head/channel 选若干 token 的合法单码步，优先考虑单步本身不降目标的候选；以 `Delta_i + Delta_j + 2 delta_i H_ij delta_j` 评价双步并只接受负值（2539–2599）。它针对坐标局部极小，不等于连续残差补偿。
- **lv3 子树翻转**：B4:2642–2746 翻转四值共用的 lv3，再重新选这四个 mantissa；按固定 H 的精确二次变化顺序接受并更新梯度（2680–2744）。这是编码层级上的离散候选，不是任意调 FP32 输出。

根项目早已研究位置风险、收缩与静态谱：[v348/v349:5–9][oldposition]、[v356:5–21][oldspectral]、[v358:21–23][oldmetric]，其中静态谱独立迁移和扩大训练集还有负结果。因此新增研究点应限定为“短 T 完整冻结 H + 收缩 + 双 token/lv3 组合”，不能说首次引入 V token 风险。根 v419 动态以当前 V 预测小代理 P（3468–3503、4628–4640），并已有相关残差修正（4692–4743）；可先在这个已有代理目标上接入 pair/lv3 候选，而不假装当前真实 P 可得。代理 P 下的 `H_ij` 可以由列内积获取，无需生成完整 T² Hessian。

迁移条件包括：冻结位置风险在相同 T 的不同内容中稳定、假对端校准可提供有用统计、编码合法、候选成本可接受。完整 FP16 H state 占 `2*kv_heads*T²` 字节（例如 T=4096、kv_heads=8 为 256 MiB，仅 H），参考校准还先构造完整 QK 概率；“动态不访问 Q/K”不等于省去了校准 T² 成本。建议先在短 T 固定编码/预算下比较根代理单坐标、根代理+pair、根代理+lv3、冻结 H 组合，并加入同长度位置置换/跨内容、跨长度回退与独立模型留出，记录单算子耗时和最差组变化。无需以未知真实 P 为输入，也不要用本地正增益外推线上。

## 6. 历史结果与可归属程度

| 候选 | 保存的证据 | 本次能确认什么 |
|---|---|---|
| C25 / 历史 25062 | [回执][r25]：任务 `2116367`，`25062.607247732445`，`180.922s`，两类 `All cases passed`。历史 commit `368e0da7cea5e5d7f91e34d46ddfd7c0c7f6d259`。 | 历史源码与 ZIP 成员 AST 均为 `e0c791252b09`，与当前 C25 一致；[TUNING:45][h25doc] 写的 ZIP SHA `61a6393c…` 与本次提取台账一致。该分数属于 C25，不属于 A-only B4/B5 或 QG5。回执没有上传 ZIP SHA 和 case 数，不能独立闭合平台上传物身份，不能证明当前 50+250 题或当前分别校准协议。 |
| L2 / 历史 23040 | [回执][r23]：任务 `2116309`，`23040.561152615846`，`144.846s`，两类 `All cases passed`。历史 commit `89237d56b05e97b4784ca4158bb9148d76d8ea14`。 | 历史源码/ZIP AST `287209de5c37` 与当前 L2 一致；[README:59–61][h23doc] 的 ZIP SHA `046db04d…` 与提取台账一致。是严格 A-only Linear 的历史旁证，不证明 Q/K 独立假对端校准时的共同 balance；同样缺 case 数和平台上传 SHA。 |
| B0 | [remote_result:2–7][b0result] 保存任务 `2109570`，`8750.829689385218`，`131.443s`。 | 当前多分支主入口其实退回 B0，不能把子目录历史高分写成主入口成绩。该文本亦缺平台上传 SHA。 |
| B5 | [TUNING:17–41][b5doc] 保存两组本地总分 `382.0838695705` / `355.2292107866`，并提及“先前线上 25062”。 | “先前线上”是 C25，不能转移给 B5。未找到 B5 对应的可闭合新线上回执；文档本地分数是作者保存的实验报告，本次未重跑。 |
| QG5 / QG20 | [QG5 TUNING][qgdoc] 本地 Qwen3 `359.690123`、Qwen2 `387.060780`；[QG20 SOURCE-TUNING:9–14][qgsource] 分别 `405.680598`、`433.958152`。两包 Attention 同为 `159.080264` / `187.439327`，Linear 不同。 | 本地数据集名带 `online` 也不等于线上判题。文档明确局部算子评测，本次没有新线上证据；synthetic 项还记录负差，不能宣称跨分布稳健。 |

保存回执的 Git 来源与包摘要见 [history-sources/high-score-artifacts.json](history-sources/high-score-artifacts.json) 和 [remote-records.json](history-sources/remote-records.json)。本次验证了保存源码的算法身份与包摘要记录一致；没有向平台查询历史上传内容或重跑任何结果。

题量补充（2026-09-07 用户直接确认）：其他审阅笔记中的同源码 16806→24201，源于线上数据集从 250 Linear＋50 Attention 调整为 50 Linear＋250 Attention；24201 属于调整后题量，不能将分数变化归功于代码。原始回执没有 case 数或平台上传 SHA 的事实不变。该确认不能推及本节 25062、23040 等其他任务的题量，也不能用于反推固定的 Linear / Attention 分项得分。

## 7. 开发优先级与剩余证据缺口

最有价值的三项具体知识是：① K quotient 目标及正确的跨 token 耦合接受式；② 可在已有合法代理风险上复用的双 token / lv3 离散逃逸；③ 有明确自由度约束的一层块 Givens 及其失败消融。三者都只建议拆出最小假设实验；已有代码、历史线上回执和本地消融不构成本项目新增版本提分承诺。

不能直接迁移的部分：依赖分别校准却必须相同的自适应 Q/K 共享变换；把位置专属冻结 H 当成当前真实 P；把未重新生成的基础 Q/K Hessian 当成最终部署联合最优；把回执高分挂到不同源码/ZIP；把 helper 存在、flag 为 0 的路径和未采用的中间计算当作有效算法。

本次没有恢复未知的假对端生成机制，没有验证完整状态大小/长序列峰值内存，没有运行隔离脚本，也没有重建每个本地消融的完整选择历史。最先应补的不是更大调参，而是符合实际三角色校准协议的 shared-transform 一致性检查，以及独立于变换/拓扑/步数选择的留出。runtime170 和历史 24628/24201 在其他审阅笔记中，不在本文件重复裁决。

[b0]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/legal-solutions/tensor-recon-baseline-v1/solution.py:266
[b4]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v4/paper-solutions/attention-calhessian-linear-inputonly-safe/solution.py:1707
[c25]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--new-paper-implementation/paper-solutions/attention-calhessian-linear-inputonly-safe/solution.py:1284
[b5]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-calhessian-linear-inputonly-safe/solution.py:1935
[qg5]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-quotient-givens-linear-inputonly-safe/solution.py:2948
[l2]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-tensor-recon-v2/legal-solutions/legal-tensor-recon-v2/solution.py:275
[qg20]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--new-paper-implementation/_archives/paper-solutions/target20-attention-affinefit-safe/submission-attention-quotient-givens.zip/solution.py:2901
[isolation]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-quotient-givens-linear-inputonly-safe/audit_attention_isolation.py:54
[rootkt]: /Users/ee/Code/2026-huawei-alogorithm/solution/solution_v419.py:4169
[oldposition]: /Users/ee/Code/2026-huawei-alogorithm/docs/v348-v349-report.md:5
[oldspectral]: /Users/ee/Code/2026-huawei-alogorithm/docs/v356-report.md:5
[oldmetric]: /Users/ee/Code/2026-huawei-alogorithm/docs/v358-report.md:21
[qgsource]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-quotient-givens-linear-inputonly-safe/SOURCE-TUNING.md:48
[qgdoc]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-quotient-givens-linear-inputonly-safe/TUNING.md:1
[b5doc]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/paper-solutions/attention-calhessian-linear-inputonly-safe/TUNING.md:17
[b0result]: /Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/legal-solutions/tensor-recon-baseline-v1/remote_result:2
[r25]: /Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/remote-368e0da7-paper-solutions_attention-calhessian-linear-inputonly-safe_remote_result_new.txt:1
[r23]: /Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/remote-89237d56-legal-solutions_zips_remote_result_legal-aonly-smooth-rht-v2-1024.txt:1
[h25doc]: /Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/368e0da7-attention-calhessian-linear-inputonly-safe/paper-solutions/attention-calhessian-linear-inputonly-safe/TUNING.md:45
[h23doc]: /Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/89237d56-aonly-smooth-rht-v2-1024/legal-solutions/legal-tensor-recon-v2/README.md:59

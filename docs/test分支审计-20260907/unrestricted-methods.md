# unrestricted 分支算法审计（2026-09-07）

本笔记覆盖 `experiment/unrestricted-highscore-v1` 与 `experiment/unrestricted-headadaptive-v2` 中的目标拟合、runtime170、headadaptive/PSOT 方案，以及对应 ZIP 内源码。结论来自只读源码、AST 差异和已有记录；没有运行候选或重测分数。最值得裁剪试验的是**校准分位数约束连续补偿的范数**与**围绕现有目标残差的搜索预算/批量化**。大部分 Linear 数学目标已在根项目 v419 中实现；依赖当前真实 Q/K 的 Attention 路径不满足根项目已知的独立角色信息条件。

## 范围与证据约定

- H：`/Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-highscore-v1`，分支快照 SHA `5d6c1b7b92148ac3685979cfd37ab2c3b62dc9c1`。
- A：`/Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-headadaptive-v2`，分支快照 SHA `849fd2408cfce5c0bdd8a27f1ea2b25e38ecf106`。
- R：`/Users/ee/Code/2026-huawei-alogorithm/solution/solution_v419.py`。根默认 `solution.py` 为 v325 内容，不能用“默认”代称 v419。
- 下文 `H/<路径>:行号`、`A/<路径>:行号` 都以以上绝对根展开；区间是本次文本快照的一基行号。
- AST 组编号来自 [algorithm-files.json](/Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/algorithm-files.json)，本笔记负责 **8–10、13–48，共 39 组**。这些是源码 AST 等价组，不是 39 种独立算法；重复函数和仅预算不同的变体合并分析。
- 父任务已核对远端 24 heads 和快照。图谱 generation `2026-09-07T02:35:31Z` 的已有覆盖不能代表新导出的 refs；本范围使用精确源码读取、AST 解析、源码/归档比较补证，未声称快照获图谱完整覆盖。
- A 里 legal/safe 系列由另一份笔记负责；H 的六个早期 auto-commit trial 归档为组 49–54，由父任务审阅。这里不重复其算法或将其混入本组结论。

## 先纠正版本与成绩绑定

H 的 `solution/solution.py` 是 block-Hessian 基线，AST `c752cdb37bb2…`；与 `H/_archives/solution.zip/solution.py`、`H/_archives/paper-solutions/zips/baseline-block-hessian.zip/solution.py` 相同。高分演进保存在 `paper-solutions/`，不在这个主入口。A 的主入口则已回到 tensor reconstruction 基线；也不能用它代表 headadaptive 方案。

必须保留的实质源码/ZIP 差异是 runtime170：

| 对象 | AST / 实际区别 |
|---|---|
| `H/paper-solutions/combo-runtime170-ood-guard/solution.py` | `dc3982309ad3…`，增加 `_restore_linear_param_shape`（1099–1106）、动态入口 flatten（1134–1135）和异常回退（1557–1583）。 |
| `H/_archives/paper-solutions/zips/combo-runtime170-ood-guard.zip/solution.py` | `d529e269ee99…`，没有上述形状恢复与异常回退；不是同一算法入口。 |
| `A/paper-solutions/runtime170-logitbasis-headadaptive-v2-36k/runtime_backend.py` 及 V1 同名文件 | 与前述 **ZIP** 的 AST 相同。A 的 Linear runtime 分支仍使用缺少该包装的版本。 |
| H 的 PSOT 最终源码、对应 `submission.zip`，A 的 `strong_attention_backend.py` | AST 均为 `83d176219a1c…`；字节 hash 不同不代表算法不同。 |
| A 的 V1/V2 单文件源码与各自 `submission.zip` | 各对 AST 相同；`solution_multifile.py` 是另一 AST，主要区别为导入外置 backend 而非内嵌源码。 |

H runtime170 的工作源码回退会删去 `weight_error`、`correction_matrix`、`correction_ratio_limit`、`compensation_strength`、`quantized_weight`，退到变换后的普通层级量化。它只捕获列举的四类异常，不能据此宣称所有故障已解决；也不能把工作源码的健壮性套到旧 ZIP 的回执上。

`36k` 只是目录名。`A/paper-solutions/runtime170-logitbasis-headadaptive-v2-36k/remote_result:1–6` 实际记录：任务 **2118290**，分数 **-882.4132813838542**，**225.323 秒**，`RuntimeError at solution.py:4054`、Linear case **0–49 Wrong Answer**、Attention all cases passed。4054 是最终 Linear backend 调用，回执没有更深层堆栈，不能归因到某个算子，也不能据 Attention passed 推断符合根项目角色协议。

该目录 README 的本地五数据集成绩只能作为参考仓库既有实验：Qwen3 8B `451.132965 = 249.183997 + 201.948968`，Qwen2.5 profiled `460.549856`，online 数据集 `461.982872`；“online 数据集”是本地数据集名。Qwen3.5 表把 160 Linear / 50 Attention 各归一至 250 再求和，不能与早期 210 case 原始 `170.x` 总分直接比较。本次没有把这些记录换算成本项目线上收益。

## Linear：从二次代理到真实目标残差

记每行激活为 `a∈R^I`，变换后权重 `W∈R^(O×I)`，已量化并反解的权重 `B`，`E=B−W`，HiF4 激活反解为 `q`。最终关心 `q B^T−a W^T`。

### 基础变换与小规模学习

早期独立候选并非都需要迁移：

- `awq-fast` 用逐通道缩放平衡权重和激活；`qqq-fast` 对超过中位数/MAD 门槛的高能量通道选择性缩放。分别见 `H/paper-solutions/awq-fast/solution.py:89–112`、`qqq-fast/solution.py:89–120`。`rrs`、`rht`、`let`、`rrs-let` 比较平衡、Hadamard 与确定性符号预处理；`let:58–108` 的 family 3/4/5 仅乘缩放，而 `rrs-let:58–79` 在缩放后再做 FWHT64。不能仅按名称推断实现了论文全部优化。
- `rrs-fast:666–871` 在常见多校准分支收窄成平衡 + FWHT64 + 增强半径，跳过广泛 family/route 试选；`rrs-fast-fp16` 进一步把二次统计状态存 FP16。`rrs-lite:666–846` 则删减了二次/加权求解，属于质量/成本交换，不是等价加速。
- `infoquant-rank1/rank2` 在 block64 激活二阶矩中抽取大特征方向，以 Householder 反射把尖峰摊到等幅符号方向。rank2 实现见 `H/paper-solutions/infoquant-rank2/solution.py:151–197`：阈值为 `sqrt(eigenvalue)/base_rms > 1.25*(1+sqrt(64/N))`，后续方向先应用已选反射。这里是经验有限样本门槛，没有统计无误判保证。`fit-weight-fast:735–1011` 对 identity/rank2 做有条件拟合选择；`per-layer-alpha:737–1021` 扩展缩放 alpha 搜索，但记录中全部 32 层仍选原 alpha，是应保留的负结果。
- `spin-lwc:594–689` 学习一个所有 block 共享的 64×64 Cayley 正交矩阵和 clipping，使用 STE 与六步梯度更新，目标是校准 Linear 输出误差。它不是全宽每块自由旋转。既有 Qwen 总分约 `82.419051`，相对 RRS 的收益很小、耗时约 77 秒，不支持作为当前优先移植项。
- `focus-dgs:32,381–435` 枚举一次性编码系数 `(0.84,0.92,1,1.08,1.16)`，每八值组独立选择，最终仍返回合法 HiF4 scale/lv/mantissa；`696–799` 只在权重量化加入 relaxed rounding，动态激活没有新增变换。其既有本地总分 `36.947472` 只说明该独立基线较弱，不能否定局部编码系数在其他父方案下的可能性。

根项目已有缩放、符号/Hadamard、排列、全宽 polar 变换和完整目标求解（R:1718–1758、1919–2046）；把上述名称作为“新增整套算法”会重复已有能力。可研究的差异是更便宜的校准路径能否替代一部分全宽分解成本，须单独比较质量损失。

### cross、projection 和 target-centered 搜索

`combo-cross-adaptive` 在块二次误差外加入由全输出权重误差导出的线性项，相关项来自 `a E^T B`，并通过校准选择较小系数；动态见 `H/paper-solutions/combo-cross-adaptive/solution.py:1104–1148`。`cross-refine4` 与 `cross-refine20:1104–1219` 增加按 `||R B_b||²` 排名的 64 通道块更新，以全行输出残差验收。

`combo-projection-100:792–1089,1092–1134` 将初始化改成固定权重推导的连续中心：

```
P = (B B^T + λI)^−1 B
δ = a E^T P
a_center = a − δ
```

它先抵消权重量化误差，再量化 `a_center`。`projection-100` 用 `λ=1e−3*mean(diag(BB^T))`；refine 后续常用 `1e−4`。`projection-refine4` 在这个中心后做四块离散修正，`projection-target-refine20:1093–1222` 则以连续中心为块候选生成中心，仍以 `R=aE^T+(q−a)B^T` 验收，不把“离中心更近”当最终目标。

`projection-target-group128` 的实际源码先做 **top20** 块（1162–1168），再做 top128 八通道组；不是 top4。八通道包含两个 quartet：lv2 为 1/2，每个 lv3 为 1/2，有八种层级编码，但其中两种产生同一对有效倍率。之后的 runtime 版本去重成七种 `(1,1),(2,1),(1,2),(2,2),(4,2),(2,4),(4,4)`。给定块 scale，枚举这些层级并按连续中心重舍入 mantissa；这不是穷尽每个八通道的所有合法 mantissa。

演进中的几个不同预算都在源码保留：target8/group132；target8/group136/batch4/scalar512；target0/group96/batch8/scalar1024。它们是相同目标的不同搜索预算、候选批量和更新次序，不能把局部连续上升的表格当作单调定理。根项目 R:1179–1519 已有完整目标残差、64 块修正、Newton、micro4 与全 Gram 验收；这里的增量主要是低成本八通道编码枚举与预算重排。

### runtime170：可借鉴的精简实现及门控边界

以下以 `A/paper-solutions/runtime170-logitbasis-headadaptive-v2-36k/runtime_backend.py` 为准（与 H runtime170 ZIP 相同）：

1. 多校准分支采用平衡 + FWHT64，`reflectors` 为空且 `fitted_ranks=(0,)`（823–838）；不能把保留的 PSOT 函数当作该 Linear 分支实际启用。
2. 保存 `B`、`E` 和连续补偿矩阵 FP16，动态转 FP32；二次统计也为 FP16（920–959）。运行时构造 `R=aE^T+(q−a)B^T`（1176–1180）。这是按冻结低精度 state 定义的 FP32 输出代理残差，不能自动等同官方 BF16 最终算子。
3. top64-block 的代码仍在，但 `topk(k=0)`（1181–1188），实际 **不执行**该阶段。
4. 八通道 top96（1241–1254），七种不同有效倍率编码（1271–1304），**96 组一起一批**（1311）；不是沿用旧目录名里的 batch8。候选先各组评分，再对其合并更新重算全行残差，只有合并方案降低残差才接收（1376–1416）。这比把独立组收益直接相加更稳妥，但整批拒绝也可能丢失其中有益子集。
5. 标量先取 top512，再从剩余通道补下一批最多512，保持原前缀；缓存对应权重列、范数、scale/mantissa，用 batch28 做 ±一个 mantissa step（1424–1542）。每批再次以更新后的全行残差验收，避免多个坐标独立估计遗漏交叉项。

R 已经使用二次/全输出目标门控，不能把“有门控”当新点。参考的工程价值是：候选层级去重、保持 top-k 前缀的预算扩展、重复元数据缓存、控制 batch 的同一目标验收。v419 的量化器也已延迟最终 mantissa 重构（R:2277–2342）；不能再把这一点列作根项目新增。

### 校准 95% 分位数的补偿范数限制：当前较明确的差异

runtime backend:929–941 对每个校准激活行计算 `r=||δ||/max(||a||,eps)`，冻结

```
ρ = max(1.5 * quantile(r, 0.95), 1e−4)
δ_guarded = δ * min(1, ρ*||a||/max(||δ||,eps))
```

动态只限制补偿向量（1125–1148），不裁掉原激活本身。根 R:1157–1175、2125–2135 只有 ridge 拟合的连续 projection 和启用条件；精确全文检索没有 `quantile` / `correction_ratio`，且读取该连续中心调用链，未见这个校准分位数范数限制。一般 ridge、最终 MSE cap 不是同一个机制。

这可以作为小改动候选，但**必须保留未限制的父结果**：缩小中心会丢掉原来有益的搜索区域，即使后续每步验收不回退，也不能保证最终比父方案好。建议把受限中心作为附加候选，与原中心在相同冻结目标上比较；若为省时替换父中心，必须承认其质量风险。校准与动态都应使用同一变换坐标和实际冻结精度。

已有证据仅到 `H/paper-solutions/combo-runtime170-ood-guard/README.md:30–48`：发布数据中 guard 从未触发，所以同分；另一个 power-iteration 诊断在 32 组中触发，证明能压制补偿放大。README 对 30 个线上 WA 的“投影不稳定”解释仍是假说，没有异常堆栈或成功回执确认；同分不能证明修复。

### 状态压缩与单校准 row-space：有形状条件

`H/paper-solutions/combo-runtime160-online-state-factor-fp16/solution.py:920–950` 不保存 `P∈R^(O×I)`，改为保存 `M=(BB^T+λI)^−1∈R^(O×O)` 的行尺度 `s_i=max_j|M_ij|`（FP32）和 `M_ij/s_i`（FP16），运行时还原后计算 `(aE^T M)B`（1124–1136）。这是动态范围保护下的因式分解，不是低秩或对角近似。对 O=256,I=2048，仅该补偿状态由 1,048,576 降为 132,096 字节；原始 FP16 inverse 溢出、对角化质量下降都在 README:34–46 记录。

它在 **O<I** 时才有通常的尺寸优势；对 O>I 会更大、更贵。根 v419 的 O<I 已走 direct-output 路径，O>I 存输入侧 I×I continuous projection，不能机械把本例“节省87.4%”套到根方案全部 state。需要先量实际 state 复制/恢复开销，再比较 `min(O,I)` 侧求解与编码精度；不是本轮最高优先级。

A 最终 Linear 分派见 `runtime170-logitbasis-headadaptive-v2-36k/solution.py:4028–4056`：**恰好一个**校准样本走内嵌 corrected R2QK6，其他数量（包括2）走 runtime。README 中“三个及以上”的描述不完整。

内嵌 corrected 源码在该文件 3964 行，已按 AST 常量只解码、未 exec。decoded AST 为 `31f92fe8ce86…`；decoded:835–838 用 W 的 thin SVD，FP16 存 `V`，把激活投影为 `a V V^T`。它仍输出原逻辑宽度 I，没有缩成 rank 再交评分器。零空间去除的精确输出等价依赖理想正交精度；FP16 的 `V` 不再精确正交。实际单校准路径 decoded:1112–1170 还拟合 ridge 补偿，并在 `(1,0.75,1.25)` 中选强度；**强度选择评分的是连续补偿后输出，没有在那个循环重新量化激活**。动态 decoded:1194–1227 执行投影、family 变换、补偿、一次层级量化，没有 runtime170 的迭代 target/group/scalar 搜索。因而主文件前面未生效的大段 Linear 函数不能冒充最终路径。

## Attention：PSOT、精确短序列搜索与 headadaptive 条件

### runtime K-Hessian 与 PSOT backend

`H/paper-solutions/combo-runtime-khessian-efficient/solution.py:2096–2459` 使用缓存中当前原始 Q 和已量化 Q，为 K 拟合原 QK logits；五个插值强度 `(0.5,0.75,1,1.25,1.5)`，以实际量化 Q 构造 K 的 64 块 Hessian，再用实际 softmax 概率误差挑路。坐标更新增量维护 logits。V 使用缓存当前 Q/K 的 P、量化后的 Pq，解 `min_Vq ||Pq Vq−P V||²`。前者门控的是概率误差，不是最终 PV 输出，不应宣称 K 更新一定改善 Attention 输出。

强 PSOT 路径以 `A/paper-solutions/runtime170-logitbasis-headadaptive-v2-36k/strong_attention_backend.py` 为准；它与 `H/paper-solutions/combo-unrestricted-attention-psot2-mincal2-kexact16-vhigh3-vectorized-vpostpair-fast-vexact4/solution.py` 的 AST 相同：

- 校准 1840–1896：对相应 Q/GQA K 先平衡 + FWHT64，把同一 KV head 服务的 Q 展开，与 K 拼在一起估计 block64 二阶矩；学习最多两个共享 Householder，并把 KV 反射按 GQA repeat 扩展至 Q。少于两个校准样本时反射全零。函数 158–204 的噪声边缘同上述 PSOT。它利用**真实配对 Q/K 共同统计**才有预量化等价性；根项目独立校准中的假对端会破坏这一前提。
- K 2678–2826：当序列长度 `S<=4`，对当前梯度选中的一个通道枚举每 token 的 `0/+step/−step` 共 `3^S` 组合，重复16轮；原方案是4轮。是所选通道的短序列联合搜索，不是 K 全局最优。原点组合保留，验收 softmax 概率 SSE。
- V 2899–3447：先投影并与 base 比较，再做行级更新、四轮短序列精确 mantissa 搜索；scale 候选偏移为 `[-3,-2,-1,0,1,2,3]`。最终 3232–3447 对每对 token 行、每个64通道块枚举二者 scale route 的组合，候选包括各自当前解（3280–3325）；用实际 Pq 设计矩阵下的 PV 残差选择，处理 token 间耦合。这个 V 阶段才是在固定 Q/K 下直接约束 Attention 输出代理。
- shared quantizer:422–633、固定 route:661–707 把 scale 候选轴向量化。已有 README 的随机/离群输入等价性、0.344×/0.277×微基准是本仓库历史实验结果；不能换算成本机或线上整个 pipeline 加速。

基础 H `combo-cross-*` 至 projection 系列的 Attention 仍为各自 state 的动态量化；current Q/K 缓存是 runtime160 以后高分路径的额外信息。不能因它们共享一个项目就给所有早期候选贴同一缓存标签。

### logit-basis / headadaptive

A V2 最终 dispatcher:4059–4094 选 `len(calib)==1 or head_dim>=256 -> logit`，否则 strong PSOT；V1 对否则分支选 runtime。V2 新增的 strong 不会作用于上述 logit 条件。Q/K/V 包装 4097–4209 清空旧 cache，并追踪同 backend 的 `Q→K→V` 顺序；它只减少误配旧缓存，**仍需跨接口当前对端和调用次序**。

logit 路径本身并非普通 K 量化。A V2:2348–2379 检查 `basis_width = GQA_repeat*S` 可整除 head_dim、为64倍数，然后用确定性的 Hadamard 行构造 Q 基底，重复 `head_dim/basis_width` 次，返回原 shape。K:2406–2437 读取缓存真实 Q 计算当前 P；2444–2568 用 token 相似性、冻结混合、谱 shrink 和 S=8 的 token 旋转改变目标概率；2602–2647 把 `log(P_mixed)` 编成该基底下的 K，并在各副本内做残差编码。

这里未扩大评分器的 logical width：`replica_count*basis_width=head_dim`，Q 用 `reshape(source.shape)`，K 拼接总宽度也不变。但**固定宽度不代表信息合规**。同 shape 的 Q 替换依靠后续 K 读取真实 Q 才能恢复目标 logits；根项目独立角色调用时条件不成立。

V:3414–3491 用缓存真实 Q/K 与缓存编码 Q/K 分别产生 P/Pq，以多个 ridge 求输出拟合；后续有 token 级 block-scale beam，其 3366–3399 在当前目标输出残差上与旧解比较。因此该分支的 token mixing/rotation 与 V 回补是一套依赖当前三者的设计，不能只搬 K 变换或只搬某个高分表格。

## 可用信息、布局和导入审计

39 个负责 AST 组逐一解析了所有 import；普通 backend/算法源码仅 `__future__`、`math`、`typing`、`torch`。A 单文件增加 `base64`、`gzip`、`types`，多文件版导入 `runtime_backend` / `strong_attention_backend`；未发现根 `imperial_edict.md` 明列禁用的 `os/pathlib/io/.../sys/builtins` 等模块。三个 payload 均只解码为文本再 AST 解析，其内部 import 仍仅前述四类：

| payload | 所在行 / 解码 AST | 作用 |
|---|---|---|
| `_CORRECTED_R2QK6_SOURCE` | A V2 `solution.py:3964` / `31f92fe8ce86…` | 另行读取单校准 Linear 生效代码。 |
| `_RUNTIME_BACKEND_PAYLOAD` | 4004 / `d529e269ee99…` | 与外置 runtime backend 相同，无需重读重复数学。 |
| `_STRONG_ATTENTION_BACKEND_PAYLOAD` | 4012 / `83d176219a1c…` | 与外置 strong backend 相同，无需重读重复数学。 |

包装会通过 `types.ModuleType` 和 `exec` 构造 backend。当前明列禁用模块清单没有仅凭这几个名称就否定它们的依据；这里没有执行该包装或检验官方静态审核。后续若抽取方法，应采用直接、可审查的函数，不必复制编码包装。当前规则允许 Linear state 保存 W 和在本 API 内计算 A@W；不能引用 2026-08-29 的历史 input-only 清单否定上述 Linear 目标。

本审阅在正常生效路径未发现通过额外 rank 维度扩展返回逻辑宽度；内部候选轴、basis 副本与原宽重排不等于向评分端增加码本。不过 shape/纯数据/不可变 state 的全面契约仍需要专门检查，不能把只读 AST/import 检查写成正式提交验证。current QK cache 及 joint QK calibration 与根项目信息边界的冲突已由可达代码确认，无需用禁用模块来间接判断。

## 对根项目的建议与可证伪条件

| 优先级 | 具体试验 | 与 v419 的差异 / 保留条件 | 可否由当前证据预计增益 |
|---|---|---|---|
| 1 | 在已有连续投影后增加校准 95% 补偿范数 guard 的**候选中心** | R 的 ridge projection 未见该机制；用相同坐标和冻结精度，保留原中心/父结果，按同一输出目标选。记录触发、拒绝、最终参数变化和开销；自然数据零触发不冒充成功。 | 尚不能可靠估分。参考仅证能限制构造方向的放大。 |
| 2 | 将现有 micro/块搜索裁剪为八通道七层级 + 有序标量批量试验 | R 已有目标/Gram门控；增量是预算、元数据缓存、候选去重与批量。需要对最终同目标门控，报告整批拒绝率，不能只计独立组预测收益。 | 可能有速度空间；不能保证加法叠加更快。 |
| 3 | 核查小输出维度时补偿状态的行缩放 FP16 因式存储 | 只在 O<I 且仍物化 O×I correction 时有尺寸理由；R 的 down direct-output 和 up I×I projection 使直接迁移空间有限。先测 state 拷贝/还原成本。 | 不套用参考 87.4% 至总state；通常低优先。 |
| 4 | 用便宜的平衡/FWHT/少量反射初始化，与全宽 polar 的校准成本作受控对照 | 目标是减少根已有全宽分解，必须保留同样的目标求解预算区分“便宜初始化”与“删搜索”；不能混合多个改动解释收益。 | 参考提供可行实现，未证能替代 R 的质量。 |

PSOT 与短序列 V pair/subtree 搜索作为研究机制保留，但先重建**仅当前自身输入 + 独立冻结 state**下的目标，不能移植真实 Q/K 缓存来复现高分。根项目已有 `v348-v349`、`v356`、`v358` 静态位置风险/谱研究与负结果，不能笼统称发现了 token 耦合；若未来重新评估，必须明确完整短 T token Hessian、shrink、pair/subtree 与旧方案的具体差异，并在根角色协议下验证。

## 39 个 AST 组的覆盖索引

下表每个源码名均省略 `H/paper-solutions/`、`A/paper-solutions/` 前缀；同名 ZIP 的位置与完整 SHA 见 manifest。A runtime 简写为 `runtime170-logitbasis-headadaptive-36k`，V2 简写为 `runtime170-logitbasis-headadaptive-v2-36k`。函数相同仅参数/预算差异的项已合并读取，不为每个 AST 重讲同一数学目标。

| 组 | 代表源码（均为 `solution.py`，另注明者除外） | 核读定位 / 方法差异 |
|---|---|---|
| 8 | A runtime / `runtime_backend.py` | cal795–1096、dynamic1099–1542；与 H runtime170 ZIP 相同。 |
| 9–10 | A runtime / `solution.py`、`solution_multifile.py` | 最终分派单文件4020–4175、多文件4009–4164；嵌入/外置打包差异。 |
| 13–14 | A V2 / `solution.py`、`solution_multifile.py` | 最终分派单文件4028–4209、多文件4010–4191；strong/backend 改选；payload另解码。 |
| 15 | A V2 / `strong_attention_backend.py`，亦 H 最终 PSOT 长名称方案 | PSOT158–204、cal1823–2227、K2317–2896、V2899–3447；quantizer422–633。 |
| 16 | H awq-fast | smooth89–112、cal664–869；缩放快速基线。 |
| 17 | H combo-cross-adaptive | cal792–1101、dynamic1104–1148；全局 cross 项。 |
| 18–19 | H combo-cross-refine20 / combo-cross-refine4 | dynamic1104–1219；相同方法，top20/top4。 |
| 20 | H combo-projection-100 | cal792–1089、dynamic1092–1134；连续输出投影。 |
| 21 | H combo-projection-refine4 | dynamic1093–1221；投影后top4。 |
| 22 | H combo-projection-target-group128 | dynamic1093–1366；top20 target block + top128 group。 |
| 23 | H combo-projection-target-refine20 | dynamic1093–1222；target-centered block 候选。 |
| 24 | H combo-projection-target0-group96-batch8-scalar1024 | dynamic1093–1533；删target block、组batch8、scalar1024。 |
| 25 | H combo-projection-target8-group132-fast | dynamic1093–1385；target8、group132、向量化候选。 |
| 26 | H combo-projection-target8-group136-batch4-scalar512 | dynamic1093–1519；group136/batch4、scalar512。 |
| 27 | H combo-runtime-khessian-efficient | dynamic1087–1517、K2096–2459；group96全批、scalar28、当前Q Hessian。 |
| 28 | H combo-runtime160-fixed-attn4 | cal795–1084、dynamic1087–1515、V2115–2302；Linear rank0、scalar batch48、V输出拟合。 |
| 29 | H combo-runtime160-online-state-factor-fp16 | cal920–950、dynamic1124–1136；行尺度FP16 inverse。 |
| 30 | H combo-runtime170-ood-guard | cal929–959、impl1109–1554、wrapper1557–1583；范数限制与源/ZIP差异。 |
| 31 | H combo-unrestricted-attention-kexact16-vhigh3-vexact4 | K2167–2742、V2745–3147；K16、V宽scale、V4。 |
| 32 | H combo-unrestricted-attention-kexact4-vexact4 | K2162–2737、V2740–3142；K4与V4。 |
| 33 | H combo-unrestricted-attention-vexact4 | K2162–2525、V2528–2930；只新增V4。 |
| 34 | H focus-dgs | coef32、quant381–435、cal696–779；编码系数基线。 |
| 35 | H infoquant-fit-weight-fast | cal735–1011；条件identity/rank2拟合。 |
| 36 | H infoquant-per-layer-alpha | cal737–1021；扩alpha，既有记录无自然选择收益。 |
| 37 | H infoquant-rank1 | reflector150–187、cal724–941；一反射。 |
| 38 | H infoquant-rank2 | reflector151–197、cal734–951；两反射。 |
| 39、44 | H let / rrs-let | family58–79、smooth82–108、cal660–828；是否追加FWHT。 |
| 40 | H qqq-fast | smooth89–120、cal672–882；outlier-selective缩放。 |
| 41 | H rht | signs58–68、family71–94、cal674–836；确定性符号Hadamard。 |
| 42–43 | H rrs-fast-fp16 / rrs-fast | cal666–871；同数学，状态精度不同。 |
| 45 | H rrs-lite | cal666–846；删二次/加权成本，有质量代价。 |
| 46 | H rrs | family58–86、smooth89–114、cal666–826；基线对照。 |
| 47 | H spin-lwc | learn594–689、cal803–1004；共享Cayley + STE clipping。 |
| 48 | H `solution/solution.py`（不是paper子目录） | cal659–819、dynamic822–848；block-Hessian基线，与两个基线ZIP相同。 |

重复 ZIP 不逐份深读；AST 等价已核。例外组30/8明确分别读取；组13的 corrected payload 是顶层 AST 分组不会自动展开的另一份算法内容，已单独解码和读取生效 Linear 路径。

## 补充：24628 / 24201 历史版本的实际迁移价值

本节仅审父任务已绑定回执的两份历史源码，不重排成绩。以下简称：

- S24628：`/Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/66e91152-smoothrot-19k-qwfit-final-safe/_archive_submission-qwfit-final/solution.py`。
- S24201：`/Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/history-sources/5505f64e-rrs-fast-fp16-online-refresh-safe/_archive_rrs-fast-fp16-online-refresh-safe/solution.py`。
- B17：`/Users/ee/Code/2026-huawei-alogorithm/docs/papers/hif4-public-references/test/solution/solution.py`，父任务确认的 base_1.7 / `52b4904` Attention Compensated V2；不是根 v419。

| 历史身份 | 源码核对和证据限制 |
|---|---|
| 2026-08-29 task2113961，24628.5938，191.831秒，All passed | S24628 归档字节 SHA `e4b2a0c7d666…`；与同历史目录 `paper-solutions/smoothrot-19k-qwfit-final-safe/solution.py` 仅换行不同，LF归一 SHA `710a31429b8c…`、AST `8a6a20fbeb38…`。回执未给 case 数。 |
| 2026-08-29 task2114038，24201.5860，204.227秒，All passed | S24201 归档字节 SHA `caa4b54d9e5f…`；LF归一后等于源码 SHA `3140ce9bc5d5…`，AST为前述组42 `71dd190f02a1…`。与8月27日 task2110910 的16806.4497是同一源码。2026-09-07 用户直接确认：分数变化源于线上数据集由 250 Linear＋50 Attention 调整为 50 Linear＋250 Attention，24201 属于调整后题量，**不能将分数跃升归功于代码**。原始回执未给 case 数和平台上传 SHA。 |

以上题量来自用户直接确认，区别于回执原文。它不证明 24628 或其他历史任务使用了同一配比，也不允许从 16806、24201 两个总分反推出固定的 Linear / Attention 分项得分；源码与实际平台上传物的绑定仍受缺少上传 SHA 的证据限制。

**24628 的有效增量是小规模学习缩放与部署后 offset 窗口选择，动态投影并没有生效。** S24628:864–966 在16通道组上做两步近似梯度/Adam式缩放优化，前3个校准样本用于训练（1214–1218），其余样本用于接受门槛（1250–1259）；有非零观测可靠性屏蔽和每64通道的 log-scale 去均值。固定/学习缩放都接 FWHT64 和最多一个 PSOT 反射。`1265` 的“Skip PSOT”注释过时：此前1174–1183、1230–1238已计算并施加反射，返回state1383仍保存它。

这里的 holdout 有明确数量前提：多校准分支仅检查 `len>=2`（1122），但 `validation_gains=learned_gains[3:]`（1251）。当校准数为2或3时该列表为空，最终返回state1390的除法和1393的 `min` 无法成立；这是静态可见的缺口，不能把历史 All passed 扩展为任意合法校准数量均可用。迁移时需独立确定小样本策略，而不是直接复制门槛。

“final QW fit”只有部分符合最终部署条件：最终权重的半径为12（14、1282–1290），反解 `weight_value=B` 后重算列能量importance（1291–1293），并使用真实最终 B 对三个动态 scale-offset 窗口的量化激活输出评分（1344–1371）。但学习缩放的 `_probe_linear_losses:845–861` 用的是便宜 radius1 的 W/A 量化；其接受结论没有在最终 radius12 权重上重做“固定缩放与学习缩放”的完整比较。因此值得保留的是**用部署 B 刷新后续状态/候选目标的原则**，而不是将现有 learned gate 视为部署后的严格验收。

S24628:1295–1340 虽计算 correction matrix、补偿强度和 output-error importance，实际 `blended_importance` 只取 `dynamic_importance`（1341–1343）；state1372–1395没有保存 correction/error/strength，`quadratic=None`。单校准分支1515–1571也计算补偿却未存入state。动态1592–1629只做平衡/FWHT/反射和层级量化，不做投影、输出目标迭代或全输出 Gram 门控；`_global_weight_sketch:1015–1029` 也没有调用。不要把残留代码注释或辅助函数列成其高分实际机制。现存收益最可能值得重新检验的部分是**两步小参数缩放学习的条件选择**，而且要改成最终 B 和实际冻结精度下验收；根 v419 的更完整目标求解不是由该版本新增。

**24201 的 `fp16/refresh` 名称不能替代逐状态核对。** S24201:691–718 是平衡 + FWHT64 后的权重量化；719–721的importance确实由最终 B 列能量计算，但722的动态 block二阶矩仍来自未量化的 `family_weight=W'`，随后730转FP16。它的目标近似 `e^T H(W')e`，没有权重量化误差的交叉项，也不等于 `||q B^T−a W'^T||²`。因此它可作为便宜的块二阶代理对照，不是完整“最终Q(W)刷新”新方案。B17 的常见多校准路径已经在1206–1237用最终 B构造importance、block二阶矩和辅助/output state；R:1093–1135直接从返回weight_params的反解 B构造block/micro Gram及逆，覆盖得更完整。B17单校准路径的辅助/output state也用最终选中的B（1379–1413），但generic importance仍沿用候选1（1311–1313、1389），不能把B17概括成每个代理都刷新。

两份历史源码的缩放、FWHT和反射都以FP32操作/保存，S24628甚至没有 `to(float16)`；S24201的FP16是**二次统计压缩，不是FP16变换矩阵**。FP16舍入会改变代理二次型，不能宣称严格保持量化排序。与此不同，B17:1130–1142在适用shape下将QR得到的row-space基底转为FP16，再用 `aVV^T`；动态1832–1834用相同冻结基底，但 `V^TV` 不再精确为I，理想零空间投影的输出等价有数值误差。若把历史学习缩放/反射也压成FP16，需用实际解冻后的变换重建 W/A 变换结果、最终B和候选评分；仅刷新B的Gram无法自动修复变换互逆误差。R的连续polar/目标路径也应以实际返回与冻结数据为比较对象，不能用数学等价口号跳过舍入。

两者均**没有当前Q/K跨接口缓存**：S24628:2246–2279、S24201:1396–1429直接调用只读自身输入/state的动态函数。但两者校准都读真实配对Q/K/V，并据最终Attention输出选择bundle、balance、permutation及共享transform（S24628:1775–2126，尤其1833只取第一校准样本作route比较；S24201:1038–1342）。根项目独立假对端校准时，各角色可能选择不匹配的变换，故历史passed不证明此条件下可直接迁移。S24628动态K的mean/midrange两个中心候选（2210–2237）只用自身K、在理想softmax下有平移等价性，但选择指标仍是K重构误差，不是最终PV输出门控。

本节结论：24628可抽取**有限参数缩放学习 + 最终部署量化器复核**作为有界试验；24201主要补充历史基线和性能对照价值。二者都不支持替换现有B17/v419整个实现，也没有证据将历史总分差当作本项目可获得的增量。

## 解码后源码留档

已将 `_CORRECTED_R2QK6_SOURCE` 的原始解码字节保存为 [corrected_r2qk6_linear.py](/Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/decoded-backends/corrected_r2qk6_linear.py)，SHA256 `6ce14e81a49e8d9e5cbd70bb5e57383c360a96df0999fff2aaa43c146443d87b`，与前文 decoded 行号对应。完整来源变量、外层行号、解码字节SHA、AST SHA和另两个payload的外置等价映射保存在 [decoded-backends/manifest.json](/Users/ee/Code/2026-huawei-alogorithm/docs/test分支审计-20260907/decoded-backends/manifest.json)。过程只做 AST Literal提取、base64/gzip解码和文本写入，未导入或执行这些源码。

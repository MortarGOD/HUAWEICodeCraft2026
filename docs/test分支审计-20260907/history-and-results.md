# Mrxxx7/test：全分支历史实现与回执来源审计

审计日：2026-09-07。范围：已获取的 24 个远端分支及其可达 Git 历史；静态读取 Git 对象、源码、JSON 和 ZIP 成员。未执行候选、评测或提交。这里的“线上回执”均指仓库保存的文本，不表示本次向官方系统核验过。

本轮将前轮只覆盖 `base_1.7` 当前系列的约 18700 锚点，扩展为全分支历史成绩追溯：历史里确有 **25062.6072、24628.5938、24201.5860、23040.5612** 的带任务号、运行秒数、两类 All passed 文本，并可恢复同一记录提交里的源码和 ZIP。它们比目录标题或 README 估算更有出处，但尚不能证明优于当前 v419：原始回执未写分项题数，也没有官方上传散列把任务号绑定到 ZIP。**2026-09-07 用户直接确认，同源码 16806→24201 源于线上数据集从 250 Linear＋50 Attention 调整为 50 Linear＋250 Attention**；因此 24201 属于调整后题量，分数变化不归功于源码改进。该确认不能扩展为其他历史任务的题量或当前校准协议已得到证明。

## 1. 去重与版本恢复

24 个分支的 `solution/solution.py` 实际只有 3 个 AST：22 个是 `f13c814558c8…`，`base_1.7` 是 `8c07a5fce4c1…`，`experiment/unrestricted-highscore-v1` 是 `c752cdb37bb2…`。这里采用 `ast.dump(ast.parse(text), include_attributes=False)`，忽略格式与注释，保留函数体和常量。完整 HEAD SHA、源码 SHA、AST SHA 与逐分支主文件历史见 [branch-main-provenance.json](history-sources/branch-main-provenance.json)。主文件相同不意味着各分支附带的 `paper-solutions/`、`legal-solutions/` 也相同；这些候选由另两份方法报告逐项审计。

大量旧分支在 8 月 29 日执行 `Remove failed and constraint-violating solutions`，覆盖主解并删除历史候选；另一些从 `30a2f78` 起把主解切换到共同 RHT 基线。清理说明的历史原文见 [7fa03a5 的清单](history-sources/7fa03a5/doc/违规实现清单.md)。这是解释版本被覆盖的证据，不能把当时的作者合法性判断替代当前接口事实。

下面把分支名称映射到恢复出的实际历史实现。`R` 表示当前主文件属于 22 个共同 AST；“末次”是本次识别的最后实质算法阶段，不包含后续文档、回执、归档或清理提交。历史日志见 [branch-history.json](history-sources/branch-history.json)，其筛选日志是辅助索引，源码版本链以 [main-source-history.json](history-sources/main-source-history.json) 为准。

| 分支 | 当前主文件 | 恢复出的实质历史／归属 | 结果证据边界 |
|---|---|---|---|
| `base3.07` | R | `0817734` 主解混合尺度与 Attention 等价变换；还保留根目录备份 | 名字 3.07 不是回执分数 |
| `base_1.7` | 独立 | `52b4904` Q 误差补偿到 K、长序列 V；此前 `fc87886` 联合 V 重建 | 当前系列另由主报告对齐 18700；本地 500/500 不等于线上 |
| `experiment/attention-fisher-objective` | R | 清理前主解为 `66b754d`，Q/K 能量重要性、平衡、置换、输出损失选路 | 未在清理前候选树确认独立 Fisher 实现；不按名字另计 |
| `experiment/attention-learned-rotation` | R | `a75bca8` 每 KV head 的 16 通道协方差特征向量×Hadamard | 本地留出集 Attention 下降，明确归档拒绝 |
| `experiment/block-hessian-objective` | R | `9f7745d` 64×64 二阶矩收缩、块二次型量化损失 | 恢复了代码；无专属可绑定的线上增益回执 |
| `experiment/joint-output-optimization` | R | 清理前主解同 `9f7745d` | 清理前候选树未发现独立于该主解的同名实现 |
| `experiment/legal-inputisolated-v4` | R | `e09c83c` A-only Smooth-RHT；继承 `03147f9/3bc375d` input-isolated 候选 | 其历史也包含 24201 回执，不等于 V4 候选得分 |
| `experiment/legal-inputisolated-v5` | R | `39ca2a0` strict quotient-Givens 候选 | 3.6w 是候选名；没有据此成立的 36000 回执 |
| `experiment/legal-tensor-recon-v2` | R | `ca688b5` A-only Smooth-RHT，`89237d56` 保存 23040 回执 | 同提交 ZIP 与源码逐字节相同；当前角色隔离另审 |
| `experiment/linear-residual-fit-v2` | R | `6a9eeba`按量化后误差选择残差补偿强度 | 本地 500/500；251.9 秒是估算，不是线上实测 |
| `experiment/new-paper-implementation` | R | 历史 `368e0da7` 校准 Hessian＋Linear input-only；末端 `4c42e35` quotient-Givens ZIP | 25062 属于前述历史提交；不能贴给末端候选 |
| `experiment/non-rrs-paper-baselines` | R | `e5faf21` 等 AWQ/FOCUS/QQQ/PSOT 扩展，到 `4c5c97e` guarded full-residual output basis | 多个候选有 WA；见原回执表，不按前缀合并成成功方法 |
| `experiment/output-aware-rounding` | R | 清理前主解同 `66b754d`；`901116a` 合并到 channel-balance | 清理前候选树无额外独立同名主解 |
| `experiment/paper-solution-suite` | R | `0cbdba4→a3f6436` LET/RHT/RRS/RRS-LET，RRS-fast/FP16/lite、Spin-LWC | README 表为本地 20/210 case；不能作为线上排名 |
| `experiment/pca-hadamard-attention` | R | 清理前主解同 `66b754d`，其变换是能量排序，不是代码中的 PCA 特征分解 | 真正协方差特征分解见 learned-rotation 的 `a75bca8` |
| `experiment/qwen-channel-balance` | R | `66b754d`；与 output-aware 共用 `901116a` 合并历史 | 不能数成另一份独立算法证据 |
| `experiment/smoothrot-19k` | R | `6fd5509→c513611→741c145` 纯 quantile、手工 STE、Q(W)-fit | `66e91152` 记录 24628；19k 不是最终分数 |
| `experiment/smoothrot-32k` | R | 同一末次 Q(W)-fit 算法提交 `741c145` | 没有据分支名字成立的 32000 回执 |
| `experiment/smoothrot-32k-v2` | R | `8208886` 向量化 Q(W) Hessian 激活坐标更新 | 本地 315.6422 / 56.08 秒；其继承的 24201 属于另一候选 |
| `experiment/soar-hif4` | R | SOAR 候选在 `972abc6`；后续 `734323c` 已转向 strict input-only fast | SOAR 只有很小本地增益，文档说未提交 |
| `experiment/unrestricted-headadaptive-v2` | R | `13d4a18/980209e` head-adaptive 与形状后端组合 | `849fd240` 的“36k”候选实际 -882.4133，RuntimeError/Linear WA |
| `experiment/unrestricted-highscore-v1` | `9f7745d` 旧主解 AST | 附带 `4caa5e8→78f89ac→87a2514→5d6c1b7` exact attention、V 搜索、PSOT | 重点算法在附带目录，不在未更新的主文件；见另一方法报告 |
| `feature/hif4-solution` | R | 清理前 `0817734` 主解 | 共用历史，不能另算成绩 |
| `main` | R | 清理前 `0817734`，此后清理与根目录备份 | 8750 基线回执被多分支继承，不是多次复现 |

上述四个“名称并未对应独立实现”的判断限定在：该分支主解变更链，以及清理提交第一父节点中名称含 solution/fisher/pca 的 Python 候选树。它们当时仅有 baseline、profiling baseline 和 `solution/solution.py`。这不是对未发布、未提交或已不可达对象的存在性判断。

## 2. 高分回执、代码和 ZIP 的绑定程度

日期均为**保存回执的 Git 提交日期**（+08），不是官方任务完成时间；原文没有官方日期。四条成功记录都只说 Linear/Attn All cases passed，**原文未写分项题数**。题量另有用户于 2026-09-07 的直接确认：16806 对应调整前 250 Linear＋50 Attention，24201 对应调整后 50 Linear＋250 Attention。其余历史任务不能据此填写题量；也不能仅按 300 题总量比较，或从这两个总分反推出固定的 Linear、Attention 分项得分。完整原文和元数据见 [remote-records.json](history-sources/remote-records.json)，全量表见 [remote-record-table.md](history-sources/remote-record-table.md)。

| Git 记录时间 | 任务 ID | 分数 | 秒 | 记录提交／候选 | 当前能确认 |
|---|---:|---:|---:|---|---|
| 08-31 11:39 | 2116367 | 25062.607247732445 | 180.922 | `368e0da7` / attention-calhessian-linear-inputonly-safe | 两类 All passed；同提交代码/ZIP/TUNING SHA 对上 |
| 08-29 10:12 | 2113961 | 24628.593822132687 | 191.831 | `66e91152` / smoothrot-19k-qwfit-final-safe | 两类 All passed；同提交代码/ZIP/README SHA 对上 |
| 08-29 12:36 | 2114038 | 24201.58599981876 | 204.227 | `5505f64e` / rrs-fast-fp16-online-refresh-safe | 两类 All passed；同源码；50 Linear＋250 Attention 由 2026-09-07 用户确认 |
| 08-31 10:00 | 2116309 | 23040.561152615846 | 144.846 | `89237d56` / legal-aonly-smooth-rht-v2-1024 | 两类 All passed；源文件与 ZIP 成员逐字节相同 |
| 08-25（详见元数据） | 2107805 | 23328.74556016715 | 272.813 | `4a91b906` / baseline-r2qk6-v-output-ridgels-fast-safe | 同任务另存到数据集目录，不是独立重复实验 |
| 09-01 09:32 | 2118290 | **-882.4132813838542** | 225.323 | `849fd240` / runtime170-logitbasis-headadaptive-v2-36k | RuntimeError at solution.py:4054；Linear case 0–49 WA，Attention All passed |

`7c82acbc` 的 `remote_result——legal-aonly-smooth-rht-v2-1024.zip` 实际是文本文件，不是 ZIP；下一提交将它改名为 `remote_result:legal-aonly-smooth-rht-v2-1024`。两份任务号同为 2116309，不能计两次成绩。`bdcb9187` 的 23328 / 244 秒没有任务号，精度与 2107805 的 23328.7456 / 272.813 秒不同，也不能强行当成同一次运行。

四个归档的 SHA-256 如下。前三个仓库源文件与 ZIP 成员的字节散列不同，但逐行内容与 AST 相同，差异仅为换行表示；第四个字节完全相同。完整源和压缩包散列见 [high-score-artifacts.json](history-sources/high-score-artifacts.json)。

| 候选分数 | 同提交 ZIP SHA-256 | ZIP 中 solution.py SHA-256 | 恢复的原始 ZIP 成员 |
|---:|---|---|---|
| 25062 | `61a6393c548899fa7de11aee60c6abb2154474d7337d885ed7eb70863b52991e` | `7ec5e2998df9bf2a5cbdc404a5506173cf192e50b857220251cc1f2d0e6b6fcb` | [源码](history-sources/368e0da7-attention-calhessian-linear-inputonly-safe/_archive_submission/solution.py) |
| 24628 | `e8ea09af5c67d9c33ca5082e841c3199360396dbebd082ab9a8825c7a6863514` | `e4b2a0c7d66639827ec6d153f9e385a3c0beb00fe45fab5c7cdf88669ebf8144` | [源码](history-sources/66e91152-smoothrot-19k-qwfit-final-safe/_archive_submission-qwfit-final/solution.py) |
| 24201 | `6d285f6e56bad230fba20b15cf044f985a274e3b0d1dc33a1451d810518f5dec` | `caa4b54d9e5fbb26c67fcdf3ec7d843dc8ec5149a217325ef2fb3c5d261821f7` | [源码](history-sources/5505f64e-rrs-fast-fp16-online-refresh-safe/_archive_rrs-fast-fp16-online-refresh-safe/solution.py) |
| 23040 | `046db04d817ad0efc0511656e57a2c344cc0aadaf790d63a54532190aa603db9` | `144eae003c3b243955fffddf70531480db1680a7ab51a37ce1fb079755ea1061` | [源码](history-sources/89237d56-aonly-smooth-rht-v2-1024/_archive_legal-aonly-smooth-rht-v2-1024/solution.py) |

这建立的是“回执文本与提交档案在 Git 中共存、README/TUNING 指定归档散列、源和 ZIP 内容一致”的链路。缺少官方任务 ID→上传 ZIP 散列的证据，故不能写成已核实该任务执行了这份 ZIP。safe、input-only、legal 只是作者命名，当前接口兼容性仍要读代码。

## 3. 不能横向换算的原因

最直接的对照是 [source-comparisons.json](history-sources/source-comparisons.json)：`44d08fb9` 与 `5505f64e` 的 `paper-solutions/rrs-fast-fp16-online-refresh-safe/solution.py` 均为 SHA-256 `3140ce9bc5d52a9e65d84601d82460ab50cde5c10ccb10b5a0cfc0201ec8723e`。前者任务 2110910 得 16806.44973116135 / 218.617 秒，后者任务 2114038 得 24201.58599981876 / 204.227 秒。**根据 2026-09-07 用户直接确认，这次分数变化的原因是线上数据集由 250 Linear＋50 Attention 调整为 50 Linear＋250 Attention**；前者属旧题量，后者属新题量，不能把跃迁解释为源码改进。原始回执没有 case 数和上传 SHA 的证据事实不变；该确认不提供固定分项得分，也不应据此反解或外推其他历史任务的题量。

`4ae604f` 在 8 月 31 日 12:32 保存的 [doc/最新公告](history-sources/4ae604f/doc/最新公告) 第 1、4 行明确写下调 Linear 占比、超时改 5 分钟、要求重新提交获取最新评分；没有给出新旧题数。这份文件自身也是仓库存档，保存日期不证明公告何时生效。25062/23040 的记录提交时间甚至早于此公告入库，不能按入库时间机械划分“新规则”。失败记录列出 Linear 0–49 只能证明它报告了这 50 个失败编号，不能证明所有其他历史成功任务的完整题数。

私有 judge 的 500/500、Linear 248.8174 / Attention 212.9853 也不可拿来与我们 300 题、v419 21258 / 259 秒横比：

- `test/tests/judge_src/fourbit_judge/evaluator.py:237` 一次把配对 Q/K/V 校准给一个 calibration 调用，随后同模块依次调用 Q→K→V；当前线上事实是三次各真一项、另两项为假，只有自己的 state。数据可见性不同。
- 500 case 版本是 250 Linear＋250 Attention；每个 Attention case 是整个输出，不是把 Q、K、V 各算一题。Qwen3.5 的 160+50 则在私有 judge 内按各分项 reference_cases=250 归一。这只是私有评分定义。
- `test/tests/judge_src/fourbit_judge/references/hierarchical_v1.py:122` 的基线在 `:159` 枚举邻域 E6M2 top scales、`:167–185` 选择最小 SSE，同时搜索 lv2/lv3。它与根项目峰值 Algorithm1 基线不同，**连 improvement 分母也不一样**，不是按题量乘一个常数就能修复。
- profiled-v2 的四 token trace 是本地代理数据；拼接已有 trace 构造的长序列不等于真实连续序列。文档自己把长序列实验列为诊断，而非准确线上预测。
- 本地 worker 耗时、README 按 244 秒比例外推耗时、官方回执耗时应分开。相同 500/500 只表示通过本地有效性检查，不证明正收益或线上兼容。

## 4. 恢复历史里真正值得学习的少数方法

以下是代码与历史实验形成的可借鉴点，不是承诺提分，也不意味着可以照搬其状态与跨接口数据流。

| 恢复的机制 | 代码或文档证据（文件内行号） | 历史结果／可迁移边界 |
|---|---|---|
| **误差二次型加入收缩** | [9f7745d 主解](history-sources/9f7745d/solution/solution.py)：272–295 块二次型；548–588 用二阶矩估计噪声，收缩非对角；799–811 把统计放入 activation_state | 少量校准时把 noisy full Hessian 收缩到对角，是可以单独证伪的想法。是否允许统计进入动态接口须按当前事实判断，不能因为旧代码能传就假定今天能传 |
| **等价变换与输出损失选路** | [66b754d 主解](history-sources/66b754d/solution/solution.py)：82–107 Linear RMS 平衡；737–814 Q/K 共同能量、置换、平衡；888–900 同时读取真 Q/K/V；919–921 计算参考 Attention 输出 | 数学上的互逆 scale、共同正交变换不难成立；实际双方能否一致选择才是关键。三次假数据分离校准不能共享真实 Q/K 能量，也不能假设能先分别估计再合并。只可固定双方都能推导的共同规则，或另行证明状态选择一致 |
| **协方差旋转必须看留出集** | [learned-rotation 源码](history-sources/a75bca8/paper-solutions/baseline-r2qk6-attention-headrot-safe/solution.py)：1362–1395 使用 Q/K 平均协方差特征向量×Hadamard；[README](history-sources/a75bca8/paper-solutions/baseline-r2qk6-attention-headrot-safe/README.md) | 校准只接受 1/10 group，但留出 Attention 从 44.3483 降到 42.8226，Linear 不变，作者明确拒绝。不要把 PCA/Hadamard 名称当正向先验；当前还缺共同真实统计 |
| **量化后再选择补偿强度** | [residual-fit 源码](history-sources/6a9eeba/paper-solutions/baseline-r2qk6-linear-residual-fit-v2-safe/solution.py)：917 起补偿搜索、983 起重新量化候选、1010 起 ≥8% mean gain 门控；[README](history-sources/6a9eeba/paper-solutions/baseline-r2qk6-linear-residual-fit-v2-safe/README.md) | 本地 profiled-v2 总分 385.3351，94.54 秒；251.9 秒是比例估算。可借鉴“通过合法量化后的真实损失选参数”，但其 W 派生残差/动态状态不能不经审查搬过来 |
| **SOAR 风格编码尺度与解码尺度解耦** | [SOAR README](history-sources/972abc6/paper-solutions/rrs-fast-fp16-soar-safe/README.md) 与[源码](history-sources/972abc6/paper-solutions/rrs-fast-fp16-soar-safe/solution.py) | 编码端尺度可连续搜索，输出仍回到合法 E6M2 HiF4 解码。记录的本地 Qwen2.5 增益仅 +0.0341，Qwen3.5 +0.006；增加耗时，作者未提交。可学习搜索方法，不可说 SOAR 已证明有效 |
| **SmoothRot 并非直接照抄论文参数** | [纯版本](history-sources/6fd5509/paper-solutions/smoothrot-paper-n3-pure/README.md)、[调整版](history-sources/c513611/paper-solutions/smoothrot-paper-n3-pure/README.md)、[手工 STE](history-sources/c513611/paper-solutions/smoothrot-learned-scale-strong-manual-ste-safe/README.md) | alpha=.475 / q=.994 版本本地严重负向；调到 alpha=.1 / q=.977 仍弱。手工 STE 版本有 16219 / 259.144 秒回执。论文方法必须适配 HiF4 固定层级，不能用方法名字替代实测 |
| **Q(W)-fit 与计算量约束** | [741c145 README](history-sources/741c145/paper-solutions/smoothrot-19k-qwfit-final-safe/README.md)；[8208886 源码](history-sources/8208886/paper-solutions/rrs-fast-fp16-linear-qwhessian-actcd1-vector-fast-safe/solution.py)：520 起块候选 refinement，1026 由量化 W 构造动态二次型；[TUNING](history-sources/8208886/paper-solutions/rrs-fast-fp16-linear-qwhessian-actcd1-vector-fast-safe/TUNING.md) | 局部向量化坐标更新本地 56.08 秒，对照更完整 CD 119.98 秒：可学习“一次计算多个合法候选再更新一个块”。其 Q(W) 统计是否允许出现在动态 A 中由当前信息边界决定；局部结果不可归给 24201 回执 |

23040 的 [README](history-sources/89237d56-aonly-smooth-rht-v2-1024/legal-solutions/legal-tensor-recon-v2/README.md):9–22 已清楚暴露一个容易漏掉的区别：Linear scale 是 A-only，但 Attention scale 仍由 Q/K 两者的 RMS 共同生成；`:33–34` 也承认 state 为 Q/K 派生。这不足以满足今天三次各自真/假分离后的共同变换一致性。其 `safe` 不是我们当前接口的证明。

后续开发可以先做两项有明确反例标准的实验设计，当前审计没有运行：

1. 对同一合法候选，在完全固定源码、输入、评分分母下比较对角损失与收缩块二次型；报告校准/留出损失、每种长度、累计耗时。若优势只在配对校准或本地代理集出现，则停止把它解释为线上可迁移增益。
2. 对所有需要 Q/K 共同 scale、rotation 或路由的方案，先按真实线上三次校准模式独立生成三个 state，直接检查 Q、K 对应变换一致性。假数据分布未知时，依赖假 Q/K 的选择没有普遍一致性保证；无法证明一致就不进入提分比较。固定公共变换是可测的对照。

## 5. 原始记录的保留与解释

在全部可达历史中按 `*remote_result*` 的新增/修改路径恢复了 36 个不同文本内容（其中有占位、TBD、无任务号、同任务不同换行和报错）；明确任务号共 25 个。这是路径限定的回执检索范围，不宣称覆盖图片、聊天或站外页面上的成绩。元数据保存了每条记录所在提交的完整 SHA、记录日期、原路径、文本 SHA 及包含该记录的远端分支。

不要只摘成功数字：历史中有 19683、18497、17879、17721、17700、14824 等带 Linear WA 的记录；`combo-refine20-stepgain-batch4-fixed-attn4-safe` 的 13803 / 294.695 秒有 Attention case 0–49 WA；低秩 Hessian 某版本原文只写 timeout。全部保留于 [回执表](history-sources/remote-record-table.md)。

8 月 19 日的自动调参 JSON 是旧试验档案；清理说明明确保留 JSON 供历史参考、删除旧提交包。它们没有随着 22 个同主解分支重新运行，不能把残留 JSON 分数贴到当前 HEAD。`solution_backups/2103477.txt` 和 `solution_backups/2103499` 实际是 Python 代码备份，数字文件名不构成带任务号的回执。

可确认的结论是：历史中存在可追溯程度较高、数值高于 18700 的成功文本回执；当前可复现、同规则、同接口的最佳算法仍需靠同条件验证判定。没有证据支持“32k/36k 分支已经达到标题分数”，也没有证据支持私有 500/500 可以换算成当前线上分数。

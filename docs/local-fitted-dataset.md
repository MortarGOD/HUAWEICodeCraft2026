# 拟合线上消融结果的本地数据集

> **2026-09-03 反馈反演修复**：新增 `tools/fit_attention_feedback.py` 与独立种子 bank 生成/验证流程；不再把 Linear 的步长直接视为 Attention 的已知长度。冻结的共享分布权重须同时解释 GPTQ、Q/K refinement、V mean 和 V polish，并通过另外四条算法边。新 bank 的最终资格以 manifest 的 `score_prediction_allowed` 为准，未通过不能替换成新的精确线上预估器。完整实验和限制见 [反演报告](attention-feedback-identification.md)。旧 covariance_v1 保留，Linear 不变。

> **本轮最终状态**：`local_fitted_models_attention_feedback_v2/` 已生成 50 组/250 道，10 版回归、860/860 格式检查及 8 项单测完成。新种子 GPTQ +1929.54、Q/K +108.00、V mean +334.01、polish −5635.89；8 条边仅 4 条满足容差，绝对量级审计也未通过。因此 `score_prediction_allowed=false`，仅作补充诊断，不单独驱动精确线上预估或删除既有有效策略。

> 用户现已确认v311/v312全部用例PASS。代理中polish大幅正向而线上−4995，是实质性的精度方向失配，不是格式/罚分混入。v314全量原始增量+5.526476仍不可直接映射成21200+；固定bank继续保留作机制回归，不修改其权重去制造这个目标。

> **最新隔离反馈**：Q/K增强+115，V均值反馈+283，逐token polish−4995；v312确认提速不改损失。均值反馈现已获线上正反馈，撤回整类V否决。covariance_v1仍无法预测polish方向，保留为机制/性能/格式回归，不修改样本权重或种子强行拟合。下文等待隔离回执的描述已过时。

> **v311线上14979/244后的最新政策**：covariance_v1无法复现v307→v311的-4597分，当前不能作为新Attention算法线上得分预测器。保留为机制/性能/格式回归集，暂停V类跨算法斜率外推。以下“活动主bank”仅指本地回归用途，不表示其分布已识别线上数据。先取得Q/K-only与V隔离回执，不通过修改样本权重或种子拟合这个单一反馈。

## 2026-09-03活动数据集与验收口径（覆盖下文历史政策）

当前主bank为`local_fitted_models_covariance_v1/`：真实checkpoint Weight/QKV投影，10个Linear权重/50题、50个Attention组/250题，存储行数1/10/128/512/1024。**hidden activations仍是模型化生成，不是完整模型实际前向采集**；Linear最大维度压到2048，Qwen仅8个Q head，因此不能直接外推96核线上耗时。

Attention在同一组的独立校准/测试样本间共享固定的总体协方差方向，rank128/strength4/drift0.05；共享的是分布，不共享样本或答案。三个种子分别用于探索、留出和全量回归。参数由历史算法反馈选择，留出种子不能被称为未见算法的独立验证。

旧`local_fitted_models_realw/`和`local_fitted_models_realw_v41/`保留为反例及数值一致性检查，不再驱动新版本精度预测。另以固定真实checkpoint、强通道漂移构造历史RRS诊断场景；不能混入当前bank按版本重加权。旧单边stress预估器已阻止v303及以后预测。

完整结果、全部失败点、生成/复现命令见[v308报告](v308-data-and-cleanup.md)。当前可复现六条大Linear边、四条当前Attention大边；RRS在独立诊断场景复现，但在当前主bank仍反向；v303→v305的184分小边仍未通过。不得宣称已覆盖所有历史版本或已还原线上私有分布。

以下为历史记录，涉及“当前”“以后”的旧政策均以上述新政策为准。

> **2026-09-02封存状态**：下述历史`local_fitted*`合成数据已移动到
> `local_fitted_archive/`，只用于结果追溯。从此版本发布、参数选择和线上分数
> 预估只使用`local_fitted_models_realw/`；它采用Qwen3.6与DeepSeek-V4固定
> checkpoint的真实Weight。模型式Activation仍属于代理数据，所有预估必须报告
> 留一误差区间，不能视为线上保证。

当前活动集分为两个均使用真实checkpoint Weight/QKV投影的bank：

- `local_fitted_models_realw/`：core，context drift=0，服务常规策略回归；
- `local_fitted_models_realw_v41/`：stress，context drift=0.8，服务
  rotation/GPTQ/QKS类策略。该强度使v302−v301本地Attention增量约20.20，
  对应线上18.30个归一化总分（1830分），但不单独用于普通策略预测。

以后候选必须同时报告core/stress结果；预估器将stress作为附加特征。不得为拟合
单个线上点覆盖core，也不得重新启用`local_fitted_archive/`中的历史合成集。

## 目的与边界

**2026-09-03纠偏**：v303/v305/v306已证明stress上的GPTQ/refinement增量
方向不能代表线上。该bank中GPTQ守门50组全部拒绝，refinement只在少数组
生效，不能据此删除真实线上有效分支。v307起该算法族的线上分数改用已确认
父版分项和/消融区间估算；core/stress保留为数值一致性与相对回归检查，暂停
使用单边斜率将这些结构变更映射为精确的线上收益。真实Weight不等于真实
Activation，也不保证校准Hessian的特征谱和跨样本迁移率一致。

`local_fitted` 不是对线上私有数据的还原，而是一个由既有线上消融结果监督的行为代理集。官方 `mini_sample` 仅用于确认公开的容器结构和 NVFP4 carrier/scale 格式，不作为数值分布先验；这样可避免 mini_sample 被刻意构造成反向特征时误导优化。

数据生成器把下列线上事实作为约束：

- Linear：v11 输出感知平滑显著优于 v10；v20 优于 v17；v21 低于 v20；v24/v25 小幅增益；v26/v27 回退；v29 大幅跃升；v30 低于 v29。
- Linear 最新反馈：固定 Weight residual feedback step 从 0→0.25→0.50→0.75 时，线上分数从 18051→18514→18807→18945 单调上升；v73/v113 在 step=0 父链上只能部分追回损失。
- Attention：v6 RRS 显著优于 v1；v7 微增；v8/v9 回退；v10 为该链最高。
- 当前新集结构：10 个 Linear Weight，每个对应 5 个校准和 5 个测试 Activation，共 50 个 Linear 测试；逻辑 stride 为 1/10/128/512/1024。Attention 为 50 个组、每组 5 个测试，共 250 个测试。

## 两种 preset

### correlation

默认快速集，保持题目数量和 profile 混合，但 Linear 基础维度缩为 512，实际行数为 1/10/32/64/128。它用于一次运行十几个历史版本并计算与线上增量的相关性，不用于估计绝对耗时或 4K shape gate。

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_fitted_dataset.py \
  --preset correlation --output local_fitted
```

当前固定种子结果：

| 子集 | 历史版本 | Pearson | Spearman |
|---|---|---:|---:|
| Linear 历史主链 | v10/v11/v13/v17/v20/v21/v24/v25/v26/v28/v29/v30 | 0.937123 | 0.993007 |
| Linear 最新反馈全组 | v32/v33/v36/v37/v39/v40/v42/v73/v113/v117/v119 | 0.987565 | 0.954545 |
| Linear feedback 主轴 | v32/v39/v40/v42 | 0.988103 | 1.000000 |
| Attention | v1/v6/v7/v8/v9/v10 | 0.998326 | 1.000000 |

Linear 的本地顺序正确复现 v20>v21、v25>v26、v28>v26、v29 大跃升、v29>v30，以及关键的 `v32 > v39 > v40 > v42`；Attention 顺序为 `v10 > v7 > v6 > v9 > v8 > v1`，与线上一致。v32/v33只有1分线上差异，不把本地两者的微小反序当作有效信号。

### full

全尺寸集使用 4096×4096、8192×4096 和 4096×8192 Weight，Activation 行数严格为 1/10/128/512/1024。Linear优先阶段只生成10个Linear权重，避免额外生成250道数GB规模的Attention；它用于最终候选的shape/state/内存/耗时复核。

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_fitted_dataset.py \
  --preset full --linear-only --output local_fitted_full
```

### rectangular hypothesis

v278只修改square却与v277线上同分，因此保留原mixed集用于历史相关性，同时新增5个up、5个down的矩形假设集。v279的up-only线上仅比v278增加9分且耗时相同，v280的down连续逆映射也仅增加1分/4秒，说明本地连续中心收益迁移率很低；代理集后续不再用这类绝对增量外推线上，只用于检查机制激活和逐profile非退化。v281转用历史已在线上正迁移的128通道跨块离散Newton，并继续用4K真实规模收益/成本配对决定强度。

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_fitted_dataset.py \
  --preset correlation --linear-shape-mix rectangular \
  --output local_fitted_rectangular
```

## 数据 profile

生成器覆盖 square、MLP up、MLP down 三类矩阵，并组合以下线上消融所要求的困难特征：

- Activation/Weight 通道范围反相关，使等价平滑和输出感知校准有发挥空间；
- 低秩列相关和非对角 Gram，使 Q(W) 输出度量优于纯对角 SSE；
- 同一 Weight 的五份校准和五份测试 Activation 共享一个跨全部64通道块的低秩基底，但每份 Activation 使用独立latent系数；这样只共享分布协方差，不共享样本内容；
- 平衡/块异常 profile 使用强度1.0的稳定协方差，复杂低秩 profile保留0.65，避免只拟合feedback轴而破坏v29>v30；测试域对角漂移缩放为原profile的0.5；
- block outlier、稀疏值和 heavy tail，区分 clipping、旋转和层级 scale 策略；
- 校准—测试通道漂移，使增加候选可能发生过拟合，能够区分 v29 与 v30；
- Attention GQA 的 Q/K 互逆通道尺度、稀疏 spike 和测试漂移，区分 RRS、zigzag、sign ensemble 与多 holdout。

## 验证命令

```bash
/opt/miniconda3/envs/codecraft/bin/python self_check.py \
  --solution_dir solution --datasets_dir local_fitted

/opt/miniconda3/envs/codecraft/bin/python tools/evaluate_fitted_correlation.py \
  --datasets-dir local_fitted \
  --versions 10,11,13,17,20,21,24,25,26,28,29,30 \
  --json local_fitted/linear_correlation.json

/opt/miniconda3/envs/codecraft/bin/python tools/evaluate_fitted_correlation.py \
  --datasets-dir local_fitted \
  --versions 32,33,36,37,39,40,42,73,113 \
  --json local_fitted/feedback_correlation.json

/opt/miniconda3/envs/codecraft/bin/python tools/evaluate_attention_correlation.py \
  --datasets-dir local_fitted \
  --json local_fitted/attention_correlation.json
```

Attention 默认仍使用历史 `legacy` profile，以复现既有版本的排序。研究“校准
Q/K 与测试 Q/K 来自同一固定模型投影”的跨集合预训练时，另生成隔离代理：

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_fitted_dataset.py \
  --preset correlation --attention-profile same_model \
  --output local_fitted_attention_model
```

`same_model` 在每个 Attention group 内固定同一组通道投影，校准与测试只更换
latent activation；它用于检验可迁移关系和留出门控，不替代 `legacy` 回归集。

## 使用规则

- 新候选先看逐 profile 和历史版本相关性，不把本地总分当线上绝对分预测。
- 若修改数据 profile，必须同时重新计算 Linear 与 Attention 的相关性；不能为了单个新版本调数据而破坏既有消融顺序。
- `local_fitted/` 和 `local_fitted_full/` 是可复现生成物，不提交 Git；生成器、固定种子和评估脚本提交版本管理。
- 线上新反馈仍是最高优先级。新版本若与本地预测冲突，应新增约束并重新拟合 profile，而不是选择性忽略线上结果。

## Qwen3.6 / DeepSeek-V4 模型代理集

`tools/build_model_proxy_dataset.py` 是与历史消融拟合集隔离的新数据源。它固定
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) 和
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
的公开模型 revision，使用真实的投影尺寸、GQA 头数、head_dim、partial RoPE
维度与 TP-local 形状，同时模拟 RMSNorm 后激活、共享低秩协方差、Q/K 互逆
通道尺度、稀疏离群值和校准/测试漂移。

直接读取固定revision的safetensors头后，Qwen第3层`q_proj.weight`确认为
`[8192, 2048] BF16`，与题目`mini_sample`的Linear权重形状完全一致；这是
Q与attention output gate的联合投影，运行时Q仍为`16×256=4096`通道。
DeepSeek-V4-Flash的`wq_a/wq_b/wo_a/wo_b`分别为`[1024,4096]`、
`[32768,1024]`、`[8192,4096]`、`[4096,8192]`，代理集按8路TP使用对应
local shard形状。其expert FP4 checkpoint以打包I8存储，生成器采用解包后的
逻辑维度。

该集合是“模型结构与统计代理”，不是从完整 checkpoint 前向提取的真实张量。
公开仓库索引显示两套权重约为71.9GB和166.9GB；在没有GPU前向环境时，直接
下载完整模型并不能得到真实激活。后续若取得官方 checkpoint 运行环境，应将
相同prompt的不同样本通过layer hook采集，再替换本生成器中的统计激活。
生成器也支持HTTP range只读取上述固定revision中的目标tensor，而不下载整个
checkpoint；此时Weight是真实模型权重，Activation仍是模型结构统计代理：

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_model_proxy_dataset.py \
  --preset correlation --weight-source checkpoint \
  --output local_fitted_models_realw
```

2026-09-02已生成core/stress两个活动集：各含10个真实checkpoint Weight、50个
Linear测试和250个Attention测试，分别占236MB/249MB；共享tensor cache约
230MB。两套数据对当前`solution.py`的格式检查均为860/860通过。数据分数只
作为后续版本的相对对照，不直接映射为线上绝对分。

线上预估使用`tools/calibrate_online_estimator.py`。常规族回归点为新题集成功版本
v277/v278/v279/v280/v281/v283/v284/v285/v287/v293/v296；v301/v302只用于
v41族锚定。模型仅以准确总分训练，文档中的估算Linear/Attention拆分值只做
诊断。每个新版本输出中心值、经验误差区间和是否超出训练域：

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/calibrate_online_estimator.py

/opt/miniconda3/envs/codecraft/bin/python tools/calibrate_online_estimator.py \
  --reuse-training --predict-version 303 --prediction-family v41
```

回归按算法族拆分，禁止把唯一的v41强点与常规版本混成一个全局直线：

- 常规v277–v296使用core总分ridge：留一MAE/RMSE/最大误差为
  17.97/23.78/61.16分；
- v41族使用v302父版锚定：stress上`v302−v301=+20.203554`，线上为
  `+1830`，即约90.58线上分/stress点；Linear用`v301−v285`单变量锚定；
- v41族当前只有一条线上边，所有预测至少保留±100分，不允许跨族外推；
- core正确复现`v277=v278`、v279/v280微增、v284负向及v293负向；
  v283线上+64在core仍为0，是当前明确未拟合项，只能引用线上父边。

快速集保持当前线上题量，Linear 为两个模型各5个典型投影，共10组/50题；
Attention 为两个模型各25组，共50组/250题：

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_model_proxy_dataset.py \
  --preset correlation --output local_fitted_models
```

`online-shape` 使用1/10/128/512/1024真实行数。Qwen Attention为16/2/256；
DeepSeek按8路TP使用8/1/512。该集合较大，建议分Linear和Attention构建：

```bash
/opt/miniconda3/envs/codecraft/bin/python tools/build_model_proxy_dataset.py \
  --preset online-shape --linear-only --output local_fitted_models_full_linear

/opt/miniconda3/envs/codecraft/bin/python tools/build_model_proxy_dataset.py \
  --preset online-shape --attention-only --output local_fitted_models_full_attention
```

# 完整 Q 信息与校准输出蒸馏

线上最佳仍是 v314=20071/241 秒；目标 22000+ 未达，v315 等待回执。本轮不发布新 solution、不改 Linear、不重配数据集权重。

## 1. 完整 Q 能否重建当前 K/V

之前的 `cross_role_predictor.py` 只测试了128/256个随机通道。本轮把 feature 上限设为65536，实际覆盖所有 Q 通道：group1/14/49 为4096，group5为2048。仍只用前四份校准拟合、每份最多512行，最后一份校准和五份测试用于不同阶段的评估。

| 组 | ridge=0.01，K预测误差/HiF4误差 | V预测误差/HiF4误差 | 预测教师输出误差/官方输出基准误差 |
|---|---:|---:|---:|
| 1 | 25.138 | 25.138 | 12.510 |
| 5 | 32.103 | 12.892 | 16.872 |
| 14 | 25.756 | 27.799 | 124.958 |
| 49 | 24.221 | 24.221 | 10.722 |

以上均是五份测试的均值。将ridge降至0.0001，四组预测教师输出误差分别为15.171/31.589/125.085/13.401，没有解决问题。用末份校准拟合一个输出残差减法系数，测试上的**无约束诊断**净收益也没有普遍改善；它不是可提交HiF4实现。

结论仅限这个全通道线性回归器和所测代理。它排除了“原先只选256通道，因此只要全量Q就能得到高精度K/V”的直接推断，不证明任意非线性预测都不可能。没有将测试K/V传给预测器；真实其他角色只用于离线打分。预测器state检查通过，拟合函数保持原样；CLI新增ridge和实际feature计数，默认行为不变。

## 2. 不重建 K/V，直接学习 Q 的输出补偿

新原型 `query_distill_core.py` 在父版部署坐标 x 上学习：

`x_corrected = x * (1 + gain) + normalized_bias * calibration_RMS`。

gain、bias均逐Q头逐通道；归一化参数限制在±0.2。随后重新搜索合法HiF4元数据、执行父版GPTQ和原序列长度对应的refinement预算。

这不是保持原始QK不变的共享正交旋转，也不同于之前只拟合原始logit的非对称线性重构。它允许Q有目的地偏离原输入，优化 `softmax(Q_quant K_quantᵀ/√d) V_quant` 对原始输出的误差；K/V采用父版完整动态量化结果。

数学含义：Q的公共偏置会产生所有query共享的逐key logit偏置，等价于改变key先验权重；逐通道gain则改变查询方向，不仅是每头一个温度参数。这两者不必能泛化，必须由真实量化前向和留出输出验证。

### 训练与部署边界

- 前四份校准中的可用样本均参与训练（2≤T≤4096），保留全部key，只抽样query。每份/每头按父版误差归一，避免大能量样本完全主导。
- 使用真实GPTQ/refinement输出作为前向，STE只提供近似梯度；每步两个步长都重新执行真实量化、按头保留训练损失下降者。无动量/随机重采样，全部步长不改善则早停。
- 最后校准样本取最多64个query，奇偶两折均至少改善0.5%才按Q头采用。两折来自同一序列，不等于独立序列泛化证明；父版本身也使用该最后样本守门。
- 动态函数仅读取当前Q和已冻结state，无当前K/V、全局跨接口缓存或输入输出文件。
- 新增state只有 `qd_gain`、`qd_bias` 两个CPU float32张量，每个为 `[Qheads,d]`，无教师样本/梯度/优化器。未激活头与T=1完全回退父版。
- 原型保留一次完整父Q调用，以确保回退头逐值不变，再计算候选；尚未消除重复工作，不宣称低开销。

## 3. 结果

| 诊断 | 查询数/训练上限 | 本地题数 | 相对v314原始分增量 | 采用Q头 |
|---|---|---:|---:|---:|
| 缩放＋偏置，组1/5初测 | 32/4步 | 10 | +0.042045 | 3 |
| 缩放＋偏置，组1/5/14/49 | 64/8步 | 20 | 0 | 0 |
| 仅偏置，同上 | 64/8步 | 20 | 0 | 0 |
| 仅缩放，同上 | 64/8步 | 20 | +0.049270 | 9 |
| 仅缩放，未参与初筛组7/19/27/36 | 64/8步 | 20 | −0.001309 | 3 |
| 仅缩放，covariance组1/7/16/49 | 64/8步 | 20 | −0.002391 | 1 |

初测的32query/4步不是64query/8步的严格单变量对照。增加训练样本量与步数后，组合收益消失；训练更充分不保证量化泛化更好。首批gain有11题改善、4题下降、5题不变；独立批为1改善/4下降/15不变。没有足够证据发布小版本或给出线上+50/+200预测。

两套bank均是checkpoint投影加模型化hidden及人工压力，不是完整真实模型前向激活。旧bank的Attention长度设置也不是线上已知事实。这里保留它们作机制反例，不恢复其线上预测资格。

记录计时包括校准、动态、输出评分和检查，部分进程并行；不按比例折算线上241秒。所有负结果保留，不能根据这些小样本证明某策略在线上100%无效。

## 4. 验证

`test_query_distill.py`通过：

- STE前向逐值等于真实量化输出；解析链式梯度与独立构造的平滑输出梯度差为0。这只验证所定义的代理梯度，不是离散量化真梯度。
- 训练记录按头单调、参数有界、全局RNG不变、原校准输入不变；改变最后一份三角色数据不会改变训练参数。
- 在外部 `inference_mode` 下训练仍能工作，返回state无梯度。
- 零参数、未激活头、单token均逐值等于父版；强制启用分支确实改变输出。
- T=1/65/129/1025、GQA、父旋转及joint_transform+通道尺度路径共5个动态返回格式通过。
- 在所测T=65/129/1025，保留原T预算的抽样query结果与完整动态Q相同行最大差0，覆盖NumPy/Torch切换的一组边界；不是所有shape的普遍证明。
- 四份回归报告合计36次state检查、824次公开动态返回格式检查通过，包含校准内调用，不是824道独立题目。内部每个训练候选没有逐个调用格式检查器。
- 研究文件编译和diff检查通过。没有生成新生产提交，自然也没有声称新版本完整self_check或96核时间验证已完成。

## 5. 下一步

不扩大这条仿射蒸馏的参数网格，不用训练下降代替泛化。保留原型，等待v315线上回执。

源码仍有一个更直接的部署偏差待做消融：旧GPTQ开关用原始V评分，refinement开关抽样key且用原始V；后续joint增强又要求原父版已开启GPTQ。由此可能漏掉在真实量化V下有效的候选。这个逻辑差异已由源码确认，**是否导致线上失分尚未验证**。下一轮优先做候选开关/守门的部署一致性审计，保留完整key和实际V，不能未经证据整类关闭线上已证明有效的GPTQ。

复现命令与原始记录：

```sh
python tools/probe_cross_role.py --groups 1,5,14,49 --features 65536 --sources q --output docs/full-query-prediction-screen.json
python tools/probe_cross_role.py --groups 1,5,14,49 --features 65536 --sources q --ridge .0001 --output docs/full-query-prediction-small-ridge.json
python tools/test_query_distill.py
python tools/probe_query_distill.py --groups 1,5 --steps 4 --output docs/query-distill-first.json
python tools/probe_query_distill.py --modes none,both,bias,gain --steps 8 --queries 64 --output docs/query-distill-screen.json
python tools/probe_query_distill.py --groups 7,19,27,36 --modes none,gain --steps 8 --queries 64 --output docs/query-distill-independent.json
python tools/probe_query_distill.py --dataset local_fitted_models_covariance_v1 --groups 1,7,16,49 --modes none,gain --steps 8 --queries 64 --output docs/query-distill-covariance.json
```

使用 `/opt/miniconda3/envs/codecraft/bin/python`。

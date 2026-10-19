# v307反馈：有效路径清理、代理分布纠偏与因果边界

日期：2026-09-03。线上实际v307 **19576分/237秒**，相对v306 +1819/+43秒，相对v302 +184/-44秒。

## 结论先行

1. 保留v307完整GPTQ、校准Hessian、短序列refinement、Q/K平衡/旋转与高速Linear。不能再依据旧stress集的GPTQ零收益删除有效分支。
2. v308只做等输出清理，不是涨分版本。删除37个不可达/被覆盖的顶层函数，5180行压至3430行；保留历史版本以便追溯。删除历史不可达V refinement、无意义的乘全1张量、没有后续消费者时的unit重建，不减少有效搜索候选。
3. 新主bank能够复现本次回归的六条大Linear边、四条当前Attention大边。历史RRS需独立漂移诊断场景；当前主bank不复现它。v303→v305的184分恢复尚未通过。因此**没有完成“所有历史优化点准确预测”**，也不能保证下一个新算法线上方向正确。
4. 旧绝对分项拆分可能漏加失败题罚分；只用隔离版本差值做机制判断，暂停“Linear只拿到约9500、还剩2000”的推论。

## 为什么此前代理集误判

真实Weight不等于真实Activation。旧Attention生成器每个样本重新抽一组弱语义方向，再叠加强度0.8的独立通道漂移。即使QKV来自真实checkpoint，校准Hessian也不一定适用于测试。此前为拟合v301→v302单点选择stress强度，只匹配了总增量，没验证GPTQ是否启用：旧bank中50/50组拒绝GPTQ，导致v306和v307被预测同分。

新生成器在每组固定抽取真实checkpoint投影的归一化行作为总体协方差方向；不同校准/测试样本使用独立潜变量。没有复制校准样本到测试，没有逐版本生成数据，也没有回填目标分。

同一行数、同一种子、同样真实投影的二因素对照（每配置10题，完整结果见`v308-covariance-controls.json`）：

| 持久结构rank/strength | 通道漂移 | GPTQ采用组数 | v306→v307本地增量 |
|---|---:|---:|---:|
| 无 | 0.8 | 0/2 | 0 |
| 无 | 0.05 | 0/2 | 0 |
| 128/4 | 0.8 | 1/2 | -0.05827 |
| 128/4 | 0.05 | 2/2 | +0.87337 |

这支持“显著的通道相关性 + 跨样本可迁移的二阶结构”这一联合解释；只增加样本数、只降低漂移或只制造低秩都不够。这里是对生成机制的干预实验，**不是测得线上真实协方差**。

## 当前主bank及回归结果

路径`local_fitted_models_covariance_v1/`，约1.5GB（数据不提交Git）。10个Linear权重×5测试，50个Attention组×5测试；每组均5校准，存储行数1/10/128/512/1024。真实checkpoint版本与张量记录在bank的`manifest.json`和缓存元数据。

- 真实Qwen3.6/DeepSeek-V4 Weight和QKV投影，但hidden activations仍为模型化代理，不是完整模型前向采集。
- 主bank rank128/strength4/drift0.05，种子20262903；探索种子20260903、留出种子20261903。留出50题的GPTQ方向保持正向；主bank为第三种子250题。
- Linear压缩最大维度至2048；Qwen保留8个Q head而非完整16；DeepSeek为TP局部布局。行数按stride实例化，相当于只覆盖部分T，不能称为覆盖线上全部T或所有层。
- 50个Attention组复用少数实际层的投影，变化的是样本/协方差方向；不能视为50个独立实际模型层。
- 主bank K二阶矩有效秩均值58.54、非对角能量占比0.7414、校准/测试矩阵余弦0.9276；这些数值只描述代理。二阶矩未中心化，不把它冒称为严格中心化协方差或线上谱。

本地分数为`Σ(1−MSE_player/MSE_standard)`原始单位，**不等于线上分，也没有按版本调权**。

| 分项/版本边 | 线上差值 | 主bank本地差值 | 判断 |
|---|---:|---:|---|
| Linear v10→v11 | +1810 | +64.20018 | 正向 |
| Linear v28→v29 | +3888 | +8.28058 | 正向 |
| Linear v31→v32 | +2228 | +3.77970 | 正向 |
| Linear v229→v239 | 总分+5551 | +18.53340 | 仅方向参照，非纯单因素 |
| Linear v254→v255 | +423 | +1.78040 | 正向 |
| Linear v269→v270 | +204 | +1.45353 | 正向 |
| Linear v301→v285 | +171 | +3.53197 | 正向 |
| Linear v303→v304 | +12 | +0.39099 | 正向，但幅度仍失配 |
| Attention v285→v302 | +1830 | +21.84090 | 正向，已扣除Linear变更影响 |
| Attention v302→v303 | -2333 | -27.76909 | 大回退复现 |
| Attention v303→v305 | +184 | -0.07125 | **不通过** |
| Attention v305→v306 | +331 | +5.43923 | 正向，扣除Linear +12 |
| Attention v306→v307 | +1819 | +22.40111 | 大跃迁复现 |
| Linear/Attention v307→v308 | 尚未线上 | 0 / 0 | 300题逐题分数一致 |

GPTQ在主bank50/50组采用；v306→v307为198题提高、2题下降、50题相同。50题相同主要对应长度1：softmax只含一个元素时Q/K不改变注意力结果。这是后续必须按长度分组观察的结构因素，不是“所有题都同幅受益”。

旧Linear边来自不同题量/规则阶段，只检验机制方向，不用其绝对差值共同拟合一个系数。v10等负的本地原始总分合法，说明低精度误差可以差于标准基线，不作截断隐藏。

### 预估体系的限度

仅以Attention v285→v302拟合一个公共斜率83.78776，其他边回放为：

| 验证边 | 公共斜率回放 | 线上差值 |
|---|---:|---:|
| v302→v303 | -2326.7 | -2333 |
| v303→v305 | -6.0 | +184 |
| v305→v306 | +455.7 | +331 |
| v306→v307 | +1876.9 | +1819 |

公共斜率能解释主要跃迁量级，但184分反例与125分量级偏差仍在；此结果是**回顾性拟合**，算法版本参与了分布选择，不是盲测预测精度。不据此声称误差仅几十点，不给v308虚构涨分；其目标为与v307同分。旧`calibrate_online_estimator.py`已阻止v303及之后沿用错误stress锚点预测。

本地Attention v307/v308约52.75/54.89秒、Linear均约16.48秒（同一回归）；没有证据声称这次清理已经提速。96核线上237秒不直接按本地时长缩放。

## 历史RRS：不能强行混成同一线上分布

v1→v6历史线上+864，代码差异在Attention。当前主bank52.89826→52.23417，**反向**，保留失败记录`v308-attention-earlier-edges.json`。

独立真实投影漂移诊断rank0/strength0/drift0.8能够复现RRS收益；额外留出种子20263903、10组50题：-0.44779→14.14839（+14.59618）。见`v308-rrs-diagnostic.json`和`v308-rrs-holdout.json`。

解释：RRS处理可被旋转摊薄的通道能量不均衡；跨样本漂移较强时，比依赖可迁移全Hessian的策略更稳健。但历史题目分布与当前不同，不能用此场景占比任意拼出当前总分；其作用是保留算法机制覆盖，不取代当前主bank。

## 大优化背后的机制与可以推断的数据特征

### Linear

- v10→v11：用校准真实输出而非激活/权重单独重构选择平滑尺度。对应通道重要性不均且与W输出敏感方向相关；只是减少张量MSE不足以优化任务MSE。
- v28→v29：高秩全局输出度量、多候选尺度树、跨块坐标下降和校准反馈的组合。若误差各通道独立且W各向同性，全局度量价值小；其大收益支持输出方向及跨通道误差耦合重要。该边同时变更多项，不能给每项单独分配3888分。
- v31→v32：源码实际包含块内rank16→4回退及5%对角trust-region校准选择。不能再把+2228全部归因于“首次增加残差反馈”（两版均已有feedback0.75）。它表明更精确的Q(W)代理度量不一定更接近原W任务目标；正则化/真实输出门控抑制代理失配。
- v229→v239及后续Newton：设部署权重B、激活误差δA、权重误差δW，输出误差含`δA·Bᵀ + A·δWᵀ`，两项交叉项不可忽略。以真实`A·Wᵀ`为目标搜索允许二者抵消；只最小化`||δA||²`会漏掉这一大块收益。
- v269→v270：联合polar变换改变可量化坐标系/码本，而非只增加同一码本搜索轮数。收益支持量化误差受基底/条件数影响，但不能从总分反推线上矩阵的确切谱。
- v304：输出稀疏基底在局部down代理明显获益、线上仅+12，却省28秒。说明本地该shape收益强度/误差占比不代表线上；不能由“方向通过”宣称Linear幅度已拟合好，也不能由square小收益证明线上不存在square。

### Attention

一阶logit扰动为`δS≈(δQ·Kᵀ + Q·δKᵀ)/sqrt(d)`；其平方误差对Q量化误差的度量含`KᵀK`，对K含`QᵀQ`。最终softmax/V输出还会重新加权，因此这里只是任务代理，不是完全等价目标。

GPTQ在逐列舍入后用二阶结构向未量化列传播补偿；refinement在合法格点附近继续按Hessian改善。v305→v306 +331是无GPTQ条件下的refinement收益；v306→v307 +1819是采用GPTQ及其与refinement交互的条件收益，**不是独立GPTQ贡献的精确分解**。当前证据说明主要缺口不是再换一次旋转，而是删除了误差补偿路径。

线上较可能具备：显著非对角通道关系、组内校准/测试二阶统计可迁移、对任务敏感方向的误差比单通道峰值更重要。线上总分无法唯一确定rank、漂移、Qwen/DeepSeek占比或序列长度分布；新bank的128/4/0.05只是一组能解释主要反馈的假设。

## 删除与保留的边界

- `v308-pruned-functions.json`逐项记录静态删枝函数；`prune_v307_candidate.py`负责AST阶段，不包含后续手工等价简化。所有模块级赋值/别名捕获在静态闭包中保守保留。
- 不删除旧版本文件、不覆盖用户资料/旧预测修改，旧bank不物理删除；方便核查失败与回退。
- 过去线上否决的单独探针不合入当前父版；本次不把“未获得单独增益证据”等同于“可从完整组合安全删除”。
- 验证：旧core 750常规+500强制GPTQ返回参数一致；新bank300题逐题分数一致；mini self_check22/22，新bank860/860。生产提交只使用`solution.py`，不携带研究工具中的文件读取/下载模块。

## 分项口径待核实

任务书若按每题`100×(1−MSE/MSE_STD)`，50题Linear上限5000，与旧拆分9378冲突。一个可能原因是把含50题失败罚分的提交总分当成纯Attention：若每题罚-100，则Attention需加5000，Linear相应减5000。此时v285为4366/13367，v307为4378/15198，和总分吻合。

尚需原始分项回执或官方权重确认：全部失败-30000本身并不能证明部分失败一定逐题扣分。不能把上述条件修正当确定事实。后续采用固定分项版本差值，既不依赖旧9378也不依赖未确认的新4378。

## 复现

使用codecraft Python；从仓库根目录执行。默认生成器现已指向新配置并禁止无显式`--overwrite`覆盖已有数据。当前bank最初以显式参数构建，manifest中的preset仍为correlation，不修改既有来源记录。

```sh
python tools/build_model_proxy_dataset.py --preset correlation --rows 1,10,128,512,1024 --weight-source checkpoint --attention-covariance-rank 128 --attention-structured-strength 4 --attention-context-drift 0.05 --seed 20262903 --output local_fitted_models_covariance_v1
python tools/replay_major_edges.py --datasets-dir local_fitted_models_covariance_v1 --component attention --versions 285,302,303,305,306,307,308 --output docs/v308-attention-major-edges.json
python tools/replay_major_edges.py --datasets-dir local_fitted_models_covariance_v1 --component linear --versions 31,32,229,239,254,255,301,285,303,304,307,308 --output docs/v308-linear-major-edges.json
python tools/replay_major_edges.py --datasets-dir local_fitted_models_covariance_v1 --component linear --versions 10,11,28,29,269,270 --output docs/v308-linear-earlier-edges.json
python tools/probe_persistent_covariance.py --configs 0:0:0.8 --versions 1,6 --rows 1,10,128,512,1024 --groups 5 --seed 20263903 --output docs/v308-rrs-holdout.json
python tools/audit_v307_attention.py --reference 307 --candidate 308 --datasets-dir local_fitted_models_realw
python self_check.py --solution_dir solution --datasets_dir local_fitted_models_covariance_v1
```

探索、留出、全量结果及失败场景均保留，未来不能悄悄按新版本重新选种子。下一轮应优先验证新的算法方向在固定bank和独立种子上的逐组增量，先解释机制再申请线上验证；尚未完成的旋转小边与真实激活采集仍是明确缺口。

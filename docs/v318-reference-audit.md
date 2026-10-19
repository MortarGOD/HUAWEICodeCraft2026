# v318 参考仓库审计与迁移决策

> **最终发布选择**：下面标P0的精度方向是研究优先级，不表示全部已合入。metadata、阻尼、V仿射和抽query改守门收益不足，均保留研究；正式v318只合入signed GPTQ scratch、完整query分块执行和长校准Gram覆盖。验证与失败证据见 [v318报告](v318-report.md)。

## 范围与证据边界

本次梳理26个本地源码入口，不是声称逐行读完26个大型仓库，也不是把26种算法全部合入。固定提交、获取范围和路径见本地 `/Users/ee/.agent-reach/hif4-references-20260903/manifest.json` 与 README；FourOverSix复用 `/Users/ee/Code/hif4-public-references/mit-han-lab-fouroversix`。未安装或执行第三方项目，未复制未经许可代码到提交文件。以下是针对本题具体算子的实现审计。

v317线上20208/260秒；v315为20189/259。Attention函数完全相同，若判题和分项口径不变，可推算Linear9397、Attention10811。不能把本地Linear原始分37.9256/50直接映射为线上9397/10000的误差构成。

**重要纠正**：完整部署坐标重选已在v316线上验证为20189/262、净增0。虽然它在模型化sink族本地有大增益，本轮不将它重新包装为新策略。v311/v312全部PASS但损失约4995，说明代理目标求得更精确也可能更差；因此必须区分数学目标、校准守门和测试迁移。

## 逐仓库可迁移内容

路径均相对各仓库根目录。P0为可立即实验，P1为后续独立消融，P2为受接口/成本限制暂缓。

| 仓库、入口 | 具体学到的实现点 | 与当前实现的差距 / 决策 |
|---|---|---|
| SamsungLabs/BoA `quantizers/boa.py`, `minmax.py` | 不仅用H传播舍入误差，还可按H评价尺度；行/列双侧H将输出方向耦合 | **P0**：本题Q/K为被量化数据而非投影权重，不能整类照搬；迁移Hessian尺度评价，补足当前固定尺度GPTQ之后只改整数码的局限 |
| Vahe1994/AQLM `src/beam_search_xtx.py` | 保留当前解，替换局部码后只算受影响的二次误差项，候选可并行；beam保留多个离散路径 | **P0/P1**：本轮迁移精确增量目标与候选比较，**不是完整beam**；自定义加性码本不能作为HiF4输出。Linear已有精确残差搜索，不把同一公式重复计为新Linear算法 |
| Cornell-RelaxML/quip-sharp `lib/algo/quip.py` | LDLQ_buffered转置连续布局、块内更新和prod_cache，块界再做大GEMM | 当前GPTQ已有32列连续scratch与块界GEMM，不能宣称第一次实现；P0复用新搜索的梯度缓冲，P1考虑小块联合候选。不能直接替换HiF4为D4/E8码本 |
| ModelCloud/GPTQModel `gptqmodel/quantization/gptq.py` | 工作区租借、按几何分块累积FP32 H；阻尼重试要恢复原对角，避免逐次叠加 | P0新小块求解使用有界Cholesky和失败回退。全局缓存有并发隔离问题，不引入跨接口状态缓存；已有scratch优化不重复计收益 |
| Intelligent-Computing-Lab-Yale/GPTQv2 `fake_quant/gptaq_utils.py` | H来自量化输入；dXXT=(原始−量化)输入交叉矩，三角补偿与GPTQ更新结合 | **P1**：已有非对称矩阵重构不是这份GPTAQ逐列实现，不能说全算法已落地。其当前研究收益很小，不直接叠加；无明确根许可，仅独立推导。需要量化对端统计及实际PV守门 |
| IST-DASLab/FP-Quant `src/quantization/gptq.py` | microscaling静态分组、act-order后仍按原通道选择group尺度 | P0分组边界审计；本题64/8/4层次不能被排序破坏。不是允许任意码本或任意浮点scale |
| Asker0Lee/Hif4-gptq `src/quantization/quant_ops.py` | 本次HEAD与FP-Quant相同 | 不作为第二种独立算法/独立证据；同上 |
| ggml-org/llama.cpp `ggml/src/ggml-quants.c` | 固定码下加权尺度闭式解Σwxl/Σwl²，按真实损失选择局部搜索结果 | P1可用连续尺度建议生成附近E6M2候选；不能输出GGUF零点/任意scale。当前量化器已枚举合法E6M2，本轮新增的是H重排而非重新发明scale搜索 |
| intel/auto-round `auto_round/data_type/nvfp.py` | scale先舍入至FP8，随后才舍入元素；可学习舍入偏移和importance加权尺度 | P0坚持真实合法前向；P1校准尺度/舍入交替。101档尺度扫描和整模型STE成本不可直接放到每次CPU动态接口 |
| mit-han-lab/fouroversix `src/fouroversix/quantize/pytorch/reference.py` | 比较已舍入scale的4/6候选，评价实际反量化误差 | 当前HiF4树已按候选真实重构误差选择；不是把NVFP4的4/6参数原样迁移。P0验证元数据联动，而非仅改float权重 |
| HiFloat/Hif4 `hif4_gpu/quant_cy/base/QFuncs/hifx.py` | BF16舍入时点、E6M2及两级微指数、64→8→4层次 | P0用于格式/数值边界；比赛hif4_quant.py和self_check仍是判定依据。该HEAD与既有HiFloat4副本相同，非新算法 |
| pytorch/ao `torchao/prototype/mx_formats/nvfp4_tensor.py` | FP32缩放运算、乘倒数与除法的舍入顺序、两级scale | P0数值检查；不能假设代数相同即bit-exact，也不能将AO CUDA吞吐外推本题CPU |
| microsoft/microxcaling `mx/elemwise_ops.py` | 舍入方式、饱和、极小值和非规格化边界 | P0格式测试；保持赛题round规则，不能直接使用另一格式的denormal策略 |
| bitsandbytes-foundation/bitsandbytes `csrc/kernels.cu` | 分段阈值树和编码查表、向量化pack | P1替代重复比较需先证明边界逐值相同；其FP4网格不同于NVFP4和HiF4，不能仅因都4bit复用码字 |
| NVIDIA/Model-Optimizer `examples/hf_ptq/cast_mxfp4_to_nvfp4.py` | 特定指数跨度下可闭式选择二级scale，减少转换搜索 | P1仅在满足可表示性条件时适用；MXFP4→NVFP4的精确条件不证明NVFP4→HiF4总能无损 |
| chiparon/HscCon `solution.py` | HiF4微指数树的局部动态规划、Smooth统计稳定性门控 | 当前已有同类树和稳定性判断；未发现可直接替换就保证更好的新算法。无明确许可不复制实现 |
| thu-ml/SageAttention `sageattention/core.py` | K中心化保持softmax不变；V中心化必须补回输出常量 | K中心化已研究、未显示足够增益；V输出补回不属于本题独立量化接口可自由操作的步骤。CUDA/Triton内核不提交 |
| facebookresearch/SpinQuant `train_utils/optimizer.py` | Cayley迭代保持正交、动量和QR回缩 | P1校准训练旋转可迁移，但全模型训练损失需换本题PV；当前旋转/密集配平不等于训练了SpinQuant。避免随机QR和全局seed带来的安全/复现问题 |
| Hsu1023/DuQuant-v2 `quantize/quantizer.py` | 旋转细粒度与微尺度块相匹配，训练clip与重排协作 | P1新的受约束旋转候选，须保持GQA成对变换；既有64Hadamard不等于完整DuQuant++，不可引用GPU结果预测收益 |
| EasonGZY/AutoAPTQ `auto_gptq/quantization/aptq.py` | 保留Attention模块及投影信息，构造注意力感知重构/Hessian | P1算子级目标思想，非直接可运行替换；本题没有整个层/投影网络、位置编码和下游监督。现有完整key输出GN实验要保留负结果 |
| hahnyuan/PTQ4ViT `quant_layers/matmul.py` | Hessian/梯度平方加权误差；A/B交替搜索时缓存另一操作数，分批候选 | P0缓存固定K/V/基准输出的工程习惯；P1重新推导本题Jacobian。分类损失梯度不能直接当本题误差权重；无明确许可只读参考 |
| GoatWu/APHQ-ViT `utils/block_recon.py` | 对称扰动和真实/量化模型对比估计敏感度，再做块重构 | P2只借估计方法；全模型反向与KL监督缺失，不能声称已有Q/K Gram就是APHQ实现 |
| dropbox/hqq `hqq/core/optimize.py` | Lp近端收缩、尺度更新、保存最优与早停 | P1候选方向和接受策略；HiF4无自由zero point，任意零点拟合不可输出。逐元素Lp最优也未必改善PV |
| Dao-AILab/fast-hadamard-transform `fast_hadamard_transform_interface.py` | CUDA布局、归一化和非2幂特殊阶变换 | P2当前CPU/Python不能导入自定义扩展；本题维度不可随意补零改变输出接口。可学习布局，不承诺直接加速 |
| naver-aics/lut-gemm `README.md` | 用输入构建小表，按量化码索引实现GEMM | P2针对CUDA低比特GEMM；不是预存测试答案，不是用LUT替代搜索就能免费得到最优码。本题Python索引成本必须实测 |
| lsjfy-open-com/hif4-nvfp4 `docs/00-formal-model.md` | 将格式、scale、舍入与可表示性分开建模 | P0作为理论核对，不是成熟求解器；仍以本地官方量化函数检查边界，不将文档推导当已验证性能 |

## 本轮实质缺口与实验

现有Q/K：先按逐元素误差选合法scale，再固定scale做GPTQ和相邻整数码修正。GPTQ降低的是`(q−x) H (q−x)^T`，但其整数码搜索不会重新选择E6M2/lv2/lv3；因此固定scale会限制可达解。

新实验将每个64通道块作为一次合法重编码单元。设当前误差e=z−x、梯度g=eH，块b的连续条件中心为`z_b−g_b H_bb^−1`。缓存小块逆矩阵，合法量化得到候选；接受依据为：

`ΔJ = 2 Δ_b g_b^T + Δ_b H_bb Δ_b^T`。

接受后更新全梯度`g += Δ_b H_b,:`，保留已选块。第二个实验进一步用完整H而非对角MSE重排六个合法scale候选。微指数树仍用对角权重建议，不宣称对所有64通道元数据全局最优，更不是完整AQLM beam。每次只一遍，不添加第二遍GPTQ。

**初筛否定证据保留**：仅对角MSE选建议再做完整H接受，第一批30题净增−0.007099原始分；不能以数学目标逐行下降代替Attention收益。完整数据、第二个完整H重排实验及最后发布决策见v318报告和原始JSON。

校准守门保留全部K/V和真实量化V，按相同序列的奇偶query两折比较父/Q改/K改/双改；它不是独立序列外推保证。测试阶段Q、K各自只用自身输入和校准state，不引入跨接口通信、文件访问、原始测试对端。Linear完全冻结。

## 后续顺序

1. 完成本轮合法scale重编码的精度、格式与成本验证；若净收益不足，继续保留研究而不凭愿望写预估。
2. 对GPTAQ逐列交叉补偿单独立项，与已有非对称矩阵重构区分；不把二者共同算“已用过”。
3. 受约束beam或成对块搜索：优先检测既有单码局部最优陷阱、限定候选数，记录真实PV而非仅H指标。
4. 训练旋转/尺度和近端重构必须用部署实际量化前向验收；动态成本、校准成本分别计时。
5. 全部外部项目需要CUDA/编译/文件IO的运行框架都不能直接进入提交。只迁移独立推导的算子算法，并核验许可证及规则。

本轮使用codebase-memory确认生产与研究入口及覆盖；覆盖是best-effort，重复定义/alias仍逐段读源码。未使用子代理。没有已验证的线上+200保证。

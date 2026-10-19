# 可利用论文第二轮精读

## 方法

本轮不按论文数量堆叠功能，而是把论文结论拆成满足竞赛接口的单变量候选。每个候选必须回答四件事：保持什么不变、只改变什么、线上失败时回退到哪里、下一轮从反馈中学到什么。

当前线上事实：

- v1：9450 分 / 104 秒；
- v5（仅 RRS Linear）：8951 分 / 137 秒，证明该 Linear 旋转分支负收益；
- v6（仅 RRS Attention）：10314 分 / 144 秒，证明 Attention 旋转是当前主线；
- v7：修正 Attention RRS 候选目标与部署平滑路径的错配，等待线上反馈。

## 第一优先级：DuQuant

论文：[DuQuant](https://arxiv.org/abs/2406.01721)；[官方实现](https://github.com/Hsu1023/DuQuant)。

二次精读结论：

- 第一阶段 block rotation 将集中 outlier 扩散到相邻通道；
- zigzag permutation 的目标不是简单升序聚类，而是平衡不同 block 的 outlier 强度；
- 第二次 rotation 再降低重排后的局部峰值；
- 本题可让 Q/K 使用同一 permutation/rotation，严格保持 FP32 logits；不能给 V 套用同一逻辑，因为没有输出端逆变换。

拟议 v8：

- 父版本：v7 若线上高于 10314，否则 v6；
- 唯一变量：在 identity、当前 sorted permutation 之外增加 peak-balanced zigzag permutation；
- 不增加动态 state 类型，只保存最终选中的 permutation；
- holdout 继续用真实 Attention 输出选择候选；
- AC：隔离 self-check 22/22；FP32 logits 等价误差 <2e-4；线上高于父版本且 <300 秒；
- 回退：无增益则放弃 zigzag，转随机 sign 单变量扩展。

## 第二优先级：BASE-Q

论文：[BASE-Q](https://arxiv.org/abs/2506.15689)。

二次精读结论：

- rotation 可能无法对齐通道均值，并会让分布更接近 Gaussian，从而增加 rounding/clipping 能量损失；
- 原论文使用 bias correction 与 asymmetric scaling；
- HiF4 返回格式只有共享无符号 scale，且 Q/K 平移不能在评分端严格抵消，因此 bias/asymmetric 机制不能原样照搬。

可利用部分：

- 在 calibration 中统计旋转前后均值、峰均比、clipping 与 zero-collapse；
- 把 mean-aware error 作为 permutation/sign 的选择指标；
- 只允许改变候选排序，不向 state 引入无法抵消的 bias。

这一路线必须先做诊断；没有证据时不直接成为 v9。

## 第三优先级：AdaRound

论文：[AdaRound](https://arxiv.org/abs/2004.10568)。

二次精读结论：

- RTN 只在忽略 Hessian 非对角项时自然成立；
- 真正收益来自协调多个元素的上下舍入方向，而不是全体向上或全体向下；
- 本题 Weight 在校准和测试间共享，且校准阶段允许使用 A@W，因此 Linear Weight 是最合法的落点；
- 动态 Q/K 无法针对测试样本拟合逐元素 rounding，只能学习低维、冻结的 role/block threshold。

拟议候选：

- 只对固定 Weight 的敏感 HiF4 block 开放 mant 上/下邻级；
- 用校准 A@W 的 sampled-row 输出误差选择；
- 限制为一轮坐标更新或少量候选，避免 300 秒超时；
- RRS Linear 已证实负收益不等于 Weight adaptive rounding 无效，两者必须独立验证。

## 第四优先级：GPTQ

论文：[GPTQ](https://arxiv.org/abs/2210.17323)；[官方实现](https://github.com/IST-DASLab/gptq)。

二次精读结论：

- 核心是近似二阶信息和顺序误差补偿，不等价于简单逐通道 importance；
- 本题最适配的是 64 维 HiF4 block 内的固定 Weight 更新；
- 完整 Cholesky/逐列 GPTQ 可能超过校准预算，先采用 sampled activation covariance 和单轮 block residual；
- Attention 方向优先使用 logit/softmax 敏感度代理，不直接套用 Weight-only GPTQ。

## 执行队列

1. 完成当前 v10 的线上 Attention 验证并冻结最佳 Attention；
2. v11：将 v3 的 holdout + proxy alpha 搜索只迁移到 Linear；
3. v12：校准输出感知地选择 Linear E6 offset 集；
4. v13：固定 Weight 的敏感 block 受限 AdaRound；
5. v14：AdaRound 有收益后加入 64维 sampled-covariance GPTQ residual；
6. Linear 主线完成后，再恢复 Attention 的 BASE-Q/alpha-gain 等候选。

每一版只引入一个主要变量；mini_sample 只负责合法性，不用于判断收益。

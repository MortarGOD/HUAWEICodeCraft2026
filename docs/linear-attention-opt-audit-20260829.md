# `linear_opt.md` / `Attention_opt.md` 深读与落地审计

## Linear

新文档提出的主轴是最终输出误差 `||AW-Q(A)Q(W)||`，而不是元素重构。当前 v215 已覆盖：SmoothQuant式可逆通道缩放、balanced permutation、Hadamard/sign rotation、per-block二阶度量、残差反馈Q(W)、最终Q(W)高秩输出度量、四轮Q(A)坐标下降、轻ridge与校准策略选择。

仍有明确空白的不是再实现一次通用AWQ/QuaRot，而是：

1. AdaRound/BRECQ式跨块输出残差教师：v222。
2. 校准学习教师约束强度：v223。
3. Output-aware clipping与非RTN rounding的等成本候选库：v224。
4. 校准覆盖度和联合学习消融：v225–v227。

2026-09-07复核修正：此前把v162–v211概括为“FlatQuant/SpinQuant类已系统验证”过宽。具体源码中，v162–164主要是至多两次有限Kronecker Givens角度候选，v172–211主要学习通道scale并使用固定Hadamard；这些实验没有覆盖一般量化感知正交/Cayley/Procrustes训练。它们的损失、收益或超时约束对应的实际实现，不能替代对整个方法家族的检验。另有learned Cayley工具作用于Attention，其报告明确未改Linear。详见`linear-strategy-audit-v423.md`；历史当轮选择不重复高成本旋转训练仍是当时资源决策，不是数学穷尽结论。

## Attention

新文档最重要的结论是必须按 `QKᵀ→Softmax→PV` 的最终输出误差优化。当前稳定路径只覆盖：per-head Q/K reciprocal scale、共享Hadamard、Q/K与V独立offset、Q/K能量漂移惩罚以及K-center。

仍未充分覆盖、待Linear收口后按优先级执行：

1. 直接最终Attention输出损失选择Q/K/V候选，避免元素MSE代理失配。
2. Q/K耦合可逆变换 `Q'=QT, K'=KT^{-T}` 的低成本对角或2×2块版本；完整稠密T会超时且状态过大。
3. Head sensitivity/Fisher近似：按softmax Jacobian给Q/K误差加权。
4. ZipCache式token显著度加权V量化；只能影响合法HiF4候选选择，不能改变接口datatype。
5. Q/K/V role-specific clipping/rounding库；已有offset边界消融表明不应简单增删offset，必须由最终输出损失选择。

自定义Softmax codebook、量化P和更高精度PV不符合当前只返回Q/K/V HiF4参数的接口，故不作为可落地候选。

## 新线上反馈的约束

v215/v218为27033，v219为27027，证明本地代理对A通道流形路由方向错误；v220为27025/219秒，证明小输出草图的独立块精确教师既慢又不泛化；v221融合两者后进一步降至27024/225秒，说明它们在线上存在负交互。因此本轮本地数据只用于确认机制活跃、格式和性能风险，不作为确定淘汰标准。

v226随后取得27050/188秒，相比v215提升17分；这证明完整A@W监督并非无效，关键是把它用于选择等成本clipping/rounding策略，而不是把原始W教师直接注入测试Q(A)坐标标签。8行覆盖也比4行/16行更接近当前代理最优点，等待v224线上反馈确认覆盖度贡献。

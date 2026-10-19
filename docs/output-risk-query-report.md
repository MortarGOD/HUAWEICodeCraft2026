# 校准输出残差 query 选择实验

目标是把校准阶段可直接观测的最终 Attention 输出误差用于 V 风险因子的 query 表示，而不是继续增加 v343/v344 的均值坐标轮数。父版固定为 v353，Linear、Q、K 与动态搜索预算不变。

## 两个独立对照

1. `top-risk`：对每个校准 query 计算 `mean((P(Q,Kq) @ (Vq-V))^2)`，每个 KV head 直接选择风险最高的 16 个 query。
2. `weighted-uniform`：保留原有均匀覆盖的 16 个 query，仅把上述风险转为平方根权重，并裁剪到均值的 `[0.5, 2.0]`，降低过度集中的风险。

两者都固定 `_AF_SEARCH_RIDGE=1.0`；校准留出守门和线上动态路径不变。数据仍是 checkpoint 投影权重配建模激活，不具有线上绝对估分资格。

## 结果

| 方法 | feedback 6组 | covariance 6组 | 合计 |
|---|---:|---:|---:|
| top-risk | -0.043290 | -0.033888 | -0.077178 |
| weighted-uniform | -0.003593 | -0.005275 | -0.008868 |

`top-risk` 在 feedback 中 0升/3降/3平，在 covariance 中 1升/5降；明显过度集中。温和加权缩小了损失，但仍为 3升/6降/3平，且没有同时改善两套分布。

## 决策

不生成 v356，不覆盖默认提交入口。这个结果否定的是“把单个校准输出残差幅度直接当测试 query 重要性”的两种具体实现，不是否定最终输出目标本身。下一线上优先提交已经完成验证并推送的 v353，独立测量非对角 V 风险因子是否迁移。

原始证据：`output-risk-query-screen.json`、`output-risk-query-screen-weighted.json`。

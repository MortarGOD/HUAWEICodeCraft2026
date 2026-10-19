# `paper_list.md` 新增论文精读与接口适配

## 新增资料

| 论文 | 可迁移机制 | 本题实现 | 不采用部分 |
|---|---|---|---|
| OBC/OBQ (`2208.11580`) | 块 Hessian 逆、逐列量化误差传播、异常值优先处理 | v21 在每个连续64维 HiF4 block 上构造阻尼 Hessian inverse-Cholesky，顺序修正尚未量化的 Weight，再把结果固化为合法 HiF4 mant | 不在测试阶段保存 Hessian、原始 Weight 或反解输出 |
| SqueezeLLM (`2306.07629`) | 二阶敏感度加权、敏感值/异常值保护 | v21 用校准输出在原策略、OBQ RTN、两种符号非对称候选间选择；HiF4 clipping 保留原合法网格 | 稀疏 FP16 outlier 旁路和自定义 k-means codebook 与返回格式不兼容 |
| BitDistiller (`2402.10631`) | 正负非对称 clipping 初始化 | v21 将其压缩为固定 HiF4 网格内的 `positive/negative` 不同舍入阈值候选 | QAT、CAKLD、teacher logits 和训练更新没有接口入口 |

## v21 数学落点

对每个64维输入块，用最多128个校准 token 构造

`H = X^T X / N + 0.01 * mean(diag(H)) * I`。

使用 `chol(H^-1)` 按列量化并传播当前误差到尚未量化列。HiF4 的 E6/E1 metadata 保持固定，只有最终 sign/mant 被固化，因此输出格式不变。候选包括：

- v17 原策略；
- OBQ ties-to-even RTN；
- 正数更易向上、负数更保守；
- 负数更易向上、正数更保守。

候选只在 calibration 阶段用合法的 sampled `A@W` 输出重构选择。在线 state 不保存原始 Weight、低秩 Weight、Hessian 或输出补偿矩阵，dynamic Activation 路径不计算 `A@W`。

## 结论

这三篇新增论文最强的共同点是“利用完整 Weight 与校准输入，把补偿固化进固定 `Q(W)`”，而不是把完整 Weight 信息带到测试阶段反求 `Q(A)`。前者与题目规则和单次评分 GEMM兼容；后者触发赛题书的明确禁止条款。

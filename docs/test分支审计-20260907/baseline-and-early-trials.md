# 基础入口与早期六个调参包

本笔记补足全部分支 HEAD 算法去重表的组 0、1、49–54；仅静态阅读和 AST 比较，未执行参考代码。

## 22 个主入口实际是什么

以 [base3.07 的主入口](../papers/hif4-public-references/test-branch-snapshots-20260907/base3.07/solution/solution.py) 为例，411 行。22 个分支的这份文件与 `legal-solutions/tensor-recon-baseline-v1/solution.py` 的 AST 完全相同。

- 18–36 行保留 SrcFmt 解码 BF16 舍入。
- 43–81 行使用整数哈希生成固定 signs，再执行 FWHT，不依赖全局 RNG。
- 84–229 行以邻域 E6M2、微指数树的逐张量重构误差择优。
- 266–305 行仅从前两份校准，以 W 与 A 的归一化重构误差之和选择 identity 或块旋转；动态 A 只用一个 bool 路由。
- 308–337 行把校准 Q/K 的重构误差加和选共同旋转；V 独立逐张量量化。

它适合充当极简控制组，不是本项目 v419 的更强优化候选。还有一个必要条件：共同旋转路由来自同一次配对校准。在当前三次“真自身 / 假对端”校准条件下，Q/K 所选 bool 未必相同；函数不使用跨动态缓存，并不自动证明端到端旋转等价。

组 0 的 `examples/baseline_solution/solution.py` 只是 19 行内部判题器 API 转导示例，依赖本地 `fourbit_judge` 模块，不能按独立提交实现评价。

## 六个 auto-commit trial 不是六种算法

文件位于 [归档目录](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-highscore-v1/_archives/auto-commit/data/archive)。六份均为 528 行；所有函数的 AST 完全相同，变化仅是以下模块常量。

| trial | enhanced radius | 最小均值增益 | 最差样本增益阈值 | 文件名与日志记载分数 |
|---|---:|---:|---:|---:|
| 1 | 2 | 0.002 | −0.02 | 8972.133354277346 |
| 2 | 3 | 0.002 | −0.02 | 9545.285635925295 |
| 3 | 4 | 0.002 | −0.02 | 9545.285635925295 |
| 4 | 3 | 0.001 | −0.02 | 9545.285635925295 |
| 5 | 3 | 0.001 | −0.05 | 9731.699839527024 |
| 6 | 3 | 0.001 | −0.10 | 9731.699839527024 |

以 [trial 5](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-highscore-v1/_archives/auto-commit/data/archive/trial_005_score_9731.699839527024.zip/solution.py) 为例：

- 229–243 行同时要求平均校准增益和最差样本增益，才替换基线。这是有价值的保守候选验收形式，但本项目已具备更细的留出验证和 MSE cap；本实现的同一批校准既估计权重又选候选，不是独立留出。
- 285–342 行用校准 A 能量加权 W 量化，再用量化 W 的列能量加权 A；最后以完整 Linear 校准输出选择。属于已有对角敏感度与输出守门手段。
- 364–469 行使用校准 Q/K 能量互为重要性，选择四个 Q/K/V 组合；动态接口只读取自身 state。仍需检查假对端校准导致的重要性与组合决策失配。
- 六条分数来自 2026-08-19 自动调参归档/日志。这里核对了源码与常量，没有将日志分数视作当前数据、当前协议下的复现结果。扩大 radius 或放宽最坏样本阈值的旧局部收益不能作为当前 v419 调参依据。

结论：保留其固定 RNG 无关变换生成及“均值 + 最坏样本”验收形式作为实现参考；没有识别出优先于本项目现有求解器的新算法。

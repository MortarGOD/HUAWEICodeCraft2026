# 2026-09-05 GitHub / GitCode 第二轮公开代码扫描

## 扫描结论

本轮用赛题六个接口名、HiF4/NVFP4、华为2026量化赛题、Attention/GPTQ 等组合检索 GitHub 代码与仓库，并在 GitCode 网页端交叉检索。可确认的公开赛题实现仍主要集中在以下仓库：

- `openaddr/hif4-contest`：已知公开 v41/v42 路线。
- `suibianll/AI-Contest`：截至 2026-09-05 仍在更新，新增了 Attention 2×2 Matrix-Smooth、Jacobian/Fisher 等完整消融记录。
- `chiparon/HscCon`：Smooth-QK、H64、V 多基准 E6M2 搜索。
- `jansenin/quantization-contest-work`：Q/K 交叉能量权重与较小的 E6M2 搜索。
- GitCode `qq_41888214/quantizer`：旧评测上的 V 概率平方权重与双重安全门。
- `youxilee/hif4`、`Asker0Lee/Hif4-gptq`、`lsjfy-open-com/hif4-nvfp4`、`alvinshao0313/HiF4_Sp`：参考实现或评测框架，没有发现明显强于当前线上 20803 基线的完整新赛题方案。

GitCode 对精确接口名的检索只新增确认了 `qq_41888214/quantizer`；其余命中大多是 CANN 文档或无关项目。GitHub 代码检索新增确认 `suibianll/AI-Contest`，未发现另一个带完整六接口且有更高新榜成绩的公开仓库。外部镜像目录新增 `chiparon-HscCon`、`gitcode-quantizer`、`lsjfy-hif4-nvfp4`，并把 `suibianll-AI-Contest` 从 `f7af6d1` 更新到 `ad59a6b`。

## 与当前 Attention 的逐项差异

| 操作 | 公开证据 | 当前成熟代码 | 判断 |
|---|---:|---|---|
| GQA 相邻通道 2×2 解析 Matrix-Smooth | `suibianll` v158 在其 v86 基线上 +117 / +0.3秒 | 只有按整头 64×64 SPD balance，没有单独的小块候选 | **此前未隔离；作为 v364** |
| softmax Jacobian/Fisher Q/K 通道权重 | 早期弱基线 +721；移植到成熟 v186 后 v188 为 -4 | 旧 v235 有原型，当前 v324 没有该分支 | 有新实现细节，但成熟公开消融为负，列研究臂而非下一强提交 |
| V 的 `E[P²]` 使用度权重、plain-MSE 双门 | GitCode v7 在旧基线上 +51 | 当前已有最终输出 V kernel、mean/variance、风险/位置/尺度多轮探针 | 形式不完全相同，但历史 v341–v358 已覆盖更强目标且线上基本同分，低优先级 |
| reciprocal Smooth-QK、H64、尺度感知 K center | 多仓库重复出现 | 已有 qks、rotation、temperature、K translation/Lloyd | 高度重叠，不再换参数重试 |
| 当前测试 Q/K/V 通过模块全局缓存联合优化 | `openaddr` 日志说明本地同进程可用，线上独立 quantizer 不共享 | v361–v363 连续线上同分 | 强证据表明通道不成立，终止此族 |

已经尝试或有更强等价实现的还有：GPTQ、按头 lattice refine、整头 64×64 balance、Hadamard/rotation、logit temperature、K softmax不变平移、V 均值与条件输出核、长序列分块精确 softmax、尺度候选扩窗。继续在这些臂上调阈值已被几十个线上同分版本证伪。

## v364 的机制与风险

v364 固定继承 v324 的 Linear 与所有成熟 Attention 路径，新增一个 GQA 对齐的相邻 2 通道闭式等价变换。校准窗 1/3 拟合 Q/K 协方差，逐 pair 解 SPD 平衡；Q 乘 `M`、K 乘 `M^-T`，连续域 QK 点积不变。变换预合并到原有 head 矩阵，动态测试没有第二次矩阵乘。

候选在未参与拟合的最后校准窗上按最终 Attention 输出逐 KV head 门控：平均至少改善 0.5%，两个 query fold 任一退化不得超过 1%；否则逐头回退原状态。完整 50 组/250 题真实权重代理由 v324 的 `81.561574` 提到 `81.620715`，增益 `+0.059141`，只在 2/50 组各选择 1 个 head。回归耗时 188.20 秒；此前同机 v363 完整回归 196.80 秒，跨轮绝对耗时不可直接比较，只能判断没有显著新增动态开销。

公开 +117 来自较弱且不同的基线，不能直接外推为我们的线上收益。结合本地增益约为 v363 未迁移增益的 11.8 倍，v364 合理预估为线上 **+30～+120 分、250～260 秒**；这是一条新结构探针，不承诺达到 +200。

## 后续优先级

1. 先用 v364 判定小块等价变换能否穿透当前成熟基线。
2. 若正收益，下一版做 pair 分组重排或 4 通道稀疏 Givens，而不是扩大成已验证过的 full-64。
3. 若同分，Matrix-Smooth 路线关闭；随后只保留一个严格复刻现代 softmax Jacobian、并用最终输出门控的独立探针。
4. 不再开发依赖 Q/K/V 模块全局缓存或当前输入互见的方案。

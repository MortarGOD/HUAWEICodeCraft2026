# HUAWEICodeCraft2026

2026 年华为算法大赛初赛的 **NVFP4 → HiF4 量化研究与实现**。围绕 Linear 和 Attention 的最终输出误差，保留算法候选、六接口自检、本地诊断工具、实验结论与失败记录。

本仓库于 **2026-10-19** 从本地研究工作区整理更新，保留原有 GitHub 历史。此轮是代码与资料归档，不是新的算法优化或竞赛提交。

## 从哪里开始

| 目的 | 入口 |
|---|---|
| 理解赛题、数据布局及六个接口 | [项目理解与开发依据](docs/项目理解与开发依据-20260907.md)、[任务书](docs/2026年华为算法大赛-初赛任务书-0818.docx) |
| 查看原工作区默认实现 | [solution/solution.py](solution/solution.py)（与 v325 相同，保留原绑定） |
| 查看历史线上最佳记录对应源码 | [solution/solution_v636.py](solution/solution_v636.py) |
| 查看后续超时恢复候选 | [timeout_recovery_v636_stream.py](solution/timeout_recovery_v636_stream.py) |
| 检查格式和 state 契约 | [self_check.py](self_check.py)、[mini_sample](mini_sample/) |
| 查看输出误差与本地耗时 | [evaluate_local.py](evaluate_local.py)、[hif4_quant.py](hif4_quant.py) |
| 查看历史回执及版本取舍 | [benchmark-log.md](benchmark-log.md)、[candidate-dag.md](candidate-dag.md) |
| 查看公开整理范围与验证 | [公开整理说明](docs/publication/README.md) |

**不要按版本号最大来选择提交文件。** 截至本次导入的历史记录，v636 的用户提供线上成绩为 **21327**；这不是本次重新运行的成绩。后续候选的验证状态和已撤回方案以 benchmark log、candidate DAG 为准。默认入口没有被自动替换为 v636；部分版本保留早期文件头，请以实际源码和带日期记录为准。

## 目录

```text
solution/                 算法入口与历史候选（保留路径供研究脚本复用）
tools/                    数据构造、机制诊断、消融与评测脚本
mini_sample/              原有小样本：linear.pt、attn.pt
hif4_quant.py             NVFP4 解码、HiF4 参考量化与反量化
self_check.py             六接口、参数格式与 state 自检
evaluate_local.py         最终输出质量与时间诊断
docs/                     任务说明、研究报告和汇总实验结果
docs/publication/         导入来源、文件清单、校验和与整理记录
benchmark-log.md          历史线上反馈
candidate-dag.md          版本关系和取舍
implementation-plan.md    研究过程与假设修正
```

原 GitHub 仓库独有的 `local_eval.py`、早期资料和工具仍保留，历史上下文可从提交记录追溯。新读者优先使用上表中的入口。大量论文全文、第三方仓库镜像、模型缓存、逐单元实验日志和提交 ZIP 未从本地导入；文献索引见 [paper_list.md](paper_list.md)。历史文档中指向这些本地材料的链接可能无法在公开仓库打开。

## 环境与运行

建议使用独立 Python 3.10+ 环境。以下仅安装核心算法与样例检查所需依赖；专项研究工具还可能需要各自的依赖和本地数据。历史完整环境见 [环境说明](docs/环境说明.md)，其中 GPU/系统特有依赖不应直接照搬到其他平台。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# 原工作区默认入口的输出格式自检
python self_check.py --solution_dir solution --datasets_dir mini_sample

# 指定候选做样例上的输出误差诊断
python evaluate_local.py --solution solution/solution_v636.py --datasets-dir mini_sample
```

这些命令在仓库根目录执行。只加载可信来源的 `.pt` 数据。小样本和本地代理不代表官方隐藏数据；本地耗时不能换算为鲲鹏线上耗时。`evaluate_local.py` 不指定数据目录时会使用本地研究数据路径，公开仓库未包含该数据，所以示例显式传入 `mini_sample`。

自检工具要求目录中有名为 `solution.py` 的文件；如果要自检其他候选，请复制候选到独立目录并命名为 `solution.py`，不要覆盖已有研究入口。

## 算法与实验边界

六接口覆盖固定权重校准、动态激活量化、Attention 校准及 Q/K/V 动态量化。HiF4 参数须满足层级尺度和离散码字布局；质量评估针对最终 `A @ Wᵀ` 或 Attention 输出，张量 MSE 只是诊断指标。历史研究包括输入/权重平滑、通道排列与旋转、GPTQ/残差更新、尺度搜索及角色独立校准等。

官方规则、题量和时间限制曾发生变化，历史得分须按对应日期和口径比较。[imperial_edict.md](imperial_edict.md) 与研究记录保留当时依据，不能当作未来赛事的最新规则。研究目录包含失败、撤回及不适合提交的候选，不表示每个 `.py` 都可直接参赛。

本次整理没有改动算法实现、重新评测官方成绩或重新授权第三方材料。源码中的来源说明与现有许可保持原状；仓库公开不意味着对引用材料新增统一许可证。

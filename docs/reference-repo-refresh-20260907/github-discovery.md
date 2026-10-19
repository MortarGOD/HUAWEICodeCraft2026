# GitHub 公开赛题仓库发现记录（2026-09-07）

本记录只负责发现和公开来源核验；现有镜像更新、克隆及全分支落地由主任务负责。使用 agent-reach 的 GitHub / gh CLI 路由，辅以 GitHub 连接器与网页检索。仅做只读远端请求，没有运行仓库代码。

新增确认：**5 个独立仓库 + 2 个 fork，共 7 个来源、12 条远端分支**。其中 6 个来源有当前六 API 的真实源码声明；MortarGOD 为同赛题旧双 API 阶段。这里的“新增”相对主任务给定的原 14 个参考目录及 2026-09-02 两份索引，最终去重以实际 origin URL 为准。

## 可交给主任务下载的来源

| 仓库 | 归属 | pushed_at（UTC） | 远端 heads | 固定提交源码证据 |
|---|---|---|---:|---|
| [Tarjan01/huawei_algor_contest](https://github.com/Tarjan01/huawei_algor_contest) | 当前六 API；独立仓库 | 2026-09-06T08:58:35Z | 1 | [build/submission/solution.py](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py) |
| [xwcai98/2026-huawei-alogorithm-test](https://github.com/xwcai98/2026-huawei-alogorithm-test) | 当前六 API；独立仓库 | 2026-09-03T00:59:28Z | 2 | [v311/solution.py](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py) |
| [Lulili520/huawei-code-competition-2026](https://github.com/Lulili520/huawei-code-competition-2026) | 当前六 API；独立仓库 | 2026-09-04T07:37:25Z | 2 | [solution/v0_hessian_repair/solution.py](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py) |
| [klompn24-crypto/2026_compatition](https://github.com/klompn24-crypto/2026_compatition) | 当前六 API；独立仓库 | 2026-09-06T11:31:04Z | 1 | [code/solution.py](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py) |
| [wwwbby/code](https://github.com/wwwbby/code) | 当前六 API；独立仓库 | 2026-09-07T11:45:57Z | 4 | [solution.py](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py) |
| [qcjiang/2026-huawei-alogorithm-test](https://github.com/qcjiang/2026-huawei-alogorithm-test) | 当前六 API；xwcai98 fork | 2026-08-19T08:22:44Z | 1 | [v4/solution.py](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py) |
| [MortarGOD/2026-huawei-alogorithm-test](https://github.com/MortarGOD/2026-huawei-alogorithm-test) | 旧双 API；xwcai98 fork | 2026-08-15T15:27:33Z | 1 | [solution/solution.py](https://github.com/MortarGOD/2026-huawei-alogorithm-test/blob/aa894973e13d2be049aa181a6ec7f74bd2f76290/solution/solution.py) |

## 远端全部分支快照

这些是 `git ls-remote --heads` 返回的完整 heads 列表，不以默认分支代表整个仓库。没有创建或切换参考仓库的本地分支。

### Tarjan01/huawei_algor_contest

| 分支 | 提交 |
|---|---|
| `master` | `f3cc2417cb804633ec47b2fb186b8ba101ba7be8` |

### xwcai98/2026-huawei-alogorithm-test

| 分支 | 提交 |
|---|---|
| `before` | `a905dbdd51955d84fc5eaaadd40d50f495b4f3ed` |
| `main` | `eeb8381e8c93cb2494a7a26c116358505e64c261` |

### Lulili520/huawei-code-competition-2026

| 分支 | 提交 |
|---|---|
| `main` | `8f1b47128e05f0abe4e36d4e12faf703cf13c425` |
| `submission` | `99dae1b10e59f03d883b097d65dd573d356edbb0` |

### klompn24-crypto/2026_compatition

| 分支 | 提交 |
|---|---|
| `main` | `93dd79799f4118281cb99ac0e9f8d5d39960b480` |

### wwwbby/code

| 分支 | 提交 |
|---|---|
| `experiment/linear-output-rounding` | `be6ffae3b6dba49f0106c089576996026d8fa5fc` |
| `experiment/qk-low-rank-balance` | `a807b94a32490d7eaee4dfde72a45882750e78b1` |
| `experiment/v-output-rounding` | `988385ef2d78091b308edae432ec5914a70ed811` |
| `main` | `3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5` |

### qcjiang/2026-huawei-alogorithm-test

| 分支 | 提交 |
|---|---|
| `main` | `74ffccd454ff9a90f451b6e754945677d964a306` |

### MortarGOD/2026-huawei-alogorithm-test

| 分支 | 提交 |
|---|---|
| `main` | `aa894973e13d2be049aa181a6ec7f74bd2f76290` |

## 接口声明核验

按上表的固定提交读源码，下面的行号指向真实 `def` 声明。只用于证明同题与接口存在，不构成算法正确性、线上协议或性能验证。`wwwbby/code` 包含同名重定义，列出全部命中，后续算法分析应继续核对最终绑定。

| 来源 | 接口及行号 |
|---|---|
| Tarjan01/huawei_algor_contest | `hif4_calibration_attention` [L446](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L446)；`hif4_dynamic_quantize_q` [L510](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L510)；`hif4_dynamic_quantize_k` [L523](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L523)；`hif4_dynamic_quantize_v` [L536](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L536)；`hif4_calibration_and_quantize_weight` [L613](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L613)；`hif4_dynamic_quantize_activation` [L639](https://github.com/Tarjan01/huawei_algor_contest/blob/f3cc2417cb804633ec47b2fb186b8ba101ba7be8/build/submission/solution.py#L639) |
| xwcai98/2026-huawei-alogorithm-test | `hif4_calibration_and_quantize_weight` [L887](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L887)；`hif4_dynamic_quantize_activation` [L1100](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L1100)；`hif4_calibration_attention` [L1165](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L1165)；`hif4_dynamic_quantize_q` [L1305](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L1305)；`hif4_dynamic_quantize_k` [L1321](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L1321)；`hif4_dynamic_quantize_v` [L1340](https://github.com/xwcai98/2026-huawei-alogorithm-test/blob/eeb8381e8c93cb2494a7a26c116358505e64c261/v311/solution.py#L1340) |
| Lulili520/huawei-code-competition-2026 | `hif4_calibration_and_quantize_weight` [L1027](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1027)；`hif4_dynamic_quantize_activation` [L1102](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1102)；`hif4_calibration_attention` [L1268](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1268)；`hif4_dynamic_quantize_q` [L1546](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1546)；`hif4_dynamic_quantize_k` [L1558](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1558)；`hif4_dynamic_quantize_v` [L1570](https://github.com/Lulili520/huawei-code-competition-2026/blob/8f1b47128e05f0abe4e36d4e12faf703cf13c425/solution/v0_hessian_repair/solution.py#L1570) |
| klompn24-crypto/2026_compatition | `hif4_calibration_and_quantize_weight` [L1439](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L1439)；`hif4_dynamic_quantize_activation` [L1686](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L1686)；`hif4_calibration_attention` [L1938](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L1938)；`hif4_dynamic_quantize_q` [L2740](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L2740)；`hif4_dynamic_quantize_k` [L3052](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L3052)；`hif4_dynamic_quantize_v` [L3286](https://github.com/klompn24-crypto/2026_compatition/blob/93dd79799f4118281cb99ac0e9f8d5d39960b480/code/solution.py#L3286) |
| wwwbby/code | `hif4_calibration_and_quantize_weight` [L1638](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L1638)；`hif4_dynamic_quantize_activation` [L1773](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L1773)；`hif4_calibration_attention` [L1816](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L1816)；`hif4_dynamic_quantize_q` [L1899](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L1899)；`hif4_dynamic_quantize_k` [L1949](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L1949)；`hif4_dynamic_quantize_v` [L2018](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L2018)；`hif4_calibration_attention` [L2216](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L2216)；`hif4_dynamic_quantize_k` [L2275](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L2275)；`hif4_calibration_attention` [L2477](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L2477)；`hif4_dynamic_quantize_v` [L2489](https://github.com/wwwbby/code/blob/3a51762d1dc129a9c908a229ee4b9faeaf7a4ae5/solution.py#L2489) |
| qcjiang/2026-huawei-alogorithm-test | `hif4_calibration_and_quantize_weight` [L187](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L187)；`hif4_dynamic_quantize_activation` [L220](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L220)；`hif4_calibration_attention` [L230](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L230)；`hif4_dynamic_quantize_q` [L273](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L273)；`hif4_dynamic_quantize_k` [L278](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L278)；`hif4_dynamic_quantize_v` [L283](https://github.com/qcjiang/2026-huawei-alogorithm-test/blob/74ffccd454ff9a90f451b6e754945677d964a306/v4/solution.py#L283) |
| MortarGOD/2026-huawei-alogorithm-test | `hif4_quantize` [L589](https://github.com/MortarGOD/2026-huawei-alogorithm-test/blob/aa894973e13d2be049aa181a6ec7f74bd2f76290/solution/solution.py#L589)；`hif4_quantize_attn` [L771](https://github.com/MortarGOD/2026-huawei-alogorithm-test/blob/aa894973e13d2be049aa181a6ec7f74bd2f76290/solution/solution.py#L771) |

## 检索范围与分页

原始请求、返回 JSON、读取时间、候选 metadata/tree/README、固定源码和 fork 枚举保存在 [`github-discovery.json`](github-discovery.json)。共 50 次 GitHub REST 搜索请求、38 个不同的 `(类型, query)`；另有连接器代码搜索和 7 条网页搜索。

关键查询与覆盖：

| 查询 | GitHub total_count | 覆盖 |
|---|---:|---|
| `hif4 fork:true` | 13 | 第 1 页；查询结果已取完 |
| `HiFloat4 fork:true` | 16 | 第 1 页；查询结果已取完 |
| `NVFP4 HiFloat4 fork:true` | 2 | 第 1 页；查询结果已取完 |
| `NVFP4 huawei fork:true` | 1 | 第 1 页；查询结果已取完 |
| `"华为算法大赛" 2026 fork:true` | 0 | 第 1 页；查询结果已取完 |
| `"4bit" "数值转换" fork:true` | 0 | 第 1 页；查询结果已取完 |
| `huawei2026 fork:true` | 4 | 第 1 页；查询结果已取完 |
| `2026-huawei-algorithm fork:true` | 2 | 第 1 页；查询结果已取完 |
| `NVFP4 HiF4 in:readme fork:true` | 31 | 第 1 页；查询结果已取完 |
| `hif4 in:readme fork:true` | 119 | 第 1, 2 页；查询结果已取完 |
| `"hif4_calibration_attention" in:readme fork:true` | 4 | 第 1 页；查询结果已取完 |
| `"数值转换" in:readme fork:true` | 11275 | 第 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 页；到 GitHub 1000 条结果上限；非穷尽 |

六 API 分别执行不带 `fork:true` 的精确代码检索，`total_count` 依次为 weight 1360、activation 1316、attention calibration 1328、Q 1284、K 1264、V 1280。每个原始查询读取第 1 页（100 条）；为防止大仓库的历史版本淹没其他来源，进一步对每个 API 排除已知及新发现的 9 个 owner，最终均返回 0 条索引结果。两条临时 403 已重试成功，错误也保留。

该补充查询只说明这些 owner 以外当时没有可见的索引命中，**不证明不存在其他仓库**：本次 README 检索找出的 Lulili520、klompn24-crypto 和 wwwbby 已有六 API 源码，但没有进入该轮代码索引结果。带 `fork:true` 的代码查询对已知存在的函数也返回 0，故不将其当作缺失证据。

对原有 8 个直接相关 GitHub 仓库及新发现 5 个独立仓库逐个调用 `/repos/{owner}/{repo}/forks`（每页 100，13 个端点均一页取完）：只有 xwcai98 返回 qcjiang、MortarGOD 两个 fork，其余为空。fork 本身的分支也分别枚举。

## 相邻项目与排除理由

- `shch-y/icme`、`Reopen-AI/Wan2.2-I2V-14B-HiF4`、`GCC-HiFloat/Percentile_Wan2.2`、`Charles2530/GCC_Competition_EfficientAI` 属于 ICME/Wan2.2 视频量化或其流程，不计作本题参赛方案。
- `ecastel1-sudo/NoWei` 是 MoE 动态负载均衡赛题；`Wang-Ruibin/huawei2026` 自述软件精英挑战赛；不计入本题。
- `Huaiqing-Teng/Huawei2026` 无远端 heads；`HidalgoFloresMarcos/Huawei2026` 默认树仅有标题 README；`IcySkyhy/Huawei2026KunpengTensor` 推送时间为 2025-12-03，未见本题声明。没有因名称接近而纳入。
- `yuuoo-oreo/hif4-numpy` 是官方 HiFloat4 转换器的 NumPy 移植；`Asker0Lee/Hif4-gptq` 是 MR-GPTQ 路线；`Evans-Z/sepquant` 为 DAFQ 通用研究框架；`Futuresxy/floating-ponit-quant-dequant` 是数值格式交互演示。这些可作为一般格式/论文相关线索，未计作直接参赛仓库。
- `MingZwhy/FAQUANT-Public` 是 Qwen3-8B HiF4 W4A4 PTQ/QAD/LongBench；`Areache/MRBD` 为 DeepSeek-V2-Lite FlatQuant/MRBD；`lloo099/mission44` 是研究面板；`Charles2530/fake_quant_ops` 是多格式算子库。
- `Novachevski/snowflake_demo` 的命中来自 README 内嵌数据中的 HIF4 字符串；其正文为数据库 CI/CD。`KanghoonYoon/morphoquant_hif4` 的 README 仅标题，不能据此确认同赛题。

网页搜索没有增加直接相关的 GitHub 来源；GitCode 命中已由主任务的另一发现子任务负责。

## 局限

公开搜索存在默认分支索引、延迟、结果上限与仓库无 README 等盲点，本记录是当日公开快照，不宣称已经找全互联网的所有方案。源码读取范围仅支持赛题归属与接口声明；未执行、评测或吸收算法。所有作者自述规则、历史分数和估时均不覆盖本项目较新的官方/用户依据。

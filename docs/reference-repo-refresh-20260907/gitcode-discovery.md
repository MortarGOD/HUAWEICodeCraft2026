# GitCode 公开仓库发现记录（2026-09-07）

本轮发现并在线核验 **3 个此前清单之外的相关公开仓库**：`Curiosi/quantizer`、`yuchunfei/quantizer`、`Austrini/Quantize`。前两者与已知 `qq_41888214/quantizer` 构成一个上游和两个 fork，不能合并为 URL 别名。`Austrini/Quantize` 已由主任务读取固定提交的静态源码，确认包含本赛题六个 API。

本发现子任务只做公开检索、网页阅读及 `git ls-remote`，没有 clone、改动镜像、执行参考算法或推送。全部分支获取和静态审计由主任务整合。机器可读查询、状态及分支 SHA 见 [gitcode-discovery.json](gitcode-discovery.json)。

## 已确认仓库

| 仓库 | 本轮身份 | 公开分支数 | 默认分支与 HEAD | 证据 |
| --- | --- | ---: | --- | --- |
| [Curiosi/quantizer](https://gitcode.com/Curiosi/quantizer) | 新发现，上游 | 2 | `main` / `4e916451f682748fc35bde0baa3f8f2a21005c30` | 站内 quantizer 搜索、公开六 API README、live Git refs |
| [yuchunfei/quantizer](https://gitcode.com/yuchunfei/quantizer) | 新补齐，Curiosi 的 fork | 8 | `main` / `f7985440100ffe3a494c88b425e45c8a3494cdb9` | 本地参考资料线索，经公开主页和 live Git 独立核验 |
| [qq_41888214/quantizer](https://gitcode.com/qq_41888214/quantizer) | 已知，Curiosi 的 fork | 9 | `main` / `8b85f229cc673dc421f98eeae8d068da6ca29e4e` | 精确 API 网页检索、live Git refs |
| [Austrini/Quantize](https://gitcode.com/Austrini/Quantize) | 新发现，六 API 静态源码确认 | 1 | `main` / `675c4b69c82d3fc08a21e33180ad639d1801a561` | 站内搜索、live Git refs、主任务固定提交源码核验 |
| [Mrxxx7/test](https://gitcode.com/Mrxxx7/test) | 已知 | 24 | `main` / `2ffcfe109f5b0d5be373aa252b65c7a52b8cef10` | live Git refs；`base_1.7=916c56b93dd0fb9e5e1dd797e7758b7b6b629f4b` |

以上 5 个公开相关远端合计 **44 个分支**。这是本轮发现范围内的实测远端引用数量，不是 GitCode 全站赛题仓库总量。

[Curiosi 的公开 forks 页](https://gitcode.com/Curiosi/quantizer/forks)显示 2 项，已逐项读取：`yuchunfei/quantizer`、`qq_41888214/quantizer`。上游当前主页显示 217 次提交；两个 fork 各自的分支集和 HEAD 不同。

`Curiosi/quantizer` 的两个分支为 `main`、`submission`；后者固定在 `467925a8ddaf435003148b02ce4d4d3c35ddda17`。`yuchunfei/quantizer` 包含 `main`、`codex/hif4-v10-kunpeng`、六/七轮扫描、diagonal、block/local-Hessian 性能预算分支，以及 `codex/hif4-v11-canary-20260902`。分支名只用于定位，不能据此认定实现有效或有线上收益。其 main 提交主题为 “Fix large-data timeout: drop block-Hessian weight search”。

## 搜索范围与分页

使用 agent-reach 的搜索/网页工作流，联网后端为 `web.run` 与 GitCode 公开网页。外部网页搜索共 **16 个查询**，涵盖六个完整接口名、`高精度4Bit数值转换`、`2026华为算法大赛`、HiF4 / NVFP4 / HiFloat4 及组合。完整字面查询保存在 JSON。外部索引中的相关仓库命中主要是已知 `qq_41888214/quantizer`；无穷尽总数或可用分页，所以不作全网不存在其他实现的推断。

GitCode 站内项目搜索实际完成如下；以下总数只取结果加载完成后的显示值：

| 查询 | 显示项目总数 | 已查页 | 实际可见结果 | 判断 |
| --- | ---: | --- | --- | --- |
| `hif4_calibration_and_quantize_weight` | 0 | 1 | 0 | 项目检索未命中，不代表代码不存在 |
| `hif4` | 6 | 1 | 6 | HiFi、HiFV3Dsolver、Wan14b-hif8 等模糊命中 |
| `nvfp4` | 2 | 1 | 2 | Tutel、neural-compressor，通用框架 |
| `华为算法大赛` | 7 | 1 | 7 | CANN 比赛入口/地区页面，其他赛道 |
| `quantizer` | 17 | 1、2 | 6 + 5 | 找到 Curiosi、Austrini，后者由主任务静态源码确认。总数与渲染数不一致 |
| `高精度4Bit数值转换` | 1 | 1 | 1 | DynaFEM-GPH，有限元/SPH 转换，无关 |
| `hifloat4` | 4 | 1 | 4 | 官方格式仿真、HiFloat8、resources 等背景资料 |
| `2026华为算法大赛` | 17 | 1、2 | 10 + 7 | CANN、终端、开发者、EDA、RISC-V 等；两页间有重复项 |

`quantizer` 两页其余可见名称为 `gitmylo/bark-voice-cloning-HuBERT-quantizer`、`ai-dev/ai-edge-quantizer`、`rickyzhu/mixed-moe-quantizer`、`lucidrains/vector-quantize-pytorch`、`gcw_RRBWdjVD/modelSlim-quantize`、ECCV2022 VQFR 镜像、CVPR2023 Q-DETR 镜像、`qq_56418536/Quantize`、`Ppsjq/quantize_gcn`；其展示说明或时间未提供本赛题相关依据，未计为相关实现。

## 静态源码补证与限制

[Austrini/Quantize](https://gitcode.com/Austrini/Quantize) 已通过 Git transport 确认为公开远端：1 个 `main`，HEAD `675c4b69c82d3fc08a21e33180ad639d1801a561`；主页显示 7 次提交、Python 100%、pytorch 标签、README 为空。网页目录未正常渲染，提交列表要求登录，发现阶段曾将相关性标为待核。随后主任务读取该固定提交的 `solution.py`：第 21–24 行标明 NVFP4 / HiF4，六个 API 定义分别位于第 960、1258、1319、1722、1747、1763 行，已确认属于本赛题。镜像位于 `docs/papers/hif4-public-references/gitcode-Austrini-Quantize`；核验只读取 Git blob，没有执行参考算法。

GitCode 全局搜索界面只显示项目、模型、数据集、Issues、PR、组织、用户、标签，未提供本轮可用的全站代码搜索。其模糊项目搜索会漏掉已经公开可访问的已知仓库，所以检索结果不构成源码覆盖证明。

从当前搜索页面实际加载的 `index-0e021784.js` 和 `index-e12b0320.js` 观察到公开前端读取端点 `/api/v1/search/nauth/query` 与参数，普通匿名 HTTP 请求返回 **418**；没有猜测端点或尝试绕过，继续使用公开网页。`web.run` 打开 GitCode 页面还出现间歇超时/安全错误，均与“仓库不存在”分开记录。

外部搜索缓存曾把 `yuchunfei/quantizer` 标成 2 个分支，live Git 与当次浏览器主页均为 8 个，因此分支数量以 live Git refs 为准。部分网页目录显示“暂无内容”，但同页有 README、有效 HEAD 和提交计数，不能据此认定 Git 树为空。

本轮不声称覆盖私有仓库、隐藏/删除分支、未推送代码、登录后才能查看的内容，也不把分支名或自述当作已验证算法收益。

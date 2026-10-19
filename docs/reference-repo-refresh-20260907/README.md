# 2026-09-07 公开代码仓检索与刷新证据

综合结论见[搜索与全分支更新报告](../赛题公开代码仓搜索与全分支更新-20260907.md)。本目录将发现、抓取、文件可读性和算法分析分开记录；当前完成前述资料工作，没有执行外部算法。

| 文件 | 内容 |
|---|---|
| [all-repositories.json](all-repositories.json) | 24 个仓库的统一入口、75 条固定分支 SHA、分支变化与源码范围 |
| [remote-heads.tsv](remote-heads.tsv) | 每条远端分支一行，含 URL、完整提交和新/旧仓库分组 |
| [github-discovery.md](github-discovery.md) / [JSON](github-discovery.json) | GitHub 查询、分页、fork、同题源码证据；JSON 包含原始响应，约 15 MB |
| [gitcode-discovery.md](gitcode-discovery.md) / [JSON](gitcode-discovery.json) | GitCode 站内/外部搜索、公开 fork 与 live refs |
| [existing-refresh.json](existing-refresh.json) | 原有 14 个镜像初始状态、fetch、完整分支、源码 hydration、工作区保留；保留已恢复的失败尝试 |
| [new-repos-refresh.json](new-repos-refresh.json) | 10 个新镜像、23 条分支、源码和初步 ZIP hydration、精确 sparse checkout |
| [additional-tree-scope.json](additional-tree-scope.json) | 全部 75 个提交目录的补充 ZIP 文件名扫描、第三方 gitlink 列表 |
| [supplemental-archives.json](supplemental-archives.json) | 文件名筛选之外的 ZIP 补齐、对象可读性和原工作区保留 |
| [archive-inventory.md](archive-inventory.md) / [JSON](archive-inventory.json) | 全部已确认 ZIP 的成员清单、Python 源码哈希、静态读取结果和限制 |
| [final-verification.md](final-verification.md) / [JSON](final-verification.json) | 独立只读验收：引用、源码对象、工作区和项目算法哈希 |

维护工具：[refresh_reference_repos.py](../../tools/refresh_reference_repos.py)、[materialize_new_reference_repos.py](../../tools/materialize_new_reference_repos.py)。必须显式传入仓库名单，工作区不干净时仅 fetch，不执行参考程序。工具后来扩展 ZIP 分类；初始抓取 JSON 的工具 SHA 保留历史值，新 SHA 见补充记录。

“全部分支”指相应抓取时间公开远端返回的全部 heads。完整提交历史不等于所有二进制/历史 blob 已下载；仅有链接的通用框架和第三方子模块不计入 24 个镜像。

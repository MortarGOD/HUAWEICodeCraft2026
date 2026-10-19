# 最终离线验收（2026-09-07）

结论：通过。本轮独立复核 **24 个仓库、75 个记录远端分支**，核对本地 `refs/remotes/origin/*`、每个分支提交与根树对象，以及每个选定文件在对应提交中的路径和 blob 绑定。

**5494 个选定 blob（各仓库去重后求和）全部在本地**；跨仓库全局去重为 5327 个。其中源码 blob 各仓库去重求和 1998 个。覆盖各分支文件出现次数 10175，其中源码文件出现次数 4280。选定集合还包含文档和配置，不能把总 blob 数全称为算法源码。

全部仓库均为 non-shallow。23 个工作树干净；`chiparon-HscCon` 的原有 10 个修改逐项保留，HEAD、分支、NUL 分隔状态、diff 大小、diff SHA256 与 staged diff SHA256 均与刷新前 initial snapshot 一致。

`test` 仍在 `base_1.7`，HEAD `916c56b93dd0fb9e5e1dd797e7758b7b6b629f4b`。主项目两个算法文件 SHA256 均与给定基线一致：

| 文件 | SHA256 |
| --- | --- |
| `solution/solution.py` | `bd4848d2b9ce1340bef64bcbcf9e4d1b6a225b52b66b7044ff6adaa15b824d86` |
| `solution/solution_v425.py` | `a6e53f37abf2c7c51c99e6439df3e554fa57d75fbdc99f0c12f780a9e5691fcb` |

| 仓库 | 分支 | 选定 blob（仓库内去重） | 验收 |
| --- | ---: | ---: | --- |
| `abdelfattah-NVFP4-RaZeR` | 3 | 79 | 通过 |
| `actypedef-ARCQuant` | 1 | 52 | 通过 |
| `analytistic-T4bit` | 1 | 37 | 通过 |
| `chiparon-HscCon` | 3 | 72 | 通过 |
| `gitcode-quantizer` | 9 | 661 | 通过 |
| `global-computing-consortium-HiFloat4` | 1 | 49 | 通过 |
| `jansenin-quantization-contest-work` | 1 | 66 | 通过 |
| `lsjfy-hif4-nvfp4` | 2 | 42 | 通过 |
| `mit-han-lab-fouroversix` | 3 | 263 | 通过 |
| `openaddr-hif4-contest` | 1 | 222 | 通过 |
| `suibianll-AI-Contest` | 1 | 2557 | 通过 |
| `test` | 24 | 360 | 通过 |
| `youxilee-hif4` | 1 | 7 | 通过 |
| `zhaow22-NVFP4-HiFloat4` | 1 | 15 | 通过 |
| `Tarjan01-huawei_algor_contest` | 1 | 30 | 通过 |
| `xwcai98-2026-huawei-alogorithm-test` | 2 | 383 | 通过 |
| `Lulili520-huawei-code-competition-2026` | 2 | 79 | 通过 |
| `klompn24-crypto-2026_compatition` | 1 | 171 | 通过 |
| `wwwbby-code` | 4 | 19 | 通过 |
| `qcjiang-2026-huawei-alogorithm-test` | 1 | 12 | 通过 |
| `MortarGOD-2026-huawei-alogorithm-test` | 1 | 60 | 通过 |
| `gitcode-yuchunfei-quantizer` | 8 | 96 | 通过 |
| `gitcode-Curiosi-quantizer` | 2 | 161 | 通过 |
| `gitcode-Austrini-Quantize` | 1 | 1 | 通过 |

本轮设置 `GIT_NO_LAZY_FETCH=1`、空的 `GIT_ALLOW_PROTOCOL`、`GIT_OPTIONAL_LOCKS=0` 和 `PYTHONDONTWRITEBYTECODE=1`，仅调用本地读取命令，没有联网或触发下载，没有修改镜像、主算法或其他报告。

范围边界：远端新鲜度来自 `existing-refresh.json` 与 `new-repos-refresh.json` 中保存的联网快照，本轮不再次联网；non-shallow 不等于所有历史 blob 或大型二进制数据已下载。压缩包由独立步骤验收，本报告不重复覆盖。没有执行参考算法，也不据此声称质量或线上收益。

完整逐分支哈希、对象与保留状态检查见 [final-verification.json](final-verification.json)。

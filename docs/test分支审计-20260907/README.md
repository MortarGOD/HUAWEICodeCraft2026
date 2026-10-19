# 全分支源码快照台账

[总分析与开发优先级](../Mrxxx7-test全分支分析-20260907.md) · [算法覆盖矩阵](coverage.md) · [历史回执](history-and-results.md) · [源码校验](verification.json)

抓取时间（UTC）：2026-09-07T02:55:37.140239+00:00。远端 24 个 heads 均与本地 origin 引用逐 SHA 相同；`git fsck --connectivity-only --no-dangling` 通过，仓库不是浅克隆。

所有分支的完整 Git 对象已保存在原参考仓库；另导出 3290 份 UTF-8 文本文件（约 80.47 MB）及 82 份 ZIP 内 Python 成员，保持原始字节。未导出的 `.pt` 和 `.zip` 二进制仍可从 Git 对象恢复。没有执行参考算法或其脚本。

HEAD 主入口经 AST 去重只有 3 类：22 个分支均为 tensor-recon-baseline-v1；base_1.7 为 Compensated V2；unrestricted-highscore-v1 为 block-Hessian 基线。包含子目录、后端和归档后，本次分支 HEAD 共 55 个算法 AST 组。此数字不包括另外恢复的历史版本或后端内嵌载荷，不代表 55 个独立数学方法。

[完整文件/函数/hash台账](manifest.json) · [算法去重位置映射](algorithm-files.json) · [远端heads原始记录](remote-heads.tsv) · [导出脚本](export_snapshot.py)

| 分支 | 固定 commit | 主入口类型 | 文本文件 / ZIP Python | 源码入口 |
|---|---|---|---|---|
| `base3.07` | `cbc6d10a119a` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/base3.07/solution/solution.py) |
| `base_1.7` | `52b490459d11` | Compensated V2 | 130 / 4 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/base_1.7/solution/solution.py) |
| `experiment/attention-fisher-objective` | `532b022f9165` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--attention-fisher-objective/solution/solution.py) |
| `experiment/attention-learned-rotation` | `850f9e6b3f78` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--attention-learned-rotation/solution/solution.py) |
| `experiment/block-hessian-objective` | `51dbfe5fb46e` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--block-hessian-objective/solution/solution.py) |
| `experiment/joint-output-optimization` | `bd9735c76c69` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--joint-output-optimization/solution/solution.py) |
| `experiment/legal-inputisolated-v4` | `307f1e6beda6` | 独立重构基线 | 145 / 3 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v4/solution/solution.py) |
| `experiment/legal-inputisolated-v5` | `39ca2a0496ec` | 独立重构基线 | 171 / 6 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-inputisolated-v5/solution/solution.py) |
| `experiment/legal-tensor-recon-v2` | `d1aa97f53fd4` | 独立重构基线 | 143 / 2 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--legal-tensor-recon-v2/solution/solution.py) |
| `experiment/linear-residual-fit-v2` | `2d0d18ee48cc` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--linear-residual-fit-v2/solution/solution.py) |
| `experiment/new-paper-implementation` | `4c42e3598ae0` | 独立重构基线 | 136 / 3 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--new-paper-implementation/solution/solution.py) |
| `experiment/non-rrs-paper-baselines` | `d03e612e3271` | 独立重构基线 | 129 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--non-rrs-paper-baselines/solution/solution.py) |
| `experiment/output-aware-rounding` | `5f04c8f17667` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--output-aware-rounding/solution/solution.py) |
| `experiment/paper-solution-suite` | `fbedb77fb124` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--paper-solution-suite/solution/solution.py) |
| `experiment/pca-hadamard-attention` | `b512632f931a` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--pca-hadamard-attention/solution/solution.py) |
| `experiment/qwen-channel-balance` | `e0ee0d5d2d96` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--qwen-channel-balance/solution/solution.py) |
| `experiment/smoothrot-19k` | `98e66ceb0785` | 独立重构基线 | 133 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--smoothrot-19k/solution/solution.py) |
| `experiment/smoothrot-32k` | `c24a35be3588` | 独立重构基线 | 133 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--smoothrot-32k/solution/solution.py) |
| `experiment/smoothrot-32k-v2` | `389307549993` | 独立重构基线 | 133 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--smoothrot-32k-v2/solution/solution.py) |
| `experiment/soar-hif4` | `a407b665db32` | 独立重构基线 | 132 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--soar-hif4/solution/solution.py) |
| `experiment/unrestricted-headadaptive-v2` | `849fd2408cfc` | 独立重构基线 | 175 / 6 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-headadaptive-v2/solution/solution.py) |
| `experiment/unrestricted-highscore-v1` | `5d6c1b7b9214` | block-Hessian 基线 | 194 / 41 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/experiment--unrestricted-highscore-v1/solution/solution.py) |
| `feature/hif4-solution` | `92a44857432e` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/feature--hif4-solution/solution/solution.py) |
| `main` | `2ffcfe109f5b` | 独立重构基线 | 128 / 1 | [solution.py](../papers/hif4-public-references/test-branch-snapshots-20260907/main/solution/solution.py) |

快照按分支名将 `/` 编码为 `--` 存放，源码与压缩包版本不同的情况分别保留。当前参考仓库仍停留在原 `base_1.7` 工作分支，根项目算法入口未修改。

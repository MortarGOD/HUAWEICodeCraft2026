# `solution_cxw.py` 差异审计

## 结论

- Linear 不替换 v120：cxw 是较早的 v22 型 global-output + feedback 方案，global rank 仅32、offset仅0..4，缺少 v32/v117-v120 已线上验证的 rank2048、四轮 Q(W) 扫描、shape/stride 候选替换与稳健门控。本地前三个权重15题为8.644498，低于v120的9.154595（-5.57%），虽然快约36%，但v120线上仍有约32秒余量。
- Attention 最值得借鉴的是 per-head K-center。对每个head所有token的K减同一向量，只给固定query的全部key logits减同一常数，因此softmax严格不变；量化前中心化可缩小K动态范围。
- cxw 的 residual-K RMS blend 只在中心能量比≥0.69时启用；修正本地集中心能量比仅约0.004，因此本地实际差异几乎全部来自K-center。
- cxw 全量 Attention 本地5.196944，等于隔离迁移的v150，低于v120的5.513893；但v124/v148已证明当前本地Attention代理与线上RRS方向相反，所以v150仍作为低成本单变量线上探针，不把cxw整包并入。

## Attention 机制拆分

| 机制 | 合法性 | 成本 | 判断 |
|---|---:|---:|---|
| per-head Q/K RMS互逆平衡 | 严格保持QK logits | 低 | v120已有 |
| 稀疏度门控full-head Hadamard | Q/K成对，严格保持logits | 中 | v120已有，当前线上RRS整体负向 |
| K-center | softmax对每行logits常数平移不变 | 极低 | v150隔离线上验证 |
| residual-K RMS blend | 与Q做互逆scale，严格合法 | 低 | cxw阈值在本地不触发；待K-center线上正向后再测 |
| Q/K能量漂移scale损失 | 输出仍为合法HiF4 | 中 | v120已有 |

## Linear 机制拆分

cxw 可识别的机制——64维Hadamard/sign、Q(W)低秩输出度量、局部residual feedback、校准A@W门控——均已在现有主线以更高rank、更多候选和线上反馈门控实现。可借鉴的只有“rank32极速回退档”，它属于性能折中而非精度新算法；在v120仍未超时时不应牺牲约5.57%的本地Linear收益。

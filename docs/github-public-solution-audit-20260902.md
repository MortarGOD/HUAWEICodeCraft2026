# GitHub 公开 HiF4 赛题方案审计（2026-09-02）

## 范围与边界

本次通过 GitHub 公共代码搜索检索六个赛题 API、`NVFP4 -> HiF4`、Linear 与 Attention 相关实现。只阅读公开仓库；没有访问 GitHub 私有仓库，也不能访问其他 Codex 用户的私有任务或对话。

除特别说明外，下列参赛仓库均未声明开源许可证。因此这里只做算法审计和独立重实现依据，不直接复制其源码。

## 本地完整镜像

2026-09-02 已将五个公开仓库以非 shallow 的完整 Git clone 下载到：

`/Users/ee/Code/hif4-public-references/`

| 本地目录 | 下载时 HEAD |
|---|---|
| `openaddr-hif4-contest` | `f40932101ac2` |
| `suibianll-AI-Contest` | `11c53f70d409` |
| `youxilee-hif4` | `dd5ee6515323` |
| `analytistic-T4bit` | `4a2c4c744f38` |
| `jansenin-quantization-contest-work` | `a81b34c2f0b6` |

这些仓库保留各自的 `origin` 和完整历史，未嵌套进比赛主仓库。后续更新可在对应目录执行 `git pull --ff-only`。

## 找到的公开仓库

| 仓库 | 可核验成绩/证据 | 主要机制 | 对当前代码的价值 |
|---|---|---|---|
| [openaddr/hif4-contest](https://github.com/openaddr/hif4-contest) | 旧250 Linear/50 Attention数据集公开记录最高 v41 `25268 / 261s`；v42 未记录线上结果 | Weight/Activation GPTQ、64-block Hadamard、holdout gate、部署 Weight Gram 激活精化、Q/K logit Hessian、QKS、按 T 分档的精化轮数、大量性能消融 | 旧数据分数不能与当前新数据比较；其 holdout/cross-fold 思路和性能工程最有价值 |
| [suibianll/AI-Contest](https://github.com/suibianll/AI-Contest) | 新权重可复现最好 v86 `16744 / 222.7s`；README 提到 `17816`，但明确缺少对应源码/SHA/耗时 | BOAT、Weight/Activation HSDQ、`Q(W)^TQ(W)` Gram、`Q(W)^TW` cross、output-supervised W refine、Attention 端到端选择 | 当前分数低于我们，但 cross-fold 接受规则和误差分解评测器值得吸收 |
| [youxilee/hif4](https://github.com/youxilee/hif4) | 无可核验线上新数据成绩；CHANGELOG 给出固定 GPT-2 A/B | block SmoothQuant、完整协方差量化、Attention 端到端候选、Q/K 等价块变换、X/W Gauss-Seidel 联合残差 | v2.6 联合残差补偿与我方精确目标搜索同源；可作为实现正确性参照 |
| [analytistic/T4bit](https://github.com/analytistic/T4bit) | 无公开线上成绩 | 携带 `Q(W)`、`W-Q(W)`、back-projector；动态阶段以完整 `A@W` 目标做 64-block 层级/码值坐标下降 | 与 v287 最接近；验证“连续回投影中心 + 离散精确目标搜索”路线 |
| [jansenin/quantization-contest-work](https://github.com/jansenin/quantization-contest-work) | 公开最高约 `5326 / 120s` | 精确层级求解、role gate、H64、真实模型 capture/evaluator | 算法强度明显低于当前版本；数据捕获和负结果记录方法可借鉴 |

## 与当前 v287/v296 的实质对照

### 已经覆盖或更深入

1. **完整权重精确目标**：v287 已保存完整变换后原始 W 与部署 `Q(W)`，动态阶段直接构造 `A@W`，最小化 `||A@W^T-Q(A)@Q(W)^T||^2`。这覆盖 T4bit 的 exact-AW 主线。
2. **连续回投影 + 离散投影**：v287 已有 block inverse、Newton proposal、合法 HiF4 lookup、精确二次型接受；还包含 128-channel super-block 和 4-value microgroup 精化，比公开基础实现更深。
3. **Q(W)-Q(A) 联合校准**：v275 起已在五种 layout 上用真实输出选择替代 `Q(W)`，v285 对 up-projection 做 fixed-point Weight/encoder 联合更新。
4. **Attention 真实输出选择**：v287 已按校准 softmax 输出蒸馏三种 Q/K 策略，并按测试序列长度路由；不是只优化 Q/K operand MSE。
5. **运行时候选融合**：v212 以后共享候选准备，v254 以后把高成本 Newton 网格压缩成 lookup；公开仓库没有展示比当前主干更强且更快的在线搜索结构。

### 尚未充分利用、值得优先验证

1. **真正的 cross-fold candidate admission**（最高优先级）
   - 公开实现让 fold-0 生成的 Weight 候选必须改善 fold-1，反之亦然；最终再按 `mean + lambda * worst` 选。
   - 当前代码虽有五-layout 赢面和 worst-layout 门槛，但多数候选仍由同一组合校准数据生成并在这些数据上选择，不能完全阻止拟合泄漏。
   - 建议先作为 workbench，不立刻发版本：对 up/square/down 分别测“同折生成同折选”与“异折准入”的线上预测稳定性。

2. **Attention 双口径安全选择**
   - `youxilee`/`suibian` 用真实 causal softmax 输出作主目标，同时把另一 mask/窗口输出作为 worst-case safety 项；候选评分还接入最终实际量化器。
   - 当前按长度蒸馏策略仍可能在有限校准序列上过拟合。应把 5 个校准长度拆成 train/validation，策略由一部分产生，另一部分做端到端准入。

3. **误差源分解驱动预算**
   - 公开 evaluator 将 Linear 拆成 W-only/A-only/both/interaction，将 Attention 拆成 Q/K/V/QK/QKV、logits、softmax。
   - 这对当前最重要：先确认新数据的 Attention 剩余误差主要在 Q/K logits、softmax 放大还是 V，避免继续用总分给微策略做盲反馈。

4. **候选顺序而不是继续扩候选数**
   - T4bit 的顺序是：连续 back-projection center -> 层级 scale 候选 -> 精确 64x64 Gram 接受 -> 固定 hierarchy 后逐 mantissa coordinate descent。
   - v287 机制更丰富，但候选族和 micro passes 很重。可以测试把最强两类候选按上述顺序串行收敛，替代“一开始堆宽候选 bank”，目标是在相同时限内增加有效 sweep。

5. **按测试行数分配迭代预算**
   - `openaddr` 的真实线上曲线显示 T=1/10/128 的深 sweep 性价比明显高于 T=512/1024；最终采用 tiered sweep，而不是统一深度。
   - 我方已有类似门控，但应依据当前 50 Linear case 的新占比重新核算。新数据 Linear 只有 10 个权重组，省下的 Attention 时间可以重新分给短 T 的 exact-AW search。

## 公开仓库明确否定的方向

- 只在校准集拟合 PCA/数据驱动旋转，再原样迁移测试：多个仓库均观察到校准收益、测试回归。
- 继续扩大 SmoothQuant alpha、offset、seed 等小网格：边际收益远低于改目标函数或联合残差结构。
- V 的纯通道 Gram 精化：Attention 中 V 误差的关键耦合在 token 维的 `P^T P`；动态 V 接口拿不到当次 Q/K 时，单位通道 Gram 基本退化为普通重建 MSE。
- 无门控的深 Attention Hessian 精化：结构化 mini 可正收益，线上/独立样本常反转；必须 cross-fold 或端到端 holdout。
- 大规模精度路径直接堆叠：公开 v97、v98、v115-v125 等虽有本地增益，但 300s/新规则时限下大量超时；复杂度必须按动态调用计价。

## 下一步建议

1. 不把任何公开 `solution.py` 直接并入正式版本；先实现独立的 **cross-fold exact-output gate** workbench。
2. 用现有本地数据生成逐 case 的 Linear W/A/interaction 与 Attention Q/K/V/logits/softmax 分解表。
3. 若 cross-fold gate 能保住 v285 的本地收益并拒绝 v286/v296 这类线上回退候选，再基于 v285/v287 形成一个大版本；否则不出小版本。
4. Attention 下一版优先测试“按校准长度拆折的端到端 policy admission”，而不是再增加 predictor 或 scale bank。

## 关于“其他 Codex 用户”

GitHub 公共搜索能找到带 `AGENTS.md`、持续实验日志和 agent 工作流痕迹的仓库，但这些证据不足以确认作者具体使用了 Codex。Codex 不提供跨用户私有会话检索能力；因此只能引用作者主动公开到 GitHub 的内容，不能拿取其他用户未公开的解决方案。

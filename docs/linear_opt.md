Linear 4-bit 数据类型转换：优化思路与论文地图
1. 问题定义
目标是在两种 4-bit 数据类型之间做转换。
设：
	•	输入激活矩阵：(A)
	•	权重矩阵：(W)
	•	原始数据类型：4-bit type A
	•	目标数据类型：4-bit type B
	•	计算目标：
[Y = A W]
整个任务分为两个阶段。
1.1 Calibration 阶段
输入：
[A_{\text{calib}}, W]
其中：
	•	(W) 是完整权重矩阵；
	•	(A_{\text{calib}}) 是校准数据；
	•	输入数据使用 4-bit type A 表示。
Calibration 阶段允许输出：
[Q(W)]
以及任意允许的辅助信息：
[\mathcal M]
例如：
	•	scale
	•	zero point
	•	clipping threshold
	•	channel/block scale
	•	permutation
	•	rotation matrix
	•	affine transform
	•	codebook
	•	sensitivity information
因此 calibration 可以抽象为：
[(W,A_{\text{calib}})\rightarrow(Q(W),\mathcal M)]
1.2 Test 阶段
输入：
[A_{\text{test}}, Q(W), \mathcal M]
要求计算：
[

\hat Y
Q_B(A_{\text{test}})Q(W)]
或者等价的目标 4-bit 计算流程，使：
[\hat Y \approx A_{\text{test}}W]
最终优化目标是：
[\boxed{\min\left|A_{\text{test}}W-\hat A\hat W\right|}]
重点不是单独最小化：
[|W-\hat W|]
或者：
[|A-\hat A|]
而是直接最小化最终 GEMM 输出误差。

⸻

2. 最重要的优化思路
2.1 从 element-wise quantization 转向 output-aware quantization
最基础的方法通常优化：
[\min |W-Q(W)|_F^2]
但真正评价指标是：
[AW]
所以更合理的 calibration objective 是：
[\boxed{\min_\theta\sum_{A_i\in D_{\text{calib}}}|A_iW-Q_A(A_i;\theta_A)Q_W(W;\theta_W)|_F^2}]
其中：
[\theta]
可以包含：
	•	scale
	•	clipping
	•	transformation
	•	codebook
	•	rotation
这类思路和 GPTQ、AWQ、OmniQuant、AdaRound 等方法高度相关。

⸻

3. Scale 优化
3.1 Per-tensor scale
最简单：
[x_q=\operatorname{round}(x/s)]
整个 tensor 使用一个 scale：
[s.]
优点：
	•	简单
	•	metadata 少
缺点：
	•	对 outlier 非常敏感
	•	4-bit 情况通常精度不足

⸻

3.2 Per-channel scale
对于权重：
[W[:,j]]
分别使用：
[s_j.]
即：
[

\hat W_{:,j}
s_jQ(W_{:,j}/s_j)]
一般显著优于 per-tensor quantization。

⸻

3.3 Group-wise / Block-wise scale
把权重划分为 block：
[W={W_g}_{g=1}^{G}]
每组一个 scale：
[s_g.]
例如：
	•	group size 128
	•	group size 64
	•	group size 32
	•	group size 16
随着 group size 下降：
	•	精度通常提高
	•	metadata 增加
4-bit 场景下 group-wise quantization 是一个非常重要的 baseline。
相关思想：
	•	GPTQ
	•	AWQ
	•	QServe
	•	MXFP4
	•	microscaling quantization

⸻

4. Clipping 优化
直接使用：
[s=\frac{\max |x|}{q_{\max}}]
通常容易被 outlier 破坏。
可以寻找：
[

\alpha^*
\arg\min_\alphaL(AW,Q_\alpha(A)Q_\alpha(W))]
然后使用：
[

x’
\operatorname{clip}(x,-\alpha,\alpha)]
再量化。
可搜索：
	•	percentile clipping
	•	MSE clipping
	•	output-aware clipping
	•	learnable clipping
相关论文：
	•	AWQ
	•	OmniQuant
	•	AdaRound
	•	BRECQ

⸻

5. Activation-aware Weight Quantization
对于：
[Y=AW]
不同 weight channel 的重要性取决于输入 activation。
可以根据 calibration activation 计算：
[

m_j
E[|A_j|]]
或者：
[m_j=E[A_j^2]]
对重要 channel 给予更好的量化精度。
典型思路：
[W’_j=s_jW_j]
同时：
[A’_j=A_j/s_j]
于是：
[A’W’=AW.]
选择：
[s_j]
使 weight 更容易量化。
核心论文：
AWQ
AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration
核心思想：
activation 中的重要 channel 应该保护对应的 weight。
非常适合作为 calibration-driven scale learning 的参考。

⸻

6. SmoothQuant：等价变换
对于：
[Y=AW]
插入：
[S^{-1}S=I]
得到：
[

Y
(AS^{-1})(SW).]
于是可以重新分配：
	•	activation dynamic range
	•	weight dynamic range
找到：
[S]
使两边都更容易映射到低 bit。
核心论文：
SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
适合：
	•	W4A4
	•	activation outlier
	•	channel scaling

⸻

7. Rotation：消除 outlier
设：
[R^TR=I.]
那么：
[

AW
(AR)(R^TW).]
因为：
[RR^T=I.]
于是可以先对 activation 做：
[A’=AR]
同时权重：
[W’=R^TW.]
Full precision 输出完全不变：
[A’W’=AW.]
但 rotation 可以显著改善数值分布。
典型 rotation：
	•	Hadamard
	•	random orthogonal
	•	learned orthogonal
核心论文：
QuaRot
QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs
核心思想：
[A’=AR]
[W’=R^TW]
通过 orthogonal rotation 平摊 outlier。
非常适合 4-bit activation + weight。

⸻

8. Learned Rotation
固定 Hadamard 不一定是最优。
可以通过 calibration 学：
[

R^*
\arg\min_{R^TR=I}\sum_i|A_iW-Q(A_iR)Q(R^TW)|^2]
代表论文：
SpinQuant
SpinQuant: LLM Quantization with Learned Rotations
重点：
	•	rotation 可训练
	•	calibration-driven
	•	W4A4KV4
对于你的 calibration API，非常匹配。

⸻

9. Rotation + Permutation
如果 block quantization 使用固定 group size，outlier 集中在某些 group 会导致很大误差。
可以：
[A\rightarrowAR_1\rightarrowP\rightarrowR_2]
其中：
[P]
为 channel permutation。
目标是让不同 block 的 dynamic range 更均匀。
代表论文：
DuQuant
DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs
核心：
	•	rotation
	•	permutation
	•	second rotation
	•	W4A4
非常适合 block-wise 4-bit conversion。

⸻

10. General Affine Transformation
Orthogonal rotation 只是：
[R^{-1}=R^T]
的一种特殊情况。
更一般地，可以学习：
[T]
并使用：
[A’=AT]
[W’=T^{-1}W.]
于是：
[

A’W’
ATT^{-1}W
AW.]
这提供更大的 optimization space。

⸻

FlatQuant
FlatQuant: Flatness Matters for LLM Quantization
核心：
	•	learnable affine transformation
	•	calibration optimization
	•	Kronecker 
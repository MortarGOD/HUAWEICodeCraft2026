# Linear 无 activation_state 接口迁移预案

## 状态与启用条件

- 记录日期：2026-08-29。
- 本文仅为赛题组拟于2026-08-31取消Linear `activation_state` 后的预案。
- 在新函数签名、进程隔离和返回结构正式确认前，不落地代码、不创建迁移版本。
- 当前正式开发冻结v229 Linear，全力优化Attention。

## 新的核心目标

测试激活量化器必须是只依赖当前激活的固定函数 `Q0(A)`。校准阶段利用完整权重和五份校准激活，直接优化最终HiF4权重：

`Q(W)* = argmin_B sum_i ||A_i W^T - Q0(A_i) B^T||^2, B in HiF4`。

这会把原本测试阶段由`Q(A)`补偿`W-Q(W)`的自由度，迁移成校准阶段由`Q(W)`同时吸收权重误差和固定激活量化误差。

## 论文路线重排

1. 直接保留：GPTQ、OBC/OBQ、HiGPTQ、MR-GPTQ、AdaRound、BRECQ、AWQ敏感度、SqueezeLLM/BitDistiller敏感舍入、Four Over Six、MXFP4 OAS/MBS、MX+/M2XFP、OmniQuant LWC。所有结果必须固化进合法`Q(W)`。
2. 固定化使用：QuaRot/QuIP#/RRS的公共Hadamard与deterministic sign、ButterflyQuant固定结构、BASE-Q的均值和zero-collapse诊断。A/W两端只能执行代码中写死的同一变换。
3. 仅保留部分思想：SmoothQuant、Trainable Smooth-Rotation、SpinQuant、DuQuant、RPTQ、FlatQuant、AffineQuant。每个W不同的scale、rotation和permutation无法传给测试A，不能部署。
4. 放弃：AQLM自定义码本、SpQR/稀疏FP旁路、RILQ/LoRA、ARCQuant/DecDEC残差通道、Model Expansion及需要训练或额外GEMM的格式外方案。

## 无状态 Q(A) 自由度

- 当前A自身的HiF4 E6邻域、block SSE、clipping、受控overflow和舍入阈值搜索。
- peak/RMS、kurtosis、稀疏率、zero-collapse等当前张量统计。
- 按行数1/10/128/512/1024写死的全局规则。
- 固定Hadamard/sign；禁止per-W参数、Q(W)度量、校准布局标签、shape标签和candidate bank。

## 判题切换后的开发顺序

1. Stateless baseline：固定`Q0(A)`，修复接口与坐标一致性。
2. 比较identity、H64、fixed sign+H64、全通道Hadamard和低成本固定butterfly。
3. 针对固定`Q0(A)`训练residual-feedback `Q(W)`。
4. 求解带ridge的连续有效权重，再投影到HiF4；优先对角/64维块/低秩近似。
5. 加入GPTQ/HiGPTQ Hessian补偿、AdaRound mant、三级metadata和output-aware clipping。
6. 加入BRECQ跨块输出残差，最后做性能裁剪。

五种布局必须等权计算，使用拟合/验证行分割、至少3/5布局改善和最大布局回退门控，避免1024行布局支配目标。

## 切换前必须确认

- 两个Linear函数的新签名和准确返回结构。
- Dynamic Activation是否完全看不到`Q(W)`与Weight ID。
- Attention q/k/v state是否保持。
- 校准与测试是否分进程；是否禁止全局变量、文件、共享内存和native持久状态。
- 是否随机交错不同Weight的测试，以及总时间是否仍为420秒。

## Attention 4-bit 数据类型转换：优化思路与论文地图  
## 1. 问题定义  
Attention 输入为：  
[ Q,K,V ]  
数据使用 4-bit type A。  
目标是在 calibration 数据帮助下，将其转换到 4-bit type B，并尽量保证最终 attention 输出精度。  
标准 attention：  
## [  
##    
## S  
\frac{QK^T}{\sqrt d} ]  
## [  
##    
## P  
Softmax(S) ]  
## [  
##    
## O  
PV. ]  
低比特计算：  
## [  
##    
## \hat S  
\frac{\hat Q\hat K^T}{\sqrt d} ]  
## [  
##    
## \hat P  
Softmax(\hat S) ]  
## [  
##    
## \hat O  
\hat P\hat V. ]  
最终目标：  
[ \boxed{ \min |O-\hat O| } ]  
而不是仅仅优化：  
[ |Q-\hat Q| + |K-\hat K| + |V-\hat V|. ]  
   
⸻  
   
## 2. Calibration 阶段  
输入：  
[ Q_c,K_c,V_c ]  
可以输出任意允许的辅助信息：  
[ \mathcal M. ]  
例如：  
## [  
##    
## \mathcal M  
{ s_Q,s_K,s_V, R, T, P, \alpha, C }. ]  
包括：  
* per-head scale  
* per-channel scale  
* per-block scale  
* clipping threshold  
* rotation  
* permutation  
* affine transform  
* learned codebook  
* head sensitivity  
* token sensitivity  
测试阶段：  
[ Q_t,K_t,V_t,\mathcal M \rightarrow \hat O. ]  
   
⸻  
   
## 3. Attention 为什么比 Linear 更难  
Linear 只有：  
[ AW. ]  
Attention 多了一层：  
[ QK^T \rightarrow Softmax \rightarrow PV. ]  
尤其 Softmax：  
## [  
##    
## P_{ij}  
\frac{e^{S_{ij}}} {\sum_ke^{S_{ik}}} ]  
会放大某些很小的 score error。  
因此：  
[ |Q-\hat Q| ]  
很小，并不代表：  
[ Softmax(QK^T) ]  
误差也小。  
这意味着 calibration 应该直接围绕最终 attention output 优化。  
   
⸻  
   
## 4. 推荐的 Attention Calibration Loss  
建议定义：  
## [  
##    
## L  
\lambda_1L_S + \lambda_2L_P + \lambda_3L_O. ]  
## Score loss  
## [  
##    
## L_S  
| QK^T-\hat Q\hat K^T |_F^2. ]  
## Softmax loss  
## [  
##    
## L_P  
D_{KL} (P|\hat P) ]  
或者：  
[ |P-\hat P|_F^2. ]  
## Final output loss  
## [  
##    
## L_O  
| PV-\hat P\hat V |_F^2. ]  
如果最终只评价 attention 输出：  
[ \boxed{ \lambda_3 \gg \lambda_1,\lambda_2 } ]  
甚至可以直接：  
[ \boxed{ L=L_O. } ]  
   
⸻  
   
## 5. 路线一：直接 4-bit Attention  
## SageAttention  
**SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration**  
研究：  
[ QK^T ]  
和：  
[ PV ]  
两个 attention GEMM 的数值特征。  
重要结论：  
Q/K 和 P/V 不一定应该使用同一种 quantization strategy。  
   
⸻  
   
## SageAttention2  
**SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization**  
ICML 2025。  
直接研究：  
[ Q,K\rightarrow INT4. ]  
然后：  
[ Q_{INT4}K_{INT4}^T. ]  
主要方法：  
* Q smoothing  
* per-thread INT4  
* fine-grained quantization  
* PV 使用更高精度  
这和 4-bit Q/K attention 非常接近。  
   
⸻  
   
## SageAttention3  
**SageAttention3: Microscaling FP4 Attention for Inference**  
直接研究：  
[ \boxed{FP4 Attention} ]  
尤其是 microscaling FP4。  
形式：  
## [  
##    
## x_g  
s_gq_g. ]  
不同 block：  
[ g ]  
有不同：  
[ s_g. ]  
Calibration metadata 可以直接设计成：  
[ { s_{Q,g},s_{K,g},s_{V,g} }. ]  
如果 type A/type B 中有 FP4，这篇属于必读。  
   
⸻  
   
## Attn-QAT  
**Attn-QAT: 4-Bit Attention With Quantization-Aware Training**  
研究 FP4 attention 中：  
* heavy-tailed activation  
* dynamic range  
* score error  
* softmax sensitivity  
虽然属于 QAT，但对 PTQ calibration 也很有参考价值。  
   
⸻  
   
## 6. 路线二：Q/K Coupled Transformation  
这是 Attention 最值得研究的一个方向。  
   
⸻  
   
## Orthogonal Rotation  
设：  
[ R^TR=I. ]  
定义：  
[ Q’=QR ]  
[ K’=KR. ]  
那么：  
## [  
##    
## Q’K’^T  
QR(KR)^T ]  
## [  
QRR^TK^T ]  
## [  
QK^T. ]  
所以 full-precision attention score 完全不变。  
但：  
[ Q_B(QR) Q_B(KR)^T ]  
可能显著优于：  
[ Q_B(Q) Q_B(K)^T. ]  
   
⸻  
   
## 7. QuaRot  
**QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs**  
NeurIPS 2024。  
核心：  
* orthogonal rotation  
* Hadamard  
* 4-bit activation  
* 4-bit weight  
* KV cache  
Attention 中可以直接使用：  
[ Q’=QR ]  
[ K’=KR. ]  
Calibration 输出：  
[ R. ]  
Test 时直接应用。  
这是最容易实现、风险最低的 transformation baseline 之一。  
   
⸻  
   
## 8. SpinQuant  
**SpinQuant: LLM Quantization with Learned Rotations**  
ICLR 2025。  
相比 QuaRot：  
* QuaRot 使用固定/random rotation  
* SpinQuant 用 calibration 学最优 rotation  
优化：  
## [  
##    
## R^*  
\arg\min_{R^TR=I} \sum_i L( O_i, \hat O_i(R) ). ]  
这和允许 calibration 输出 auxiliary information 的任务非常匹配。  
   
⸻  
   
## 9. DuQuant  
**DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs**  
NeurIPS 2024 Oral。  
使用：  
[ Rotation + Permutation + Rotation. ]  
目的：  
将 outlier 更均匀地分配到不同 block。  
对于：  
* group-wise FP4  
* INT4  
* microscaling  
尤其有效。  
   
⸻  
   
## 10. OSTQuant  
**OSTQuant: Orthogonal and Scaling Transformations for Better Distribution Fitting**  
同时学习：  
[ R ]  
和：  
[ S. ]  
例如：  
[ Q’=QRS ]  
并设计对应的 K 逆变换保持：  
[ QK^T ]  
不变。  
它提出 QSUR：  
Quantization Space Utilization Rate。  
核心思想：  
不仅减少 outlier，还要充分使用有限的 16 个 quantization levels。  
   
⸻  
   
## 11. General Q/K Affine Transformation  
Orthogonal rotation 可以推广。  
令：  
[ Q’=QT ]  
同时：  
[ K’=KT^{-T}. ]  
那么：  
## [  
##    
## Q’K’^T  
QT(KT^{-T})^T ]  
## [  
QTT^{-1}K^T ]  
## [  
QK^T. ]  
因此：  
[ \boxed{ Q’=QT,\qquad K’=KT^{-T} } ]  
是一个非常重要的 attention-specific invariance。  
相比 rotation：  
[ T ]  
不要求正交，所以 optimization space 更大。  
   
⸻  
   
## 12. FlatQuant  
**FlatQuant: Flatness Matters for LLM Quantization**  
使用 learnable affine transformation。  
可以借鉴到 attention：  
## [  
##    
## T^*  
## \arg\min_T  
##    
## \sum_i  
##    
## |  
##    
## Attention(Q_i,K_i,V_i)  
Attention( Q_B(Q_iT), Q_B(K_iT^{-T}), Q_B(V_i) ) |^2. ]  
这是非常值得尝试的高级方案。  
   
⸻  
   
## 13. AffineQuant  
**AffineQuant: Affine Transformation Quantization for Large Language Models**  
同样研究可逆 affine transformation。  
可以作为：  
[ T,T^{-1} ]  
optimization 的参考。  
   
⸻  
   
## 14. Q/K/V 不应该使用同一种 quantization  
Attention 中：  
[ Q,K,V ]  
功能不同。  
Q/K 决定：  
## [  
##    
## P  
Softmax(QK^T). ]  
V 只参与：  
[ PV. ]  
因此可以分别使用：  
[ Q: {s_Q,\alpha_Q,T_Q} ]  
[ K: {s_K,\alpha_K,T_K} ]  
[ V: {s_V,\alpha_V,T_V}. ]  
尤其 Q/K 对 score 和 softmax ranking 更敏感。  
   
⸻  
   
## 15. Per-head Quantization  
Multi-head attention：  
[ Q= [Q_1,\dots,Q_H]. ]  
每个 head 分布不同。  
因此推荐：  
[ s_{Q,h}, s_{K,h}, s_{V,h}. ]  
比整个 tensor 一个 scale 更合理。  
还可以 calibration 得到：  
[ \alpha_h ]  
或者：  
[ R_h. ]  
   
⸻  
   
## 16. Per-channel / Per-block Quantization  
对于每个 attention head，再进一步做：  
[ s_{h,g}. ]  
例如 block size：  
* 128  
* 64  
* 32  
* 16  
目标：  
[ Q_g\approx s_gq_g. ]  
这与 SageAttention3 / microscaling FP4 高度一致。  
   
⸻  
   
## 17. Softmax Quantization  
Softmax output：  
[ P ]  
通常极度 non-uniform。  
大量值接近：  
[ 0. ]  
因此 uniform INT4：  
[ 0,\frac1{15},…1 ]  
通常效率很低。  
   
⸻  
   
## 18. FQ-ViT  
**FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer**  
IJCAI 2022。  
提出：  
**Log-Int-Softmax**  
核心思想：  
用 logarithmic representation 对 softmax probability 量化。  
可以理解为：  
[ q(P) \approx -\log_2P. ]  
它证明 attention map 可以做到非常低 bit。  
如果 type B 支持非均匀 4-bit datatype，这篇非常重要。  
   
⸻  
   
## 19. RepQ-ViT  
**RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers**  
ICCV 2023。  
针对：  
* LayerNorm  
* Softmax  
特殊 activation 分布设计 quantizer，并通过 scale reparameterization 让推理更硬件友好。  
   
⸻  
   
## 20. I&S-ViT  
提出：  
**Shift-Uniform-Log2 Quantizer**  
相比直接：  
[ \log_2 ]  
进一步改善 Softmax quantization。  
说明：  
4-bit B datatype 不一定应该是 uniform datatype。  
   
⸻  
   
## 21. AdaLog  
**AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer**  
ECCV 2024。  
将固定：  
[ \log_2 ]  
变成：  
[ \log_b. ]  
Calibration 搜索：  
[ b^*. ]  
即：  
[ P_{\text{calib}} \rightarrow b^*. ]  
测试阶段：  
[ P \rightarrow Q_{\log_{b^*}}(P). ]  
这是很典型的 calibration-learned datatype。  
   
⸻  
   
## 22. PTQ4ViT  
**PTQ4ViT: Post-training Quantization for Vision Transformers with Twin Uniform Quantization**  
ECCV 2022。  
重要思想：  
MSE 不一定能够准确评价 Transformer quantization quality。  
因此使用 Hessian-guided metric。  
例如：  
[ L \approx \Delta x^TH\Delta x. ]  
对你的启发：  
不要单独优化：  
[ |Q-\hat Q|. ]  
应该更加关注：  
[ \Delta Q ]  
对最终：  
[ O ]  
的 sensitivity。  
   
⸻  
   
## 23. FIMA-Q  
**FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation**  
CVPR 2025。  
使用 Fisher Information 评价 quantization sensitivity。  
可以进一步设计：  
[ w_h ]  
表示每个 attention head 的 sensitivity。  
优化：  
[ \sum_hw_hL_h. ]  
或者：  
[ w_{token}. ]  
   
⸻  
   
## 24. KV Cache 系列  
即使你的任务不是 autoregressive KV cache，这些论文仍然很值得参考，因为它们深入分析了：  
[ K ]  
和：  
[ V ]  
的分布差异。  
   
⸻  
   
## 25. KIVI  
**KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache**  
重要经验：  
[ K:\ per-channel ]  
[ V:\ per-token ]  
通常更合适。  
说明：  
[ K,V ]  
不能默认使用完全相同的量化方法。  
   
⸻  
   
## 26. KVQuant  
**KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization**  
关键技术：  
* per-channel K  
* Pre-RoPE quantization  
* non-uniform datatype  
* sensitivity-weighted quantization  
* dense + sparse representation  
尤其值得参考：  
[ \boxed{\text{calibration learned non-uniform datatype}} ]  
与 4-bit A → 4-bit B 非常相关。  
   
⸻  
   
## 27. SKVQ  
**SKVQ: Sliding-window Key and Value Cache Quantization**  
使用：  
* channel reorder  
* clipping  
* group quantization  
* calibration  
对于你的辅助信息：  
[ P,\alpha,s_g ]  
设计很有参考价值。  
   
⸻  
   
## 28. QServe  
**QServe: W4A8KV4 Quantization and System Co-design**  
提出：  
**SmoothAttention**  
用于改善 KV4 attention 精度。  
可以作为：  
[ K,V ]  
activation smoothing 的参考。  
   
⸻  
   
## 29. WKVQuant  
研究：  
* weight  
* KV cache  
* attention reconstruction  
包括 2-D quantization 和 reconstruction-based optimization。  
适合 calibration objective 设计。  
   
⸻  
   
## 30. ZipCache  
**ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification**  
核心：  
利用 attention score 判断 token importance。  
可以定义：  
## [  
##    
## w_i  
f(P_i). ]  
然后对 V quantization 做：  
## [  
##    
## L_V  
\sum_i w_i |V_i-\hat V_i|^2. ]  
说明：  
不同 token 对最终 attention output 的重要性完全不同。  
   
⸻  
   
## 31. Learned 4-bit Codebook  
如果 type B 可以自定义 16 个 representable values：  
## [  
##    
## C  
{c_0,\dots,c_{15}}. ]  
可以针对：  
* Q  
* K  
* V  
* P  
分别学习不同 codebook：  
[ C_Q,C_K,C_V,C_P. ]  
Calibration：  
## [  
##    
## C_Q^*  
\arg\min_C L_O. ]  
因为 Attention 是高度非线性的，所以最佳 codebook 未必是 element-wise MSE 最优 codebook。  
   
⸻  
   
## 32. Microscaling FP4  
如果 type B 是：  
* E2M1  
* MXFP4  
* NVFP4  
则重点应该放在：  
[ x_g=s_gq_g. ]  
推荐研究：  
* SageAttention3  
* MXFP4  
* NVFP4  
* MR-GPTQ  
* AdaMX  
核心方向：  
不同 block 根据数据分布使用不同 scale，甚至不同 4-bit representation。  
   
⸻  
   
## 33. 推荐 Attention 实验路线  

| 阶段         | 实验                                     |
| ---------- | -------------------------------------- |
| Baseline 0 | Q/K/V MinMax                           |
| Baseline 1 | percentile clipping                    |
| Baseline 2 | Q/K/V 独立 scale                         |
| Baseline 3 | per-head scale                         |
| Baseline 4 | per-channel scale                      |
| Baseline 5 | block/microscaling                     |
| Exp 1      | Q smoothing                            |
| Exp 2      | QuaRot Hadamard                        |
| Exp 3      | SpinQuant learned rotation             |
| Exp 4      | DuQuant rotation + permutation         |
| Exp 5      | OSTQuant rotation + scaling            |
| Exp 6      | (QT,\\ KT^{-T})                        |
| Exp 7      | FlatQuant-style affine transform       |
| Exp 8      | Q/K/V independent codebook             |
| Exp 9      | output-aware calibration               |
| Exp 10     | head sensitivity-aware quantization    |
| Exp 11     | token sensitivity-aware V quantization |
| Exp 12     | non-uniform Softmax 4-bit datatype     |
  
   
⸻  
   
## 34. 最推荐的 Attention Calibration Parameterization  
定义：  
[ \theta= { R, S, P, T, \alpha, C, s_{h,g} }. ]  
其中：  
* (R)：rotation  
* (S)：scaling  
* (P)：channel permutation  
* (T)：affine transformation  
* (\alpha)：clipping  
* (C)：16-level codebook  
* (s_{h,g})：per-head/block scale  
求解：  
## [  
##    
## \boxed{  
##    
## \theta^*  
## \arg\min_\theta  
##    
## \sum_{(Q,K,V)\in D_{\text{calib}}}  
##    
## |  
##    
## Attention(Q,K,V)  
\widehat{Attention}(Q,K,V;\theta) |^2 } ]  
   
⸻  
   
## 35. 推荐优先阅读的论文  
## 第一梯队：直接影响实验设计  
1. SageAttention2  
2. SageAttention3  
3. QuaRot  
4. SpinQuant  
5. DuQuant  
6. OSTQuant  
7. FlatQuant  
8. OmniQuant  
9. PTQ4ViT  
10. FQ-ViT  
## 第二梯队：Attention 特殊分布  
11. RepQ-ViT  
12. AdaLog  
13. I&S-ViT  
14. FIMA-Q  
15. ERQ  
## 第三梯队：K/V 和系统实现  
16. KVQuant  
17. KIVI  
18. SKVQ  
19. QServe  
20. WKVQuant  
21. ZipCache  
   
⸻  
   
## 36. 与 Linear 数据集最大的区别  
Linear：  
[ Y=AW. ]  
主要优化：  
[ | AW-\hat A\hat W |. ]  
Attention：  
[ QK^T \rightarrow Softmax \rightarrow PV. ]  
因此必须考虑三层误差传播：  
[ Q/K \rightarrow Score ]  
[ Score \rightarrow Softmax ]  
[ Softmax/V \rightarrow Output. ]  
所以 Attention 中最值得优化的不是单一 tensor reconstruction，而是：  
[ \boxed{ \text{end-to-end attention output error} } ]  
   
⸻  
   
## 37. 最值得研究的 Attention 特有方向  
我认为以下三个方向最值得优先尝试。  
## 方向一：Learned coupled Q/K transform  
[ Q’=QT ]  
[ K’=KT^{-T}. ]  
保证：  
[ Q’K’^T=QK^T. ]  
Calibration 学习：  
[ T^*. ]  
这是 rotation 的严格推广。  
   
⸻  
   
## 方向二：Attention-aware datatype conversion  
不要优化：  
[ \min_x |x-Q_B(x)|. ]  
而直接优化：  
# [  
##    
## \boxed{  
##    
## \min  
##    
## |  
##    
## Attention(Q_A,K_A,V_A)  
Attention( Q_B(Q_A), Q_B(K_A), Q_B(V_A) ) | } ]  
即：  
**Attention-Aware 4-bit Datatype Conversion**。  
   
⸻  
   
## 方向三：Q/K/V 使用不同 4-bit representation  
例如：  
[ Q\rightarrow C_Q ]  
[ K\rightarrow C_K ]  
[ V\rightarrow C_V. ]  
甚至：  
[ P\rightarrow C_P. ]  
因为四种 tensor 的分布完全不同。  
Calibration 可以学习：  
[ C_Q,C_K,C_V,C_P. ]  
如果规则允许自定义辅助信息，这可能显著优于一个统一的 type A → type B 映射。  
   
⸻  
   
## 38. 一个推荐的最终框架  
Calibration 阶段：  
[ (Q_c,K_c,V_c) ]  
先学习：  
[ R,T,S,P,\alpha,C. ]  
构造：  
## [  
##    
## Q’  
QT_Q ]  
## [  
##    
## K’  
KT_K ]  
并约束：  
[ T_QT_K^T \approx I. ]  
然后 block-wise quantize：  
## [  
##    
## \hat Q  
Q_B(Q’;s_{Q,h,g}) ]  
## [  
##    
## \hat K  
Q_B(K’;s_{K,h,g}) ]  
## [  
##    
## \hat V  
Q_B(V;s_{V,h,g}). ]  
最终直接优化：  
[ \boxed{ L= | O-\hat O |^2 + \lambda D_{KL}(P|\hat P) } ]  
这可以同时吸收：  
* SageAttention  
* QuaRot  
* SpinQuant  
* OSTQuant  
* FlatQuant  
* PTQ4ViT  
* KVQuant  
等多条研究路线。  
从课题角度，可以把这个问题概括为：  
[ \boxed{ \text{Calibration-aware Cross-Format 4-bit Attention Quantization} } ]  
或者：  
[ \boxed{ \text{Attention-Aware 4-bit Datatype Conversion} } ]  

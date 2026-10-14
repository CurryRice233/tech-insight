# Over-Tokenized Transformer / Over-Encoding(OE) / Over-Decoding(OD) 机制与谱系调研笔记

> **核实对象**：*Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling*，[arXiv:2501.16975](https://arxiv.org/abs/2501.16975)
> **作者/机构**：Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, Xun Zhou（ByteDance Seed）【事实】[arXiv abs 页](https://arxiv.org/abs/2501.16975)
> **版本**：v1 2025-01-28；v2 2025-05-23（本笔记以 v2 为主，v1 差异单列）【事实】[arXiv abs 页 submission history](https://arxiv.org/abs/2501.16975)
> **会议**：accepted by ICML 2025【事实】[arXiv abs 页 Comments](https://arxiv.org/abs/2501.16975)；ICML 2025 poster 页 [icml.cc/virtual/2025/poster/44467](https://icml.cc/virtual/2025/poster/44467)、ACM DL 收录于第 42 届 ICML 论文集 [10.5555/3780338.3781371](https://dl.acm.org/doi/10.5555/3780338.3781371)、官方报告视频 [SlidesLive](https://slideslive.com/39042441/overtokenized-transformer-vocabulary-is-generally-worth-scaling)
> **核实方式**：arXiv abs 页 + arXiv HTML v2/v1 全文逐节比对 + ar5iv 全文交叉核对 + Semantic Scholar 引文接口 + 各后续论文全文；核实日期 2026-10-14。
> **标注约定**：【事实】= 原文/权威页面可直接支撑；【未找到】= 该资料中未检索到；【推测】= 笔者的推算或推断，已在文中给出依据。

---

## 1. 核心机制：什么叫 "over-tokenized"、层次化 n-gram 嵌入怎么构造

**1.1 总纲：解耦输入词表与输出词表，只放大输入侧。**
【事实】论文的核心主张是 "a novel framework that decouples input and output vocabularies to improve language modeling performance. Specifically, our approach scales up input vocabularies to leverage multi-gram tokens."（[arXiv:2501.16975 Abstract](https://arxiv.org/abs/2501.16975)）
【事实】动机来自合成实验结论："the large input vocabulary is always positive while the large output vocabulary can be negative for smaller models"（[§3.1](https://arxiv.org/html/2501.16975v2#S3.SS1)）。所谓 **Over-Tokenized Transformer(OT)** = Over-Encoding(OE) + Over-Decoding(OD) 的总称（[§3.2 Over-Tokenized Transformer(OT)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px4)）。

**1.2 n-gram 输入 token 的构造（式 1–2）。**
【事实】给定基础 tokenizer 的 id 序列 $x_1,\dots,x_t$，$n$-gram 输入 token 定义为 $x^{(-n)}_i=f(x_i,x_{i-1},\dots,x_{i-n+1})$，越界位置用 0 补齐（$x_i=0,\forall i\notin[1,t]$）；$x^{(-1)}_i=x_i$ 即标准 Transformer 输入（[§3.2 式(1)](https://arxiv.org/html/2501.16975v2#S3.SS2)）。
【事实】$f(z_1,\dots,z_n)=\sum_{i=1}^{n}z_ip^{i-1}$，即把 $(z_1,\dots,z_n)$ 视作 $p$ 进制数，$p\ge V$ 保证双射，通常取 $p=V$ 以让取值范围最紧凑（[§3.2 式(2)](https://arxiv.org/html/2501.16975v2#S3.SS2)）。

**1.3 关键工程化：tiled matrix parameterization + 取模哈希（式 3）。**
【事实】因为原生 $n$-gram 词表是 $V^n$（$V\sim10^5$）不可行，论文用 "tiled matrix parameterization" 把一个 $m\times d$ 的 embedding 表平铺扩展成 $V^n\times d$：查表时对 token id 取模，$\mathbf{h}=\mathbf{E}\left(x^{(n)}\ \%\ m\right)$，$m$ 是**可配置的词表规模**（[§3.2 General $n$-gram Embedder, 式(3)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px1)）。
【事实】该 $n$-gram hashing 技术明确归功于先前工作 (Huang et al., 2021; Roy et al., 2022)，并声明与并发的 BLT 思路类似（[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2)）。
【事实】副作用是哈希冲突：当 $m$ 恰好是 $V$ 的整数倍时增益显著下降，论文建议 **$m$ 取与 $V$ 互质的值**（[§4.2 What's Bad for OE](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px3)、[Table 4](https://arxiv.org/html/2501.16975v2#S4.T4)）。

**1.4 层次化（hierarchical）n-gram 嵌入与低秩切片（式 4–5）。**
【事实】OE 把 1-gram、2-gram、…、$n$-gram 的 embedding **相加**作为 GPT 的输入 embedding：
$$\texttt{OE}(x)=\mathbb{E}^{V\times d}(x^{(-1)})+\sum_{i=2}^{n}\mathbb{E}^{m\times\frac{d}{n}|k}(x^{(-i)})$$
其中 1-gram 表 $\mathbb{E}^{V\times d}$ 与原 Transformer 实现一致，以配合权重绑定（tied weight）设计；$m$ 一般远大于 $V$，且 $m$ 增大时性能持续改善（[§3.2 Over-Encoding(OE), 式(5)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px2)）。
【事实】"sliced into $k$ low-rank decomposed embedders"：$\mathbb{E}^{m\times d|k}(x^{(-n)})=\sum_{i=1}^{k}\mathbb{E}^{m\times\frac{d}{k}}_{i}(x^{(-n)})\mathbf{W}_i$，$\mathbf{W}_i\in\mathbb{R}^{\frac{d}{k}\times d_{model}}$ 负责投影到模型维度；在**同等 embedding 参数量**下只多出 $k$ 个小稠密矩阵，性能显著更好（[§3.2 式(4)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px2)）。
【事实】为让多个 embedder 的映射互不重合，论文对每个 embedder 的行数做微调（"replacing $m$ with $m+2$"），否则切片技巧不起作用（[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px2)）。附录 A 的 PyTorch 伪代码印证了这一点：`oe_embedders = nn.ModuleList([nn.Embedding(m+i*2, D//(k*(n-1))) for i in range(k*(n-1))])`，且最后做 `x /= 1 + k*(n-1)` 归一（[Appendix A, Algorithm 1](https://arxiv.org/html/2501.16975v2#A1)）。

**1.5 具体取值与选取准则。**
【事实】默认配置：$n=3$、$m=1.28\times10^{7}$（即 **OE-12.8M**），并随 $d_{model}$ 调整 $k$，使 $\frac{d_{model}}{nk}\approx 256$（[§4 Experiments — Implementation](https://arxiv.org/html/2501.16975v2#S4.SS0.SSS0.Px1)）。该式在 v1、v2 中一致（v1 同节亦有原文）。
【事实】论文把 "OE-$m$" 定义为**总共 $m\times d_{model}$ 个 embedding 参数**的 over-encoded 模型（[§4 Implementation](https://arxiv.org/html/2501.16975v2#S4.SS0.SSS0.Px1)）。
【事实】词表放大倍数：baseline $V=100278\approx0.1$M，OE-1.2M 与 OE-12.8M 相对 baseline 分别为 **12×** 与 **128×**（[Figure 1 caption](https://arxiv.org/html/2501.16975v2#S1.F1)）。
【事实】另一条选取准则是"层次不能省"：去掉 1-gram 只用 2-gram 会显著变差（2.785 vs 2.714），跳过 2-gram 只用 1+3-gram 也次优（2.684 vs 2.667）（[Table 3](https://arxiv.org/html/2501.16975v2#S4.T3)、[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px3)）。
【未找到】论文**没有**给出"如何选 $n$"的准则或公式：$n$ 只通过消融（$n=2$ vs $n=3$）说明更大的 $n$ 更好，CFG 合成实验用 $n=3$；也没有给出 $k$ 的独立选取准则（只说随 $d_{model}$ 变）。
【推测】文本中的 $\frac{d_{model}}{nk}\approx256$ 与附录 A 代码里的切片维度 `D//(k*(n-1))` **不一致**（差一个 $n\to n-1$）。用代码口径验证：OLMoE-7B 的 $d_{model}=2048$、$n=3$、$k=4$（[Figure 10 caption](https://arxiv.org/html/2501.16975v2#A2.F10)）时 $D/(k(n-1))=2048/8=256$ **恰好等于 256**；而按正文 $\frac{d_{model}}{nk}=2048/12\approx170.7$ 则对不上。因此我判断论文想表达的是"每个切片维度 $\approx256$"，正文公式少减了 1。
【推测】参数量口径自洽性：附录 A 中共 $(n-1)k$ 张表，每张 $m\times\frac{D}{k(n-1)}$，合计 $=(n-1)k\cdot m\cdot\frac{D}{k(n-1)}=m\cdot D$，与正文"OE-$m$ 共 $m\times d_{model}$ 个 embedding 参数"完全吻合；再由 [Table 1](https://arxiv.org/html/2501.16975v2#S4.T1) 的 Emb. P. 反推 $d_{model}$：OLMoE-1.3B 为 1024（$12.8\text{M}\times1024=13.1$B，且 baseline $50280\times1024\approx51$M）、OLMoE-7B 为 2048（$12.8\text{M}\times2048=26.2$B，$50280\times2048\approx103$M），与表格数字一致。

---

## 2. Over-Decoding(OD)：定义、与 MTP 的关系、损失形式

**2.1 定义与与 MTP 的关系。**
【事实】OD = 用**更大的输出词表**换取"更细粒度的监督信号"（finer-grained supervision）（[§1 Introduction](https://arxiv.org/html/2501.16975v2#S1)）：原文 "we propose the concept of Over-Decoding(OD), which leverages larger output vocabulary to provide more fine-grained supervision. We typically treat multi-token prediction methods (Gloeckle et al., 2024; DeepSeek-AI et al., 2024) as approximations of OD."
【事实】§3.2 原文进一步明确："previous researches on Multi-Token Prediction (MTP) are typically approximations of over-decoding... Generally, **MTP-like methods are viewed as over-decoding in this paper**."（[§3.2 Over-Decoding(OD)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px3)）
【事实】注意措辞层级：OE 原文说 MTP 是 OD 的 **approximation（近似）**，并未使用"有效实例（effective instantiation）"这个词；把"MTP 是 OD 的有效实例"写成结论的是后续的 VWN 论文（[VWN §2 Related Works](https://arxiv.org/html/2511.11238#S2)：原文 "Notably, Multi-Token Prediction (MTP) (Gloeckle et al., 2024) is regarded as an effective instantiation of OD for practical training."）。
【事实】OD 只在模型足够大时有效，这与 MTP "only large models benefit from future token predictions" 的既有结论一致（[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px3)）。
【事实】合成实验依据：3-gram **编码** 模型在所有规模上都提升，3-gram **解码** 模型只对较大模型（85M）有益、对小模型（2.4M）有害（[Figure 2 caption](https://arxiv.org/html/2501.16975v2#S3.F2)）；为保持推理成本不变，$n$-gram 输出模型"只保留预测出的 next-1 token"（[§3.1](https://arxiv.org/html/2501.16975v2#S3.SS1)、[Figure 3 caption](https://arxiv.org/html/2501.16975v2#S3.F3)）。

**2.2 训练损失形式（product embedding decomposition）。**
【事实】为把巨大的 $n$-gram 输出词表拆小，用 n 个 $V$ 大小的小表参数化 $V^n$ 表：$\tilde{\mathbf{e}}_x=\sum_{j=1}^{n}\mathbb{E}^{V\times d}_j(x/V^{j-1})=\sum_{j=1}^{n}\mathbf{E}_j(z_j)$（式 6）。
【事实】$n$-gram 预测的交叉熵可精确分解为多个 next-$j$ 预测的 CE 之和：$\mathcal{L}=\texttt{CE}(\hat{\mathbf{h}}\tilde{\mathbf{E}}^{\top},x^{(+n)})=\sum_{j=1}^{n}\texttt{CE}(\hat{\mathbf{h}}\mathbf{E}_j^{\top},z_j)$（式 7，附录 C 给出完整推导式 9）。
【事实】最终带权损失：$\mathcal{L}=\sum_{j=1}^{n}\lambda_j\texttt{CE}(\hat{\mathbf{h}}\mathbf{W}_j\mathbf{E}_j^{\top},z_j)$，$\mathbf{W}_j$ 是**可选的低秩投影**（把最后隐状态投影到更小维度以降低 unembedding 成本），通常 $\lambda_1=1$、$\forall i>1,\lambda_i\le1$（[Appendix C, 式(8)](https://arxiv.org/html/2501.16975v2#A3)）。
【事实】OD 实现上**去掉了 MTP 为每个预测头额外加的 Transformer 层**，直接从最后隐状态用 $n$ 套不同词表解码 next-1…next-$n$（[Appendix C Discussion](https://arxiv.org/html/2501.16975v2#A3)）。
【事实】超参 $\lambda_2\in\{0.1,0.2,0.5,1.0\}$、$n=2$；OD 在训练早期明显落后 baseline，**超过 200B tokens 后才反超**；下游任务上更小的权重更好（[Appendix C.1](https://arxiv.org/html/2501.16975v2#A3.SS1)、[Figure 14](https://arxiv.org/html/2501.16975v2#A3.F14)、[Table 10](https://arxiv.org/html/2501.16975v2#A3.T10)）。

**2.3 OT = OE + MTP-DS（DeepSeek-V3 式条件递归 MTP）。**
【事实】OT 采用 DeepSeek V3 的条件递归 MTP（论文称 MTP-DS）：第 $n$ 个 head 把 next-$(n-1)$ token 的 embedding 拼接到层输入作为条件，顺序（而非并行）预测后续 token；OE 在 MTP-DS 下既增强表示又"直接参与未来 token 预测"，两方面互相加强（[§3.2 Over-Tokenized Transformer(OT)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px4)）。
【事实】OLMoE-1.3B 上的实现只加 1 个额外 head 预测 next-2 token，未来 token 损失权重设为 0.1，得到 **OT-12.8M**（[§4.3](https://arxiv.org/html/2501.16975v2#S4.SS3)）。

---

## 3. 词表缩放定律：log-linear 关系与"最优词表随算力增长"

**3.1 OE 论文自己给出的输入词表—loss 定律。**
【事实】函数形式：$\mathcal{L}=2.6754-0.0256\times\log_{10}{m}$；数据采集条件为 **OLMoE-1.3B、训练 500B tokens**（[Figure 5 caption](https://arxiv.org/html/2501.16975v2#S4.F5)）。
【事实】口语化表述："for every 4× increase in $m$, the training loss decreases by 0.015"（[§4.2 Vocabulary Size Scaling](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px1)）。
【事实】数据点范围：$m$ 从 **20K 到 12.8M**，消融用最简单的 OE 配置 $n=2,k=1$；横轴的"input vocabulary size"定义为 **1-gram 与 2-gram 词表之和 $V+m$**（[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px1)）。
【推测】系数自洽性核对：$0.0256\times\log_{10}4=0.0154\approx0.015$，与正文"每 4× 降 0.015"一致。
【未找到】**没有**给出 $R^2$、RMSE 或拟合数据点表格（全文检索 "R²/correlation coefficient" 无命中）；也没有给出以算力 $C$ 为自变量的 OE 词表幂律拟合（$m^\*(C)$）。
【事实】重要的定性前提："the model needs to be sufficiently trained to capture these patterns… The larger vocabulary size requires longer training until the gains converge"（[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px1)、[Figure 9](https://arxiv.org/html/2501.16975v2#A2.F9)）；CFG 实验也观察到"large vocabulary 的 slow-start"（[Appendix B.2 Ablation on base tokenizers](https://arxiv.org/html/2501.16975v2#A2.SS2.SSS0.Px4)）。
【事实】输入与输出词表**缩放行为不同**：输入词表无论模型大小都持续有益，输出词表过大会伤害小模型（[Abstract](https://arxiv.org/abs/2501.16975)、[§3.1](https://arxiv.org/html/2501.16975v2#S3.SS1)）。

**3.2 与 Tao et al.（词表缩放定律）的关系与差异。**
【事实】Tao et al., *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*，**arXiv:2407.13623**，NeurIPS 2024；训练 33M–3B 模型、最多 500B 字符（[arXiv:2407.13623 abs](https://arxiv.org/abs/2407.13623)）。
【事实】其结论：最优词表参数随算力按幂律增长 $N_{\rm v}=0.20*C^{0.42}$（对比 $N_{\rm nv}=0.08*C^{0.50}$、$H=6.42*C^{0.50}$），因此 $\gamma=0.42/0.50=0.84<1$，即**词表参数应比非词表参数增长得慢**，但**最优词表仍随算力增长**；由三种方法（IsoFLOPs、导数估计、参数化拟合）收敛到 $\gamma\approx0.83$（[Tao §4.1 / Figure 5](https://arxiv.org/html/2407.13623v3#S4.SS1)）。
【事实】Tao 给出的可操作预测：Llama2-70B 的最优词表至少应为 **216K**（是其 32K 的 7 倍）；词表 32K→43K 在同样 2.3e21 FLOPs 下把 ARC-Challenge 从 29.1 提升到 32.0（[arXiv:2407.13623 abs](https://arxiv.org/abs/2407.13623)）。
【事实】OE 论文在 Related Work 与 Introduction 中把 Tao et al. 作为直接前置：承认"larger models deserve larger vocabulary"这一发现，并指出自己进一步**把 embedding（输入）与 unembedding（输出）分开**研究——"While embedding incurs only lookup costs, unembedding introduces computational costs that scale with vocabulary size"（[§1](https://arxiv.org/html/2501.16975v2#S1)、[§2.2 Scaling Vocabulary](https://arxiv.org/html/2501.16975v2#S2.SS2)）。
【推测】两者是互补而非竞争关系：Tao 的定律针对**总词表（含 tied unembedding）** 与算力 $C$，OE 的经验律针对**输入侧 $m$** 与 loss；OE 未反驳 Tao，反而是"输入/输出解耦后输入词表可以更激进地放大"的证据。

---

## 4. 全部关键实验数字（逐条给出处）

**4.1 主结果与收敛加速。**

| 结论 | 数字 | 出处 |
|---|---|---|
| 400M OE-12.8M ≈ 1B baseline | 400B tokens 的 loss 相当；论文表述为"comparable performance to a baseline model that is **2.5 times larger**" | 【事实】[Figure 1](https://arxiv.org/html/2501.16975v2#S1.F1)、[§4.1](https://arxiv.org/html/2501.16975v2#S4.SS1) |
| 摘要里的口径不一致 | Abstract/Intro 写 "performance comparable to **double-sized** baselines" | 【事实】[Abstract](https://arxiv.org/abs/2501.16975)、[§1](https://arxiv.org/html/2501.16975v2#S1) |
| OLMo2-1B 上 OE vs baseline 的 loss 增益 | 400B tokens 时 **0.12**，1T tokens 时扩大到 **0.14** | 【事实】[§4.1](https://arxiv.org/html/2501.16975v2#S4.SS1) |
| loss 收敛加速 | **5.7×** | 【事实】[Figure 4 caption](https://arxiv.org/html/2501.16975v2#S3.F4)、[§4.1](https://arxiv.org/html/2501.16975v2#S4.SS1) |
| 下游任务收敛加速 | MMLU-Var **3.2×**、Hellaswag **3.0×**、ARC-Challenge **2.6×**、ARC-Easy **3.1×**、PIQA **3.9×** | 【事实】[Figure 4 caption](https://arxiv.org/html/2501.16975v2#S3.F4) |
| 训练配置 | OLMo2-151M / 400M 训 400B tokens；OLMo2-1B 训 1T tokens；baseline 词表 $V=100278$ | 【事实】[§4.1](https://arxiv.org/html/2501.16975v2#S4.SS1) |
| CFG 合成实验 | 大模型 85M / 小模型 2.4M；3-gram 编码模型全规模有益，3-gram 解码模型只对大模型有益 | 【事实】[Figure 2 caption](https://arxiv.org/html/2501.16975v2#S3.F2) |

**4.2 MoE（OLMoE）主结果 —— Table 1，500B tokens，base 词表 $V=50280$。**

| 模型 | 额外 Emb. P. | Loss↓ | Downstream↑ |
|---|---|---|---|
| OLMoE-1.3B baseline | 51M | 2.554 | 0.510 |
| +OE-12.8M | 13.1B（与总参数量之比≈10×） | 2.472（−0.082） | 0.524（+0.014） |
| OLMoE-7B baseline | 102M | 2.305 | 0.601 |
| +OE-12.8M | 26.3B（≈3.7×） | 2.229（−0.076） | 0.608（+0.007） |

【事实】[Table 1](https://arxiv.org/html/2501.16975v2#S4.T1)；两个规模上 loss 改善幅度近似相同，但**下游指标的增益随规模衰减**，论文假设与 MoE 稀疏参数的作用重叠（[§4.1 MoE Models](https://arxiv.org/html/2501.16975v2#S4.SS1.SSS0.Px2)、[Table 1 caption](https://arxiv.org/html/2501.16975v2#S4.T1)）。

**4.3 消融 —— 输入词表设计（Table 2，OLMoE-1.3B，500B tokens；括注为 Train Loss / Eval Loss）。**
【事实】C-1 baseline 2.554/2.924；C-2 $+\mathbb{E}^{3.2M\times d}(x^{(-2)})$ 2.511/2.887；C-3 $+\mathbb{E}^{6.4M\times d/2}(x^{(-2)})$ 2.507/2.882；C-4 $+\mathbb{E}^{3.2M\times d|2}$ 2.503/2.877；C-5 $+\mathbb{E}^{3.2M\times d|4}$ 2.503/2.876；C-6（=OE-3.2M，1+2+3-gram, $k=2$）2.495/2.870；C-7 $+\mathbb{E}^{12.8M\times d}(x^{(-2)})$ 2.493/2.881；C-8（=OE-12.8M，$k=2$）2.472/2.862（[Table 2](https://arxiv.org/html/2501.16975v2#S4.T2)、[§4.2 What's Good for OE](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px2)）。
【事实】结论：沿 $d$ 维切片有额外收益（C-2<C-4<C-5）；$n$ 从 2 增到 3 有收益（C-5→C-6）；$m$ 放大到 12.8M 收益更大（C-7→C-8）；论文把切片收益部分归因于**访存更少**（C-4 每 token 访存 $2d_{\text{model}}$，C-3 只有 $1.5d_{\text{model}}$）（[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px2)）。

**4.4 层次设计与哈希冲突（Table 3 / Table 4，50B tokens，$m=3.2$M）。**
【事实】Table 3（Loss）：仅 1-gram 2.714；仅 2-gram **2.785（变差）**；1+2-gram $k{=}1$ 2.678；1+2-gram $k{=}4$ 2.670；1+3-gram $k{=}4$ 2.684；1+2+3-gram $k{=}4$ **2.667**（最优）（[Table 3](https://arxiv.org/html/2501.16975v2#S4.T3)）。
【事实】Table 4（Loss/PPL/Eval Loss）：baseline 2.714/15.094/3.094；$+\mathbb{E}^{64V\times d}$（即 $m$ 恰为 $V$ 的整数倍，$64V\approx3.218$M）2.702/14.892/3.077；$+\mathbb{E}^{3.2M\times d}$ 2.678/14.555/3.054——**同样词表规模下，与 $V$ 整数倍对齐会显著削掉增益**（[Table 4](https://arxiv.org/html/2501.16975v2#S4.T4)、[§4.2 What's Bad for OE](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px3)）。

**4.5 OE + OD 协同（Table 5，OLMoE-1.3B，500B tokens）。**

| 模型 | Loss↓ | Eval Loss↓ | Downstream↑ |
|---|---|---|---|
| baseline | 2.554 | 2.924 | 0.510 |
| +MTP | 2.556 (+0.002) | 2.925 (+0.001) | 0.508 (−0.002) |
| +MTP-DS | 2.555 (+0.001) | 2.926 (+0.002) | 0.511 (+0.001) |
| OE-12.8M | 2.472 | 2.862 | 0.524 |
| OT-12.8M | 2.481 (+0.009) | 2.869 (+0.007) | **0.537 (+0.013)** |

【事实】[Table 5](https://arxiv.org/html/2501.16975v2#S4.T5)、[§4.3](https://arxiv.org/html/2501.16975v2#S4.SS3)。论文自陈："MTP shows no improvement, as well as its stronger version MTP-DS. However, with over-encoding plugged, MTP-DS presents greater power."OT 相对 OE 提升下游 1.3%（同时略牺牲 loss 收益）——注意该处 "1.3%" 实为**绝对差值 0.013**。
【事实】OD 单独结果（Table 10，OLMoE-1.3B 500B tokens）：baseline 2.554/2.924；$+\text{OD }\lambda_2{=}0.1$ 2.549/2.920；$\lambda_2{=}0.2$ 2.549/2.918；$\lambda_2{=}0.5$ 2.549/2.918；$\lambda_2{=}1.0$ 2.555/2.923；OLMoE-7B baseline 2.306/2.670，$+\text{OD }\lambda_2{=}0.1$ 2.304/2.672（[Table 10](https://arxiv.org/html/2501.16975v2#A3.T10)）。

**4.6 速度与开销。**

| 项目 | 数字 | 出处 |
|---|---|---|
| 训练吞吐（OLMoE-1.3B，32×A100） | baseline 1.211 → +OE 1.155 M tokens/s，**−4.63%** | 【事实】[Table 6](https://arxiv.org/html/2501.16975v2#S4.T6) |
| 训练吞吐（OLMoE-7B，64×A100） | baseline 0.494 → +OE 0.453，**−8.3%**（论文称因未仔细调优工程配置） | 【事实】[Table 6](https://arxiv.org/html/2501.16975v2#S4.T6)、[§4.4](https://arxiv.org/html/2501.16975v2#S4.SS4) |
| 前向 GFLOPs/token（OLMoE-1.3B） | baseline 0.5409 → +OE 0.5430，**+0.38%** | 【事实】[Table 8](https://arxiv.org/html/2501.16975v2#S4.T8) |
| 前向 GFLOPs/token（OLMoE-7B） | baseline 2.3578 → +OE 2.3662，**+0.35%** | 【事实】[Table 8](https://arxiv.org/html/2501.16975v2#S4.T8) |
| 理论额外 FLOPs | "less than 0.5%"，实测开销主要来自 all-to-all 通信（正比于 DP 规模） | 【事实】[§4.4](https://arxiv.org/html/2501.16975v2#S4.SS4) |
| 推理吞吐（seq len 2048，单 A100） | batch=1：Dense-1B prefill 20728.7→19446.4、decode 136.2→126.6；MoE-1B/7B prefill 6303.8→6189.0、decode 28.2→27.9；batch=64 decode：Dense-1B 1422.1→1407.4、MoE 860.3→826.7（部分设置 OOM 留空） | 【事实】[Table 7](https://arxiv.org/html/2501.16975v2#S4.T7) |
| 训练总开销 | "resulting an additional training overhead **less than 5%**" | 【事实】[Abstract](https://arxiv.org/abs/2501.16975)、[§1](https://arxiv.org/html/2501.16975v2#S1) |
| FSDP 对照（$m=10^{7}$） | 使用所提优化后吞吐下降 **<5%**；**不做该优化**时 FSDP 出现 **25% slowdown**，且 $m>5\times10^{6}$ 很容易 OOM | 【事实】[§3.3](https://arxiv.org/html/2501.16975v2#S3.SS3) |

**4.7 内部（in-house）大模型验证 —— Table 9。**
【事实】baseline 为 MoE（400M 激活参数 / 4B 稀疏参数总计），OE 把词表放大到 $m=36$M，训练 600B tokens；reasoning：ARC-C 65.7→67.9、Drop 34.4→36.3、BBH 37.1→39.5、WinoGrande 63.2→65.5、Hellaswag 66.2→67.2；knowledge：MMLU 54.8→57.9、C-Eval 61.3→68.3、TriviaQA 39.7→49.0、MMLU-Pro 21.1→24.1、AGIEval 39.1→43.2；math：Ape210K 63.7→63.8、GSM8K 40.6→46.2、MATH 22.7→25.3（[Table 9](https://arxiv.org/html/2501.16975v2#A2.T9)、[Appendix B.5](https://arxiv.org/html/2501.16975v2#A2.SS5)）。

**4.8 v1 → v2 的差异（版本核实）。**
【事实】v1（[HTML](https://arxiv.org/html/2501.16975v1)）中 §2.3 标题为 "Multi-Token Prediction **and n-Gram Modeling**"，v2 改为 "Multi-Token Prediction"；v1 的主干只到 §4.3 + §5，**没有 §4.4 Speed Analysis**，正文表格只到 Table 6（附录 C 的 OD 权重表在 v1 记为 Table 6），**没有训练吞吐表、推理速度表、GFLOPs 表（v2 的 Table 6/7/8）和 in-house 表（v2 的 Table 9）**；v2 行数显著增加（提取文本 1218 行 → 1470 行）。
【事实】v1 已经包含 Appendix C Over-Decoding 与 Algorithm 2，核心数字（$5.7\times$、$3.9\times$、$\mathcal{L}=2.6754-0.0256\log_{10}m$、$d_{model}/nk\approx256$、"less than 5%"、"25% slowdown"）在 v1 中即已存在（[v1 HTML](https://arxiv.org/html/2501.16975v1)）。
【推测】v2 相对 v1 的实质增量主要是**系统/效率证据链**（训练吞吐、推理吞吐、GFLOPs、in-house 规模验证）与 Related Work 的重写，用于回应审稿对"开销是否真的可忽略"的追问。

---

## 5. 实现细节（词表构建、参数量与 FLOPs 的精确账）

**5.1 词表如何构建。**
【事实】基础 tokenizer **完全冻结**："The original tokenizer used in baseline models is preserved as the base tokenizer in our method."（[§4 Implementation](https://arxiv.org/html/2501.16975v2#S4.SS0.SSS0.Px1)）
【事实】**不做 n-gram 频次统计、不设频次阈值、不新增 tokenizer merge**：$n$-gram id 用 base-$V$ 进制现算（`n_gram_ids += F.pad(input_ids, i-1) * V ** (i-1)`），再用 `% (m + 2*index)` 映射到固定大小的表；词表规模 $m$ 是纯超参（3.2M / 12.8M / 36M），越界 token 用 0 补齐（[Appendix A Algorithm 1](https://arxiv.org/html/2501.16975v2#A1)、[§3.2 式(1)(3)](https://arxiv.org/html/2501.16975v2#S3.SS2)）。
【事实】这与 SCONE 的"保留原始词表 + 只给**高频** n-gram 建 embedding"路线形成对照；SCONE 论文亦确认 OE"把 n-gram 哈希进固定数量的 embedding"（[SCONE §5 / §4.2](https://arxiv.org/html/2502.01637v3)、[arXiv:2502.01637 abs](https://arxiv.org/abs/2502.01637)）。
【事实】哈希冲突治理：$m$ 与 $V$ 互质（避免 $m$ 为 $V$ 的整数倍），每张表行数按 $m+2i$ 微调错开映射（[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2)、[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px3)）。

**5.2 参数量账。**
【事实】论文口径：OE-$m$ = 总共 $m\times d_{model}$ 个 embedding 参数（[§4](https://arxiv.org/html/2501.16975v2#S4.SS0.SSS0.Px1)）；实测 OLMoE-1.3B：51M → **13.1B**；OLMoE-7B：102M → **26.3B**（[Table 1](https://arxiv.org/html/2501.16975v2#S4.T1)）。
【推测】与附录 A 代码一致：$(n-1)k$ 张表 × $m$ 行 × $\frac{d_{model}}{k(n-1)}$ 维 $=m\cdot d_{model}$，故"放大倍数"随模型规模下降（1.3B 上约 10×，7B 上约 3.7×）。
【事实】额外只增加 $k(n-1)$ 个稠密投影矩阵 $\mathbf{W}_i\in\mathbb{R}^{\frac{d}{k}\times d_{model}}$（[§3.2 式(4)](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px2)）；1-gram 表与输出层保持原样（tied weight），因此 **OE 本身不让 unembedding/logits 变贵**。
【事实】变贵的只有 OD：unembedding 是稠密激活的、"computational bottleneck"，尤其对小模型；论文用 product decomposition + **可选低秩投影** $W_j$（式 8）来压成本，并明确指出 OD "likely only practical for sufficiently large models"（[Appendix C](https://arxiv.org/html/2501.16975v2#A3)）。
【未找到】主文未给出 400M/1B dense 模型的绝对 embedding 参数量（只给了相对倍数 12×/128× 与 OE-12.8M 命名），也未给出 memory footprint（GB）的具体数字。

**5.3 FLOPs 与工程实现。**
【事实】前向额外 FLOPs **+0.35% ~ +0.38%**（"less than 0.5%"），主要实测开销来自 all-to-all 通信，正比于数据并行规模（[Table 8](https://arxiv.org/html/2501.16975v2#S4.T8)、[§4.4](https://arxiv.org/html/2501.16975v2#S4.SS4)）。
【事实】并行实现：**对 OE embedding 层单独用 tensor parallelism**——embedding 表按行切分到所有 DP rank，token 被发到持有该 embedding 的 DP rank，查完再发回原 rank；前向 **2 次 all-to-all**、反向 **1 次 all-to-all**，总通信量显著低于 FSDP（[§3.3](https://arxiv.org/html/2501.16975v2#S3.SS3)）。
【事实】推理时把额外 embedding 参数 **offload 到 CPU**，不占 GPU 显存（[§4.4](https://arxiv.org/html/2501.16975v2#S4.SS4)）。
【事实】论文认为当前实现未到极限：可把 embedding lookup 做成流水并行的独立 stage，与当前 micro-batch 的 Transformer 前向重叠通信，并把 OE 参数 offload 到 CPU，从而"maintain training throughput without any performance degradation"（[§3.3](https://arxiv.org/html/2501.16975v2#S3.SS3)）——**注意这是展望，不是已实现结果**【推测】。

---

## 6. 局限与失败模式

**6.1 论文自陈的问题（论文没有独立 Limitations 章节）。**
【未找到】全文没有 "Limitations" 小节（检索无命中；v1/v2 皆由 Conclusion → Acknowledgements → Impact Statement 结束）。以下为散落在正文的自陈限制：
- **显存与通信**：大 embedding 表给 GPU 显存造成压力；FSDP 对稀疏参数的通信会严重拖慢训练，从而"further constraining the choice of $m$"；未做优化时 $m>5\times10^6$ 很容易 OOM、吞吐掉 25%（[§3.3](https://arxiv.org/html/2501.16975v2#S3.SS3)）。
- **开销口径不完全一致**：摘要/§3.3 说额外训练开销 "<5%"，但 [Table 6](https://arxiv.org/html/2501.16975v2#S4.T6) 测得 OLMoE-7B 为 **−8.3%**，论文解释为"we did not carefully optimize engineering configurations"（[§4.4](https://arxiv.org/html/2501.16975v2#S4.SS4)）。
- **MoE 上下游收益衰减**：OE-12.8M 在 OLMoE-1.3B/7B 上 loss 各降约 0.08，但 downstream 只 +0.014/+0.007；假设与 MoE 稀疏参数收益重叠（[§4.1 MoE Models](https://arxiv.org/html/2501.16975v2#S4.SS1.SSS0.Px2)）。
- **哈希冲突**：$m$ 与 $V$ 整数倍对齐时增益显著减少；论文的对策是取互质（[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px3)、[Table 4](https://arxiv.org/html/2501.16975v2#S4.T4)）。
- **层次不可省**：去掉 1-gram（只剩大 $n$-gram 表）会显著退化，原因是无法消歧、查找结果冲突（[Table 3](https://arxiv.org/html/2501.16975v2#S4.T3)）。
- **slow-start / 需更长训练**：大词表的收益需要足够训练 token 才兑现；OLMoE-7B **500B tokens 时 OE 收益尚未完全收敛**，论文推测大模型需要更多训练才能充分发挥 OE（[§4.2](https://arxiv.org/html/2501.16975v2#S4.SS2.SSS0.Px1)、[Appendix B.4](https://arxiv.org/html/2501.16975v2#A2.SS4)）。
- **输出侧（OD）成本与规模门槛**：大输出词表对小模型有害；OD 前 200B tokens 落后 baseline；unembedding 是计算瓶颈，OD 只对足够大的模型实用（[§3.1](https://arxiv.org/html/2501.16975v2#S3.SS1)、[Appendix C](https://arxiv.org/html/2501.16975v2#A3)）。
- **未讨论项**：论文没有分析长尾（低频）$n$-gram 的训练充分性、没有给出词表规模的上界理论、也未给出内存随 $m$ 增长的精确定量表【未找到】。

**6.2 后续工作指出的失败模式（外部核实）。**
- 【事实】DeepSeek Engram（[arXiv:2601.07372](https://arxiv.org/abs/2601.07372)）把 OverEncoding 作为 baseline，并称 "OverEncoding (Huang et al., 2025a) fails to yield meaningful improvements on sparse MoE backbones even under a non-isoparametric setting"；同时批评 OE 把 embedding 放在 Layer 0 造成访存/计算串行（[Engram §1](https://arxiv.org/html/2601.07372#S1)、[§2 Related Work](https://arxiv.org/html/2601.07372#S2)）。
- 【事实】SCONE（[arXiv:2502.01637](https://arxiv.org/abs/2502.01637)）指出 OE 的额外 embedding 层 "must be fully instantiated and resident on accelerators during training, leading to significant memory challenges despite tensor sharding"（[SCONE §5](https://arxiv.org/html/2502.01637v3)）。
- 【事实】Forking（[arXiv:2610.00394](https://arxiv.org/abs/2610.00394)）报告 over-encoding $n$-gram memory 分支在 **data replay** 下出现训练/验证 loss 突变的 "forking" 过拟合现象，并归因于 $n$-gram 模块形成的弱交互、上下文特定子空间；低频上下文贡献主要 gap，更大的训练预算与更拥挤的表可抑制它（[arXiv:2610.00394 abs](https://arxiv.org/abs/2610.00394)）。
- 【事实】X-GRAM（[arXiv:2604.21724](https://arxiv.org/abs/2604.21724)）把这类方法的实际收益受限归因于 "Zipfian under-training of the long tail、heterogeneous demand across layers、slot collapse"——即长尾 $n$-gram 训练不足正是 OE 路线的核心痛点之一（[arXiv:2604.21724 abs](https://arxiv.org/abs/2604.21724)）。
- 【推测】"词表过大带来的 softmax 成本"在 OE 中**不构成**问题（输入侧解耦、tied unembedding 不变），真正继承该成本的是 OD/MTP 与 Engram 式输出侧或后置记忆模块；这一点 Engram 用"U 型稀疏分配律 + 严格 iso-FLOPs"来正面处理（[Engram abs](https://arxiv.org/abs/2601.07372)）。

---

## 7. 谱系：OE 继承谁、影响了谁

### 7.1 上游：OE 明确继承/对话的工作

| 方向 | 工作 | OE 的关系 |
|---|---|---|
| 词表缩放定律（**重点核实项①**） | **Tao et al., arXiv:2407.13623**, *Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies*, NeurIPS 2024：$N_{\rm v}=0.20C^{0.42}$、$N_{\rm nv}=0.08C^{0.50}$、$\gamma\approx0.83$–$0.84$，Llama2-70B 最优词表≥216K、32K→43K 使 ARC-C 29.1→32.0（[abs](https://arxiv.org/abs/2407.13623)、[§4.1](https://arxiv.org/html/2407.13623v3#S4.SS1)） | 【事实】OE 把 Tao 列为直接前置，引用其"larger models deserve larger vocabulary"，并推进为**输入/输出解耦**的独立缩放（[§1](https://arxiv.org/html/2501.16975v2#S1)、[§2.2](https://arxiv.org/html/2501.16975v2#S2.SS2)） |
| 层次化/多 gram 嵌入（**重点核实项②**） | **Huang et al., arXiv:2104.04552**, *Lookup-Table Recurrent Language Models for Long Tail Speech Recognition*, Interspeech 2021：首个"可扩展 n-gram embedding 表 + modular hashing 控制词表规模"（[abs](https://arxiv.org/abs/2104.04552)、[Interspeech](https://www.isca-archive.org/interspeech_2021/huang21f_interspeech.html)） | 【事实】OE 明确承认哈希 n-gram 技术"commonly used in previous works (Huang et al., 2021)"（[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2)、[§2.2](https://arxiv.org/html/2501.16975v2#S2.SS2)） |
| 层次化/多 gram 嵌入 | **Roy et al., arXiv:2207.06366**, *N-Grammer: Augmenting Transformers with latent n-grams*（用离散隐表示构造 n-gram 增强 Transformer）（[abs](https://arxiv.org/abs/2207.06366)） | 【事实】OE 列为 n-gram embedding 的直接前作（[§2.2](https://arxiv.org/html/2501.16975v2#S2.SS2)、[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2)） |
| 大规模 n-gram LM | **Liu et al., arXiv:2401.17377**, *Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens*（在 **5 万亿 tokens** 上训练 $\infty$-gram，可补充神经 LM 降 PPL）（[abs](https://arxiv.org/abs/2401.17377)） | 【事实】OE 引用其"trained on 5 trillion tokens"，作为 n-gram 在超大规模下仍有价值的证据（[§2.2](https://arxiv.org/html/2501.16975v2#S2.SS2)） |
| MTP | Gloeckle et al. 2024 *Better & Faster LLMs via Multi-Token Prediction*；DeepSeek-V3（arXiv:2412.19437）的 MTP-DS | 【事实】前者被视作 OD 的近似，后者的条件递归形式被 OT 直接采用（[§2.3](https://arxiv.org/html/2501.16975v2#S2.SS3)、[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2.SSS0.Px4)） |
| 合成实验范式 | Allen-Zhu & Li 2024 *Physics of Language Models: Part 1*（cfg3f 语法、CFG 建模结论） | 【事实】CFG 实验完全沿用其 setup（[§3.1](https://arxiv.org/html/2501.16975v2#S3.SS1)、[Appendix B.2](https://arxiv.org/html/2501.16975v2#A2.SS2)） |
| 并行/训练框架 | FSDP (Zhao et al. 2023)、Fang et al. 2022 / Li et al. 2023 的 offload 流水框架 | 【事实】作为对照与展望引用（[§3.3](https://arxiv.org/html/2501.16975v2#S3.SS3)） |

### 7.2 字符/字节级与 tokenizer-free 路线（**重点核实项③**）

- 【事实】**MegaByte（arXiv:2305.07185，NeurIPS 2023）**：把序列切成 patch，patch 内用局部子模型、patch 间用全局模型，实现百万字节级端到端建模，是"tokenization-free 可行"的早期证据（[abs](https://arxiv.org/abs/2305.07185)）。OE 的 Related Work 认为它"performance at scale remains inferior to tokenizer-based approaches"（[§2.1](https://arxiv.org/html/2501.16975v2#S2.SS1)）。
- 【事实】**BLT（arXiv:2412.09871，Byte Latent Transformer）**：按下一字节熵动态切 patch，是首个在规模上追平 tokenization 的字节级架构；OE 将其列为**并发工作**，并明确二者机制相近："BLT also has the same finding on byte-level models"，BLT 也采用 n-gram hashing embedding（[§2.1](https://arxiv.org/html/2501.16975v2#S2.SS1)、[§3.2](https://arxiv.org/html/2501.16975v2#S3.SS2)）。
- 【推测】**OE 与这两条路线的本质差异**：MegaByte/BLT 改变的是**计算单元的粒度**（把字节聚合成 patch，变的是序列长度与注意力成本），OE 保持原始 BPE 序列与计算图不变，只把**输入 embedding 的查表空间**放大（查表拼接 + 哈希）；因此 OE 与 tokenizer-free 是正交的——OE 论文也承认 BLT 在字节级取得了同类发现。后续 Bolmo（[arXiv:2512.15586](https://arxiv.org/abs/2512.15586)）与 H-Net/动态分块（[arXiv:2507.07955](https://arxiv.org/abs/2507.07955)）延续的是 BLT 一支（均为字节级 + 层次化 chunking），与 OE 的"保留 tokenizer、放大输入词表"形成两条平行路线。

### 7.3 下游：引用/复用 OE 的后续工作（**重点核实项④**）

以 Semantic Scholar 引文接口检索，截至 2026-10-14 共有 **45 篇**引用 OE 的论文，其中 2025 年 22 篇、2026 年 23 篇（[Semantic Scholar citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.16975/citations)）。关键条目：

1. **VWN / Virtual Width Networks（arXiv:2511.11238，2025-11，ByteDance Seed，必查项）**
   - 【事实】VWN 在 Related Work 的 "Embedding Expansion" 段完整转述 OE 的两大组件："The Over-Tokenized Transformer framework (Huang et al., 2025) introduced **Over-Encoding (OE)** to scale input representations using multi-gram tokenization and **Over-Decoding (OD)** to enhance output supervision via multi-token prediction objectives. Notably, Multi-Token Prediction (MTP) is regarded as an **effective instantiation of OD** for practical training."（[VWN §2](https://arxiv.org/html/2511.11238#S2)）
   - 【事实】VWN 在自身方法里复用 OD 思想："previous research Huang et al. (2025) has demonstrated that MTP serves as an approximation of $k$-gram decoding. Building upon this insight, we leverage MTP… following DeepSeek-AI et al. (2025)"，并引用 OE 作为"per-input 注入更多信息以增强加宽表示"的依据（[VWN §3](https://arxiv.org/html/2511.11238#S3)）。
   - 【事实】VWN 的差异化定位：OE 放大的是**词表维度的广度（over-encoding / 稀疏查表）**，VWN 放大的是**每个 token 的 embedding 宽度（over-width embedding）+ Generalized Hyper-Connections**；VWN 报告 8× 加宽带来 next-token loss 用 **2.5×** 更少 token、next-2-token loss 用 **3.5×** 更少 token 达到 baseline 水平（摘要口径："over 2 times" / "3 times"），并发现 virtual width 与 loss 近似 log-linear（[VWN abs](https://arxiv.org/abs/2511.11238)、[VWN §1](https://arxiv.org/html/2511.11238#S1)）。
2. **SCONE（arXiv:2502.01637，NeurIPS 2025 camera ready）**——"Scalable, Contextualized, Offloaded, N-gram Embedding"：保留原词表 + 只为**高频 n-gram** 建 embedding，训练后预计算并 offload 到加速器外存；1B 加速器参数模型超过 1.9B baseline，推理只用约一半 FLOPs/显存。论文把 **OE-12.8M 作为正式 baseline**，报告 OLMo-1B + 10M f-gram 超过 OLMo-1B + OE-12.8M，并指出 OE 的表必须常驻加速器（[SCONE §4.2/§5](https://arxiv.org/html/2502.01637v3)、[abs](https://arxiv.org/abs/2502.01637)）。这是把 OE 当作"被超越的并发方法"的最直接证据。
3. **Engram / Conditional Memory via Scalable Lookup（arXiv:2601.07372，DeepSeek，ACL）**——把 $N$-gram embedding 现代化为 O(1) lookup 的 conditional memory 模块（tokenizer 压缩、多头哈希、上下文化 gating、多层注入），提出 MoE 计算 vs 静态记忆的 **U 型稀疏分配律**，扩到 27B 记忆参数并超过严格 iso-parameter/iso-FLOPs 的 MoE baseline。文中把 **OverEncoding 作为 baseline**，并称其在稀疏 MoE backbone 上无显著提升，且批评"embeddings 只在 Layer 0"导致访存串行（[Engram §1/§2](https://arxiv.org/html/2601.07372#S1)、[abs](https://arxiv.org/abs/2601.07372)）。**这是 OE 之后最重要的同源思想继承者。**
4. **Tensorizing Engram（arXiv:2606.08347）**——明确把 "Over-tokenized Transformers 和 Engram" 并列为"用 n-gram 记忆显式建模多 token 模式"的两条代表，指出它们"每个 n-gram 阶数用独立哈希表，带来哈希冲突且无法共享嵌套 n-gram 的潜在结构"，提出 CP 分解共享因子的 TN-gram（[abs](https://arxiv.org/abs/2606.08347)）。
5. **X-GRAM（arXiv:2604.21724）**——"Beyond N-gram: Data-Aware X-GRAM Extraction for Efficient Embedding Parameter Scaling"：频率感知的动态 token 注入 + 混合哈希与 alias mixing 压长尾，0.73B/1.15B 上平均准确率最多 +4.4，表更小（[abs](https://arxiv.org/abs/2604.21724)）。
6. **Scaling Embeddings Outperforms Scaling Experts / LongCat-Flash-Lite（arXiv:2601.21204）**——把 embedding 缩放当作与 MoE 正交的稀疏维度，68.5B 总参数 / ~3B 激活、>30B 参数分配给 embedding（[abs](https://arxiv.org/abs/2601.21204)）。
7. **JTok（arXiv:2602.00800）**——token-indexed 参数作为新的缩放轴（检索出的调制向量调制主干），650M–61B 上验证并给出幂律（[abs](https://arxiv.org/abs/2602.00800)）。
8. **Forking: Sudden Overfitting Under Replay（arXiv:2610.00394）**——直接以 over-encoding n-gram memory 分支为研究对象，报告 replay 下的突变过拟合失败模式（[abs](https://arxiv.org/abs/2610.00394)）。
9. **EngramEdit（arXiv:2610.10533）**——基于 Engram 式条件记忆做事实知识的解耦更新（[abs](https://arxiv.org/abs/2610.10533)）。
10. **Mixture of Lookup Key-Value Experts（arXiv:2512.09723）**、**MoME: Mixture-of-Memory Embeddings（arXiv:2609.15126）**、**UltraMemV2（arXiv:2508.18756）**——分别从"token-id 检索专家的上下文化""上下文化多槽记忆""记忆层架构"角度扩展同一 conditional-memory 家族（[2512.09723](https://arxiv.org/abs/2512.09723)、[2609.15126](https://arxiv.org/abs/2609.15126)、[2508.18756](https://arxiv.org/abs/2508.18756)）。
11. **ShortSWA（Yifan Zhang，2026-01-12 技术博客，非 arXiv）**——自称是 "a dynamic, parameter-efficient realization of the 'Over-Encoding' framework"，用窗口受限的短滑窗注意力替代静态词表查找以获得 n-gram 缩放收益，并引用 "Huang et al. (ICML 2025)"（[ShortSWA 项目页](https://yifanzhang-pro.github.io/ShortSWA-Ngram-Embedding/)）。【事实】为博客自称，**未经同行评审**。

### 7.4 字节 Seed 内部同源线（同一批作者，可解释 OE 的方法来源）
【事实】Hyper-Connections（[arXiv:2409.19606](https://arxiv.org/abs/2409.19606)，ICLR）与 Frac-Connections（[arXiv:2503.14125](https://arxiv.org/abs/2503.14125)）的作者均包含 Defa Zhu、Hongzhi Huang、Banggu Wu、Yutao Zeng、Qiyang Min、Xun Zhou，即 OE 的核心作者群；VWN 明确把 HC/FC 作为 Generalized Hyper-Connections 的直系前作（[S2 引文核验](https://api.semanticscholar.org/graph/v1/paper/batch)、[VWN §1/§2](https://arxiv.org/html/2511.11238#S2)）。
【推测】由此形成清晰的一条内部演进链：**Hyper-Connections → Frac-Connections → Over-Tokenized Transformer（OE/OD）→ Virtual Width Networks**，主题始终是"用极低 FLOPs 开销换取表示容量/稀疏容量"；OE 提供的是**词表/查表维度**的稀疏容量，VWN 提供的是**embedding 宽度维度**的容量，两者在 VWN 中显式合流（VWN 同时引用 OE 与 HC/FC，并叠加 MTP）。

---

## URL 汇总清单

**核心论文（OE/OT）**
1. arXiv abs 页（含 v1/v2 日期、ICML 2025 接收）：https://arxiv.org/abs/2501.16975
2. arXiv HTML v2 全文（本笔记主要依据）：https://arxiv.org/html/2501.16975v2
3. arXiv HTML v1 全文（版本差异比对）：https://arxiv.org/html/2501.16975v1
4. ar5iv 全文（交叉核对）：https://ar5iv.labs.arxiv.org/html/2501.16975
5. ICML 2025 poster：https://icml.cc/virtual/2025/poster/44467
6. ACM DL（第 42 届 ICML 论文集）：https://dl.acm.org/doi/10.5555/3780338.3781371
7. SlidesLive 官方报告视频：https://slideslive.com/39042441/overtokenized-transformer-vocabulary-is-generally-worth-scaling

**机构/前置工作**
8. Tao et al., arXiv:2407.13623（abs / HTML）：https://arxiv.org/abs/2407.13623 ｜ https://arxiv.org/html/2407.13623v3
9. Huang et al. 2021, Lookup-Table RLM：https://arxiv.org/abs/2104.04552 ｜ https://www.isca-archive.org/interspeech_2021/huang21f_interspeech.html
10. Roy et al. 2022, N-Grammer：https://arxiv.org/abs/2207.06366
11. Liu et al. 2024, Infini-gram：https://arxiv.org/abs/2401.17377
12. MegaByte：https://arxiv.org/abs/2305.07185
13. Byte Latent Transformer（BLT）：https://arxiv.org/abs/2412.09871
14. H-Net 动态分块：https://arxiv.org/abs/2507.07955
15. Bolmo 字节化：https://arxiv.org/abs/2512.15586
16. Hyper-Connections：https://arxiv.org/abs/2409.19606
17. Frac-Connections：https://arxiv.org/abs/2503.14125
18. DeepSeek-V3（MTP-DS 来源）：https://arxiv.org/abs/2412.19437

**后续/引用 OE 的工作**
19. VWN：https://arxiv.org/abs/2511.11238 ｜ https://arxiv.org/html/2511.11238
20. SCONE：https://arxiv.org/abs/2502.01637 ｜ https://arxiv.org/html/2502.01637v3
21. Engram / Conditional Memory：https://arxiv.org/abs/2601.07372 ｜ https://arxiv.org/html/2601.07372
22. Tensorizing Engram：https://arxiv.org/abs/2606.08347
23. X-GRAM：https://arxiv.org/abs/2604.21724
24. Scaling Embeddings Outperforms Scaling Experts（LongCat-Flash-Lite）：https://arxiv.org/abs/2601.21204
25. JTok：https://arxiv.org/abs/2602.00800
26. Forking: Sudden Overfitting Under Replay：https://arxiv.org/abs/2610.00394
27. EngramEdit：https://arxiv.org/abs/2610.10533
28. MoLKV：https://arxiv.org/abs/2512.09723
29. MoME：https://arxiv.org/abs/2609.15126
30. UltraMemV2：https://arxiv.org/abs/2508.18756
31. ShortSWA 项目页（博客，非同行评审）：https://yifanzhang-pro.github.io/ShortSWA-Ngram-Embedding/

**引文检索接口**
32. Semantic Scholar 引文列表（45 篇）：https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.16975/citations
33. Semantic Scholar 批量论文接口（用于核验作者/摘要）：https://api.semanticscholar.org/graph/v1/paper/batch

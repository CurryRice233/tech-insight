# 与 ByteDance Seed「Virtual Width Networks (VWN)」相关的技术脉络调研笔记

> 调研方式：web_search / web_fetch 直读 arXiv abs 页 + arXiv HTML 全文 + ar5iv 全文。
> 调研时间：2026-10-13。所有事实均附可点击 URL；凡未能从一手来源证实者，明确标注「未找到」。
> **重要更正：用户猜测的 `arXiv 2502.12962` 不是 Over-Tokenized Transformer。** 该编号实为 *Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing*（Ye 等，2025-02-18），与本题无关。Over-Tokenized Transformer 的正确编号是 **arXiv 2501.16975**。

---

## 0. 先说结论：VWN 与各工作的继承/差异关系（一张图）

VWN（arXiv 2511.11238，ByteDance Seed，Defa Zhu 通讯）本身是一篇**综述式统一框架**论文。它把此前几条独立线索缝合成一个整体：

| 维度 | 代表工作 | 与 VWN 的关系 |
|---|---|---|
| 深度方向的多路连接 | Hyper-Connections (HC) | 被 VWN 的 **GHC 继承**并推广；HC 是 VWN 的 `m=1` 特例 |
| 分段而非加宽 | Frac-Connections (FC) | 被 VWN 的 **GHC 继承**；FC 是 VWN 的 `n<m`（压缩）侧的思路来源 |
| 只加宽 embedding、骨干不变 | AltUp | VWN 明确称 AltUp 是 **VWN family 的简化实例**（predict-and-correct 版） |
| 输入侧多 gram 词表放大 | Over-Tokenized (OE) | VWN 的 **Over-Width Embedding 的替代/补充路线**；VWN 引其 n-gram embedding 作为输入增强 |
| 输出侧多 token 监督 | MTP (Gloeckle) / DeepSeek-V3 MTP | VWN **直接采用 DeepSeek-V3 的串联式 MTP**，并针对宽表示改成 block-level 线性 |
| 宽度为何不能直接放大 | Kaplan 2020 / Chinchilla | VWN 的**动机引用**：宽度→算力平方增长，故要「虚拟」加宽 |

一句话：**VWN = Over-Width Embedding（借 AltUp/OE 的「加宽表示」） + Generalized Hyper-Connections（统一 HC 与 FC 的连接矩阵） + DeepSeek-V3 式 MTP（压缩后的分块版）**，并额外给出「虚拟宽度 r 与 loss 近似对数线性」的经验缩放律。

---

## 1. Hyper-Connections (HC)

- **链接**：[arXiv:2409.19606](https://arxiv.org/abs/2409.19606) ｜ [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2409.19606v1) ｜ [PDF](https://arxiv.org/pdf/2409.19606)
- **编号确认**：`arXiv:2409.19606`（cs.LG）。v1 提交 **2024-09-29**，v3 最后修订 **2025-03-18**（用户所说的「2409.19606 一带」正确）。
- **作者与机构**：Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou —— **Seed-Foundation-Model Team, ByteDance**（论文首页 affiliation 明写；通讯邮箱 `zhudefa@bytedance.com`）。来源：[abs 页](https://arxiv.org/abs/2409.19606)、[ar5iv §1](https://ar5iv.labs.arxiv.org/html/2409.19606v1)
- **发表时间**：2024-09-29（arXiv v1）。
- **一句话核心思想**：用可学习的 **depth-connections（深度连接）+ width-connections（宽度连接）** 取代固定的残差连接，让网络自己决定「不同深度特征之间该有多强的连接」，从而打破 Pre-Norm/Post-Norm 在「梯度消失 ↔ 表示坍缩」之间的跷跷板效应。

### 关键机制要点

1. **扩张率 (expansion rate) n 与 hyper hidden matrix**：把输入 `h⁰∈ℝ^d` 复制 n 份，构成 `H⁰ = (h⁰ h⁰ … h⁰)ᵀ ∈ ℝ^{n×d}`。层与层之间传递的是这个 `n×d` 矩阵，而非单个向量。最后一层按行求和得到输出向量。原文：`n is the expansion rate`（[ar5iv §2.1](https://ar5iv.labs.arxiv.org/html/2409.19606v1)）。
2. **统一连接矩阵 HC ∈ ℝ^{(n+1)×(n+1)}**：形式为 `HC = [[0_{1×1}, B], [A_m, A_r]]`，其中 `B` 是当前层输出的权重、`A_m` 把 n 个隐向量加权求和成喂给本层的 `h₀`、`A_r` 是残差/横向混合矩阵。前向为 `Ĥ = Bᵀ·T(HᵀA_m)ᵀ + A_rᵀH`（原文 Eq.2–5）。
3. **可解耦为深度连接 (DC) 与宽度连接 (WC)**：`DC = [B; diag(A_r)] ∈ ℝ^{2×n}`（决定「当前层输出」与「历史输入」的权重），`WC = [A_m  A_r] ∈ ℝ^{n×(n+1)}`（决定同一层内 n 个隐向量之间的信息交换）。原文 Eq.6–7。
4. **动态 HC（DHC）**：把 HC 矩阵的每个元素变成输入的函数 `HC(H)`。实现上先 `norm(H)`，再做线性变换 + `tanh`，乘一个小的可学习缩放因子，最后与静态矩阵相加：`B(H) = s_β ∘ tanh(H̄W_β)ᵀ + B`，`A_m(H) = s_α ∘ tanh(H̄W_m) + A_m`，`A_r(H) = s_α ∘ tanh(H̄W_r) + A_r`（原文 Eq.10–13）。
5. **初始化（关键，保证起点等于 Pre-Norm 残差）**：动态参数 `W_β, W_m, W_r` **初始化为 0**；静态矩阵初始化为 `[[0, 1_{1×n}], [e_{k mod n}, e_{n×n}]]`（k 为层索引）——即第 k 层挑一个隐槽走残差、其余为恒等，使初始等价于 Pre-Norm。原文 Eq.14。**注**：论文明确写「we do not perform any dedicated tuning of the initialization」，即该初始化未做专门调优。
6. **顺序-并行对偶（Sequential-Parallel Duality）**：HC 矩阵学成特定形式时，网络可分别表现为「顺序堆叠」（Eq.17）或「每隔一层并行」（Eq.18–19），因而 HC 是在顺序与并行之间做软混合；DHC 还能对每个 token 动态调整排列。
7. **Pre-Norm / Post-Norm 是 n=1 的不可训练特例**：`HC_PreNorm = [[0,1],[1,1]]`；`HC_PostNorm` 的权重由输入输出方差/协方差决定（原文 Eq.15–16）。

### 关键数值

- **扩张率 n 的消融**：论文在 OLMo-1B 上测了 `DHC×1 / ×2 / ×4 / ×8`（500B tokens）。原文 Table 1：

  | 方法 | V2 Eval Loss↓ | V3 Eval Loss↓ | Downstream Acc.↑ |
  |---|---|---|---|
  | OLMo-1B (baseline) | 2.811 | 2.544 | 62.5 |
  | DHC×1 | 2.819 | 2.556 | 62.3 |
  | DHC×2 | 2.802 | 2.534 | 63.0 |
  | **DHC×4** | **2.781** | **2.514** | **63.8** |
  | DHC×8 | 2.778 | 2.516 | 62.8 |

  来源：[ar5iv Table 1](https://ar5iv.labs.arxiv.org/html/2409.19606v1)。可见 **n=4 是常用的性价比拐点**（n=8 的 loss 略优但下游回落），这印证了你提到的「HC 的扩张率常用 4」。论文并未声明「4 是唯一推荐值」——它说「increasing the expansion rate leads to improved training loss」，最终取值需自行权衡。
- **收敛加速**：OLMoE-1B-7B-DHC×4 相比 baseline「converges 1.8 times faster」，ARC-Challenge 提升 6 分（500B tokens）。来源：[ar5iv §1 Figure 1](https://ar5iv.labs.arxiv.org/html/2409.19606v1)。
- **n=1 无效**：原文强调 `n>1` 才必要，`n=1` 时跷跷板效应依旧、性能无提升。
- **额外开销**：论文称参数与计算增量「almost negligible」（见 Appendix B）。具体每层参数量公式见 [ar5iv Appendix B](https://ar5iv.labs.arxiv.org/html/2409.19606v1)。

### 与 VWN 的继承/差异

- **继承**：VWN 的 GHC 直接沿用 HC 的 `[[0,B],[A_m,A_r]]` 矩阵范式、动态参数的 `norm→linear→tanh→缩放→加静态矩阵` 流程、以及「动态参数初始化为 0」的策略。VWN 原文：*"we adopt the hybrid strategy from [Zhu et al. 2024; Zhu et al. 2025]"*。
- **差异 1（维度语义）**：HC 把 **隐藏状态复制 n 份**（`n×d`，宽度真变宽，memory access 上升）；VWN 把它拆成 **m 段 × n 槽**、总预算 `D' = (n/m)·D`，且**骨干宽度 D 保持不变**——加宽只发生在「虚拟」的 over-width 表示上。
- **差异 2（m 的作用）**：HC 没有 `m` 这个概念；VWN 的 `m`（fraction rate）控制「每层保真度 vs 记忆层数」的权衡（见第 2、7 节）。
- **差异 3（是否支持非整数倍）**：HC 的 n 必须是整数；VWN/GHC 允许 `n/m` 为非整数（如 1.5×）。
- **VWN 对 HC 的批评**：*"in large hidden spaces, HC often under-utilizes the expanded representations, since each extension is updated using only a few scalar weights"*（[VWN arXiv HTML §2](https://arxiv.org/html/2511.11238v1)）。

---

## 2. Frac-Connections (FC)

- **链接**：[arXiv:2503.14125](https://arxiv.org/abs/2503.14125) ｜ [arXiv HTML 全文](https://arxiv.org/html/2503.14125v1) ｜ [PDF](https://arxiv.org/pdf/2503.14125v1) ｜ [Semantic Scholar 条目](https://www.semanticscholar.org/reader/69c3497a9088802b79315e06922ef0481fb81038)
- **编号确认**：`arXiv:2503.14125`（cs.LG）。提交 **2025-03-18**（仅 v1）。
- **作者与机构**：Defa Zhu, Hongzhi Huang, Jundong Zhou, Zihao Huang, Yutao Zeng, Banggu Wu, Qiyang Min, Xun Zhou —— **ByteDance Seed**（通讯 `zhudefa@bytedance.com`、`qiyangming@bytedance.com`）。来源：[abs 页](https://arxiv.org/abs/2503.14125)、[HTML 首页](https://arxiv.org/html/2503.14125v1)
- **发表时间**：2025-03-18。
- **一句话核心思想**：既然 HC 的代价来自「把隐藏状态复制 n 份导致宽度/访存上升」，那就反过来——**不加宽，而是把现有隐藏维度切成 m 段**，每段配一套类 HC 的连接权重，从而把 HC 的扩张率推广到 `n ≤ 1`（分数域）。

### 关键机制要点

1. **扩张率推广到分数域**：定义 `0 < n < 1`。**当 n=1 时，FC ≡ HC**。FC 用 `m = 1/n` 段代替 HC 的 n 份复制，`m` 称为 **frac-rate（分片数）**。原文：*"frac-connections can be viewed as a fractional variant of hyper-connections that divides the hidden states into m=1/n parts instead of replicating them n times, where m (referred to as the frac-rate) represents the number of partitions"*（[HTML §4.1](https://arxiv.org/html/2503.14125v1)）。
2. **分片构造**：`H = (h₁ h₂ … h_m)ᵀ = Reshape(h, (m, d/m))`。每段独立处理，处理完 **concatenate（Cat）** 回主网络通路。原文 Fig.3(c)。
3. **连接标量**：FC 使用 `γ_{1,2}, γ_{2,1}, γ_{2,2}` 这类权重（可学习或由网络预测），与 HC 的 `β/α` 角色对应。原文 Fig.3 caption。
4. **静态 (SFC) vs 动态 (DFC)**：动态版沿用 DHC 的组件——**归一化 + tanh 激活 + rescaling（s_β / s_α）**。消融结论：三者中**去掉 rescaling 损失最大**，其次是去掉 tanh，去掉归一化影响最小。原文 [HTML §5.1](https://arxiv.org/html/2503.14125v1)。
5. **表示能力排序**：由相邻层隐状态余弦相似度（Fig.2）得出 **HC ≫ FC ≫ Pre-Norm**——即 FC 位于 HC 与普通残差之间，用更少显存换回 HC 的部分收益。
6. **与 FractalNet 的异同**：FractalNet 也把隐藏状态分段，但分给**不同深度**；FC 分给**不同连接权重**。原文 [HTML §2](https://arxiv.org/html/2503.14125v1)。

### 关键数值

- **示例配置**：论文正文 Fig.3 以 **`n = 1/2`（即 frac-rate `m = 2`）** 作为讲解示例。
- **消融（OLMoE-1.3B）**：对比 `DFC×2` 与 `DFC×4` —— 二者都优于 baseline，**`DFC×4` 相对 `DFC×2` 仅边际增益**；论文给出 **OLMoE-1.3B-DFC×4 相对 baseline 训练 loss 降低约 0.014**。来源：[HTML §5.1 Ablation Study](https://arxiv.org/html/2503.14125v1)。
- **SFC vs DFC**：`SFC×4` 与 `DFC×4` 均优于 baseline，**`DFC×4` 更好**（动态预测机制提供额外建模能力）。同上来源。
- **最大实验规模**：**7B MoE 模型，训练最多 3T tokens**（摘要明写）。来源：[abs 页](https://arxiv.org/abs/2503.14125)。
- ⚠️ 注意：FC 论文里的 `×2 / ×4` 指的是 **frac-rate m**（分片数），不是 HC 的扩张率 n；不要与 HC 的 `DHC×4` 混淆。

### 与 VWN 的继承/差异

- **继承**：VWN 的 GHC 直接复用 FC 的「分段 + 每段独立连接权重」思想，以及 DFC 的归一化/tanh/rescaling 三件套。
- **差异**：FC 是**纯压缩**（`n<1`，宽度不变甚至每片更窄）以省显存；VWN 的 GHC 是**加宽 + 细分并存**——`D' = (n/m)·D`，`n>m`。VWN 原文：*"GHC integrate the advantages of both—expanding the hidden dimension while further subdividing it into structured sub-states."*
- **VWN 里 FC 的角色**：VWN 用 FC 的语言把 `m` 解释为「每层压缩到 D/m 维」的压缩率，从而把 FC 的「分数扩张率」变成 GHC 的一个自由度。

---

## 3. AltUp (Alternating Updates)

- **链接**：[arXiv:2301.13310](https://arxiv.org/abs/2301.13310) ｜ [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2301.13310) ｜ [NeurIPS 2023 版（VWN 引用为 NeurIPS 36: 76718–76736）](https://arxiv.org/abs/2301.13310)
- **编号确认**：`arXiv:2301.13310`。v1 提交 **2023-01-30**，v2 **2023-10-03**。
- **作者与机构**：Cenk Baykal, Dylan Cutler, Nishanth Dikkala, Nikhil Ghosh（UC Berkeley，Google Research 实习）, Rina Panigrahy, Xin Wang —— **Google Research**（通讯 `baykalc@google.com`）。来源：[ar5iv 标题页](https://ar5iv.labs.arxiv.org/html/2301.13310)
- **发表时间**：2023-01-30（arXiv）；NeurIPS 2023。
- **一句话核心思想**：token 表示可以加宽到 `K·d`，但**每层只用宽度为 d 的子块做运算**，其余子块用「预测-计算-校正」(predict-compute-correct) 来估计——于是既有宽表示的好处，又不付宽层的平方算力。

### 关键机制要点

1. **动机（本调研中「宽度=二次成本」论证的一手来源之一）**：注意力成本 `O(N²·d_model)`、FFN 成本 `O(N·d_model²)`。原文明确写：把 token 维度从 `d` 增到 `2d`，模型「at least 2 times (and closer to 4 times for small N) slower」。来源：[ar5iv §3.1 Background](https://ar5iv.labs.arxiv.org/html/2301.13310)
2. **扩张因子 K**：表示宽度 `d → K·d`，但层宽保持 `d_model = d`（**不是** `K·d`）。典型取值 **K=2 或 K=4**（原文：「typically K=2 or K=4, see Sec. 5」）。
3. **三步算法**（原文 Algorithm 1）：
   - **Predict**：`x̂ⁱ = Σ_{j=1}^{K} p_{i,j} · x_old^j`，`p_{i,j} ∈ ℝ` 是可学习标量；
   - **Compute**：选一个子块 `j*`，用宽度 d 的层更新它 `x̃^{j*} = L(x_old^{j*})`；
   - **Correct**：`x_new^i = x̂ⁱ + g_i (x̃^{j*} − x̂^{j*})`，`g_i ∈ ℝ` 可学习标量。
4. **子块选择策略**：`(i) same`（所有层用同一子块）与 `(ii) alternating`（层 ℓ 用子块 `ℓ mod K`，**默认**）。实验表明 alternating 更好。
5. **开销**：每层额外 `O(d·K²)` 计算、`K²+K` 个额外可学习参数；相对于 FFN 的 `O(d²K²)` 可忽略。激活内存增量（K=2）为 `3sbhL`，「less than 10% of the vanilla transformer's」。embedding 表需加宽 K 倍，额外参数 `(K−1)|V|d`。
6. **两个扩展**：**Recycled-AltUp**（把 d 维 embedding 复制 K 份，零额外参数；末层把 K 个 d 维子块逐元素相加降维）；**Sequence-AltUp**（把 predict-compute-correct 用到序列维，stride=k 采样计算 + 校正，保持同算力下让被跳过的 token 也能拿到上下文）。
7. **与 MoE 正交/互补**：可与 SMoE 叠加获得更高容量。

### 关键数值

- **K = 2 或 4**（原文 §3.1、§5）。来源：[ar5iv](https://ar5iv.labs.arxiv.org/html/2301.13310)
- **加速比**：SuperGLUE 与 SQuAD 上，在**同等精度**下相对 dense baseline **最高 87% 加速**（摘要）。
- 实验设置：T5 small/base/large/3B，C4 预训练 500,000 steps，batch size 256，finetune 50,000 steps。

### 与 VWN 的继承/差异

- **继承**：核心命题完全一致——**「加宽 token 表示（embedding）而不是加宽骨干」**。VWN 原文把 AltUp 列为 VWN 家族的简化实例：*"models employing methods such as Hyper-Connections or AltUp can be regarded as simplified instances within the broader VWN family"*（[VWN §1](https://arxiv.org/html/2511.11238v1)）。
- **差异 1（加宽发生在哪）**：AltUp 加宽的是**每层之间传递的表示**（K·d 表示贯穿全网络，靠 predict/correct 维持）；VWN 加宽的是 **embedding 与 over-width hidden state**，并用 GHC 在每层**显式压缩回 D 再扩张回去**。
- **差异 2（跨段信息传递）**：AltUp 用**标量** `p_{i,j}`、`g_i` 做预测与校正；VWN 的 GHC 用**矩阵** `A ∈ ℝ^{n×(m+n)}`、`B ∈ ℝ^{m×n}`，并可输入条件化（DGHC）——表达力更强。
- **差异 3（是否依赖层宽不变）**：两者都保持骨干宽度不变，但 AltUp 每层只处理 1 个子块（其余靠估计），VWN 每层用 `A_m` 把所有段压缩成 D 维后**完整**跑一次 attention/FFN。

---

## 4. Over-Tokenized Transformer / Over-Encoding / Over-Decoding

- **链接**：[arXiv:2501.16975](https://arxiv.org/abs/2501.16975) ｜ [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2501.16975) ｜ [Hugging Face 论文页](https://huggingface.co/papers/2501.16975) ｜ [ICML 2025 poster](https://icml.cc/virtual/2025/poster/44467) ｜ [ACM DL（ICML'25 正式版）](https://dl.acm.org/doi/10.5555/3780338.3781371)
- **编号确认（重要）**：正确编号是 **`arXiv:2501.16975`**，标题 *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling*。v1 **2025-01-28**，v2 **2025-05-23**，Comments 注明 **accepted by ICML 2025**。
  - ❌ **`arXiv:2502.12962` 不是本文**。该编号是 *Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing*（Xiaoju Ye, Zhichun Wang, Jingyuan Wang，2025-02-18，cs.CL）——与本主题无关。[核实来源](https://arxiv.org/abs/2502.12962)
- **作者与机构**：Hongzhi Huang（通讯 `huanghongzhi.51@bytedance.com`）, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, Xun Zhou —— **ByteDance Seed**。来源：[ar5iv 标题页](https://ar5iv.labs.arxiv.org/html/2501.16975)
- **发表时间**：2025-01-28（arXiv v1）；ICML 2025。
- **一句话核心思想**：把**输入词表和解码词表解耦**——输入侧用巨大的**层次化 n-gram 词表（Over-Encoding）**来放大表示能力，输出侧用**多 gram 监督（Over-Decoding）**细化监督信号；并发现**输入词表大小与 loss 呈对数线性关系**。

### 关键机制要点

1. **n-gram 输入索引**：`x_i^{(-n)} = f(x_i, x_{i-1}, …, x_{i-n+1})`，其中 `f(z_1,…,z_n) = Σ_{i=1}^{n} z_i · p^{i−1}`（p 进制拼接，`p ≥ V`，通常取 `p = V`）。`n=1` 即标准 Transformer 输入。原文 Eq.1–2。
2. **Tiled matrix / n-gram hashing（廉价放大词表的关键）**：不做 `V^n × d` 的真实大表，而是用 `m × d` 的小表 + 取模：`h = E(x^{(n)} % m)`，`E ∈ ℝ^{m×d}`，`m` 可配置且通常 `m ≫ V`。原文 Eq.3。作者注明该 hashing 常见于前人工作（Huang 2021；Roy 2022），并承认与同期 BLT（Pagnoni 2024）的 byte-level n-gram hashing 思路相似。
3. **Over-Encoding (OE) = 层次化求和**：`OE(x) = E^{V×d}(x^{(−1)}) + Σ_{i=2}^{n} E^{m×(d/n)|k}(x^{(−i)})`。1-gram 部分与原始 Transformer 保持一致（配合 weight tying）。原文 Eq.5。
4. **低秩切片技巧**：把 embedder 切成 k 个低秩子 embedder 再加权求和 `E^{m×d|k}(x^{(−n)}) = Σ_{i=1}^{k} E_i^{m×(d/k)}(x^{(−n)})·W_i`，`W_i ∈ ℝ^{(d/k)×d_model}`。**同等 embedding 参数量下显著提升性能**。为让多个 embedder 的映射互不相同，会把 m 微调（如 `m → m+2`）。原文 Eq.4。
5. **Over-Decoding (OD)**：用更大输出词表提供更细粒度监督。**MTP 被视为 OD 的有效实例**。关键结论：**大输出词表对小模型有害**（更细粒度监督加重欠拟合），对大模型有益；而**大输入词表永远有益**。
6. **Over-Tokenized Transformer (OT) = OE + OD**：论文明确采用 **DeepSeek-V3 的条件递归式 MTP（称为 MTP-DS）**——不再并行预测，而是顺序预测，第 n 个头把第 (n−1) 个 next token 的 embedding 拼进层输入。
7. **工程方案**：对 OE embedding 层用 **tensor parallelism + DP rank 间 all-to-all**（前向 2 次、反向 1 次 all-to-all），通信量远低于 FSDP。实测 **m=10⁷ 时 FSDP 吞吐仅下降 <5%**；不做优化则 **FSDP 慢 25%**，且 `m > 5×10⁶` 时容易 OOM。原文 [§3.3](https://ar5iv.labs.arxiv.org/html/2501.16975)。

### 关键数值

- **OE-12.8M 配置**：`n = 3`（即 1/2/3-gram），`m = 1.28×10⁷`；并按 `d_model/(n·k) ≈ 256` 来选 k。原文 [§4 Implementation](https://ar5iv.labs.arxiv.org/html/2501.16975)。
- **词表放大倍数与收益**：Fig.1 中相对 baseline 词表 `V = 100278`，OE 扩到 **1.2M（12×）与 12.8M（128×）**；**OE-12.8M 的 400M 模型匹配 1B baseline 的训练 loss，且无额外训练成本**。摘要另表述为「comparable to double-sized baselines」/ 正文「2.5 times larger in model scale」。
- **对数线性律**：输入词表大小指数增长 → loss 线性下降（log-linear，摘要与 §4.1）。
- **收敛加速（OLMo2-1B）**：loss **5.7×**；下游 MMLU-Var 3.2×、HellaSwag 3.0×、ARC-Challenge 2.6×、ARC-Easy 3.1×、PIQA 3.9×。原文 Fig.4。
- **MoE 结果（500B tokens）**：OLMoE-1.3B loss 2.554→2.472（−0.082），downstream 0.510→0.524；OLMoE-7B loss 2.305→2.229（−0.076），downstream 0.601→0.608。原文 Table 1。作者指出 MoE 下下游增益变小，推测与稀疏参数收益重叠。
- **额外训练开销 < 5%**（摘要）。

### 与 VWN 的继承/差异

- **继承**：VWN **引用 OE 作为「Embedding Expansion」路线的代表**，并在 Over-Width Embedding 一节明说「one can adopt input-augmentation strategies [Huang et al., 2025] that inject more information per input than a single isolated token embedding」。VWN 也沿用 OT 的「n-gram 视角」来解释 MTP。
- **差异 1（加宽的对象不同）**：OE 放大的是**词表维度**（更多 token 类型），embedding 表变多但**每个 token 的向量维度仍是 D**；VWN 放大的是**每个 token 的向量维度 `D' = (n/m)·D`**（同词表，表示更宽）。
- **差异 2（是否分组/分段）**：VWN 的 over-width 表示被显式切成 m 段、n 槽，并有 GHC 做结构化路由；OE 的多个 n-gram embedder 只是**相加融合**，没有段间路由矩阵。
- **差异 3（输出侧）**：VWN 借用的是 DeepSeek-V3 的串联式 MTP，但为宽表示改造成 **block-level 线性**（见第 5 节）；OT 用 MTP-DS 原版。
- **潜在冲突/互补**：VWN 与 OT 都在放大「输入侧表示容量」但维度不同（词表 vs 向量宽度），理论上可叠加（VWN 的 `W_expand` 低秩分解就与 OE 的低秩切片思路同源）。

---

## 5. Multi-Token Prediction (MTP)

### 5.1 Gloeckle 等 2024（FAIR at Meta）

- **链接**：[arXiv:2404.19737](https://arxiv.org/abs/2404.19737) ｜ [ar5iv 全文](https://ar5iv.labs.arxiv.org/html/2404.19737) ｜ [PDF](https://arxiv.org/pdf/2404.19737)
- **编号确认**：`arXiv:2404.19737`，提交 **2024-04-30**，ICML 2024。
- **作者与机构**：Fabian Gloeckle（FAIR at Meta；CERMICS, Ecole des Ponts ParisTech）, Badr Youbi Idrissi（FAIR at Meta；LISN Université Paris-Saclay）, Baptiste Rozière（FAIR）, David Lopez-Paz（FAIR）, Gabriel Synnaeve（FAIR）。来源：[ar5iv 标题页](https://ar5iv.labs.arxiv.org/html/2404.19737)
- **发表时间**：2024-04-30。
- **一句话核心思想**：在每个位置同时预测未来 n 个 token（**n 个独立输出头 + 共享 trunk**），提升样本效率；且好处随模型规模增大而增强。

**关键机制要点**

1. **损失**：`L_n = −Σ_t log P_θ(x_{t+n:t+1} | x_{t:1})`；因式分解为共享 trunk `f_s` + n 个独立头 `f_{h_i}` + 共享 unembedding `f_u`：`P_θ(x_{t+i}|x_{t:1}) = softmax(f_u(f_{h_i}(f_s(x_{t:1}))))`。原文 Eq.1–3。
2. **n 个头是并行的（与 DeepSeek-V3 的关键区别）**：每个头独立从同一 trunk 表示出发并行预测，各头之间**不传递因果链**。
3. **显存优化实现**：顺序地「前向+反向」每个头，累积 trunk 梯度，及时释放 logits 与梯度，把峰值显存从 `O(nV + d)` 降到 `O(V + d)`，且**运行时间无损失**。原文 §2 Memory-efficient implementation。
4. **推理**：基础用法只用 next-token 头；附加头可用于 **self-speculative decoding**（blockwise parallel decoding / Medusa 式 tree attention）。
5. **等价参数量对比**：加 n−1 个预测头层时，从共享 trunk 里减掉 n−1 层，保证参数量可比。

**关键数值**

- **最优 n 依赖数据分布**：7B / 200B code tokens 上 `n=4` 最优（MBPP pass@1 +3.8%，HumanEval +1.2%）；APPS/Intro 上 `n=6` 略优；byte-level 模型最优窗口稳定在 **8 bytes**。原文 Table 1、§3.4。
- **推理加速**：**4-token 模型最高 3×**（code 上 3.0×，平均 2.5/3 个 token 被接受；text 2.7×）；8-byte 模型 **6.4×**。原文摘要、§3.2。
- **规模效应**：小模型上 MTP 反而更差，**13B 级别**才显著超越 baseline；13B 在 HumanEval 多解 12%、MBPP 多解 17%。原文摘要、Fig.3。
- **归纳头（induction head）**：对 ≤30M non-embedding 参数的小模型，2-token 预测显著促进归纳能力形成；≥100M 时优势消失。原文 §4.1。

### 5.2 DeepSeek-V3 的 MTP 实现

- **链接**：[arXiv:2412.19437](https://arxiv.org/abs/2412.19437) ｜ [HTML v2 §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2) ｜ [PDF](https://arxiv.org/pdf/2412.19437)
- **编号确认**：`arXiv:2412.19437`，v1 **2024-12-27**，v2 **2025-02-18**。作者：**DeepSeek-AI**。来源：[abs 页](https://arxiv.org/abs/2412.19437)
- **一句话核心思想**：不使用并行独立头，而是**顺序预测**额外的 token，并**保持完整因果链**——第 k 个 MTP 模块把「上一层深度的表示」与「第 (i+k) 个 token 的 embedding」拼接后投影，喂给一个 Transformer 块。

**关键机制要点（逐条对应你说的「1 个额外 MTP 模块、D 维嵌入 + 前一层嵌入拼接后 2D→D 投影」）**

1. **D 个顺序模块**：原文 *"our MTP implementation uses D sequential modules to predict D additional tokens"*。⚠️ **更正你的表述**：论文写的是 **D 个顺序 MTP 模块**（符号 D 在此处表示预测深度），不是「1 个额外模块」。DeepSeek-V3 实际**只训练 1 个 MTP 模块**（即 D=1，预测 1 个额外 token）——这一点论文 §4.5.1 的消融与官方实现一致，但 **§2.2 正文本身用的是 D 的通用表述**；「只用一个 MTP 模块」属于工程配置，正文未用「1」这个数字明写。**请以「D 个顺序模块」为准，不要写成论文原文说 1 个**。
2. **第 k 个 MTP 模块的组成**：shared embedding `Emb(·)`、shared output head `OutHead(·)`、Transformer block `TRM_k(·)`、投影矩阵 **`M_k ∈ ℝ^{d×2d}`**。原文 §2.2 MTP Modules。
3. **拼接投影（2d → d）**：`h'_i^k = M_k [ RMSNorm(h_i^{k−1}) ; RMSNorm(Emb(t_{i+k})) ]`，原文 **Eq.(21)**。`k=1` 时 `h_i^{k−1}` 就是主模型给的表示。
4. **Transformer 块**：`h_{1:T−k}^k = TRM_k(h'^k_{1:T−k})`，原文 Eq.(22)。
5. **共享输出头**：`P_{i+k+1}^k = OutHead(h_i^k)`，原文 Eq.(23)。每个 MTP 模块的 **embedding 层与 output head 都与主模型共享**（[§3.2.3 Shared Embedding and Output Head for Multi-Token Prediction](https://arxiv.org/html/2412.19437v2#S3.SS2.SSS3)）。
6. **损失**：`L_MTP^k = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1})`，各深度取平均后乘权重 λ。原文 Eq.(24)。
7. **与 EAGLE 的区别**：维护因果链的动机与 EAGLE 类似，但 EAGLE 主要面向投机解码，DeepSeek-V3 用 MTP 来**改进训练**。原文 §2.2。
8. **推理用途**：MTP 模块可弃用（仅用主模型）或用于**投机解码加速**。

**关键数值**

- **671B 总参数 / 37B 激活**；预训练 **14.8T tokens**；总训练 **2.788M H800 GPU hours**（预训练 2664K + 长上下文 119K + 后训练 5K），约 **$5.576M**。原文摘要与 Table 1。
- MTP 是摘要中点名的两项架构贡献之一（另一项是 auxiliary-loss-free 负载均衡）。

### 5.3 VWN 如何改造 MTP（三者的递进关系）

- **VWN 明确说它 follow DeepSeek-V3**：*"following [DeepSeek-AI et al. 2025], we concatenate the embedding of the next token with the last-layer embedding of the preceding token, applying a linear projection"*（[VWN §3.4](https://arxiv.org/html/2511.11238v1)）。
- **问题**：DeepSeek 式 `2D → D` 全连接在 VWN 下会变成 `2rD → rD`；`r = 8` 时参数与 FLOPs 都不可接受。
- **VWN 的解法（block-level linear）**：把 `rD` 维向量切成 `n = r × m` 段、每段 `D/m` 维，在**每段内局部**做同一个 `(2D/m) → (D/m)` 的小线性（所有 block **共享**该投影），使混合成本与 `r = 1` 相当。原文 §3.4。
- **VWN 与 MTP 的协同（经验）**：MTP 给更密的监督信号，VWN 给更多表示自由度，二者互补；`r` 越大对 next-2-token 的增益越明显（见第 7 节数值）。

---

## 6. 宽度缩放与缩放定律

### 6.1 Kaplan 等 2020

- **链接**：[arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **作者与机构**：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei —— **OpenAI**。
- **发表时间**：2020-01-23（arXiv）。
- **一句话核心思想**：语言模型性能（loss）与参数量 N、数据量 D、计算量 C 呈幂律关系，且在给定计算预算下存在最优分配。
- **VWN 中的角色**：VWN 在 §1、§2 两次引用为「扩大参数或语料都能得到更强模型」的依据，并紧接着指出「naively increasing hidden dimensions leads to quadratic growth in parameters and compute」。引用条目见 [VWN 参考文献](https://arxiv.org/html/2511.11238v1)（`Kaplan et al. (2020) ... arXiv preprint arXiv:2001.08361`）。

### 6.2 Hoffmann 等 2022（Chinchilla）

- **链接**：[arXiv:2203.15556](https://arxiv.org/abs/2203.15556) ｜ [HF 论文页](https://huggingface.co/papers/2203.15556)
- **作者与机构**：Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark 等 —— **DeepMind**。
- **发表时间**：2022-03-29（arXiv）；NeurIPS 2022。
- **一句话核心思想**：在给定算力预算下，模型参数量与训练 token 数应按大致等比例缩放（计算最优训练），当时的大模型普遍「参数过大、数据不足」。
- **VWN 中的角色**：与 Kaplan 并列引用，作为「scaling laws 指引下参数/数据扩容有效」的动机来源。引用条目见 [VWN 参考文献](https://arxiv.org/html/2511.11238v1)。

### 6.3 「宽度 scaling 是二次成本」的论证来源

本调研中该论证在**这些论文内部**有明确出处（不需要外推）：

| 出处 | 原话/要点 | URL |
|---|---|---|
| **AltUp (Baykal 2023) §3.1** | 注意力成本 `O(N²·d_model)`、FFN 成本 `O(N·d_model²)`；宽度 d→2d 使模型「at least 2× (closer to 4× for small N) slower」 | [ar5iv §3.1](https://ar5iv.labs.arxiv.org/html/2301.13310) |
| **VWN (2025) §3.1** | *"The computational complexity for transformer is O(D²), indicating that scaling the model width D results in a quadratic increase in computational cost."* 并指出 embedding lookup 只占极小部分算力，故可解耦 | [VWN §3.1](https://arxiv.org/html/2511.11238v1) |
| **VWN §1 / Fig.2(b)** | *"Naive width scaling expands both proportionally, causing quadratic growth in parameters and compute."* | [VWN §1](https://arxiv.org/html/2511.11238v1) |
| **Over-Tokenized (2025) §1** | *"expanding the input vocabulary incurs almost no additional computational cost, whereas expanding the output vocabulary significantly increases the training overhead"*（成本不对称的论证） | [ar5iv §1](https://ar5iv.labs.arxiv.org/html/2501.16975) |
| **HC (2024) §1 / App. B** | HC 通过扩张率提高容量，但额外参数与计算「almost negligible」（反证：真正贵的是隐藏维度本身） | [ar5iv App. B](https://ar5iv.labs.arxiv.org/html/2409.19606v1) |

> ⚠️ **未找到**：我没有在任何一手来源中找到「某篇专门论证『宽度 scaling 是二次成本』的独立论文」。该说法是上述工作（AltUp、VWN）中作为**常识性动机**陈述的，其成本公式源自标准 Transformer 复杂度分析（attention 的 `O(N²d)` 与 FFN 的 `O(Nd²)`）。如果你需要「一句可引用的原始出处」，**AltUp §3.1 与 VWN §3.1 是最合适的两个**。

---

## 7. 汇总：关键数值对照表

| 工作 | 关键超参 / 数值 | 来源 URL |
|---|---|---|
| HC (2409.19606) | 扩张率 n = 2, 4, 8 消融；**n=4 为性价比拐点**（V3 Eval Loss 2.514，Downstream 63.8）；n=1 无效；DHC×4 收敛快 1.8× | [ar5iv Table 1](https://ar5iv.labs.arxiv.org/html/2409.19606v1) |
| FC (2503.14125) | frac-rate **m = 1/n**；示例 **n = 1/2**；消融 DFC×2 / DFC×4；**DFC×4 loss 降 ~0.014**；最大 7B MoE / 3T tokens | [arXiv HTML](https://arxiv.org/html/2503.14125v1) ｜ [abs](https://arxiv.org/abs/2503.14125) |
| AltUp (2301.13310) | 扩张因子 **K = 2 或 4**（默认 K=2）；最高 **87% 加速**；额外参数 `K²+K`/层 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2301.13310) |
| Over-Tokenized (2501.16975) | **n=3, m=1.28×10⁷**（OE-12.8M）；`d_model/(n·k) ≈ 256`；词表放大 12× / 128×；**400M OE ≈ 1B baseline**；loss 加速 5.7×；开销 <5% | [ar5iv §4](https://ar5iv.labs.arxiv.org/html/2501.16975) |
| MTP Gloeckle (2404.19737) | 最优 **n=4**（32k 词表）/ **n=8**（byte-level）；**3×** 推理加速（byte 模型 6.4×）；13B 才显现规模收益 | [ar5iv](https://ar5iv.labs.arxiv.org/html/2404.19737) |
| DeepSeek-V3 MTP (2412.19437) | 投影矩阵 **`M_k ∈ ℝ^{d×2d}`**（即 2D→D）；Eq.(21)；embedding 与 output head 与主模型共享；671B/37B 激活 | [HTML §2.2](https://arxiv.org/html/2412.19437v2#S2.SS2) |
| **VWN (2511.11238)** | **MoE-A3.3B 用 (m,n)=(8,64) → r=8**；1.5× 配置用 **(m,n)=(2,3)**；`D' = (n/m)·D`；`r ∈ {2,4,8}` 对应 `n = r·m = {16,32,64}`（固定 m=8）；对数线性拟合系数 **−0.0069**（r 每翻倍，loss 降约 0.0069） | [VWN HTML](https://arxiv.org/html/2511.11238v1) |
| VWN 训练/精度数值 | **2.5× fewer tokens**（next-token）、**3.5× fewer tokens**（next-2-token）；VWN×8 相对 baseline：Δ=0.035（next-token loss）、Δ=0.058（next-2 loss）、**+4.16 分**（Collection B）；MoE-A3.3B 上 next-token 差距 Δ=0.025→0.032 @3T，next-2 Δ=0.049→0.056，下游峰值 **+2.16 分** | [VWN HTML §5.2](https://arxiv.org/html/2511.11238v1) |
| VWN 成本 | (m,n)=(2,3) 时：归一化+动态参数+宽度连接共 **42D FLOPs**，深度连接 **6D FLOPs**；VWN 额外开销约为 `(n/m − 1)` 倍 layernorm+残差加法 | [VWN §3.5](https://arxiv.org/html/2511.11238v1) |
| VWN 可行性边界 | 论文自述：**1.5×–4× 在当今技术栈上更可行**；8× 受通信/访存/硬件限制 | [VWN §6](https://arxiv.org/html/2511.11238v1) |

---

## 8. 关于 VWN 论文版本身份的一点澄清（避免误引用）

- VWN 的 **arXiv 编号 2511.11238 与 v1 提交日期 2025-11-14** 已由 abs 页确认（[abs](https://arxiv.org/abs/2511.11238)）。
- ⚠️ 但从 arxiv.org / ar5iv 拉取的 HTML/PDF 页眉上出现了 **"August 10, 2026" / "August 24, 2026"** 的日期字样，且正文引用了 `arXiv:2609.33895`、`arXiv:2601.20994` 等**编号远高于 2511 的文献**。
- 这意味着该论文在被抓取的版本中**仍在持续更新**（或该 HTML 是较晚的修订版）。**引用时请务必写「arXiv:2511.11238（v1, 2025-11-14）」，并对具体数值注明是哪个版本**；本次调研的所有 VWN 数值均取自 `arxiv.org/html/2511.11238v1` 与 `arxiv.org/abs/2511.11238`。

---

## 9. 全部 URL 列表（按工作分组）

**VWN（核心）**
- https://arxiv.org/abs/2511.11238
- https://arxiv.org/html/2511.11238v1
- https://arxiv.org/pdf/2511.11238v1
- https://ar5iv.labs.arxiv.org/html/2511.11238
- https://huggingface.co/papers/2511.11238

**HC**
- https://arxiv.org/abs/2409.19606
- https://arxiv.org/html/2409.19606v3
- https://arxiv.org/pdf/2409.19606
- https://ar5iv.labs.arxiv.org/html/2409.19606v1
- https://huggingface.co/buckets/huggingchat/papers-content/tree/2409/2409.19606.md

**FC**
- https://arxiv.org/abs/2503.14125
- https://arxiv.org/html/2503.14125v1
- https://arxiv.org/pdf/2503.14125v1
- https://export.arxiv.org/abs/2503.14125
- https://www.semanticscholar.org/reader/69c3497a9088802b79315e06922ef0481fb81038
- https://arxiv.gg/abs/2503.14125

**AltUp**
- https://arxiv.org/abs/2301.13310
- https://ar5iv.labs.arxiv.org/html/2301.13310
- http://xxx.itp.ac.cn/abs/2301.13310v1
- https://scixplorer.org/abs/2023arXiv230113310B/abstract

**Over-Tokenized Transformer**
- https://arxiv.org/abs/2501.16975
- https://ar5iv.labs.arxiv.org/html/2501.16975
- https://huggingface.co/papers/2501.16975
- https://icml.cc/virtual/2025/poster/44467
- https://dl.acm.org/doi/10.5555/3780338.3781371
- https://www.semanticscholar.org/reader/f0b5b7c450f0e3c8e1ab6653695629f85426e534
- https://www.marktechpost.com/2025/01/30/decoupling-tokenization-how-over-tokenized-transformers-redefine-vocabulary-scaling-in-language-models/amp/
- （错误编号，仅供核对）https://arxiv.org/abs/2502.12962

**MTP（Gloeckle）**
- https://arxiv.org/abs/2404.19737
- https://ar5iv.labs.arxiv.org/html/2404.19737
- https://arxiv.org/pdf/2404.19737

**DeepSeek-V3**
- https://arxiv.org/abs/2412.19437
- https://arxiv.org/html/2412.19437v2
- https://arxiv.org/pdf/2412.19437
- https://github.com/deepseek-ai/DeepSeek-V3
- https://github.com/deepseek-ai/DeepSeek-V3/issues/930

**缩放定律**
- https://arxiv.org/abs/2001.08361 （Kaplan 2020）
- https://arxiv.org/abs/2203.15556 （Chinchilla / Hoffmann 2022）
- https://huggingface.co/papers/2203.15556
- https://arxiv-org.ezproxy.obspm.fr/html/2203.15556v1

**MTP 工程实现参考（第三方，非一手）**
- https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.multi_token_prediction.html
- https://docs.nvidia.com/megatron-core/developer-guide/0.19.2/user-guide/features/multi_token_prediction.html
- https://github.com/NVIDIA/Megatron-LM/blob/540e91ea59d9f487d4e83b02c852d1c3a8f82839/docs/source/api-guide/multi_token_prediction.md

---

## 10. 明确标注「未找到」的项

1. **arXiv 2502.12962 与 Over-Tokenized Transformer 无关** —— 已核实为 *Infinite Retrieval*（长上下文检索）。OT 正确编号为 **2501.16975**。
2. **「HC 扩张率常用 4」在 HC 论文中没有被写成推荐值**。论文只给出 n∈{1,2,4,8} 的消融表，n=4 在 loss 与下游之间是经验拐点；没有一句话说「默认用 4」。**请不要把「常用 4」当作 HC 论文的原文主张引用。**
3. **DeepSeek-V3 的 MTP 模块数量**：论文 §2.2 正文写的是「**D 个顺序模块**」，**没有**在正文里用数字「1」明写「只用一个额外 MTP 模块」。「只用 1 个」是官方实现/工程配置层面的常见做法，**未在 §2.2 正文找到「1」的字面依据**（§4.5.1 的消融也是通用表述）。若必须引用「1 个模块」，建议标注为实现细节而非论文原文。
4. **「宽度 scaling 是二次成本」没有独立的专门论文**；最接近的一手论证是 AltUp §3.1 与 VWN §3.1（见 6.3 节表格）。
5. **FC 的具体 frac-rate 主实验取值**：论文正文明确给出的是消融用的 `DFC×2` 与 `DFC×4`（对应 m=2、m=4）以及讲解示例 `n=1/2`；**我没有找到它在一个「最终推荐配置」上给出单一 m 值的明确陈述**。
6. **VWN 的最终发表日期/期刊**：目前仅有 arXiv 预印本信息；未找到会议接收信息。

# PHD-Transformer（Efficient Pretraining Length Scaling）机制精读 — PHD-SWA / PHD-CSWA

> 调研对象：**Parallel Hidden Decoding Transformer (PHD-Transformer)**，arXiv:2504.14992，ByteDance Seed + 北京大学 + 香港大学。
> 定位：ByteDance Seed 语境下与 **SWA（sliding window attention）** 强相关的预训练长度缩放（pretraining length scaling）工作。
> 精读依据：arXiv **HTML v2 全文**（LaTeXML 渲染版，含全部表格数字），辅以 arXiv API 元数据、HF/GitHub API。
> 全文基准链接：[arxiv.org/html/2504.14992v2](https://arxiv.org/html/2504.14992v2)｜摘要页：[arxiv.org/abs/2504.14992](https://arxiv.org/abs/2504.14992)｜PDF：[arxiv.org/pdf/2504.14992](https://arxiv.org/pdf/2504.14992)
>
> **标注约定**：【事实】= 论文/官方元数据可查证；【推测】= 由事实推导的分析，非论文原文；【未找到】= 在工作区可见范围内未检索到。

---

## 0. 一页速览（核心结论）

| 问题 | 结论 | 出处 |
| --- | --- | --- |
| 是什么 | 把同一段输入 token **重复 K 次**再并行喂入同一 Transformer，用"计算深度"换性能；架构不变 | §2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2) |
| KV cache 怎么管 | 只有第 1 份（original token）建 KV 且全局可见；第 2..K 份（hidden decoding token）的 KV 用完即丢 | §2.2 式(2) [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2) |
| 显存 | 与 vanilla **相同 KV cache 规模**，且保持相同的 memory access pattern | §2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2) |
| PHD-SWA 挂在哪 | 滑窗**只给 hidden decoding tokens**（W=16）；original tokens 仍是**全注意力/全局 KV**。**不是**全层滑窗 | §2.4、§4.1 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1) |
| PHD-CSWA | 把滑窗**限制在独立 chunk（C=32）内**，把 pre-filling 的线性/超线性增长压到"只有最后一个 chunk 内线性" | §2.4、§3.3.1 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1) |
| 收益 | 1.2B：Avg **54.7 → 56.2（K=2）/ 56.7（K=3）**；550M：**49.3 → 51.1（K=5）** | 表 1 [#S3.T1](https://arxiv.org/html/2504.14992v2#S3.T1)、表 2 [#S3.T2](https://arxiv.org/html/2504.14992v2#S3.T2) |
| 代价 | K=256 时解码延迟增幅 **≤20%** | §3.5、图 9(b) [#S3.SS5](https://arxiv.org/html/2504.14992v2#S3.SS5) |
| 开源 | **【未找到】** 官方代码/权重；论文全文无 code release 链接 | §7 |
| 会议 | **【未找到】** 会议接收信息（v2 无 journal-ref / comments） | §1 |

---

## 1. 论文基本信息

### 1.1 作者与机构【事实】

- 作者（7 人，按 arXiv 顺序）：**Bohong Wu、Shen Yan、Sijun Zhang、Jianqiao Lu、Yutao Zeng、Ya Wang、Xun Zhou**。
  出处：arXiv 摘要页作者列表 [arxiv.org/abs/2504.14992](https://arxiv.org/abs/2504.14992)；HTML 作者块 [#S1](https://arxiv.org/html/2504.14992v2#S1)。
- 机构（HTML 脚注 1/2/3）：**1] ByteDance Seed；2] Peking University；3] Hong Kong University**（原文即写作 "Hong Kong University"，**未**写作 "The University of Hong Kong"）。
  出处：HTML `ltx_authors` 块 <https://arxiv.org/html/2504.14992v2>。
- 贡献声明：`[*] Work done at ByteDance Seed`（全部作者的工作均在 ByteDance Seed 完成）；`[†] Corresponding authors`。
  对应邮箱：`bohongwu@bytedance.com`、`zhouxun@bytedance.com` → 通讯作者为 **Bohong Wu** 与 **Xun Zhou**。出处：同上。
- 版本页眉标注日期：`(April 24, 2025)`，与 v2 提交日一致。出处：HTML 首页 [#S1](https://arxiv.org/html/2504.14992v2#S1)。

### 1.2 版本时间线【事实】

| 版本 | 时间（UTC） | 说明 |
| --- | --- | --- |
| v1 | **2025-04-21 09:41:26 UTC** | 首次提交 |
| v2 | **2025-04-24 04:13:49 UTC** | 最新版本（本次精读对象，4,973 KB） |

出处：arXiv 摘要页 Submission history [arxiv.org/abs/2504.14992](https://arxiv.org/abs/2504.14992)；arXiv API `published=2025-04-21T09:41:26Z`、`updated=2025-04-24T04:13:49Z`（<http://export.arxiv.org/api/query?id_list=2504.14992>）。
注：API 返回的另一个 `updated=2026-10-14T07:15:20Z` 是 arXiv 记录元数据本身的维护时间，**不是**新的论文版本；截至精读时 **v2 仍为最新版**。

### 1.3 会议接收情况【未找到】

- arXiv 摘要页**无** `Journal reference`、**无** `Comments: accepted at ...` 字段。
  出处：[arxiv.org/abs/2504.14992](https://arxiv.org/abs/2504.14992)（页面 "Subjects / Cite as" 区块中只有 cs.CL 与 DOI）。
- arXiv API 查询结果的 `<arxiv:journal_ref>` 与 `<arxiv:comment>` 均为空。
  出处：<http://export.arxiv.org/api/query?id_list=2504.14992>。
- 论文正文/参考文献中出现的 ICLR / NeurIPS / ICML / ACL 字样，**全部**是其引用的**其他**论文的会议信息（如 FlashAttention-2 ICLR 2024、StreamingLLM ICLR 2024），与本文自身接收无关。
  出处：HTML 参考文献段 <https://arxiv.org/html/2504.14992v2>。
- **结论【事实】**：截至精读时，该工作**仅以 arXiv preprint 形式存在，未检索到会议/期刊接收记录**。若需最终确认，建议后续复核 OpenReview / ACL Anthology 的作者检索页（本次检索未命中）。

### 1.4 主题与代码基座【事实】

- arXiv 分类：**cs.CL**（Computation and Language）。出处：[arxiv.org/abs/2504.14992](https://arxiv.org/abs/2504.14992)。
- 实验代码基座：**OLMo2**（OLMo 系列，参考文献 [37]）。出处：§3 开头 <https://arxiv.org/html/2504.14992v2#S3>（"We use OLMo2 [37] as the codebase of all our experiments."）。
- 命名规范：`model-type-K-W-C`（如 `PHD-CSWA-2-16-32` = 重复 2 次 / 窗口 16 / chunk 32）。出处：§3 开头。

---

## 2. PHD-Transformer 核心机制

### 2.1 记号与序列构造【事实】

训练样本 $x=\{x_1,\dots,x_t\}$（t 个 token），隐表示 $\mathbf h=\{h_1,\dots,h_t\}$，隐维度 $d$。原始自注意力写作：

$$\text{Self-Attention}(\mathbf h)=\text{softmax}\!\left(\frac{(W_q\mathbf h)(W_k\mathbf h)^{\top}\otimes M}{\sqrt d}\right)(W_v\mathbf h) \tag{1}$$

出处：§2.1 式(1) [#S2.SS1](https://arxiv.org/html/2504.14992v2#S2.SS1)。

PHD 把每个位置**重复 K 次**，序列被扩展为**交错**形式：

$$\{x_1^1,x_1^2,\dots,x_1^K;\ \ x_2^1,x_2^2,\dots,x_2^K;\ \ \dots;\ \ x_t^1,x_t^2,\dots,x_t^K\}$$

- **original tokens（原 token）**：$x_j^1,\ 1\le j\le t$（即**第 1 份**）
- **hidden decoding tokens（隐藏解码 token）**：$x_j^s,\ 1\le j\le t,\ 2\le s\le K$（即**第 2..K 份**）
- 记 $M_{mn}^{ij}$ 为 $x_m^i$ 与 $x_n^j$ 之间的注意力掩码。

出处：§2.1 末段 [#S2.SS1](https://arxiv.org/html/2504.14992v2#S2.SS1)。

> **【定义要点】hidden decoding tokens 不是新采样的 token，也不是插入的可学习 token**：它们是**同一批输入 token 的重复副本**，在同一层内被并行前向，充当"额外的隐式计算/推理步"。注意与 Goyal et al. 的 pause token、Coconut 的 latent embedding、CoTFormer 的层间隐状态复用**不同**——PHD 明确强调自己"simply repeat the input tokens 1/2/3/4 times **without post-processing on middle layer hidden states**"。出处：§1 [#S1](https://arxiv.org/html/2504.14992v2#S1)、§4.3 [#S4.SS3](https://arxiv.org/html/2504.14992v2#S4.SS3)、图 2 说明 [#S2.F2](https://arxiv.org/html/2504.14992v2#S2.F2)。

### 2.2 KV cache 管理策略（核心）【事实】

论文给出的**注意力掩码策略**（PHD 基础版的判别式）：

$$M_{mn}^{ij}=\begin{cases}
1, & \text{if } i=1 \ \text{and}\ m<n \quad\text{（original token 之间：标准因果全注意力）}\\[2pt]
1, & \text{if } i\le j \ \text{and}\ m=n \quad\text{（同一位置的 K 份副本之间：互相可见）}\\[2pt]
0, & \text{otherwise}
\end{cases} \tag{2}$$

出处：§2.2 式(2) [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)。

**逐项解读（【事实】+【推测】标注）**：

1. **`i=1 and m<n`**：只有 **original token 生成 KV cache**，且对**其后所有 token（含全部 hidden decoding tokens）全局可被注意**。这正是"只保留原 token 的 KV"——长程依赖完全由 original tokens 承载。【事实，§2.2 原文】
2. **`i≤j and m=n`**：在**同一位置 j** 上，第 i 份副本可以看到第 i 份及**更早份**（$i\le j$ 是论文原文，按上下文语义应理解为同位置的副本序号有序可见，即副本内部保持顺序依赖）。效果是：位置 j 的 hidden decoding token 能看到**同位置更早的副本**，从而获得"隐式多步计算"的效果，而**不会**污染其它位置的 KV。【原文如此；此处论文用了 $i\le j$ 而 $j$ 同时是位置下标，符号存在轻微歧义 —— 见 §8 局限】
3. **`0 otherwise`**：位置 m 的 hidden decoding token **看不到位置 n≠m 的任何 hidden decoding token**（无论份数）。→ 这是"**用完即丢**"的严格定义：hidden decoding token 的 KV **从不被后续 token 复用**，因此**无需驻留**。【事实】
4. **loss 只用最后一份**：`use the first copy of token for KV cache generation, while use the final copy of token for next token prediction`。即 KV 来源是第 1 份、预测目标是第 K 份。出处：§2.2 末段 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)；图 2 说明 [#S2.F2](https://arxiv.org/html/2504.14992v2#S2.F2)。

**结论（论文原文表述）**：
> "Our design achieves **identical KV cache size and memory access patterns** as the original Transformer during inference." —— §2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)

> **【推测｜重要澄清】** 严格说，"identical KV cache size" 指 **PHD 基础版**（式 2，无滑窗）。一旦启用 **PHD-SWA**，需要额外驻留一个窗口的 hidden decoding KV，论文自己表述为 "requiring only $\mathcal{O}(1)$ additional KV cache memory"（§1、§2.4），即"常数级额外"而非"绝对相同"。`K=256` 只影响**瞬时**（transient）显存与 FLOPs，**不影响**持久 KV。见 §3 复杂度分析。

### 2.3 Kernel 设计：如何避免 O(K²) 爆炸【事实】

- **朴素实现**的代价：attention 层计算量涨 $K^2$ 倍，FFN 层涨 $K$ 倍。出处：§2.3 [#S2.SS3](https://arxiv.org/html/2504.14992v2#S2.SS3)。
- **关键观察**：式(2) 的注意力矩阵**极其稀疏**，且交错排布（original/hidden 交替）导致矩阵**非设备友好（not device friendly）**，无法高效利用连续块。
- **重排（rearrange）方案**：把输入**拆成两组并拼接**——1 组 = $t$ 个 original tokens，1 组 = $(K-1)t$ 个 hidden decoding tokens（$K=3$ 时即 $t$ + $2t$）。重排后，**被 mask 掉的位置聚成一个连续块**，从而把 attention 计算复杂度从 $\mathcal{O}(K^2)$ 降为 $\mathcal{O}(K)$。
  出处：§2.3 + 图 3 说明 [#S2.SS3](https://arxiv.org/html/2504.14992v2#S2.SS3) [#S2.F3](https://arxiv.org/html/2504.14992v2#S2.F3)。
- **FLOPs 与延迟**：`While requiring K times FLOPs, these computations can be processed in parallel, resulting in minimal latency overhead in memory-bound inference scenarios.` 出处：§2.2。

### 2.4 复杂度与显存分析

符号：$L$ 层数、$b$ batch、$t$ 序列长度、$n_{kv}$ KV 头数、$d_h$ 头维、$W$ 窗口、$C$ chunk、$K$ 重复次数。

| 指标 | Vanilla | naive token repeat（无 KV 管理） | **PHD（基础）** | **PHD-SWA** | **PHD-CSWA** |
| --- | --- | --- | --- | --- | --- |
| 持久 KV cache 规模 | $2Lbt\,n_{kv}d_h$ | $\approx K\times$ vanilla（**线性膨胀**） | $2Lbt\,n_{kv}d_h$（**与 vanilla 相同**） | vanilla $+\,2LbW\,n_{kv}d_h$（窗口常驻，$\mathcal O(1)$ 额外） | 同 PHD-SWA（$\mathcal O(1)$ 额外） |
| 前向有效 token 数 | $t$ | $Kt$ | $Kt$（但 attention 重排后仅 $\mathcal O(K)$ 倍注意力算力） | $Kt$（窗口内注意力，算力 $\mathcal O(K)$） | $Kt$（chunk 内，算力 $\mathcal O(K)$，prefill 更省） |
| FLOPs | $1\times$ | $\sim K\times$（KV 读取放大） | $\approx K\times$ FLOPs（论文原文） | $\approx K\times$ | $\approx K\times$ |
| Pre-filling 时间 | $T$ | 随 $K$/长度**线性→超线性**增长 | **≈ $T$，且与 K 无关（constant regardless of K）** | **超线性增长**（随序列长度，论文 §3.5 原文 superlinear） | ≈ $T$ + **仅最后一个 chunk** 内线性（$K$ 次重复） |
| 解码延迟 | $1\times$ | 随 $K$ 线性增长 | 极小开销（memory-bound） | 极小开销 | 极小开销（K=256 时 ≤20%） |

**出处**：
- "identical KV cache size / memory access patterns"：§2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)
- "pre-filling time is the same as the vanilla transformer and **remains constant regardless of the scaling factor K**"：§2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)
- "$K$ times FLOPs, these computations can be processed in parallel"：§2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)
- $\mathcal O(K^2)\to\mathcal O(K)$：§2.3 [#S2.SS3](https://arxiv.org/html/2504.14992v2#S2.SS3)
- SWA "$\mathcal O(1)$ additional KV cache memory"：§1 [#S1](https://arxiv.org/html/2504.14992v2#S1)、§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)
- naive repeat 导致 "linearly increased KV cache size ... **super linear increase in pre-filling time** (Figure 1(c)) ... linear increase in decoding latency (Figure 1(d))"：§1 [#S1](https://arxiv.org/html/2504.14992v2#S1)
- CSWA "reduces the extra pre-filling overhead to just $K$ repetitions **within the final chunk**, rather than across the entire sequence"：§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

**【推测｜更细的显存拆解】** 若按标准 MHA/GQA 计量：持久 KV $=2Lbt\,n_{kv}d_h$（与 vanilla 逐项相同）；PHD-SWA 的常驻窗口 KV $=2LbW\,n_{kv}d_h$，因 $W=16$ 固定，对 $t$ 而言是 $\mathcal O(1)$（故"等价 KV 规模"的直觉成立——**相对序列长度的增长率完全相同，只是多一个常数项**）。pre-fill 阶段的**瞬时** KV 需要容纳 $(K-1)t$ 个 hidden decoding token 的 K/V：$2Lb(K-1)t\,n_{kv}d_h$，这部分**不持久**，故不进入推理期显存占用，但会抬高 pre-fill 的峰值显存——**论文未报告该峰值显存数字**【未找到】。

---

## 3. PHD-SWA：滑窗注意力挂在哪里？

### 3.1 挂载位置【事实】

> "we introduce PHD-SWA, which implements **sliding window attention restricted to $W$ preceding hidden decoding tokens**." — §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

> "the attention pattern combines **global access to original tokens** with **local access to the $W$ most recent hidden decoding tokens**." — §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

**因此（明确回答本调研问题 3）**：
- 滑窗**不是**全层滑窗，**不是**给 original tokens 用。
- **original tokens 永远保持全注意力 / 全局 KV**（承载长程依赖）；
- **滑窗只作用于 hidden decoding tokens**，窗口是"其前面 $W$ 个 **hidden decoding** token"（外加对全部 original tokens 的全局访问）。

出处：§2.4、§4.1、图 4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1) [#S2.F4](https://arxiv.org/html/2504.14992v2#S2.F4)。

论文在 §4.1 亲口与既有 SWA 划界：
> "While our PHD shares the use of predefined attention patterns, particularly in our PHD-SWA variant which employs sliding window attention for hidden decoding tokens, we **uniquely apply these patterns only to the repeated tokens, maintaining full attention for the original tokens** and thus preserving model quality while gaining efficiency." — §4.1 [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1)

### 3.2 窗口大小与为什么能"保持局部依赖"【事实】

- **窗口大小 W**：主实验/消融用 **W=16**；消融扫描 $W\in\{1,2,4,16\}$（图 6）；图 4 示意用 W=4。见 §3.3.1、§3.4。
- **为什么需要**：论文原文表明动机是**工程经验观察**，而非理论必然——
  > "However, we **empirically observe** that preserving some KV cache for hidden decoding tokens yields significant performance benefits." — §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)
- **为什么"保持局部依赖"**：hidden decoding tokens 被重复注入计算后，如果不让它们彼此可见，就丢失了**相邻副本之间的局部连贯性**；给一个 $W$ 的局部窗口，等于让"同一输入位置的 K 个副本 + 相邻若干位置的副本"形成**局部连续语境**，从而补偿"用完即丢"造成的信息损失。论文的表述为 "To capture these benefits while maintaining efficiency"（§2.4）与 "To better preserve the performance benefits from the KV cache of hidden decoding tokens"（§1）。【事实（引文）+ 推测（机制解释）】
- **代价**：仅 $\mathcal O(1)$ 额外 KV（§1、§2.4），但引入 **pre-filling 的超线性开销**（§2.4、§3.5）。
  > "While the sliding window approach in PHD-SWA enhances model performance, it incurs a **$K$ times pre-filling overhead caused by sequential dependencies in the KV cache of hidden decoding tokens**." — §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

### 3.3 实现方式【事实 + 未找到边界】

- **实现层面**：论文只给出**注意力掩码/模式**层面的描述（图 4 给出 PHD / PHD-SWA / PHD-CSWA 三种注意力矩阵对照，K=3、W=4、C=4），以及 §2.3 的**重排 + 连续块**优化思路（该重排是为 **PHD 基础版**的 $M^{ij}$ 服务的）。
  出处：图 4 [#S2.F4](https://arxiv.org/html/2504.14992v2#S2.F4)、§2.3 [#S2.SS3](https://arxiv.org/html/2504.14992v2#S2.SS3)。
- **是否基于 FlashAttention / 自研 kernel**：【未找到】。论文提到 "Our PHD approach is **orthogonal to these kernel optimizations** [FlashAttention 系列] and **could leverage them** for implementation"（§4.2），即**暗示但未明说**已用 FlashAttention 实现。
  出处：§4.2 [#S4.SS2](https://arxiv.org/html/2504.14992v2#S4.SS2)。
- **是否使用 sink / 特殊位置编码**：【未找到】，论文未提及。
- **训练与推理一致性**：PHD 系列把效率模式**内建到预训练阶段**（"integrates efficient attention patterns during training"），与 post-hoc 稀疏化不同。出处：§4.1 的 Training-Native Sparse Attention 段 [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1)。

---

## 4. PHD-CSWA（chunk-wise SWA）：如何消除 pre-filling 线性增长

### 4.1 机制【事实】

> "we introduce PHD-CSWA which processes attention **within independent chunks**. ... PHD-CSWA **constrains the sliding window attention to operate within individual chunks**. This architectural innovation reduces the extra pre-filling overhead to just **$K$ repetitions within the final chunk**, rather than across the entire sequence, making the additional computational cost **practically negligible** while preserving the benefits of local attention patterns." — §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

**因果链（【事实】层面可完整复原）**：
1. 朴素"重复 K 次" → KV 随序列线性膨胀 → pre-filling **超线性**增长（图 1(c)）；解码延迟线性增长（图 1(d)）。§1 [#S1](https://arxiv.org/html/2504.14992v2#S1)
2. PHD 基础版 → 丢弃 hidden KV → pre-filling **与 vanilla 相同且与 K 无关**，但**损失了局部依赖带来的收益**。§2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)
3. PHD-SWA → 加回 $W$ 局部窗口 → 收益回来，但窗口 KV 存在**顺序依赖（sequential dependencies）**，导致 pre-filling 又变成 **$K$ 倍 / 超线性**。§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)
4. PHD-CSWA → 把顺序依赖**切断在 chunk 边界内** → 只有**最后一个 chunk** 仍需 $K$ 次顺序重复 → 全局 pre-filling 增长从"随整个序列"降为"随一个 chunk"→ **实践中可忽略**。§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)

### 4.2 chunk 大小与分块方式【事实】

- **chunk size C = 32** 为主实验选定值（`PHD-CSWA-2-16-32`、`PHD-CSWA-3-16-32`）。出处：§3.2 Training Details [#S3.SS2](https://arxiv.org/html/2504.14992v2#S3.SS2)。
- **消融**：固定 $K=2$、$W=16$，令 $C\in\{16,\,32,\,\text{no chunk}\}$。结论：
  > "the introduction of chunks causes only **negligible differences** in both training and validation losses. We also observe a consistent trend where **larger chunk sizes yield progressively better results**. Based on this analysis, we select **$C=32$** as the optimal balance." — §3.3.1 [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1)
- **分块方式**：chunk 是**沿序列切分的独立区间**，滑窗不跨 chunk；窗口在 chunk 内仍为 $W$。图 4 以 K=3、W=4、C=4 示意三者的注意力矩阵差异。出处：§2.4、图 4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) [#S2.F4](https://arxiv.org/html/2504.14992v2#S2.F4)。
- **补充图示**：另有 `chunkwise_swa_rearrange` 图（图 3 位置，文件名 `extracted/6385455/figures/chunkwise_swa_rearrange.png`）用于说明 chunk 内的重排。
  出处：<https://arxiv.org/html/2504.14992v2>（HTML `<img src>` 列表）。

**【推测】** C 越大 → chunk 内顺序依赖的代价越高、但局部一致性越好；C=32 与 W=16 的组合意味着"每个 chunk 内约 2 个窗口长度"，是效率/性能的经验折中。

---

## 5. 全部实验数字（逐条给出处）

> **口径说明**：论文报告了**两个规模**的模型 —— **1.2B（OLMo2-1.2B，500B tokens，主结果）** 与 **550M（消融/缩放实验，300B tokens）**；另有 **151M** 仅用于图 1 的缩放曲线。**所有数字均直接取自 arXiv HTML v2 的表格文本，未做任何推算或臆造。**

### 5.1 训练配置【事实】

**1.2B 主实验**（§3.2 Training Details [#S3.SS2](https://arxiv.org/html/2504.14992v2#S3.SS2)）：
- 16 层 dense；hidden dim **2048**；FFN hidden **16384**；GQA **32 query heads / 8 KV heads**，head dim **64**；训练 **500B tokens**。
- 变体：`PHD-CSWA-2-16-32`（K=2, W=16, C=32）与 `PHD-CSWA-3-16-32`（K=3, W=16, C=32）。

**550M 消融/缩放实验**（§3.3、§3.4 [#S3.SS3](https://arxiv.org/html/2504.14992v2#S3.SS3) [#S3.SS4](https://arxiv.org/html/2504.14992v2#S3.SS4)）：
- 16 层 dense；hidden dim **1536**；FFN hidden **8192**；GQA **16 query heads / 4 KV heads**；训练 **300B tokens**。

**151M 缩放曲线**（图 1 [#S1.F1](https://arxiv.org/html/2504.14992v2#S1.F1)）：同一架构重复 1/2/3/4 次，训练 **100B tokens**（loss 与 HellaSwag）。

**评测集**（§3.1 [#S3.SS1](https://arxiv.org/html/2504.14992v2#S3.SS1)）：ARC、HellaSwag、PIQA、Winogrande、**MMLU**、CommonsenseQA（1.2B 表 1 用 "MMLU Var (MMLU-V)"；550M 表 2 用 "MMLU"）。
注：§3.1 列出的评测集**不含 GSM8K / MATH / BBH 等推理基准**，也**不含长上下文基准**（如 RULER、LongBench）——见 §8。

### 5.2 表 1：1.2B 主结果【事实｜逐格数字】

出处：**Table 1** [#S3.T1](https://arxiv.org/html/2504.14992v2#S3.T1)

| 模型 | loss ↓ | MMLU-V ↑ | Hella. ↑ | ARC-C ↑ | ARC-E ↑ | PIQA ↑ | Wino. ↑ | Comm. ↑ | **Avg. ↑** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla 1.2B | 2.601 | 35.9 | 62.3 | 36.1 | 70.3 | 74.6 | 60.5 | 43.4 | **54.7** |
| PHD-CSWA-**2**-16-32 1.2B | 2.576 | 35.8 | 63.7 | 38.8 | 70.2 | 75.2 | 62.5 | 47.4 | **56.2** |
| PHD-CSWA-**3**-16-32 1.2B | 2.567 | 36.7 | 64.5 | 40.2 | 73.5 | 75.2 | 62.1 | 45.0 | **56.7** |

**论文自陈的汇总提升**（§3.2 [#S3.SS2](https://arxiv.org/html/2504.14992v2#S3.SS2)）：
- `PHD-CSWA-2-16-32`：**平均 +1.5% 准确率**，训练 loss **−0.025**。
- `PHD-CSWA-3-16-32`：**平均 +2.0% 准确率**，训练 loss **−0.034**。

**逐项差值（本次核对计算）**：

| 对比 | loss Δ | MMLU-V | Hella. | ARC-C | ARC-E | PIQA | Wino. | Comm. | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=2 vs Vanilla | −0.025 | −0.1 | **+1.4** | **+2.7** | −0.1 | **+0.6** | **+2.0** | **+4.0** | **+1.5** |
| K=3 vs Vanilla | −0.034 | **+0.8** | **+2.2** | **+4.1** | **+3.2** | **+0.6** | **+1.6** | **+1.6** | **+2.0** |

- 单点最大提升：**K=3 的 ARC-Challenge +4.1**（36.1→40.2）、**K=2 的 CommonsenseQA +4.0**（43.4→47.4）。
- **唯一负向项**：K=2 的 MMLU-V **−0.1**（35.9→35.8）、ARC-E **−0.1**（70.3→70.2）——属于噪声量级，论文未单独讨论。
- **【事实｜论文内部不一致】** 表 1 中 `PHD-CSWA-3-16-32` 一行的 Avg. 印为 **56.7**，但该行 7 个 benchmark 的算术平均为 **57.23**；同一版式中 `PHD-SWA-3-16-∞`（表 2）印为 Avg. **50.2**，其 7 项算术平均为 **50.1**。→ **论文表格的 Avg. 列与分项存在轻微对不齐**，引用时建议以分项为准或直接标注"论文原值"。出处：表 1 [#S3.T1](https://arxiv.org/html/2504.14992v2#S3.T1)、表 2 [#S3.T2](https://arxiv.org/html/2504.14992v2#S3.T2)。

### 5.3 表 2：550M 解码计算缩放（PHD-SWA）【事实｜逐格数字】

出处：**Table 2** [#S3.T2](https://arxiv.org/html/2504.14992v2#S3.T2)（均为 `PHD-SWA-K-16-∞`，即无 chunk）

| 模型 | Loss ↓ | MMLU ↑ | Hella. ↑ | ARC-C ↑ | ARC-E ↑ | PIQA ↑ | Wino. ↑ | Comm. ↑ | **Avg. ↑** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla 550M | 2.782 | 31.4 | 51.5 | 30.1 | 62.5 | 71.9 | 56.8 | 40.6 | **49.3** |
| PHD-SWA-2-16-∞ 550M | 2.753 | 31.6 | 52.3 | 31.5 | 64.2 | 71.5 | 56.6 | 42.7 | **50.1** |
| PHD-SWA-3-16-∞ 550M | 2.739 | 32.1 | 53.1 | 31.5 | 64.6 | 71.5 | 56.9 | 41.7 | **50.2** |
| PHD-SWA-5-16-∞ 550M | 2.725 | 32.7 | 54.5 | 34.7 | 65.0 | 72.3 | 56.2 | 42.6 | **51.1** |

**论文自陈**：K=5 时 loss 下降 **"near 0.06"**（§3.4 [#S3.SS4](https://arxiv.org/html/2504.14992v2#S3.SS4)）。
**核对**：2.782 − 2.725 = **0.057**（= −0.057）；论文写 "near 0.06" 属四舍五入表述。
**论文自陈**："an average accuracy improvement of **1.8%** across all benchmarks when scaling to K=5"。
**核对**：49.3 → 51.1 = **+1.8** ✓ 完全一致。

**逐项差值（本次核对计算）**：

| 对比 | loss Δ | MMLU | Hella. | ARC-C | ARC-E | PIQA | Wino. | Comm. | Avg. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=2 | −0.029 | +0.2 | +0.8 | +1.4 | +1.7 | −0.4 | −0.2 | +2.1 | +0.8 |
| K=3 | −0.043 | +0.7 | +1.6 | +1.4 | +2.1 | −0.4 | +0.1 | +1.1 | +0.9 |
| K=5 | **−0.057** | +1.3 | +3.0 | **+4.6** | +2.5 | +0.4 | −0.6 | +2.0 | **+1.8** |

- 单点最大提升：**K=5 的 ARC-Challenge +4.6**（30.1→34.7）、**K=5 的 HellaSwag +3.0**（51.5→54.5）。
- **负向项**：PIQA 在 K=2/3 时 **−0.4**；Winogrande 在 K=5 时 **−0.6**（56.8→56.2）。论文未逐项讨论这些回退。
- **【事实】** 表 2 显示**性能随 K 单调/近似单调上升**（Avg 49.3→50.1→50.2→51.1），这是"解码计算缩放有效"的核心证据。出处：§3.4 [#S3.SS4](https://arxiv.org/html/2504.14992v2#S3.SS4)。

### 5.4 图 6 / 图 7：窗口与 chunk 消融【事实｜定性】

出处：**Figure 6**（window size）、**Figure 7**（chunk size）[#S3.F7](https://arxiv.org/html/2504.14992v2#S3.F7)；正文结论见 §3.3.1 [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1)。

- **窗口**：固定 $K=2$，$W\in\{1,2,4,16\}$。
  > "expanding the window from $W=1$ to $W=4$ yields **significant reductions in both training and validation loss**, further increasing to $W=16$ provides **only marginal improvements**. This suggests that maintaining a **small window** of local KV cache ... achieves **nearly optimal performance** while remaining computationally efficient and hardware-friendly." — §3.3.1
- **chunk**：固定 $W=16$，$C\in\{16,32,\text{no chunk}\}$。
  > "the introduction of chunks causes only **negligible differences** in both training and validation losses ... **larger chunk sizes yield progressively better results** ... we select **$C=32$**." — §3.3.1

**【事实｜重要边界】** 图 6/图 7 是**曲线形式**（训练/验证 loss vs step），**论文正文未给出逐点数值表**。因此"$W=1\to4$ 的 loss 降幅具体是多少"、"C=16 vs 32 的差值是多少"**在论文中无数字可引** → 标注【未找到具体数值】。上图 6 的四个子图标注 (a)(b)(c)(d)，但 HTML 中仅 (a)(b) 标题与图题 `Figure 6: Ablation studies on window size.` 关联，与 (c)(d) 一起归入 `Figure 7: Ablation studies on chunk size.`（原文图题排版存在错位，见 HTML [#S3.F7](https://arxiv.org/html/2504.14992v2#S3.F7)、`x7.png`–`x10.png`）。

### 5.5 图 8：PHD-SWA 的 K 缩放行为【事实｜定性】

出处：**Figure 8** [#S3.F8](https://arxiv.org/html/2504.14992v2#S3.F8)、§3.4 [#S3.SS4](https://arxiv.org/html/2504.14992v2#S3.SS4)。四个子图：(a) Training loss、(b) Validation loss、(c) HellaSwag accuracy、(d) MMLU-V accuracy。
> "using a fixed window size, the loss curve and downstream performance **scales effectively** w.r.t. the token repeating times." — §3.4

**【未找到】** 图 8 的逐点数值（论文未给表）。

### 5.6 图 1：151M 基础缩放与"没有 KV 管理会怎样"【事实｜定性】

出处：**Figure 1** [#S1.F1](https://arxiv.org/html/2504.14992v2#S1.F1)、§1 [#S1](https://arxiv.org/html/2504.14992v2#S1)：
- (a) Training loss、(b) HellaSwag accuracy：**loss 与下游准确率随重复次数 1/2/3/4 robustly 缩放**。
- (c) Prefilling time、(d) Decoding latency：**naive 重复**导致 "linearly increased KV cache size ... super linear increase in pre-filling time ... linear increase in decoding latency"。
**【未找到】** 图 1 的坐标轴数值（论文正文未列数字）。
**【推测】** (b) 的 HellaSwag 曲线正文未给具体点数，引用时应避免给数字。

### 5.7 图 9：pre-filling 时间与解码延迟【事实｜定性 + 一个硬数字】

出处：**Figure 9** [#S3.F9](https://arxiv.org/html/2504.14992v2#S3.F9)、§3.5 [#S3.SS5](https://arxiv.org/html/2504.14992v2#S3.SS5)。
**实验设置**（§3.5 原文）：**550M 模型、单张 A100、decoding batch size = 1**。

- **图 9(a) Pre-filling time**：
  > "**PHD-SWA** introduce **superlinear** pre-filling overhead when the sequence length increases, while **PHD-CSWA** only incurs **a small increase** compared to the vanilla baseline." — §3.5
- **图 9(b) Decoding latency**：K 扫描 $\{1,2,4,8,16,32,64,128,256\}$。
  > "Since decoding latency is limited by memory bound, both PHD-SWA and PHD-CSWA introduces minimal overhead when K is increased. Specifically, **setting K to 256 only leads to no more than 20% increase in decoding latency**." — §3.5
  → **这是论文给出的唯一量化延迟界：K=256 时解码延迟增幅 ≤ 20%**。出处：§3.5 [#S3.SS5](https://arxiv.org/html/2504.14992v2#S3.SS5)。

**【未找到】** 图 9(a) 各条曲线的**绝对 pre-filling 时间数值（ms/s）与横轴序列长度取值**——论文**只给图、不给数**。本次精读已下载该图（`https://arxiv.org/html/2504.14992v2/x15.png`，1662×1815）但**当前模型不支持图像输入**，无法从图中读数，故**不给出任何坐标值**（避免臆造）。
**【未找到】** 论文**没有**给出"pre-filling 时间降低百分比"这一常被引用的数字，只有 "small increase" / "practically negligible" / "superlinear" 的定性表述。

### 5.8 KV cache 大小对比【事实｜只有定性 + 公式级结论】

| 对象 | KV cache 规模 | 出处 |
| --- | --- | --- |
| Vanilla | $2Lbt\,n_{kv}d_h$（基准） | — |
| naive token repeat | 随 K **线性膨胀**（"linearly increased KV cache size"） | §1 [#S1](https://arxiv.org/html/2504.14992v2#S1) |
| **PHD（基础）** | **"identical KV cache size ... as the original Transformer"** | §2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2) |
| **PHD-SWA** | vanilla + **$\mathcal O(1)$ 额外**（窗口 $W$ 常驻） | §1 [#S1](https://arxiv.org/html/2504.14992v2#S1)、§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) |
| **PHD-CSWA** | 同 PHD-SWA（$\mathcal O(1)$ 额外） | §2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) |

**【未找到】** 论文**没有**给出以 MB/GB 为单位的实测 KV cache 占用表（无 "KV cache size (GB)" 类表格）。**"same KV cache size" 是公式/设计层面的断言，不是实测数字**。引用时应如此表述。
**【推测】** 若在 550M/1.2B 配置上自行换算：1.2B 模型 $L{=}16,\ n_{kv}{=}8,\ d_h{=}64$ → 每 token 的 KV ≈ $2\times16\times8\times64 = 16384$ 元素；以 bf16（2 B）计 ≈ **32 KB/token**；PHD 与 vanilla 相同，PHD-SWA 额外 $2LbWn_{kv}d_h$ 在 $b{=}1,W{=}16$ 时 ≈ 512 KB/层组（一次性常数）。**此段为本次自行推算，论文未给出，标注为【推测】。**

### 5.9 长上下文（32K / 128K）表现【未找到 —— 明确回答】

- 在 arXiv HTML v2 全文中检索 **"32K"、"128K"、"128k"、"long context"、"RULER"、"LongBench"** 等关键词：**均无与 PHD 实验结果相关的命中**（"32K/128K/128k" 命中数为 **0**）。
  出处：全文关键词扫描 <https://arxiv.org/html/2504.14992v2>。
- 实验最长上下文：**1.2B 用 500B tokens 预训练**，**未报告训练序列长度**（论文只给 token 总量、给 max_position_embeddings 之类的信息都没有）；图 9 的横轴是 pre-filling 序列长度但**未标注具体区间**。
- **结论【事实】**：**该论文没有报告任何 32K/128K 长上下文（needle-in-haystack / RULER / LongBench 类）实验**。若下游用途需要长上下文结论，**必须视为空白**，不能从本论文外推。
- 反例澄清：搜索结果中出现的 "131072"（128K）来自 **Tencent Sequential-Hidden-Decoding 的 SGLang `--context-length 131072` 示例**，与 PHD 论文无关。出处：<https://huggingface.co/tencent/Sequential-Hidden-Decoding-8B-n2>。

---

## 6. 与 SWA 既有工作的关系

> 本节分两块：**(A) 论文自己写明的关系（【事实】，带出处）**；**(B) 论文未提及、但调研问题要求的对照（【未找到】+ 公开资料【事实】+ 差异分析【推测】）**。

### 6.A 论文自身的划界【事实】

**§4.1 把稀疏注意力分为三类，并逐一划界**（出处：§4.1 [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1)）：

| 类别 | 论文列举的代表工作 | 论文对 PHD 的差异化陈述 |
| --- | --- | --- |
| **Fixed Pattern Sparse Attention** | **Mistral** 与 **Phi-3** 的 sliding window attention；dilated attention；**Longformer** 与 **BigBird** 的 local+global 混合模式 | "we **uniquely apply these patterns only to the repeated tokens, maintaining full attention for the original tokens**" |
| **Data-Dependent Sparse Attention** | Quest、SpAtten、SparQ | PHD 用**静态模式**，**无需运行时稀疏模式估计开销** |
| **Training-Native Sparse Attention** | **NSA**（Native Sparse Attention） | 同属"训练期就内建稀疏模式"，但 NSA 面向**通用块稀疏**，PHD **专门针对 hidden decoding tokens**，从而**保持与原始 Transformer 相同的 KV 占用** |

**§4.2 对 KV cache 优化的划界**（出处：§4.2 [#S4.SS2](https://arxiv.org/html/2504.14992v2#S4.SS2)）：

| 工作 | 论文归类 | 论文对 PHD 的差异化陈述 |
| --- | --- | --- |
| **H2O** | KV Cache Reduction（丢弃不重要 KV） | 这些方法多为**事后（post-hoc）压缩既有 KV**；PHD **重新定义"计算量"与"KV"的关系**——跨重复 token **共享 KV** |
| **StreamingLLM** | KV Cache Reduction（**sink attention** 处理流式输入） | 同上："fundamentally rethinks the relationship between computation and KV cache by **sharing KV cache across repeated tokens**" |
| SnapKV / LongLLMLingua | KV Cache Reduction / 压缩 | 同上 |
| PagedAttention、FlashDecoding / FlashDecoding++ | KV Cache Management | PHD **互补**，可组合 |
| FlashAttention 系列、RingAttention | Attention Kernels / 硬件优化 | PHD **正交（orthogonal）**，可叠加 |
| MInference | 注意力模式分析（借鉴其 insight） | PHD 借其分析思路做**预训练**长度缩放 |

**§4.3 对 latent thinking 的划界**（出处：§4.3 [#S4.SS3](https://arxiv.org/html/2504.14992v2#S4.SS3)）：

| 工作 | 论文的差异化陈述 |
| --- | --- |
| Goyal et al.（**pause tokens**） | 插入**可学习** token；PHD 直接重复输入 token |
| Quiet-Star | RL 在线探索插入 thinking token；PHD 无需 RL |
| Coconut / COCOMix / 其他 continuous-signal 方法 | "their applicability is **largely limited due to the linear increase of KV cache size**"；PHD 用 KV 管理保持可接受的增量 |
| CoTFormer | 复用**层间**隐状态；PHD 在**同层内**重复输入 token |
| Inner Thinking Transformer / Geiping et al.（recurrent latent layers） | "recurrent models **cannot be used in parallel generation**, leading to limited efficiency during inference" |

### 6.B 调研问题点名、但论文**未**提及的对象【未找到】

在 arXiv HTML v2 全文中做关键词扫描，**命中数为 0**：**Gemma / Gemma-2 / Gemma-3**（0）、**GPT-OSS**（0）、**Qwen**（0）。
出处：全文扫描 <https://arxiv.org/html/2504.14992v2>。
→ **【事实】PHD 论文完全没有讨论 Gemma 系列、GPT-OSS、Qwen 系列的 SWA / 混层方案。**

下面对照信息来自**公开模型配置**（非 PHD 论文），并给出 PHD 与它们的差异分析（【推测】部分已标注）：

| 方法 | SWA 用法（【事实】，来自公开 config） | 与 PHD-SWA 的差异（【推测】） |
| --- | --- | --- |
| **Longformer**（Beltagy et al. 2020, arXiv:2004.05150） | 局部窗口 + **全局 token**（少数 token 全可见）的混合模式 | PHD-SWA 的"全局"部分**不是少数 token，而是全部 original tokens**（占序列的 $1/K$，K=2 时即 50%），且是**因果**全局；Longformer 是 encoder 式双向 + 少量预选 global token。出处：PHD §4.1 提及 Longformer |
| **Mistral 7B**（arXiv:2310.06825） | 全层（每层同类）**局部滑窗**注意力，配合 rolling buffer KV cache；**所有 token 都被滑窗约束** | PHD-SWA **只在 hidden decoding tokens 上滑窗**，original tokens **保持全注意力**；Mistral 的 KV cache 是 $\mathcal O(W)$ **有损**压缩全部历史，PHD 的全局 KV **无压缩、完整保留** |
| **StreamingLLM**（ICLR 2024, arXiv:2309.17453） | 保留 **attention sink（前几个 token）+ 最近窗口**，丢弃中间 | PHD **不丢任何 original token 的 KV**（不依赖 sink 假设）；StreamingLLM 是**推理期**改造既有模型（post-hoc），PHD 是**预训练期**内建模式（training-native）。出处：PHD §4.2 提及 StreamingLLM |
| **Gemma-2 9B** | `sliding_window=4096`（来自 `mlx-community/gemma-2-9b-4bit` config；`model_type=gemma2`，42 层，`max_position_embeddings=8192`） | 层间**交替**：一半层滑窗、一半层全局；PHD-SWA **每一层的 original tokens 都全局可见**，不依赖层间混合 |
| **Gemma-3 27B** | `sliding_window=1024`、`sliding_window_pattern=6`（来自 `unsloth/gemma-3-27b-it` / `mlx-community/gemma-3-27b-it-4bit` config；62 层，`model_type=gemma3_text`） | 同上；且 Gemma-3 的局部窗口 **1024** 远大于 PHD-SWA 的 **W=16** |
| **GPT-OSS-20B** | `sliding_window=128`；24 层 `layer_types` = `[sliding, full]` 严格交替（来自 `openai/gpt-oss-20b` config.json） | 同上；GPT-OSS 的滑窗层 **KV 被截断到 128**，PHD 的全局 KV 不被截断 |
| **Qwen 系列层间混合** | 【未找到】PHD 论文未提及；本次未在权威 config 上核验 Qwen 的 SWA 混层细节，**不给出数字** | **【推测】** 若为"线性/滑窗层 + 全注意力层"混合，其思路是**层维度**的稀疏化，PHD 是**token 维度**的稀疏化（同一层内区分两类 token）——两者**正交、可组合** |

**一句话概括差异（【推测】）**：
> 既有 SWA（Mistral / Gemma-2 / Gemma-3 / GPT-OSS / StreamingLLM）都是**对所有 token 一视同仁地压缩 KV**（把 KV 从 $\mathcal O(n)$ 降到 $\mathcal O(W)$，或用 sink+window 近似），代价是**长程信息有损**；
> **PHD-SWA 反其道而行**——**长程信息一点不丢**（original tokens 全量全局 KV），只在**本来就要丢弃的 hidden decoding tokens** 上加一个 $W$ 的局部窗口，因此是"**在免费预算上加局部性**"，而不是"**牺牲长程换显存**"。这是它与所有既有 SWA 最本质的区别。

**与 NSA 的差异（论文自陈，【事实】）**：两者都是训练原生稀疏，但 NSA 做**通用块稀疏**，PHD **专攻 hidden decoding tokens** 且**KV 占用不增**。出处：§4.1 [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1)。

---

## 7. 开源情况

### 7.1 论文本身【事实】

- **论文正文/摘要/HTML 全文未提供任何 code / weights 链接**。
  - 关键词扫描 `github`：唯一命中来自**参考文献**（SWE-bench 标题里的 "github issues"）；`huggingface`：**0 命中**；`code will be released` / `open-source`：**无命中**；`Code` 唯一命中是 "OLMo2 as the codebase"。
  出处：全文扫描 <https://arxiv.org/html/2504.14992v2>。
- **【结论】**：论文**没有**声明开源计划，也**没有**给出仓库或权重地址。

### 7.2 GitHub `ByteDance-Seed` 组织【事实】

- GitHub API：`https://api.github.com/search/repositories?q=org:ByteDance-Seed+PHD` → **`total_count = 0`**。
- 该组织当前共 **63 个仓库**（`/orgs/ByteDance-Seed/repos?per_page=100`），**逐名核对后没有 PHD / hidden-decoding 相关仓库**。同批仓库中**有**其他 SWA/长上下文相关工作（可作为对照）：
  - `FlexPrefill` — *[ICLR 2025 Oral] FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference*
  - `ShadowKV` — *[ICML 2025 Spotlight] ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference*
  - `AHN` — *Artificial Hippocampus Networks for Efficient Long-Context Modeling*
  - `Triton-distributed`、`VeOmni` 等
  出处：<https://api.github.com/orgs/ByteDance-Seed/repos?per_page=100>。
- **【结论】**：**`github.com/ByteDance-Seed` 下没有 PHD-Transformer 的官方代码。**

### 7.3 HuggingFace `ByteDance-Seed` 组织【事实】

- HF API：`https://huggingface.co/api/models?author=ByteDance-Seed&limit=100` → 返回 **59 个模型**，逐名核对**无 PHD / PHD-SWA / PHD-CSWA / hidden-decoding 相关权重**。列表包含 `Seed-OSS-36B-Base/Instruct`、`Seed-Coder-8B-*`、`UI-TARS-*`、`BAGEL-7B-MoT`、`SeedVR2-*`、`academic-ds-9B`、`Tar-*` 等。
  出处：<https://huggingface.co/api/models?author=ByteDance-Seed&limit=100>。
- 以 "PHD" 为关键词搜 HF 全站：**无** ByteDance 相关命中（命中的是无关用户的 `phd` 用户名字段）。
  出处：<https://huggingface.co/api/models?search=PHD&limit=30>。
- **【结论】**：**HF `ByteDance-Seed` 组织下没有 PHD 的官方权重。**

### 7.4 外溢线索（重要，供父任务判断）【事实】

- **腾讯（Tencent）有一条独立的、思路相近的开源线**：`tencent/Sequential-Hidden-Decoding-8B-n2 / n4 / n8`（HF，含权重），论文题 **"Hidden Decoding: Scaling Sequence Length in Pretraining"**（2026），基于 **Qwen3-8B-Base**，n× 缩放，需 patch 过的 SGLang 才能推理（示例 `--context-length 131072`）。
  - 数据点：8B baseline → n=2/n=4/n=8：BBH 78.8→81.3/83.0/83.9；MMLU 79.8→80.9/81.9/82.2；MATH 56.0→58.2/60.0/61.1；Hellaswag 79.7→83.1/85.0/85.3；GSM8K 92.5→93.3/93.9/94.6。
  - 出处：<https://huggingface.co/tencent/Sequential-Hidden-Decoding-8B-n2>、项目页 <https://github.com/Tencent/Sequential-Hidden-Decoding>。
- **【推测】** 该工作是 PHD 思想的**同期/后续独立演化**，但它**不引用 PHD 的 KV 管理**（其 n=2 变体额外引入 **1.9B embedding 参数**，走的是"独立 embedding 矩阵"路线），**两者不是同一实现**。可作为"hidden decoding 思路已被更大规模（8B）与更长上下文（128K）验证"的**旁证**，但**不能**当作 PHD 官方开源。
- **【事实】** 本次精读**未找到** PHD 的第三方复现仓库（GitHub 搜索 `org:ByteDance-Seed+PHD` = 0；未做全站穷尽搜索，标注为【未找到】而非"不存在"）。

### 7.5 开源结论汇总

| 检查项 | 结果 | 证据 |
| --- | --- | --- |
| 论文内 code link | 【未找到】 | 全文关键词扫描 |
| `github.com/ByteDance-Seed` PHD 仓库 | 【未找到】（org 搜索 total_count=0） | GitHub API |
| HF `ByteDance-Seed` PHD 权重 | 【未找到】（59 个模型逐一核对） | HF API |
| 训练数据/数据配方 | 【未找到】 | 论文未提及数据来源 |
| **总评** | **截至精读时，PHD-Transformer 无官方开源代码与权重** | — |

---

## 8. 局限与未验证的规模

### 8.1 论文自陈的限制【事实】

1. **PHD-SWA 的 pre-filling 开销随序列长度超线性增长**（这是论文明确承认的缺陷，也是 PHD-CSWA 存在的唯一理由）。
   出处：§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)、§3.5 [#S3.SS5](https://arxiv.org/html/2504.14992v2#S3.SS5)。
2. **PHD-CSWA 用 chunk 换效率，会带来性能损失**——论文承认这是在"消除 pre-filling 开销"与"性能"之间的取舍。
   > "Since the chunk-wise sliding window attention is a **compensation for accelerating the pre-filling stage**, we are interested in the **performance drop that CSWA may introduce**." — §3.3.1 [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1)
   测得结果：**损失可忽略（negligible）**，且大 chunk 更好（§3.3.1）。
3. **PHD-CSWA 仍需"最后一个 chunk 内 K 次重复"的额外前向开销**（未能完全归零）。
   出处：§2.4 [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4)。
4. **窗口收益存在快速饱和**：$W{=}4$ 之后增益边际递减 → 论文据此选 $W{=}16$（**但不是"越大越好"**）。
   出处：§3.3.1 [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1)。
5. **K 的收益也需要付 FLOPs**：`While requiring K times FLOPs`（§2.2），PHD 只在 **memory-bound** 推理场景下"延迟开销极小"；在 compute-bound 场景下该优势会削弱。
   出处：§2.2 [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2)。

### 8.2 论文**没有**验证、但影响结论适用范围的点【事实 + 推测】

| 项 | 状态 | 证据/说明 |
| --- | --- | --- |
| **模型规模** | 最大仅 **1.2B**（主结果）；另有 550M、151M | §3.2 / §3.3 / 图 1；**【未找到】** 7B/70B+ 规模验证 |
| **训练 token 量** | 1.2B 用 **500B tokens**；550M 用 300B | §3.2 / §3.3 |
| **长上下文（32K/128K）** | **完全未验证** | 全文 "32K/128K/128k" 命中 **0**；无 RULER/LongBench/needle 测试 |
| **max_position_embeddings / 训练序列长度** | **【未找到】** 论文未报告训练序列长度 | §3.2 只给 token 总量 |
| **训练成本（GPU-hours / 墙钟时间）** | **【未找到】** 论文未报告 | 无训练时长表 |
| **pre-filling 的绝对时间数字** | **【未找到】** 只有图 9(a) 曲线 + 定性描述 | §3.5 |
| **KV cache 的绝对显存数字** | **【未找到】** 只有"same as vanilla"的公式级断言 | §2.2 |
| **下游任务覆盖** | 仅 7 个常识/知识类 benchmark（ARC、HellaSwag、PIQA、Winogrande、MMLU、CommonsenseQA）；**无数学/代码/推理重型基准**（GSM8K、MATH、BBH、GPQA 均未评） | §3.1 |
| **指令/对齐阶段** | **【未找到】** 只有 base 预训练实验，无 SFT/RL 后训练验证 | 全文 |
| **推理框架集成** | **【未找到】** 未说明 vLLM/SGLang/TensorRT-LLM 支持 | 全文 |
| **与既有 SWA 方法的直接对比实验** | **【未找到】** 只是在 §4 做文字划界，**没有** Mistral-SWA / Gemma / StreamingLLM 的对照数字 | §4.1、§4.2 |
| **kernel 实现细节** | **【未找到】** 未给 kernel 伪代码/开源实现；只给重排思路 | §2.3 |
| **公式(2)的符号严谨性** | **【推测｜原文疑点】** 式(2) 第二支写作 $i\le j$，但 $j$ 同时也是**位置下标**（$x_j^s$ 中 $j\in[1,t]$、$s\in[1,K]$）。严格地说，副本序号应为 $s$ 而非 $j$，此处论文变量命名存在**混用/歧义**，读者需按图 2/图 4 的图示理解（同位置的第 $i$ 份可见第 $\le i$ 份） | §2.1、§2.2 式(2)、图 2、图 4 |
| **表 1 / 表 2 的 Avg. 列** | **【事实｜论文内部不一致】** 见 §5.2 说明；`PHD-CSWA-3-16-32` 分项均值 57.23 vs 论文印 56.7；`PHD-SWA-3-16-∞` 分项均值 50.1 vs 论文印 50.2 | 表 1、表 2 |
| **图 6/7 的标注** | **【事实｜排版错位】** (a)(b) 标题归入 `Figure 6`，(c)(d) 归入 `Figure 7`，但正文同时引用"图 7"说明窗口与 chunk 两个话题 | §3.3.1 正文 vs HTML [#S3.F7](https://arxiv.org/html/2504.14992v2#S3.F7) |

### 8.3 未验证规模的判断【推测】

> **该论文的结论强度与其验证规模不匹配。** "pretraining length scaling with constant KV" 是一个**架构级**主张，但：
> - 参数规模只到 **1.2B**（现代 LLM 的 1/50 ~ 1/500）；
> - 上下文**完全未验证**到长上下文区间；
> - 收益（+1.5 ~ +2.0 avg）在**常识类** benchmark 上取得，**未在推理重型任务复现**；
> - **无官方开源**，无法独立复现。
>
> 因此【推测】：PHD 的**机制设计（KV 只留 original tokens + SWA 只加在 hidden tokens）在原理上是自洽且有吸引力的**，尤其"在本来就要丢弃的 token 上加局部窗口"这一思路很干净；但其**性能收益是否在 7B+ / 长上下文 / 推理任务上成立，属于论文未回答的开放问题**。若父任务需要把 PHD 与 SWA 的工程选型结合考虑，**应把 PHD 定位为"机制参考/思路来源"，而非"可直接复用的开源方案"**。

---

## 9. 关键要点汇总（可直接引用）

1. **PHD = 输入重复 K 次 + 只保留 original token 的 KV**。架构不变，改的是输入序列与注意力掩码（式 2）。hidden decoding tokens 的 KV **用完即丢**。【§2.2】
2. **持久 KV 与 vanilla 相同**，FLOPs 涨 K 倍但可并行；**pre-filling 与 K 无关**（PHD 基础版）。【§2.2】
3. **PHD-SWA：滑窗只加在 hidden decoding tokens 上（W=16），original tokens 保持全注意力**——这是与 Mistral/Gemma/GPT-OSS/StreamingLLM 的**本质差异**。额外 KV 仅 $\mathcal O(1)$。【§2.4、§4.1】
4. **PHD-CSWA：把滑窗限制在 chunk 内（C=32）**，将 pre-filling 的线性/超线性增长压缩到"仅最后一个 chunk"，实测损失可忽略、大 chunk 更好。【§2.4、§3.3.1】
5. **收益**：1.2B Avg **+1.5 / +2.0**（K=2/3），loss **−0.025 / −0.034**；550M Avg **+1.8**（K=5），loss **−0.057**。【表 1、表 2】
6. **代价**：**K=256 时解码延迟增幅 ≤20%**（唯一的量化延迟界）；PHD-SWA 的 pre-filling 是**超线性**（论文承认）。【§3.5】
7. **长上下文（32K/128K）实验：不存在**。最大模型 **1.2B**。评测集**无数学/代码/推理重型基准**。【§3.1、§8.2】
8. **无官方开源**（GitHub/HF `ByteDance-Seed` 均无）；论文无 code 声明。【§7】
9. **无会议接收记录**（v2 无 journal-ref/comment）。v1 2025-04-21、v2 2025-04-24。【§1.2、§1.3】
10. **可参考的邻近开源**：腾讯 **Sequential Hidden Decoding**（Qwen3-8B、128K、已开源权重），思路同源但实现路线不同。【§7.4】

---

## 10. 出处索引（快速跳转）

**一手来源（本次精读基准）**
- arXiv 摘要页（作者、时间线、分类）：<https://arxiv.org/abs/2504.14992>
- **arXiv HTML v2 全文（所有章节/表/图引用的基准）**：<https://arxiv.org/html/2504.14992v2>
- arXiv PDF：<https://arxiv.org/pdf/2504.14992>
- arXiv API（journal_ref / comment 为空）：<http://export.arxiv.org/api/query?id_list=2504.14992>
- arXiv 源文件（TeX）：<https://arxiv.org/src/2504.14992>

**全文锚点（HTML v2）**

| 内容 | 锚点链接 |
| --- | --- |
| §1 Introduction（贡献、naive repeat 的三大低效、151M 图 1） | [#S1](https://arxiv.org/html/2504.14992v2#S1) |
| 图 1（151M 缩放 / prefill / decode） | [#S1.F1](https://arxiv.org/html/2504.14992v2#S1.F1) |
| §2 Approach | [#S2](https://arxiv.org/html/2504.14992v2#S2) |
| §2.1 Notations（式 1、original vs hidden 定义） | [#S2.SS1](https://arxiv.org/html/2504.14992v2#S2.SS1) |
| §2.2 PHD（**式 2**、KV 策略、constant prefill） | [#S2.SS2](https://arxiv.org/html/2504.14992v2#S2.SS2) |
| 图 2（PHD block 总览、KV 可见性） | [#S2.F2](https://arxiv.org/html/2504.14992v2#S2.F2) |
| §2.3 Kernel Design（O(K²)→O(K)、重排） | [#S2.SS3](https://arxiv.org/html/2504.14992v2#S2.SS3) |
| 图 3（注意力矩阵重排） | [#S2.F3](https://arxiv.org/html/2504.14992v2#S2.F3) |
| §2.4 **PHD-SWA / PHD-CSWA**（滑窗位置、W、chunk） | [#S2.SS4](https://arxiv.org/html/2504.14992v2#S2.SS4) |
| 图 4（PHD / SWA / CSWA 三种注意力矩阵对比） | [#S2.F4](https://arxiv.org/html/2504.14992v2#S2.F4) |
| §3 Experiments（OLMo2 基座、命名规范） | [#S3](https://arxiv.org/html/2504.14992v2#S3) |
| §3.1 Evaluation Metrics（7 个 benchmark） | [#S3.SS1](https://arxiv.org/html/2504.14992v2#S3.SS1) |
| §3.2 Main Results（1.2B 配置、+1.5/+2.0） | [#S3.SS2](https://arxiv.org/html/2504.14992v2#S3.SS2) |
| 图 5（1.2B 训练曲线） | [#S3.F5](https://arxiv.org/html/2504.14992v2#S3.F5) |
| **表 1（1.2B 全部数字）** | [#S3.T1](https://arxiv.org/html/2504.14992v2#S3.T1) |
| §3.3 Ablation（550M 配置） | [#S3.SS3](https://arxiv.org/html/2504.14992v2#S3.SS3) |
| §3.3.1 CSWA 消融（W∈{1,2,4,16}、C∈{16,32,none}） | [#S3.SS3.SSS1](https://arxiv.org/html/2504.14992v2#S3.SS3.SSS1) |
| 图 7（window / chunk 消融曲线） | [#S3.F7](https://arxiv.org/html/2504.14992v2#S3.F7) |
| §3.4 Decoding Token Scaling（K∈{2,3,5}、约 0.06 loss↓、+1.8%） | [#S3.SS4](https://arxiv.org/html/2504.14992v2#S3.SS4) |
| 图 8（PHD-SWA 缩放行为） | [#S3.F8](https://arxiv.org/html/2504.14992v2#S3.F8) |
| **表 2（550M 全部数字）** | [#S3.T2](https://arxiv.org/html/2504.14992v2#S3.T2) |
| §3.5 **Pre-filling / Decoding Speed**（A100、bs=1、≤20%） | [#S3.SS5](https://arxiv.org/html/2504.14992v2#S3.SS5) |
| 图 9（prefilling time / decoding latency） | [#S3.F9](https://arxiv.org/html/2504.14992v2#S3.F9) |
| §4.1 稀疏注意力划界（Mistral / Longformer / NSA） | [#S4.SS1](https://arxiv.org/html/2504.14992v2#S4.SS1) |
| §4.2 KV cache 优化划界（StreamingLLM / H2O / FlashAttention） | [#S4.SS2](https://arxiv.org/html/2504.14992v2#S4.SS2) |
| §4.3 Latent thinking 划界（pause token / Coconut / CoTFormer） | [#S4.SS3](https://arxiv.org/html/2504.14992v2#S4.SS3) |
| §5 Conclusion | [#S5](https://arxiv.org/html/2504.14992v2#S5) |

**开源核验来源**
- GitHub `ByteDance-Seed` 仓库列表（63 个，无 PHD）：<https://api.github.com/orgs/ByteDance-Seed/repos?per_page=100>
- GitHub org 内搜索 PHD（total_count=0）：<https://api.github.com/search/repositories?q=org:ByteDance-Seed+PHD>
- HF `ByteDance-Seed` 模型列表（59 个，无 PHD）：<https://huggingface.co/api/models?author=ByteDance-Seed&limit=100>
- HF 全站 PHD 检索（无相关）：<https://huggingface.co/api/models?search=PHD&limit=30>

**对照工作（既有 SWA / 邻近开源）**
- Longformer：<https://arxiv.org/abs/2004.05150>
- Mistral 7B：<https://arxiv.org/abs/2310.06825>
- StreamingLLM（ICLR 2024）：<https://arxiv.org/abs/2309.17453>
- Gemma-2 9B config（`sliding_window=4096`）：<https://huggingface.co/mlx-community/gemma-2-9b-4bit/resolve/main/config.json>
- Gemma-3 27B config（`sliding_window=1024`, `sliding_window_pattern=6`）：<https://huggingface.co/unsloth/gemma-3-27b-it/resolve/main/config.json>
- GPT-OSS-20B config（`sliding_window=128`、层间交替）：<https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json>
- 腾讯 Sequential Hidden Decoding（8B、128K、已开源权重）：<https://huggingface.co/tencent/Sequential-Hidden-Decoding-8B-n2>、<https://github.com/Tencent/Sequential-Hidden-Decoding>

---

## 11. 对父任务（Seed 洞察 / SWA 关联研究）的处置建议

1. **PHD-SWA 是"SWA 的一种非常规用法"**：它不是"用 SWA 替代全注意力以省 KV"，而是"**在免费额度上补局部性**"。若父任务的 SWA 综述要分类，PHD-SWA 应单列一类：**token 类型分化的 SWA（type-differentiated SWA）**——同一层内，一部分 token（original，占比 $1/K$）走全局注意力，另一部分（hidden decoding，占比 $(K-1)/K$）走局部窗口。这与"层间混合 SWA"（Gemma-2 / Gemma-3 / GPT-OSS）是**正交的两个维度**，理论上可叠加。
2. **可直接引用的硬数字只有三处**：表 1、表 2、以及 "K=256 → 解码延迟 ≤+20%"。其余（prefill 时间、KV 显存、窗口/chunk 消融）**只有曲线没有数字**，引用时切勿编数。
3. **明确写"无开源"**：`github.com/ByteDance-Seed` 与 HF `ByteDance-Seed` 均无 PHD；论文未声明开源。**不要**把腾讯 Sequential Hidden Decoding 误当 PHD 开源。
4. **长上下文是空白**：**没有 32K/128K 实验**。若父任务需要"PHD 在长上下文下的表现"，答案是**论文未做**，需标注为 open problem。
5. **规模上限 1.2B / 500B tokens**：任何"PHD 能省 X% 显存/时间"的规模化结论都缺乏论文支撑。
6. **论文自身有两处小瑕疵可提示读者**：表 1/表 2 的 Avg. 列与分项微不一致；式(2) 的 $i\le j$ 用了位置下标 $j$ 表示副本序号，符号有歧义（应结合图 2/图 4 理解）。

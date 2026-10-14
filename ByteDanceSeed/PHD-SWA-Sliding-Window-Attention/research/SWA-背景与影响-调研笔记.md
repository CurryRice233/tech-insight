# 滑窗注意力（SWA）技术背景与产业采用调研笔记

> **用途**：技术洞察报告第 1 章（技术背景）与第 3 章（技术影响）。
> **落点**：ByteDance Seed《Efficient Pretraining Length Scaling》中的 **PHD-SWA / PHD-CSWA**（arXiv:2504.14992）。
> **调研时间**：2026-10-14。
> **调研方式**：直读一手来源 —— arXiv abs / arXiv HTML 全文、官方 config.json（`huggingface.co/<repo>/raw/main/config.json`）、官方 GitHub 源码、官方 docs、官方 README。
> **标注约定**：**【事实】**= 有一手 URL 支撑并已逐篇核实；**【推测】**= 分析推断；**【未找到】**= 未能从一手来源证实。
> **访问限制声明**：Google（Gemma 系列）、Meta（Llama-4）、Cohere（Command R/R+）的 HuggingFace 仓库为 **GatedRepo（401）**，本环境无账号 token。凡涉及这些模型的配置，本笔记改用：① 官方技术报告原文；② HuggingFace `transformers` 官方建模代码中的 config 默认值/派生逻辑；③ 可公开访问的官方衍生仓库。凡属此类的条目均已显式标注来源类型。

---

## 0. 一页速览：本调研的 8 条核心结论

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | SWA 的学术起点是 **Longformer（arXiv:2004.05150）**：窗口 w=512，复杂度 O(n·w)，因多层堆叠使顶层感受野达 ℓ×w | 强（原文） |
| 2 | **Mistral-7B（arXiv:2310.06825）** 把 SWA 带进主流 LLM：window_size=4096，滚动 buffer，最后一层理论注意力跨度 ≈131K | 强（原文） |
| 3 | **StreamingLLM（arXiv:2309.17453）** 揭示 SWA 单独使用的致命缺陷：PPL 从 5.40 → **5158.07**；加回 **4 个初始 token** 即修复 | 强（原文 Table 1） |
| 4 | KV cache 从 **O(L) 降到 O(W)**；Mistral 官方称 32K 序列上缓存内存 **降低 8×** | 强（原文+可复算） |
| 5 | **产业界几乎不做"全层纯 SWA"**，主流是 **SWA/全注意力层间混合**：Gemma-2 1:1、Gemma-3 5:1、GPT-OSS 1:1（窗口仅 128）、Command A 3:1 | 强（一手 config / 原文） |
| 6 | SWA 会**摧毁长上下文检索能力**：SWAA（arXiv:2512.10411）中纯 SWA 在 RULER/LongMemEval 上**归零**，必须"层间混合 + 另一项措施"才能恢复 | 强（原文 Table 1） |
| 7 | **FlashAttention / vLLM / SGLang 均已支持** SWA；但**混合层**让 KV cache 分页与 prefix cache 变得复杂（vLLM 专门写了 Hybrid KV Cache Manager 文档） | 强（官方源码/文档） |
| 8 | **PHD-SWA 不是全层滑窗**：它只对 **hidden decoding tokens** 加滑窗，original tokens 仍全局长程可见，且窗口极小（W=4~32） | 强（原文 §2.4） |

---

## 1. SWA 的历史与动机

### 1.1 前史：Sparse Transformer（局部窗口的雏形）

- **链接**：[arXiv:1904.10509](https://arxiv.org/abs/1904.10509)（Child et al., 2019, *Generating Long Sequences with Sparse Transformers*）
- Mistral-7B 论文在引用 SWA 时同时引用了本工作（`[6] Child et al., 2019`）与 Longformer，说明"固定局部窗口"这一稀疏模式的源头可追到 2019 年的 Sparse Transformer。
- **【事实】** Mistral-7B 原文引用列表可见该条目："Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509, 2019."（[Mistral 7B §References](https://arxiv.org/html/2310.06825v1)）
- 本笔记未对 Sparse Transformer 全文做逐条核实（其窗口/stride 配置与 Longformer 不同），故此处仅作为源流记录，不作为量化依据。

### 1.2 Longformer —— SWA 的学术奠基（arXiv:2004.05150）

- **链接**：[arXiv:2004.05150](https://arxiv.org/abs/2004.05150) ｜ [arXiv HTML v2](https://arxiv.org/html/2004.05150v2)
- **作者/机构**：Iz Beltagy, Matthew E. Peters, Arman Cohan（Allen Institute for AI）
- **一句话核心**：把标准自注意力换成"局部滑窗 + 任务驱动的全局注意力"的**可直接替换（drop-in replacement）**稀疏注意力，把复杂度从 O(n²) 降到 **O(n)**。

**关键结论与原文数字（均来自 [arXiv HTML v2 §3.1](https://arxiv.org/html/2004.05150v2)）：**

1. **窗口大小 w = 512**
   > "We use sliding window attention with window size of 512, therefore using the same amount of computation as RoBERTa."
   —— 512 对应 RoBERTa 的 512 位置上限，因此**计算量与 RoBERTa 完全一致**，这是 Longformer 能用 RoBERTa 权重做"最小改动继续预训练"的关键。

2. **复杂度 O(n × w)**
   > "Given a fixed window size w, each token attends to ½w tokens on each side (Fig. 2(b)). The computation complexity of this pattern is O(n × w), which scales linearly with input sequence length n."

3. **多层堆叠 → 顶层感受野 ℓ×w**
   > "In a transformer with ℓ layers, the receptive field size at the top layer is ℓ × w (assuming w is fixed for all layers). Depending on the application, it might be helpful to use different values of w for each layer to balance between efficiency and model representation capacity."

4. **空洞滑窗（dilated sliding window）把感受野放大到 ℓ×d×w**
   > "Assuming a fixed d and w for all layers, the receptive field is ℓ × d × w, which can reach tens of thousands of tokens even for small values of d."
   并且**逐头设置不同 dilation 更好**：
   > "We found settings with different dilation configurations per head improves performance by allowing some heads without dilation to focus on local context, while others with dilation focus on longer context."
   ⚠️ **重要反面证据**：脚注 6 明确指出，对已预训练的 RoBERTa 权重**加 dilation 反而掉点**：
   > "Adding dilation on a few heads as in §4.1 hurt performance, likely because it is not compatible with the pretrained RoBERTa weights. Retraining such model from scratch might be needed to improve performance."
   —— 这正是后来 SWAA（§4.2）所说"训练-推理分布错配"在 2020 年就已出现的早期证据。

5. **全局注意力（global attention）的必要性**
   > "In our case, the windowed and dilated attention are not flexible enough to learn task-specific representations. Accordingly, we add 'global attention' on few pre-selected input locations. Importantly, we make this attention operation symmetric."
   —— 全局 token 数量与 n 无关，故整体复杂度仍为 O(n)。**这是"SWA + 少量全注意力"这一产业范式的学术源头。**

6. **下游结果（[Table 7](https://arxiv.org/html/2004.05150v2)，dev set）**

   | 模型 | WikiHop | TriviaQA | HotpotQA | OntoNotes | IMDB | Hyperpartisan |
   |---|---|---|---|---|---|---|
   | RoBERTa-base | 72.4 | 74.3 | 63.5 | 78.4 | 95.3 | 87.4 |
   | **Longformer-base** | **75.0** | **75.2** | **64.4** | **78.6** | **95.7** | **94.8** |

   摘要另称 Longformer **在 WikiHop 与 TriviaQA 上创下当时 SOTA**，并在 char-level LM 的 text8/enwik8 上取得 SOTA；LED 变体用于 arXiv 摘要。

### 1.3 Mistral-7B —— SWA 进入主流 LLM（arXiv:2310.06825）

- **链接**：[arXiv:2310.06825](https://arxiv.org/abs/2310.06825) ｜ [arXiv HTML v1](https://arxiv.org/html/2310.06825v1)
- **作者/机构**：Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, William El Sayed（Mistral AI）
- **提交时间**：2023-10-10（v1）

**关键原文与数字（均来自 [arXiv HTML v1 §2 Architectural details](https://arxiv.org/html/2310.06825v1)）：**

1. **明确采用 SWA + GQA**
   > "Our model leverages grouped-query attention (GQA) for faster inference, coupled with sliding window attention (SWA) to effectively handle sequences of arbitrary length with a reduced inference cost."（Abstract）

2. **架构表（[Table 1](https://arxiv.org/html/2310.06825v1)）—— 窗口与上下文的确切数值**

   | Parameter | Value |
   |---|---|
   | dim | 4096 |
   | n_layers | 32 |
   | head_dim | 128 |
   | hidden_dim | 14336 |
   | n_heads | 32 |
   | n_kv_heads | 8 |
   | **window_size** | **4096** |
   | **context_len** | **8192** |
   | vocab_size | 32000 |

   —— 即：**窗口 W=4096，模型 context_len=8192**。这是用户所问"4096 窗口 + 滚动 buffer 达到 8K 以上上下文"的准确出处。

3. **理论注意力跨度 ≈131K（W=4096 时的 k×W 论证）**
   > "The hidden state in position i of the layer k, h_i, attends to all hidden states from the previous layer with positions between i−W and i. Recursively, h_i can access tokens from the input layer at a distance of up to W×k tokens... **At the last layer, using a window size of W=4096, we have a theoretical attention span of approximately 131K tokens.**"
   —— 131K ≈ 32 层 × 4096 = 131,072。Figure 1 caption 同样写："Each token can attend to at most W tokens from the previous layer (here, W=3). Note that **tokens outside the sliding window still influence next word prediction.** At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k×W tokens."

4. **实测加速：16K 序列、W=4096，相对 vanilla attention 有 2× 提速**
   > "In practice, for a sequence length of 16K and W=4096, changes made to FlashAttention [11] and xFormers [18] yield a **2x speed improvement** over a vanilla attention baseline."

5. **Rolling Buffer Cache（滚动缓存）——KV cache 从 O(L) 变 O(W) 的机制**
   > "A fixed attention span means that we can limit our cache size using a rolling buffer cache. **The cache has a fixed size of W**, and the keys and values for the timestep i are stored in position **i mod W** of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten, and **the size of the cache stops increasing**."
   > "**On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality.**"
   —— 32K / 4096 = **8×**，与公式完全一致。这是"KV cache 从 O(L) 降到 O(W)"最硬的官方量化背书。

6. **Pre-fill and Chunking（分块预填充）**
   > "we can chunk it into smaller pieces, and pre-fill the cache with each chunk. For this purpose, **we can select the window size as our chunk size**."（Figure 3 caption 展示了 chunk 内部 causal mask + 对 cache 的滑动窗口 mask + 窗口外不 attend 的三段结构）
   —— 这是 PHD-CSWA「chunk-wise」思路的直接前身（见 §6）。

7. **vLLM 集成**：论文明确称参考实现可用 vLLM 部署（"facilitating easy deployment ... using the vLLM inference server"），并在致谢中感谢 FlashAttention / xFormers / vLLM 团队 —— 说明 **SWA 的产业落地一开始就是与推理引擎协同设计的**。

### 1.4 StreamingLLM —— "attention sink + 滑窗"（arXiv:2309.17453）

- **链接**：[arXiv:2309.17453](https://arxiv.org/abs/2309.17453) ｜ [arXiv HTML v4](https://arxiv.org/html/2309.17453v4)
- **作者/机构**：Guangxuan Xiao（MIT）, Yuandong Tian（Meta AI）, Beidi Chen（CMU）, Song Han（MIT/NVIDIA）, Mike Lewis（Meta AI）
- **版本**：v4（2024-04-07）

**关键结论与原文数字（均来自 [arXiv HTML v4](https://arxiv.org/html/2309.17453v4)）：**

1. **纯窗口注意力（Window Attention）在长文本上崩溃 —— 这是 SWA 最著名的负面结果**

   摘要：
   > "**Window attention, where only the most recent KVs are cached, is a natural approach — but we show that it fails when the text length surpasses the cache size.** We observe an interesting phenomenon, namely **attention sink**, that keeping the KV of initial tokens will largely recover the performance of window attention."

   **[Table 1](https://arxiv.org/html/2309.17453v4)（Llama-2-13B，PG19 第一本书 65K tokens 上的 PPL ↓）：**

   | Cache config | PPL ↓ |
   |---|---|
   | **0 + 1024（纯窗口）** | **5158.07** |
   | 4 + 1020（4 初始 + 1020 近期） | **5.40** |
   | 4"\n" + 1020（4 个换行符 + 1020 近期） | 5.60 |

   —— **5158.07 → 5.40，约 955× 的差异**。这是"SWA 单独使用会灾难性失效"最直接的量化证据。

2. **为什么失效：首 token 是 attention sink，被滑出窗口即崩**
   > Figure 1(b) caption: "Window Attention caches the most recent L tokens' KV. While efficient in inference, **performance declines sharply once the starting tokens' keys and values are evicted.**"
   > "**even just evicting the KV of the first token**, as illustrated in Figure 3."

   **[Table 2](https://arxiv.org/html/2309.17453v4)（重引入初始 token 数量的影响）：**
   > "(1) Window attention (0+y) has a drastic increase in perplexity. (2) Introducing **one or two initial tokens doesn't fully restore model perplexity**, showing that the model doesn't solely use the first token as the attention sink. (3) Introducing **four initial tokens** gener[ally suffices]..."
   —— 即"**4 个初始 token 是经验最小充分值**"。

3. **机制解释：softmax 的归一化约束导致"注意力倾倒"**
   > "The nature of the SoftMax function (Equation 1) prevents all attended tokens from having zero values. This requires aggregating some information from other tokens across all heads in all layers, even if the current embedding has sufficient self-contained information for its prediction. Consequently, **the model tends to dump unnecessary attention values to specific tokens.**"
   > 该文把这一现象与量化 outlier（Xiao et al., 2023）类比，并提及 **SoftMax-Off-by-One（Miller, 2023）** 作为潜在补救方案。
   —— **这条直接连到 2025 年 GPT-OSS：其每个注意力头的 softmax 分母带可学习 bias，"similar to off-by-one attention and attention sinks"（见 §3.4）。**

4. **位置比语义更重要**
   > "Substituting the original four initial tokens with linebreak tokens '\n' (4"\n"+1020) achieves comparable perplexity restoration... This suggests that **the absolute position of the starting tokens, rather than their semantic value, holds greater significance.**"

5. **可预训练一个专用 sink token**
   > "we discover that **adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment**."
   > "we confirm our attention sink hypothesis and demonstrate that **language models can be pre-trained to require only a single attention sink token** for streaming deployment."

6. **规模与加速**
   > "StreamingLLM can enable **Llama-2-[7, 13, 70]B, MPT-[7, 30]B, Falcon-[7, 40]B, and Pythia-[2.9, 6.9, 12]B** to perform stable and efficient language modeling with **up to 4 million tokens** and more."
   > "StreamingLLM outperforms the **sliding window recomputation baseline by up to 22.2× speedup**."（该 baseline 的复杂度为 O(T·L²)，见 Figure 1(c)）
   —— 即"滑窗 + 不重算"若不加 sink 会崩，"滑窗 + 重算"正确但极慢；StreamingLLM 是二者的折中。

### 1.5 历史脉络小结（第 1 章可直接引用的线性叙事）

```
2019 Sparse Transformer (1904.10509)      固定稀疏/局部模式的开端
  ↓
2020 Longformer (2004.05150)              滑窗 w=512 + 全局注意力；O(n·w)；感受野 ℓ·w
  ↓                                       （dilation 对预训练权重有害 —— 最早的分布错配证据）
2023 Mistral-7B (2310.06825)              W=4096 + rolling buffer + chunked prefill；理论跨度 131K；32K 上省 8× 缓存
  ↓
2023 StreamingLLM (2309.17453)            纯滑窗崩（PPL 5158 → 5.40 with 4 sinks）；sink 机制；SoftMax-off-by-one
  ↓
2024 Gemma-2 (2408.00118)                 首次大规模产品化"1 层全注意力 : 1 层滑窗"层间混合
  ↓
2025 Gemma-3 (2503.19786) / GPT-OSS (2508.10925) / Command A
                                          混合比例走向 5:1、1:1(窗口仅128)、3:1；纯 SWA 基本消失
  ↓
2025 PHD-SWA (2504.14992)                 把滑窗的适用对象从"全序列"改为"hidden decoding token 子群"（见 §6）
  ↓
2025-12 SWAA (2512.10411)                 系统化"如何把 FA 模型改造为 SWA"，并给出纯 SWA 归零的量化证据
```

---

## 2. SWA 的 KV cache 与算力收益：公式与实证

### 2.1 KV cache 公式

设：层数 `n_layers`、KV 头数 `n_kv_heads`、头维度 `head_dim`、字节数 `b`（FP16 为 2，FP8 为 1）、序列长度 `L`、窗口 `W`。

**全注意力（Full Attention, FA）：**

```
KV_cache_FA(L) = 2 × n_layers × n_kv_heads × head_dim × L × b        # O(L)
```

**滑窗注意力（SWA），每层最多缓存最后 W 个 token：**

```
KV_cache_SWA = 2 × n_layers × n_kv_heads × head_dim × min(L, W) × b  # O(W)
```

**压缩比：**

```
ratio = KV_cache_FA / KV_cache_SWA = L / min(L, W)   →  当 L ≫ W 时 ≈ L / W
```

**【事实】Mistral 官方给出的正是这个比值的实例**："On a sequence length of **32k** tokens, this reduces the cache memory usage by **8x**" —— 32K/4096 = 8。（[arXiv:2310.06825 §2](https://arxiv.org/html/2310.06825v1)）

**混合层（interleaved SWA/full）的 KV cache：** 设全注意力层数 `n_full`、滑窗层数 `n_sw`（`n_full + n_sw = n_layers`）：

```
KV_cache_hybrid = 2 × n_kv_heads × head_dim × b × ( n_full × L + n_sw × min(L, W) )
```

即 **收益介于纯 SWA 与纯 FA 之间，由"全注意力层占比"线性决定** —— 这是 Gemma-3 把比例从 Gemma-2 的 1:1 改成 5:1 的直接动机（"only the global layers attend to long context"，[Gemma 3 §5.2](https://arxiv.org/html/2503.19786v1)）。

### 2.2 每 token 注意力 FLOPs

注意力核心是两次矩阵乘：`QKᵀ`（得分）与 `·V`（加权求和），每次约 `2 × d` FLOPs/元素对。

**全注意力（解码第 t 个 token，因果，可见 t 个 key）：**

```
FLOPs_attn_FA(t) ≈ 2 × (2 × n_heads × head_dim) × t = 4 × n_heads × head_dim × t     # O(L)
```

**SWA（可见 min(t, W) 个 key）：**

```
FLOPs_attn_SWA(t) ≈ 4 × n_heads × head_dim × min(t, W)                                # O(W)
```

**整个 prefill 阶段（L 个 token）的注意力 FLOPs：**

```
FA:  Σ_{t=1..L} 4·n_heads·head_dim·t   = 2 × n_heads × head_dim × L²    → O(L²)
SWA: Σ_{t=1..L} 4·n_heads·head_dim·min(t,W) ≈ 2 × n_heads × head_dim × L × W  → O(L·W)
```

（严格地：`= 2·n_heads·head_dim·(W² + 2W(L−W))`，当 L ≫ W 时 ≈ `2·n_heads·head_dim·L·W`。）

**比值：`FLOPs_SWA / FLOPs_FA ≈ W / L`。**

> ⚠️ **重要限定（务必写进报告，否则会被误读）**：`O(L·W)` 只描述**注意力矩阵乘**的算力。Transformer 每层还有 QKV/O 投影与 MLP（FFN），其 FLOPs 为 **O(L·d²)** 量级，**与 W 无关**。因此端到端加速比远小于 W/L，且受 **memory-bound 程度**支配 —— Mistral 在 16K/W=4096 时实测只有 **2×**（[arXiv:2310.06825](https://arxiv.org/html/2310.06825v1)），远低于理论 W/L=1/4 的 4×。**这正是 PHD 论文强调"decoding latency 受 memory bound 限制，故 K=256 时延迟增幅 ≤20%"的同一道理**（[arXiv:2504.14992 §3.5](https://arxiv.org/html/2504.14992v2)）。

### 2.3 完整算例（可复算，建议直接进报告表格）

**Mistral-7B 参数**（[Table 1](https://arxiv.org/html/2310.06825v1)）：`n_layers=32, n_kv_heads=8, head_dim=128, W=4096`，FP16（b=2）。

**单 token KV cache 体积：**

```
每层每 KV 头每 token = 2 (K和V) × 128 (head_dim) × 2 B = 512 B
每层每 token         = 512 B × 8 (n_kv_heads)         = 4,096 B = 4 KiB
全模型每 token       = 4 KiB × 32 (n_layers)          = 128 KiB/token
```

**不同 L 下的 KV cache 总量：**

| 序列长度 L | 全注意力 KV cache | SWA (W=4096) KV cache | 压缩比 L/W |
|---|---|---|---|
| 8,192（Mistral 训练 context_len） | 128 KiB × 8192 = **1.00 GiB** | 128 KiB × 4096 = **0.50 GiB** | 2× |
| 32,768 | 128 KiB × 32768 = **4.00 GiB** | **0.50 GiB** | **8×** ✅ 与 Mistral 原文一致 |
| 131,072（理论跨度） | 128 KiB × 131072 = **16.00 GiB** | **0.50 GiB** | **32×** |
| 1,000,000 | **122.07 GiB** | **0.50 GiB** | 244× |

（GiB = 1024³ B；128 KiB × 32768 = 2²⁰ × 2¹⁵ = 2³⁵ B = 4 GiB。）

**混合层算例（Gemma-3-27B 风格，5:1）**：`n_layers=62`（vLLM 文档给出的 52 sw + 10 full），假设每层每 token 128 KiB：

```
KV_cache_hybrid(L=128K) = 128 KiB × (10 × 131072 + 52 × 1024)
                        = 128 KiB × (1,310,720 + 53,248)
                        = 128 KiB × 1,363,968 ≈ 166.5 GiB
KV_cache_FA(L=128K)     = 128 KiB × 62 × 131072        ≈ 992.0 GiB   （纯 FA，不可行）
KV_cache_SWA 全滑窗      = 128 KiB × 62 × 1024           ≈ 7.75 GiB   （纯 SWA，质量不可接受）
```

> ⚠️ 上表数字为**按标准 MHA/GQA KV 公式的推演**（各模型 head 配置不同，Gemma-3-27B 实际为 `num_kv_heads` 等具体值需查 config），**用途是说明量级差异，不应作为具体模型的实测断言**。标注为**【推测/推演】**。

**注意力 FLOPs 算例（每 token 解码，单层）：**

```
FA  @ L=131,072: 4 × 32 heads × 128 × 131072 ≈ 2.147 GFLOP
SWA @ W=4096   : 4 × 32 heads × 128 × 4096   ≈ 0.067 GFLOP      比值 = 1/32
```

### 2.4 实证收益汇总（一手数字）

| 工作 | 收益主张（原文） | 链接 |
|---|---|---|
| Mistral-7B | 32K 序列上 cache 内存 **↓8×**，无质量损失 | [arXiv:2310.06825](https://arxiv.org/html/2310.06825v1) |
| Mistral-7B | 16K 序列、W=4096，相对 vanilla attention **2× 提速** | 同上 |
| StreamingLLM | 相比 "sliding window + recomputation" baseline **最高 22.2× 加速** | [arXiv:2309.17453](https://arxiv.org/html/2309.17453v4) |
| Gemma-2 | 推理时可把滑窗 4096→1024，PPL 仅 1.63→1.64（**"leverage for slight inference speed gain"**） | [arXiv:2408.00118 Table 10](https://arxiv.org/html/2408.00118v2) |
| SWAA | 长上下文推理 **30%~100% 加速**，质量可接受 | [arXiv:2512.10411](https://arxiv.org/abs/2512.10411) |
| PHD-SWA/CSWA | K=256（256× 解码 token 放大）时解码延迟增幅 **≤20%** | [arXiv:2504.14992 §3.5](https://arxiv.org/html/2504.14992v2) |

---

## 3. 产业采用（重点）

### 3.1 总表（一手来源核实）

> **证据类型说明**：`config` = 从 `https://huggingface.co/<repo>/raw/main/config.json` 直读并可复现；`report` = 官方技术报告原文；`code` = HuggingFace `transformers` 官方建模代码默认值/派生逻辑（仅用于 GatedRepo 无法直读 config 的场合）。

| 模型 | 是否 SWA | 窗口 | 层间模式 | 一手证据（字段=值 / 原文） | URL |
|---|---|---|---|---|---|
| **Mistral-7B-v0.1** | ✅ 是（全层） | 4096 | 全部 32 层均滑窗 | `config`：`"sliding_window": 4096`, `num_hidden_layers: 32` | [config.json](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json) |
| **Mistral-7B-v0.3** | ❌ 否（被移除） | — | 转纯全注意力 | `config`：`"sliding_window": null` | [config.json](https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json) |
| **Mistral-Nemo-Base-2407** | ❌ 否 | — | — | `config`：`"sliding_window": null`, ctx 131072 | [config.json](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/raw/main/config.json) |
| **Mixtral-8x7B-v0.1** | ❌ 否 | — | — | `config`：`"sliding_window": null` | [config.json](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json) |
| **Mistral-Small-24B-Base-2501** | ❌ 否 | — | — | `config`：`"sliding_window": null`, ctx 32768 | [config.json](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501/raw/main/config.json) |
| **Mistral-Large-Instruct-2411** | ❌ 否 | — | — | `config`：`"sliding_window": null`, 88 层, ctx 131072 | [config.json](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/raw/main/config.json) |
| **Codestral-22B-v0.1** | ❌ 否 | — | — | `config`：`"sliding_window": null` | [config.json](https://huggingface.co/mistralai/Codestral-22B-v0.1/raw/main/config.json) |
| **Gemma-2** (2B/9B/27B) | ✅ 是（混合） | 4096 | **1 滑窗 : 1 全注意力**（隔层交替） | `report`："We alternate between a local sliding window attention ... and global attention ... **in every other layer**. The sliding window size of local attention layers is set to **4096** tokens, while the span of the global attention layers is set to **8192** tokens." `code`：`gemma2` 默认 `sliding_window=4096`，`layer_types = sliding_attention if (i+1)%2 else full_attention` | [arXiv:2408.00118 §2](https://arxiv.org/html/2408.00118v2)、[transformers gemma2 config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/configuration_gemma2.py) |
| **Gemma-3** (4B/12B/27B) | ✅ 是（混合） | **1024** | **5 滑窗 : 1 全注意力**（首层为滑窗） | `report`："we interleave multiple local layers between each global layer, and assign a smaller span of only **1024 tokens** to the local layers. Therefore, **only the global layers attend to long context, and we have 1 global for every 5 local layers.**" / "We alternate between a local sliding window self-attention and global self-attention, with a pattern of **5 local layers for every global layer**, starting with a local layer as the first layer" `code`：`gemma3` 默认 `sliding_window=4096`，`_sliding_window_pattern` 默认 **6** | [arXiv:2503.19786 §5.2](https://arxiv.org/html/2503.19786v1)、[transformers gemma3 config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py) |
| **Qwen2 / Qwen2.5** | ⚠️ 形式上有字段，**实际已关闭** | `sliding_window=131072`（=满上下文） | 无层间混合 | `config`：`"sliding_window": 131072`, `"use_sliding_window": false`, `"max_window_layers": 28`（Qwen2.5-7B / 7B-Instruct 均同）。窗口等于 `max_position_embeddings` ⇒ **滑窗永不生效，等价全注意力** | [Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json) |
| **Qwen3** (8B/32B/30B-A3B) | ❌ 否 | `null` | — | `config`：`"sliding_window": null`, `"use_sliding_window": false`。技术报告原文**完全未提及**滑窗，只列 "Grouped Query Attention (GQA) ... QK-Norm"；对本笔记检索 "sliding window" **命中 0 次** | [Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)、[arXiv:2505.09388 §2](https://arxiv.org/html/2505.09388v1) |
| **Qwen3-Next-80B-A3B** | ❌ 否（改用线性注意力混合） | — | **3 Gated DeltaNet : 1 Gated Attention** | `config`：`"full_attention_interval": 4`, `"use_sliding_window": false`, `linear_num_key_heads: 16`, `linear_num_value_heads: 32` | [config.json](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json) |
| **GPT-OSS** (20B/120B) | ✅ 是（混合，**1:1**） | **128** | 奇偶交替：`sliding, full, sliding, full, ...` | `config`：`"sliding_window": 128`, `layer_types` 为 36 项（120B）/ 24 项（20B）严格交替 `sliding_attention`/`full_attention`。`report`（OpenAI 官方模型卡论文）："**attention blocks alternate between banded window and fully dense patterns, where the bandwidth is 128 tokens.** ... We apply rotary position embeddings and **extend the context length of dense layers to 131,072 tokens using YaRN**. Each attention head has a **learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks**, which enables the attention mechanism to pay no attention to any tokens." | [gpt-oss-120b config](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json)、[arXiv:2508.10925 §2.2](https://arxiv.org/html/2508.10925v1) |
| **Phi-3-small** | ❌ 否（用 block-sparse） | — | `blocksparse_block_size: 64`, `blocksparse_num_local_blocks: 16`, `blocksparse_vert_stride: 8`, `blocksparse_homo_head_pattern: false` | `config`：无任何 `sliding_window` 字段；改用 **block-sparse attention**（局部块 + 跨块 stride） | [config.json](https://huggingface.co/microsoft/Phi-3-small-8k-instruct/raw/main/config.json) |
| **Phi-3.5-mini** | ⚠️ 字段在但无意义 | 262144 | — | `config`：`"sliding_window": 262144`，远超 `max_position_embeddings: 131072` ⇒ **永不触发，实为全注意力** | [config.json](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json) |
| **Phi-4** | ❌ 否 | `null` | — | `config`：`"sliding_window": null`, 40 层, ctx 16384 | [config.json](https://huggingface.co/microsoft/phi-4/raw/main/config.json) |
| **Cohere Command A / Command A+ (2026)** | ✅ 是（混合） | **4096** | **3 滑窗 : 1 全注意力** | `config`：`"sliding_window": 4096`，`layer_types` 32 项 = 3×`sliding_attention` + 1×`full_attention` 循环；`model_type: cohere2_moe`，ctx 200000 | [command-a-plus-05-2026-bf16 config](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/raw/main/config.json) |
| **Cohere Command R / R+ (2024)** | ❌ 否（**纠正**） | — | — | 官方仓库 Gated，但社区镜像 config 的**完整 key 列表中不存在** `sliding_window` / `sliding_window_pattern` / `layer_switch` 任何一项（`model_type: cohere`, 64/40 层, ctx 131072）。**结论：Command R/R+ 不用 SWA**，SWA 是后续 Command A 世代才引入的 | [unsloth 镜像 config（R+）](https://huggingface.co/unsloth/c4ai-command-r-plus-08-2024/raw/main/config.json) |
| **Llama 4 Scout** | ⚠️ **不是 SWA**，是 chunked local attention | `attention_chunk_size: 8192` | "Local chunked attention + full"，iRoPE（每 4 层一个 NoPE 层） | vLLM 官方文档明确把 Llama4 单列为第三类："**3. Local chunked attention + full: Llama4**"（与 "1. Sliding window + full: gpt-oss, Gemma 2/3, Ministral, cohere" 并列）；`transformers` 官方 `llama4` config 默认 `attention_chunk_size=8192`, `no_rope_layer_interval=4` | [vLLM Hybrid KV Cache Manager](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md)、[transformers llama4 config](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/configuration_llama4.py) |
| **Granite 4.0-H-Small** (IBM) | ❌ 否（Mamba 混合，无 SWA） | — | **9 Mamba : 1 Attention** | `config`：`model_type: granitemoehybrid`，`layer_types` 40 项中仅 4 项为 `attention`（索引 5/15/25/35） | [config.json](https://huggingface.co/ibm-granite/granite-4.0-h-small/raw/main/config.json) |
| **Falcon-H1-34B** | ❌ 否 | — | Mamba + Attention 混合，无滑窗字段 | `config`：`model_type: falcon_h1`, 72 层, `mamba_d_state: 256`, `attn_layer_indices: null`，**无 `sliding_window`** | [config.json](https://huggingface.co/tiiuae/Falcon-H1-34B-Base/raw/main/config.json) |
| **GLM-4.5** | ❌ 否（全注意力） | — | — | `config`：`model_type: glm4_moe`, 92 层, ctx 131072，**无 `sliding_window`** | [config.json](https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json) |
| **DeepSeek-V3.2-Exp** | ❌ 否（DSA 稀疏，非滑窗） | — | — | `config`：`model_type: deepseek_v32`, 61 层, ctx 163840，**无 `sliding_window`** | [config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json) |
| **Kimi K2-Instruct** | ❌ 否 | — | — | `config`：`model_type: kimi_k2`, 61 层, ctx 131072，**无 `sliding_window`** | [config.json](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json) |
| **MiniMax-M1-80k** | ❌ 否 | `sliding_window: null` | Lightning Attention 混合 | `config`：`"sliding_window": null`, 80 层, ctx 10240000 | [config.json](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/config.json) |
| **OLMo-2** (7B/32B) | ❌ 否 | — | — | `config`：`model_type: olmo2`, **无 `sliding_window`** | [OLMo-2-1124-7B config](https://huggingface.co/allenai/OLMo-2-1124-7B/raw/main/config.json) |
| **SmolLM2-1.7B** | ❌ 否 | — | — | `config`：`model_type: llama`, **无 `sliding_window`** | [config.json](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json) |
| **AFM-4.5B** (Arcee) | ❌ 否 | — | — | `config`：`model_type: arcee`, 36 层, ctx 65536，**无 `sliding_window`** | [config.json](https://huggingface.co/arcee-ai/AFM-4.5B/raw/main/config.json) |
| **Jamba-1.5-Mini** (AI21) | **【未找到】** | — | — | HF config 请求失败（`ERR`），未能核实其注意力层是否为全注意力 | [AI21-Jamba-1.5-Mini](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini) |
| **Zamba2-7B** | **【未找到】**（疑似否） | — | 81 层，config 中无 `sliding_window`/`layer_types` | `config` 仅 `model_type: zamba2`, 81 层, ctx 4096 | [config.json](https://huggingface.co/Zyphra/Zamba2-7B-Instruct/raw/main/config.json) |
| **Step-3**、**Ling-lite**、**Command R/R+ 官方 config**、**Gemma 官方 config**、**Llama-4 官方 config** | **【未找到】（GatedRepo 401）** | — | — | 本环境无 HF token，`google/*`、`meta-llama/*`、`CohereLabs/c4ai-*` 均返回 `x-error-code: GatedRepo`。相应结论改由官方报告 / transformers 代码 / vLLM 文档替代，已在表中标注来源类型 | — |

### 3.2 从表中读出的三条趋势

1. **"纯 SWA"在主流开源模型中几乎绝迹。** 唯一属于纯 SWA 的是 **Mistral-7B-v0.1**（2023-09），而 Mistral 自家从 **v0.3 起就把 `sliding_window` 设为 `null`**，此后 Nemo / Small / Large / Codestral / Mixtral **全部回到全注意力**。这是一个被广泛忽略但极其重要的产业信号：**SWA 的首次大规模产品化在一年内被其提出者收回。**（证据：[v0.1 config](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json) vs [v0.3 config](https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json)）

2. **采用者一律走"层间混合"。** Gemma-2（1:1）→ Gemma-3（5:1）→ GPT-OSS（1:1，窗口 128）→ Command A（3:1）。**没有任何一个 2024 年后的主流开源模型采用"全层纯 SWA"。** 更关键的是**方向性**：Gemma-3 把全注意力层占比从 Gemma-2 的 1/2 降到 1/6，即**在同样硬件下把"全局层"变得更稀缺、更便宜（窗口 4096→1024）**，把省下的预算还给上下文长度（8K→128K）。这说明产业界对 SWA 的定位已经收敛为：**"用极小的窗口 + 少量全局层买长上下文"，而不是"用大窗口替换全注意力"。**

3. **"SWA 缺席"的模型走了别的稀疏路线。** Qwen3 / GLM-4.5 / DeepSeek-V3.2 / Kimi K2 / MiniMax-M1 / OLMo-2 / SmolLM2 全部**不用 SWA**；其中 Qwen3-Next、Granite-4.0、Falcon-H1、MiniMax 改用**线性注意力 / SSM(Mamba) 混合**，DeepSeek-V3.2 用 DSA，Llama-4 用 chunked local attention。**说明 SWA 不是长上下文的唯一解，且在与线性注意力/SSM 的竞争中并未占据上风。**

### 3.3 层间混合的实证理由（产业侧证据）

**(a) Gemma-2 报告：把混合归因于 Longformer，并给出比例消融**

> "We also leverage several known modifications of Transformers, namely the **interleaving of global and local attention layers from Beltagy et al. (2020a)**, and the Grouped-Query Attention (GQA) mechanism."（[arXiv:2408.00118 Abstract/§1](https://arxiv.org/html/2408.00118v2)）
> "We alternate between a local sliding window attention and global attention **in every other layer**."（§2）

⚠️ **诚实标注**：Gemma-2 报告**没有**给出"纯 SWA vs 混合"的对照消融。"为什么必须混合"的**直接实证在 Gemma-2/3 报告中缺席**（其消融只测了混合比例 1:1 vs 1:3 vs 7:1 与滑窗大小）。**"纯 SWA 会掉点"的量化证据必须从 SWAA 等文献取**（见 §4.2）。**【事实】**

**(b) Gemma-3 报告：混合比例与窗口大小的双消融**

> "**Local:Global ratio.** In Fig. 3, we compare different ratios of local to global attention layers. **1:1 is used in Gemma 2 models, and 5:1 is used in Gemma 3. We observe minimal impact on perplexity when changing this ratio.**"
> "**Sliding window size.** In Fig. 4, we compare different sliding window sizes ... **The sliding window can be reduced significantly without impacting perplexity.**"
> "**Impact on KV cache memory.** ... we interleave multiple local layers between each global layer, and assign a smaller span of only 1024 tokens to the local layers. Therefore, **only the global layers attend to long context**."（[arXiv:2503.19786 §5.2](https://arxiv.org/html/2503.19786v1)）

Gemma-3 明确表态："**The impact is minimal, even with 7-to-1 local to global.**"（Figure 3 caption）—— 即在 **perplexity** 这个指标上，把全局层压到 1/8 也几乎无损。

**(c) Gemma-2 的推理期窗口缩放实证（可直接引用）**

| sliding window | 4096 | 2048 | 1024 |
|---|---|---|---|
| perplexity (val. set) | 1.63 | 1.63 | **1.64** |

> "we show that we can change the sliding window size of the local attention layers of the models **during inference with moderate impact on perplexity**. Adjusting the size of the sliding window can thus be **a leverage for slight inference speed gain**."（[arXiv:2408.00118 Table 10](https://arxiv.org/html/2408.00118v2)）

**(d) GPT-OSS：混合 + attention sink bias 的组合拳**

GPT-OSS 是"产业界如何同时解决 SWA 的两个缺陷"的教科书样本：**层间 1:1 混合**（解决"看不到远处"） + **softmax 分母可学习 bias**（等价于永久保留一个 attention sink，解决"训练-推理分布错配"）。原文把后者明确对标 attention sink：
> "Each attention head has a **learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks**, which enables the attention mechanism to pay no attention to any tokens."（[arXiv:2508.10925 §2.2](https://arxiv.org/html/2508.10925v1)）

---

## 4. SWA 的固有缺陷

### 4.1 缺陷一：滑窗丢失长程依赖（结构性缺陷）

**机制层面（Mistral 自己承认的边界）**：Mistral 的理论跨度 131K 是 **"theoretical"**，且依赖两个隐含前提：① 每一层的信息都能恰好向前传播 W 个 token；② 模型确实学会了这种逐层接力。原文用 "**theoretical** attention span" 与 "**can** access tokens at a distance of up to W×k" 限定（[arXiv:2310.06825](https://arxiv.org/html/2310.06825v1)）。**这只是可达性的上界，不是能力保证。**

**能力层面的硬证据（StreamingLLM）**：纯滑窗在 PG19 上 PPL **5158.07**（vs 保留 4 sink 的 5.40），且"**even just evicting the KV of the first token**"就足以引发崩溃（[arXiv:2309.17453 Table 1 & Figure 3](https://arxiv.org/html/2309.17453v4)）。

**必须注意的区分**：StreamingLLM 的崩溃**混合了两个原因** —— (i) 训练-推理错配（模型预训练时用全注意力，推理时突然切滑窗，attention sink 分布被打断）；(ii) 结构性丢失远距离信息。**这两者必须分开**，否则会把"可修复的分布错配"误判为"SWA 结构上不行"。SWAA 的价值恰恰在于它把这两者显式解耦（§4.2）。

### 4.2 缺陷二的量化证据：SWAA —— 目前最系统的正反证据（arXiv:2512.10411）

- **链接**：[arXiv:2512.10411](https://arxiv.org/abs/2512.10411)（v1 2025-12-11，v5 2026-03-26）｜ [arXiv HTML v5](https://arxiv.org/html/2512.10411v5) ｜ [代码](https://github.com/yuyijiong/sliding-window-attention-adaptation)
- **作者**：Yijiong Yu, Jiale Liu, Qingyun Wu, Huazheng Wang, Ji Pei
- **一句话**：把"FA 预训练模型改造成 SWA"的适配工具包，系统解耦 SWA 失效的**两个根因**，并给出可用的配方。**这是本笔记中唯一直接、量化地回答"SWA 是否损害长上下文"的一手来源。**

**失效的两大根因（原文 Abstract）：**
> "it suffers from **catastrophic long context performance collapse**, which stems from two fundamental factors: the **training inference mismatch** when naively applying SWA to models pretrained with Full Attention (FA), and the **inherent structural inability to access distant information** when applying SWA to every module at all times."

**四项对策：**
1. **Full Attention (FA) Decode** —— prefill 用 SWA，decode 用 FA（"temporarily reopens direct pathways to distant tokens"）
2. **Interleaving FA and SWA layers** —— 层间混合
3. **preserving "sink" tokens** —— 保留前 k 个 token 的全注意力（对应 StreamingLLM）
4. **lightweight fine tuning** —— 对齐权重

**核心结论（原文 §4.2.1）：**
> "**the catastrophic degradation caused by SWA cannot be resolved by addressing the dual bottlenecks ... in isolation. Applying any single method, such as Keep First, FA Decode, Interleaving Layers, or naive SFT, yields negligible accuracy improvements over pure SWA.** Conversely, when we pair methods targeting distinct challenges ... we observe significant accuracy recoveries."

**量化（Qwen3-4B，原文 Table 1，节选；格式：窗口 | FA 层索引 | FT 步数 | FA Decode | LongMemEval | LongBenchV2 | RULER-var | RULER | 其他两列）：**

| 行 | 配置 | FA Decode | LongMemEval | LongBench V2 | RULER(var) |
|---|---|---|---|---|---|
| **24** | **Full Attention 基线** | / | **88.2** | 90.6 | 74.6 |
| 25 / 26 | 2k 窗口，无 FA 层，无 FT，**无/有 FA Decode** | ✗ | **0.0 / 0.0** | 0.0 / 0.0 | 18.8 / 15.6 |
| 3 / 4 | 2k 窗口，无 FA 层，无 FT，无 Keep First | ✗ | ~0.0 | 0.0 | — |
| 7 | 2k，无 FA 层，**Keep First=10**，**+FA Decode** | ✓ | **1.0** | 0.0 | 38.2 |
| 9 | 2k，无 FA 层，**Keep First=1000**，+FA Decode | ✓ | **3.2** | 0.2 | 50.0 |
| 11 | 2k，**偶数层 FA**，Keep First=0，+FA Decode | ✓ | 1.0 | 2.0 | 32.2 |
| **13** | 2k，**奇数层 FA**，Keep First=0，+FA Decode | ✓ | **22.8** | **32.8** | **59.2** |
| 17 | 2k，**奇数层 FA**，Keep First=10，+FA Decode | ✓ | **59.2** | **75.8** | 65.0 |
| 22 | 4k，奇数层 FA，**FT=100**，+FA Decode | ✓ | **65.4** | **83.6** | 73.0 |
| **31** | 2k，**奇数层 FA**，Keep First=0，**FT 开**，+FA Decode | ✓ | **74.0** | **85.8** | 73.2 |
| **34** | 2k，奇数层 FA，**FT=100**，+FA Decode | ✓ | **72.8** | **89.2** | 73.2 |

**从这张表能抽出四条硬结论：**

1. **纯 SWA 让检索任务归零。** 行 25/26：无 FA 层、无微调时，**LongMemEval 与 LongBench V2 均为 0.0**（完整版为 0.0/0.0 两列）。**"SWA 在检索类任务上显著退化"在本表中不是"退化"，而是"完全归零"。**
2. **任何单一措施都几乎无效。** 只加 Keep First（行 7/9）LongMemEval 仅 1.0/3.2；只加层间混合（行 11/13）LongMemEval 仅 1.0/22.8。原文判定为 "negligible"。
3. **层间混合是必要条件，但层选择极其敏感。** 原文：
   > "**configuring odd numbered layers with FA vastly outperforms even numbered ones for Qwen3-4B (row 13 vs. row 11). This asymmetry suggests that specific layers in pretrained LLMs inherently function as global routing hubs, and misaligning SWA with these hubs disrupts the semantic flow.**"
   —— 行 13 vs 行 11：LongMemEval **22.8 vs 1.0**，LongBench V2 **32.8 vs 2.0**。**同样的"每隔一层混合"，奇偶之差可达 20 倍以上。这对"层间混合比例自动搜索"方向是最强的动机。**
4. **RULER（needle-in-a-haystack）对层间混合的要求最苛刻。** 原文：
   > "While for **Ruler**, as it involves only the needle in a haystack task, it exhibits a more extreme pattern: **the accuracy can hardly be improved unless we enable Interleaving Layers, coupled with at least one another method** (row 13 and 28). This indicates that **the retrieval behavior may be concentrated in certain specific layers, therefore these layers must retain full attention.**"
   —— 即：**检索类能力高度集中在少数层，这些层必须保持全注意力。** 这是"为什么不能全层滑窗"最直接的功能性论证。

5. **训练-推理错配可用轻量微调修复，结构性缺陷不能。** 原文：
   > "**Fine Tuning: Aligning Model Weights with Sparse Attention Patterns.** ... **SFT acts as the ultimate unifier**, parametrically aligning the model's internal representations with the modified sparse attention structures. Combining structural methods (FA Decode or Interleaving Layers) with lightweight fine tuning **virtually eliminates the performance gap, restoring accuracy to perfectly match the full attention baseline** (rows 27, 28, and 31)."
   —— 行 31 vs 行 24：LongMemEval 74.0 vs 88.2（仍差 14 分）、LongBench V2 **85.8 vs 90.6**（差 4.8 分）、RULER-var 73.2 vs 74.6。**注意"virtually eliminates"是作者的表述，但表中 RULER 一列（88.2→73.2）并未完全恢复**，报告引用时应以表中数字为准，谨慎转述作者的定性措辞。

6. **效率收益量级（原文 §4.2.2）：**
   > "our performance efficiency trade off analysis identifies optimal SWAA configurations for diverse scenarios, achieving **30% to 100% speedups** for long context inference with acceptable quality retention."

> ⚠️ **诚实标注**：SWAA 是**改造（adaptation）**场景（FA 预训练权重 + 后处理），不是**预训练**场景。因此其"纯 SWA 归零"的结论**不能直接外推到"从头用 SWA 预训练"的模型**（后者无训练-推理错配问题）。**Gemma-3 的消融说明：若在预训练阶段就设计好混合结构，perplexity 对比例与窗口都不敏感。** 报告写作时须把这两个场景严格分开。

### 4.3 缺陷三：工程与系统复杂度

- **KV cache 分页复杂度**（见 §5.3）：混合层使不同层的 KV cache 长度不同，破坏"所有层共享同一 page"的假设，需要 KV cache group 机制。
- **prefix caching 语义分化**：全注意力层的 cache hit 要求**全部** token 都在缓存中；滑窗层只要求**最后 W 个**token 在缓存中（vLLM 官方文档明确列出）。
- **prefill 的额外开销**：PHD 论文自身指出，把 KV cache 用于 hidden decoding tokens 会**引入 K 倍的 prefill 开销**，因为"the KV cache of hidden decoding tokens exhibits **sequential dependencies** in PHD-SWA, which leads to a linear increase in the pre-filling time"（[arXiv:2504.14992 §1](https://arxiv.org/html/2504.14992v2)）。这正是 PHD 需要再提出 CSWA 的原因 —— **滑窗本身会把 prefill 从可并行变成串行依赖**。

---

## 5. SWA 的实现与系统支持

### 5.1 FlashAttention：`window_size_left` / `window_size_right`

- **链接**：[Dao-AILab/flash-attention `flash_attn_interface.py`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
- **【事实】** 官方接口的 `flash_attn_func` / `flash_attn_varlen_func` 等签名均包含窗口参数，默认禁用：

  ```python
  def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                      window_size_left=-1, window_size_right=-1, ...)
  ```
  ```python
  def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                             dropout_p=0.0, softmax_scale=None, causal=False,
                             window_size_left=-1, window_size_right=-1, ...)
  ```

  **约定：`-1` 表示无限制（即全注意力）。** 因果解码场景下，滑窗等价于 `window_size_left = W - 1`, `window_size_right = 0`（右侧因 causal 已为 0）。

- **【事实】历史渊源**：Mistral 论文致谢明确写 "A huge thanks to **Tri Dao** and **Daniel Haziza** for helping include **Mistral related changes to FlashAttention and xFormers** on a tight schedule."（[arXiv:2310.06825](https://arxiv.org/html/2310.06825v1)）—— **SWA 的 kernel 支持是 Mistral 与 FlashAttention/xFormers 作者共建的**，不是事后适配。

### 5.2 vLLM

| 能力 | 配置项 / 代码位置 | 证据 |
|---|---|---|
| 全局开关 | `--disable-sliding-window`（CLI），对应 `ModelConfig.disable_sliding_window` | [vllm/engine/arg_utils.py `--disable-sliding-window`](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py) |
| 从 config 读取窗口 | `ModelConfig.get_sliding_window()` → `getattr(hf_text_config, "sliding_window", None)` | [vllm/config/model.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py) |
| `sliding_window == 0` 自动禁用 | `if self.get_sliding_window() == 0: self.disable_sliding_window = True; ... sliding_window = None` | 同上 |
| 窗口 < 模型最大长度则据此收窄 max_len | `if disable_sliding_window and sliding_window is not None and sliding_window < derived_max_model_len: derived_max_model_len = sliding_window` | 同上 |
| **逐层滑窗**（混合层必需） | 模型实现里按 `layer_types` 逐层传参：`is_sliding = config.layer_types[layer_idx] == "sliding_attention"; sliding_window = config.sliding_window if is_sliding else None`，再以 `per_layer_sliding_window=sliding_window` 传入 `Attention` | [vllm/models/gemma2.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma2.py) |

**混合层的 KV cache 管理复杂度（重点，第 3 章可直接引用）**

vLLM 官方文档 [Hybrid KV Cache Manager](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md) 开篇即把混合模型分成三类，并**把 SWA 混合与 Mamba 混合、chunked local 混合并列**：

> "Many recent 'hybrid' LLMs combine multiple attention types within one model. For example:
> 1. **Sliding window attention (sw) + full attention (full): gpt-oss, Gemma 2/3, Ministral, cohere, etc.**
> 2. Mamba + full: Bamba, Jamba, Minimax, etc.
> 3. **Local chunked attention + full: Llama4**"

它必须同时满足两条要求：

> "1. Allocate different slots to different layer type, for example:
>    - Full attention layers: reserve slots for **all** tokens.
>    - Sliding window layers: reserve slots only for the most recent **`sliding_window_size`** tokens.
> 2. **Support layer-specific prefix-cache rules**, for example:
>    - Full attention: a cache hit prefix requires **all** tokens remain in the KV cache.
>    - Sliding window: a cache hit prefix only requires the **last `sliding_window_size` tokens** remain in the KV cache."

**核心难题：混合层让"每层每 token 的块数"不同，而显存池只有一个 page size。** 文档举例：Gemma-3-27B 有 **52 个滑窗层 + 10 个全注意力层**，若按"同类型同块数"分组会产生 26+5 个碎片化小组，效率极低；最终解法是**取各注意力类型层数的最小值作为组大小**（`min(52,10)=10`），分成 1 个 full 组 + 5 个 sw 组。文档还给出更细的算例：

> "Assume `block_size` 16, sliding window size 32, request length 112, then for the above example model, we need to allocate **11 blocks**（0-6 for full, 7-8 for sw group 1, 9-10 for sw group 2）."

**可直接引用的结论**：**"层间混合 SWA/FA"不是免费的 —— 它把一个统一的 KV cache 池变成需要按注意力类型分组、且 prefix cache 命中语义按层分化的问题。这是纯 SWA 或纯 FA 都不存在的系统性成本。**

### 5.3 SGLang

- **【事实】** SGLang 通过模型 config 自动解析窗口，并在运行期判断是否需要 SWA 专用 KV 池。
  - `python/sglang/srt/configs/model_config.py`：
    ```python
    def _get_sliding_window_size(self) -> Optional[int]:
        for key in ("sliding_window_size", "sliding_window", "window_size"):
            value = getattr(self.hf_text_config, key, None)
            if value is not None:
                return value
        return None
    ```
    并暴露 `self.sliding_window_size`；另有 `self.attention_chunk_size`（对应 Llama-4 的 chunked local attention）。
  - `python/sglang/srt/model_executor/model_runner.py`：`self.sliding_window_size = resolve_sliding_window_size(...)`，随后传入 attention backend。
  - `python/sglang/srt/layers/attention/flashattention_backend.py`：
    ```python
    # Window size (typically used by Gemma)
    window_size: tuple = (-1, -1)
    # Page table for Sliding Window Attention
    ...
    # For sliding window attention topk>1 spec decoding
    ...
    self.use_sliding_window_kv_pool = (...)
    # For each layer, the sliding_window_size can be different.
    # This is only used for preparing SWA metadata.
    # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
    self.sliding_window_size = model_runner.sliding_window_size
    ```
  - 后端还有 "Always use window_size=(-1, -1) because scheduler_metadata is only..." 之类的注释，说明**为了统一调度元数据，SGLang 在某些通路上必须退化成全窗口**。

  **解读（可写入报告）**：SGLang 提供的是**逐层滑窗**能力（"For each layer, the sliding_window_size can be different"）与**SWA 专用 KV pool**；但也因此出现了"某些 metadata 通路必须用全窗口"的妥协。**注意：SGLang 的 CLI 中没有一个简单的全局 `--sliding-window` 开关（本次核查未在 `server_args.py` 中找到任何 sliding/window/SWA 相关参数），窗口完全由模型 config 决定** —— 与 vLLM 提供 `--disable-sliding-window` 显式 CLI 的做法不同。**【事实】**

### 5.4 PHD 论文自身对 kernel 的描述

PHD 论文专设 §2.3 **Kernel Design**，称其把"未被注意的注意力位置聚成连续块"以便优化：
> "we group the un-attended attention positions in a continuous block, which is efficient for optimization."（[arXiv:2504.14992v2](https://arxiv.org/html/2504.14992v2)）
—— **说明 PHD 的滑窗是以自定义 mask + kernel 方式实现的，而非直接调用 FlashAttention 的 `window_size_left` 参数**（因为滑窗只作用于 token 组内的 hidden decoding tokens，不是标准因果滑窗）。

---

## 6. 与 PHD-SWA 的关系（核心）

### 6.1 PHD-SWA 到底滑什么？（直接回答）

**先说结论：PHD-SWA 不是"全层滑窗"，也不是对全部 token 滑窗。它只对 hidden decoding tokens 施加滑窗，original tokens 仍保持全局长程可见。**

**原文（[arXiv:2504.14992v2 §2.4](https://arxiv.org/html/2504.14992v2)）：**

> "To capture these benefits while maintaining efficiency, we introduce **PHD-SWA, which implements sliding window attention restricted to W preceding hidden decoding tokens.**"
> "As illustrated in Figure 4, the attention pattern **combines global access to original tokens with local access to the W most recent hidden decoding tokens.** This modified attention mechanism achieves notable performance improvements while only requiring **𝒪(1) additional KV cache memory**."

**三个必须消歧的点：**

1. **"层"的维度：全层，但每层的滑窗只作用于一组 token。** PHD 沿用与 vanilla transformer **相同的模型架构**，只改输入序列与注意力 mask（"PHD keeps the same model architecture and only differs in the input sequence and the design of the attention matrix"）。因此**不存在"哪些层用滑窗"的问题 —— 所有层都用同一套 mask**。真正的维度是 **token 类型**：`original tokens`（每个位置的**第 1 份拷贝**）vs `hidden decoding tokens`（第 2…K 份拷贝）。
   - original tokens：**全局可被所有 token 关注**（"can be globally attended to by all tokens"），其 KV 会长期保留 ⇒ **长程依赖由它们承担**。
   - hidden decoding tokens：其 KV **用完即弃**；PHD-SWA 额外保留 **W 个最近的 hidden decoding token** 的 KV，且**只有** hidden decoding tokens 能访问它们。

2. **窗口大小 W 极小（4~32），与 Mistral/Gemma 的 4096/1024 完全不同量级。** 因为长程信息已由 original tokens 的全注意力覆盖，PHD-SWA 的窗口只需捕捉 **同一位置在不同拷贝之间的局部依赖**。
   - Figure 4 示例："we set the repeating times K to 3 ... and set the **window size W to 4** and chunk size C to 4."
   - 主实验：`PHD-CSWA-2-16-32` / `PHD-CSWA-3-16-32`，**W=16, C=32, K∈{2,3}**。
   - 消融：K=2 固定，**W ∈ {1,2,4,16}**。
   - 解码缩放：**W=16 固定，K ∈ {2,3,5}**（配置名 `PHD-SWA-K-16-∞`）。

3. **窗口只覆盖 hidden decoding tokens，且限定在"最近 W 个"。** 即对 hidden decoding token `x_m^i`，其可见集合 = {所有 original tokens `x_n^1`, n<m} ∪ {最近的 W 个 hidden decoding tokens}。

### 6.2 PHD 的注意力 mask（原文 Eq.2）

$$M_{mn}^{ij} = \begin{cases} 1, & \text{if } i=1 \text{ and } m<n \\ 1, & \text{if } i \le j \text{ and } m=n \\ 0, & \text{otherwise} \end{cases}$$

（`i` 为 query 的拷贝序号，`j` 为 key 的拷贝序号，`m,n` 为位置。）
解读：① **`i=1` 的 original token 位置对更早的所有位置（`m<n`）可见** —— 这是全局长程通路；② **`i≤j` 且 `m=n`：同一位置内，靠后的拷贝可看靠前的拷贝** —— 这是 hidden decoding 的“位置内”因果；③ 其余为 0。
**PHD-SWA 在这个 mask 上追加**：允许 hidden decoding token 也访问±W 范围内的、其它位置的 hidden decoding token 的 KV（原文未给出对应公式，仅给出 Figure 4 与文字描述 —— **【事实】该公式未在论文中以数学形式单独给出**）。

### 6.3 PHD-SWA / PHD-CSWA 的关键数字

**架构与训练设置（§3.2 主实验）：**
- 模型：OLMo2-1.2B，16 层 dense，hidden 2048，FFN 16384，GQA 32 Q heads / 8 KV heads，head_dim 64，**训练 500B tokens**
- 消融基座：550M dense，16 层，hidden 1536，FFN 8192，GQA 16/4，**训练 300B tokens**

**主结果（[Table 1](https://arxiv.org/html/2504.14992v2)，1.2B 模型）：**

| 模型 | loss ↓ | MMLU-V | Hella. | ARC-C | ARC-E | PIQA | Wino. | Comm. | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla 1.2B | 2.601 | 35.9 | 62.3 | 36.1 | 70.3 | 74.6 | 60.5 | 43.4 | 54.7 |
| PHD-CSWA-2-16-32 | 2.576 | 35.8 | 63.7 | 38.8 | 70.2 | 75.2 | 62.5 | 47.4 | **56.2** |
| PHD-CSWA-3-16-32 | **2.567** | **36.7** | **64.5** | **40.2** | **73.5** | 75.2 | 62.1 | 45.0 | **56.7** |

> 原文：K=2 "introduces an average of **1.5%** accuracy improvement ... and **0.025 decrease in training loss**"，K=3 "an average of **2.0%** accuracy improvement and **0.034 decrease in training loss**"。

**窗口大小消融（原文 Figure 6/7 caption + 结论）：**
> "**Large window size leads to better performance, but the improvement converges very fast.** ... While expanding the window from **W=1 to W=4 yields significant reductions in both training and validation loss, further increasing to W=16 provides only marginal improvements.** This suggests that maintaining a **small window of local KV cache** for hidden decoding tokens achieves nearly optimal performance while remaining computationally efficient and hardware-friendly."

—— **这是一个非常有价值的"饱和点"结论：W=4 已接近饱和。** 与 Gemma-3"滑窗 1024 足够"、Gemma-2"4096→1024 PPL 仅 +0.01"完全同构：**SWA 的窗口收益快速饱和**。

**chunk 消融：**
> "**Introducing chunks for reducing pre-filling overhead leads to little performance degradation.** ... We also observe a consistent trend where **larger chunk sizes yield progressively better results.** Based on this analysis, we select **C=32** as the optimal balance."
  消融设置：W=16 固定，C ∈ {16, 32, no chunk}。

**解码 token 缩放（[Table 2](https://arxiv.org/html/2504.14992v2)，550M，W=16 固定）：**

| 模型 | loss ↓ | MMLU | Hella. | ARC-C | ARC-E | PIQA | Wino. | Comm. | **Avg.** |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla 550M | 2.782 | 31.4 | 51.5 | 30.1 | 62.5 | 71.9 | 56.8 | 40.6 | 49.3 |
| PHD-SWA-2-16-∞ | 2.753 | 31.6 | 52.3 | 31.5 | 64.2 | 71.5 | 56.6 | 42.7 | 50.1 |
| PHD-SWA-3-16-∞ | 2.739 | 32.1 | 53.1 | 31.5 | 64.6 | 71.5 | 56.9 | 41.7 | 50.2 |
| **PHD-SWA-5-16-∞** | **2.725** | **32.7** | **54.5** | **34.7** | **65.0** | **72.3** | 56.2 | 42.6 | **51.1** |

> 原文："By setting the scaling factor to 5, we achieve **near 0.06 loss decrease** with notable downstream performance improvement. Quantitative results in Table 2 reveal an **average accuracy improvement of 1.8%** across all benchmarks when scaling to K=5."

**速度（§3.5，550M，单 A100，batch=1）：**
> "PHD-SWA introduce **superlinear pre-filling overhead** when the sequence length increases, while **PHD-CSWA only incurs a small increase** compared to the vanilla baseline."
> "setting **K to 256 only leads to no more than 20% increase in decoding latency**."（K ∈ {1,2,4,8,16,32,64,128,256}）

### 6.4 PHD-SWA 对既有 SWA 的借鉴与差异（对照表）

| 维度 | 通用 SWA（Longformer/Mistral/Gemma/GPT-OSS） | **PHD-SWA / PHD-CSWA** |
|---|---|---|
| **滑窗作用对象** | 序列中的**所有 token**（每个 query 只看最近 W 个 key） | **仅 hidden decoding tokens**；original tokens 仍是全局 key |
| **长程依赖由谁承担** | 靠"层间感受野累积 k×W"**间接**实现 | 靠 **original tokens 的全注意力直接实现**（不依赖累积） |
| **窗口量级 W** | 1024 ~ 4096（Gemma-3 1024；Mistral 4096） | **4 ~ 32**（主实验 16） |
| **是否层间混合** | **是**（Gemma-2 1:1、Gemma-3 5:1、GPT-OSS 1:1、Command A 3:1）—— 必须保留部分全注意力层 | **不需要**：每层都是"全局 original + 局部 hidden decoding"的组合，**全局通路在层内、每层都有，天然不缺** |
| **实现层级** | 层类型（`sliding_attention` vs `full_attention`） | **注意力 mask 层级**（所有层同一 mask，只区分 token 类型） |
| **KV cache** | 从 O(L) 降到 O(W) | original tokens 保持 O(t)（与 vanilla 相同）；hidden decoding tokens 额外 **O(1)（即 O(W)）**。原文："**maintains the same KV cache size as the vanilla transformer**" |
| **动机** | 省显存/算力，支撑长上下文 | **不做**长上下文压缩：目的是在**已经**保持 vanilla KV cache 的前提下，**用重复 token 换取预训练长度缩放（length scaling）**；滑窗只是"别把 hidden decoding token 的 KV 全扔掉"的补救 |
| **prefill 代价** | Chunked prefill 可并行（Mistral 用窗口当 chunk size） | **PHD-SWA 引入 K 倍 prefill 串行开销**（"sequential dependencies"）；**PHD-CSWA 把串行限制在最后一个 chunk 内**，故"only the pre-filling time of the last chunk is linearly increased" |
| **与 attention sink 的关系** | StreamingLLM/GPT-OSS 需显式保留 sink token 或加 softmax bias | **PHD 不需要**：original tokens（含首 token）的 KV 始终保留且全局可见，**attention sink 天然存在**。这解释了为什么 PHD-SWA 没有 StreamingLLM 式的崩溃 |

**一句话定位（建议直接用于第 1 章小结）：**
> **PHD-SWA 是把 SWA 从"压缩长上下文 KV cache 的架构手段"借用为"在 hidden decoding token 组内部保留局部上下文的 mask 技巧"。它借了 SWA 的 mask 形式（局部窗口 + O(1) 缓存）与 chunk 化的工程思路（Mistral 的 chunked prefill → PHD-CSWA 的 chunk-wise），但拒绝了 SWA 的核心前提（"用滑窗替代全注意力"）—— 因为 PHD 的长程通路由 original tokens 的全注意力直接提供，滑窗只是在一个已经很便宜的局部区域上做锦上添花。**

**由此导出的两条洞察（【推测】，建议标注）：**
1. **PHD-SWA 的窗口饱和点（W=4~16）比通用 SWA（1024~4096）小两个数量级，正是因为窗口的职责完全不同**：通用 SWA 的窗口要承担"这就是模型能看到的最远距离"的全部压力；PHD-SWA 的窗口只需覆盖"同一位置多次拷贝之间的局部交互"。**因此把 PHD-SWA 的 W 消融结论外推到通用 SWA 是无效的**（反之亦然）。
2. **PHD-CSWA 的 chunk 机制可以被视为对"SWA 天然串行 prefill"这一缺陷的通用修补思路** —— 任何把滑窗用于"解码期新增 token 的 KV"（而非 prompt 的 KV）的方案都会遇到这个串行问题。

---

## 7. 变种方向（基于公开证据的 7 个方向）

> 每个方向给出：核心想法 / 证据 URL / 受益场景 / 风险。

### 方向 1：SWA 与线性注意力 / SSM 混合

- **核心想法**：用 SSM 或线性注意力承担"全局但压缩"的角色，用少量全注意力层或滑窗层承担"精确检索"，形成 3~4 种算子的层间编队。
- **证据 URL**：
  - Qwen3-Next-80B-A3B：[config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json) —— `full_attention_interval: 4`，即 **3 层 Gated DeltaNet : 1 层 Gated Attention**，且 `use_sliding_window: false`（**注意：这条路线完全绕开了 SWA**）
  - Falcon-H1 (TII)：[config](https://huggingface.co/tiiuae/Falcon-H1-34B-Base/raw/main/config.json) —— Mamba-2 + Attention 混合，`mamba_d_state: 256`
  - Granite 4.0-H-Small (IBM)：[config](https://huggingface.co/ibm-granite/granite-4.0-h-small/raw/main/config.json) —— 40 层中 4 层 attention，其余 Mamba（9:1）
  - Jet-Nemotron（arXiv:2508.15884）：用 PostNAS 自动学出全注意力层的位置与线性注意力块的选择，2B 模型相对 Qwen3 等达 **53.6× 生成吞吐、6.1× prefill 加速**，且 MMLU/MMLU-Pro 超过 DeepSeek-V3-Small —— [arXiv:2508.15884](https://arxiv.org/abs/2508.15884)
- **受益场景**：超长上下文 + 高吞吐服务（解码 memory-bound 场景收益最大）。
- **风险**：与 SWA 是**竞争而非互补**关系（Qwen3-Next 明确 `use_sliding_window: false`）；混合 3 种以上算子的 kernel/调度复杂度高；线性/SSM 分支在精确检索上仍弱（SWAA 已证明检索高度集中于少数层）。

### 方向 2：层间混合比例与"全局层位置"的自动搜索

- **核心想法**：不手工设定 1:1 / 5:1 / 3:1，而是搜索"哪些层必须保留全注意力"及其比例。
- **证据 URL**：
  - **最强动机来自 SWAA**：奇数层 FA vs 偶数层 FA 在 Qwen3-4B 上 LongMemEval **22.8 vs 1.0**、LongBench V2 **32.8 vs 2.0**；原文判定 "specific layers in pretrained LLMs inherently function as **global routing hubs**" —— [arXiv:2512.10411 §4.2.1](https://arxiv.org/html/2512.10411v5)
  - Jet-Nemotron 的 PostNAS 第一步就是 "**learning optimal full-attention layer placement and elimination**" —— [arXiv:2508.15884](https://arxiv.org/abs/2508.15884)
  - Gemma-3 的比例消融显示比例本身不敏感（"minimal impact ... even with 7-to-1"）—— [arXiv:2503.19786 §5.2](https://arxiv.org/html/2503.19786v1)。**与 SWAA 的"层选择极敏感"形成张力：对 perplexity 不敏感，但对检索敏感。**
- **受益场景**：把现有 FA 模型低成本改造成混合模型；异构硬件下按 KV 预算调参。
- **风险**：**比例不敏感 ≠ 位置不敏感**，两者极易被混淆（这是本方向最大的认知陷阱，也是最大的机会）；搜索结果可能过拟合到特定基准（SWAA 明确指出 RULER 与 LongMemEval 对层选择的要求不同）；需后训练配合才能兑现（SWAA 行 31 需开 FT）。

### 方向 3：把 FA 预训练模型改造为 SWA（无需重训）

- **核心想法**：SWAA 式四件套 —— FA Decode + 层间混合 + 保留 sink + 轻量微调。
- **证据 URL**：[arXiv:2512.10411](https://arxiv.org/abs/2512.10411) ｜ [代码](https://github.com/yuyijiong/sliding-window-attention-adaptation)。关键可行性证据：行 31（2k 窗口 + 奇数层 FA + FT + FA Decode）在 LongBench V2 达 85.8 vs FA 基线 90.6，RULER-var 73.2 vs 74.6；效率收益 **30%~100% 加速**。
- **受益场景**：存量 FA 模型的推理成本压缩（尤其长上下文服务），无需从头预训练。
- **风险**：**表格中 RULER 一列 88.2 → 73.2 并未完全恢复**，作者 "virtually eliminates the performance gap" 的措辞强于数字；LongMemEval 仍差 14 分；依赖 FT，且 FT 数据与目标域匹配度会影响泛化。

### 方向 4：可学习 / 自适应窗口大小

- **核心想法**：窗口不再是超参，而是按层、按头、按输入动态决定。
- **证据 URL（一手，但为间接支持）**：
  - Gemma-2/3 证明**窗口可大幅缩小几乎无损**，且在**推理期**改窗口只有 "moderate impact"（PPL 1.63→1.64）—— [arXiv:2408.00118 Table 10](https://arxiv.org/html/2408.00118v2)、[arXiv:2503.19786 §5.2](https://arxiv.org/html/2503.19786v1)
  - PHD-SWA 的 W 消融证明存在**极早的饱和点**（W=1→4 显著，4→16 边际）—— [arXiv:2504.14992 §3.3.1](https://arxiv.org/html/2504.14992v2)
  - Longformer 原始设计就主张**逐层不同 w**（"it might be helpful to use different values of w for each layer"）—— [arXiv:2004.05150 §3.1](https://arxiv.org/html/2004.05150v2)
- **受益场景**：按层/按请求动态分配 KV 预算；prefill 与 decode 用不同窗口（SWAA 的 "FA Decode" 就是这种思想的时间维版本）。
- **风险**：**【未找到】** 本次调研**未找到**一篇明确以"可学习窗口大小（learnable window size）"为主贡献、且被广泛验证的一手论文。上述证据只说明"窗口可压缩"，**不等于"窗口可学习"**。报告若写此方向，必须标注为"由现有饱和现象推断的开放方向"，不可声称已有成熟方案。

### 方向 5：SWA 与投机解码 / 前缀缓存的交互

- **核心想法**：滑窗层的 KV 是"滚动的"，投机解码的 draft/verify 与 prefix cache 的命中判定都要按层区分，二者交互会产生新的正确性/性能问题。
- **证据 URL（一手官方文档/代码）**：
  - vLLM Hybrid KV Cache Manager 明确列出**逐层分化的 prefix-cache 规则**："Full attention: a cache hit prefix requires all tokens remain in the KV cache. **Sliding window: a cache hit prefix only requires the last `sliding_window_size` tokens remain in the KV cache.**" —— [官方文档](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md)
  - vLLM 源码中针对混合层的 prefix cache 参数裁剪逻辑 —— [vllm/config/model.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)
  - SGLang 对 SWA 的专门处理，包括注释 "**For sliding window attention topk>1 spec decoding**" 与独立的 `use_sliding_window_kv_pool` —— [flashattention_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashattention_backend.py)
  - vLLM 存在针对 Gemma 系 MTP 投机解码的专门模块（`vllm/v1/spec_decode/gemma4`），说明混合滑窗模型 + 投机解码需要专门的实现路径 —— [vLLM API docs](https://docs.vllm.ai/en/latest/api/vllm/v1/spec_decode/gemma4/)
- **受益场景**：高并发长上下文服务；多轮对话的重前缀复用。
- **风险**：正确性风险高（滑窗层 cache hit 的语义弱于全注意力层，若实现不一致会静默出错）；收益取决于命中率，短对话场景可能负收益。**【事实】vLLM/SGLang 已投入工程，但本次未找到公开的量化收益/失败率数据。**

### 方向 6：Attention sink 的显式化（结构 sink / softmax bias / off-by-one）

- **核心想法**：不再依赖"输入的前 4 个 token 恰好成为 sink"，而是把 sink 变成显式可训练的结构。
- **证据 URL**：
  - StreamingLLM：sink 的发现与"加占位 token 预训练可让模型只需**一个** sink"—— [arXiv:2309.17453](https://arxiv.org/html/2309.17453v4)
  - GPT-OSS 工业级采用：**每个注意力头的 softmax 分母带可学习 bias**，原文明确对标 "off-by-one attention and attention sinks"，效果是"enables the attention mechanism **to pay no attention to any tokens**" —— [arXiv:2508.10925 §2.2](https://arxiv.org/html/2508.10925v1)
  - SWAA 把 "Keep First k tokens" 列为四大对策之一（但结论是它会被 SFT 覆盖："its effect is obviously weaker than, and will be covered by SFT ... rendering the Keep First strategy largely optional"）—— [arXiv:2512.10411 §4.2.1](https://arxiv.org/html/2512.10411v5)
- **受益场景**：流式/无限长推理；把已有模型改造成可滑窗的前提条件。
- **风险**：SWAA 的证据显示**它可能被微调"吃掉"**（如果训练数据足够，模型会自己重塑注意力权重而不再依赖 sink）；GPT-OSS 的 bias 是预训练期设计，后期无法廉价加入。

### 方向 7：SWA 与动态稀疏注意力（prefill 侧的进一步压缩）

- **核心想法**：SWA 解决的是 **decode 侧的 KV cache**；prefill 侧的 O(L²) 需要另一套动态稀疏机制。二者正交互补。
- **证据 URL**：
  - MInference 1.0：识别长上下文注意力矩阵的三种模式（**A-shape、Vertical-Slash、Block-Sparse**），逐头离线选模式 + 运行期动态构造稀疏索引，用定制 GPU kernel 加速 prefill。摘要指出 8B 模型处理 1M token prompt 在单 A100 上需 30 分钟 —— [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)
  - DuoAttention：发现只有少数 **Retrieval Heads** 需要全 KV，其余 **Streaming Heads** 只需常数长度 KV；对 retrieval heads 用全 KV、对 streaming heads 用轻量常数 KV，**长上下文显存最高降低 2.55×（MHA）** —— [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)
  - SWAA 明确把 prefill 与 decode 拆开处理（"FA Decode"），并指出 "**the accuracy can hardly be improved unless we enable Interleaving Layers**" for RULER —— 说明**prefill 压缩与检索能力直接冲突**，是硬约束。
- **受益场景**：超长 prompt（RAG、长文档、长代码库）的 prefill 加速。
- **风险**：**DuoAttention 的核心发现（少数 retrieval heads 承担检索）与 SWA 的"均匀滑窗"在假设上冲突**：若把承担检索的 head 也滑窗化，会直接重演 SWAA 的归零现象。任何 head 级稀疏化都必须先识别 retrieval heads。

### 补充：方向 8（备选）—— 把滑窗从"prompt KV"转向"新增/中间 token 的 KV"

- **核心想法**：**这正是 PHD-SWA 走的路，也是它与其他所有方向都不重叠的位置。** 不复用 SWA 的"压缩长上下文"叙事，而把它当作"为中间计算产物（hidden decoding tokens / latent thoughts / CoT 草稿）保留局部上下文"的 mask 手段。
- **证据 URL**：[arXiv:2504.14992 §2.4 & §3.3.1](https://arxiv.org/html/2504.14992v2) —— W=4 即饱和、O(1) 额外 KV、K=5 时平均 +1.8%。
- **受益场景**：latent reasoning / hidden decoding / 训练期 length scaling；任何"序列被显式拉长但不想付 KV 代价"的场景。
- **风险**：**这类"中间 token"引入串行 prefill 依赖**（PHD 自己承认），必须靠 chunk 化（PHD-CSWA）修补；且该方向目前**只有 PHD 系列一篇一手工作**，外部复现与独立验证**【未找到】**。

---

## 8. 明确的「未找到」清单

以下条目在本次调研中**未能**从一手来源证实，报告中不应作为事实陈述：

1. **【未找到】Gemma-2-2B 是否使用 SWA。** Google 官方仓库 Gated（401），无法直读其 config.json。`transformers` 的 `Gemma2Config` 默认会为**所有**层数派生 `sliding_attention if (i+1)%2 else full_attention`，但这只是**类默认行为**，不能证明 2B checkpoint 的实际取值。（用户假设"2B 仅用全注意力"—— 本次**无法证实也无法证伪**。）
2. **【未找到】Gemma-3-1B 是否使用 SWA。** 同上。已知 Gemma-3-1B 的上下文长度与其他型号不同（32K vs 128K，见 [arXiv:2503.19786 §5.2](https://arxiv.org/html/2503.19786v1)："Gemma 3 models support context length of 128K tokens, **with the exception of the 1B model that has 32K**"），但其注意力层类型未在报告中单独说明。
3. **【未找到】Cohere Command R / Command R+（2024）是否使用 SWA 的官方一手确认。** 官方仓库 Gated；社区镜像 config 的完整 key 列表中**不存在** `sliding_window` / `sliding_window_pattern` / `layer_switch`。**倾向于"不用 SWA"（【推测】，置信度中高）**，但严格意义上属"未能从官方来源证实"。
4. **【未找到】Gemma-2 报告中"纯 SWA vs 层间混合"的对照消融。** Gemma-2 只报告了混合架构与局部消融（窗口大小、deep vs wide 等），**没有**给出"若全层滑窗会掉多少"的数据。"必须混合"的直接量化证据**只能从 SWAA 取**。
5. **【未找到】Jamba-1.5-Mini 的 attention 层配置。** HF config 请求失败。已知它是 Mamba + Transformer 混合（vLLM 文档归类为 "Mamba + full"），但其 attention 层是滑窗还是全注意力**未能核实**。
6. **【未找到】Zamba2-7B 的明确结论。** config 中无 `sliding_window` 也无 `layer_types`，倾向于"不用 SWA"，但其 81 层的混合结构细节未能核实。
7. **【未找到】Step-3、Ling-lite 的 SWA 配置。** `stepfun-ai/Step-3` config 请求失败；`inclusionAI/Ling-lite` 的 config（`model_type: bailing_moe`）中**无** `sliding_window` 字段，但其官方技术报告未取到，无法确认是否在其它字段中表达。
8. **【未找到】以"可学习窗口大小"为核心贡献的一手论文**（见 §7 方向 4）。
9. **【未找到】SGLang 的全局 SWA CLI 开关。** 本次核查在 `python/sglang/srt/server_args.py`（893 行）中**未检索到**任何 `sliding` / `window` / `swa` 相关参数 —— 与 vLLM 提供 `--disable-sliding-window` 的做法不同。此为"核查后确认不存在于该文件"，但不应推断为"SGLang 完全不支持"（其 model_runner 与 attention backend 中的支持已证实）。
10. **【未找到】SWA 与投机解码交互的公开量化收益/失败率数据。** 仅找到实现层证据（vLLM `spec_decode/gemma4`、SGLang 的 "topk>1 spec decoding" 注释），**无公开 benchmark 数字**。
11. **【未找到】2025-2026 年"SWA 与投机解码/前缀缓存交互"方向的专门 arXiv 论文。** 相关能力目前只体现在推理引擎的工程实现中。
12. **【未找到】Qwen2.5 技术报告中对 sliding window 的任何论述。** 对 [arXiv:2412.15115v2](https://arxiv.org/html/2412.15115v2) 检索 "sliding window" **命中 0 次**；报告只提到 "Full Attention"（用于对比其自研的稀疏方法，且该处是讲 Qwen2.5-Turbo 的 TTFT 优化，与本笔记的 SWA 无关）。其 config 中的 `sliding_window: 131072` 只是**遗留字段**，且被 `use_sliding_window: false` 关闭。

---

## 附录 A：本笔记使用的一手来源清单

**论文（arXiv）**
- Longformer — [arXiv:2004.05150](https://arxiv.org/abs/2004.05150) ｜ [HTML](https://arxiv.org/html/2004.05150v2)
- Sparse Transformer — [arXiv:1904.10509](https://arxiv.org/abs/1904.10509)
- Mistral 7B — [arXiv:2310.06825](https://arxiv.org/abs/2310.06825) ｜ [HTML](https://arxiv.org/html/2310.06825v1)
- StreamingLLM — [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) ｜ [HTML](https://arxiv.org/html/2309.17453v4)
- Gemma 2 — [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) ｜ [HTML](https://arxiv.org/html/2408.00118v2)
- Gemma 3 — [arXiv:2503.19786](https://arxiv.org/abs/2503.19786) ｜ [HTML](https://arxiv.org/html/2503.19786v1)
- Qwen2.5 — [arXiv:2412.15115](https://arxiv.org/abs/2412.15115) ｜ [HTML](https://arxiv.org/html/2412.15115v2)
- Qwen3 — [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) ｜ [HTML](https://arxiv.org/html/2505.09388v1)
- GPT-OSS（OpenAI 官方模型卡论文） — [arXiv:2508.10925](https://arxiv.org/abs/2508.10925) ｜ [HTML](https://arxiv.org/html/2508.10925v1)
- SWAA — [arXiv:2512.10411](https://arxiv.org/abs/2512.10411)（v5 2026-03-26）｜ [HTML](https://arxiv.org/html/2512.10411v5) ｜ [代码](https://github.com/yuyijiong/sliding-window-attention-adaptation)
- PHD-Transformer / PHD-SWA / PHD-CSWA — [arXiv:2504.14992](https://arxiv.org/abs/2504.14992) ｜ [HTML v2](https://arxiv.org/html/2504.14992v2)
- Jet-Nemotron — [arXiv:2508.15884](https://arxiv.org/abs/2508.15884)
- MInference 1.0 — [arXiv:2407.02490](https://arxiv.org/abs/2407.02490)
- DuoAttention — [arXiv:2410.10819](https://arxiv.org/abs/2410.10819)
- Infini-attention — [arXiv:2404.07143](https://arxiv.org/abs/2404.07143)

**官方 config.json（可复现命令：`curl -sL https://huggingface.co/<repo>/raw/main/config.json`）**
- [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1/raw/main/config.json)（`sliding_window: 4096`）
- [mistralai/Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3/raw/main/config.json)（`sliding_window: null`）
- [mistralai/Mistral-Nemo-Base-2407](https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/raw/main/config.json)、[Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/raw/main/config.json)、[Mistral-Small-24B-Base-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501/raw/main/config.json)、[Mistral-Large-Instruct-2411](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411/raw/main/config.json)、[Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1/raw/main/config.json)（均 `null`）
- [Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json)、[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/raw/main/config.json)（`sliding_window: 131072`, `use_sliding_window: false`）
- [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)、[Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B/raw/main/config.json)、[Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B/raw/main/config.json)（`sliding_window: null`）
- [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json)（`full_attention_interval: 4`）
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b/raw/main/config.json)、[openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b/raw/main/config.json)（`sliding_window: 128`, 严格交替 `layer_types`）
- [CohereLabs/command-a-plus-05-2026-bf16](https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16/raw/main/config.json)（`sliding_window: 4096`, 3:1 模式）
- [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct/raw/main/config.json)（`blocksparse_*`, 无 SWA）
- [microsoft/Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct/raw/main/config.json)（`sliding_window: 262144` 无意义）
- [microsoft/phi-4](https://huggingface.co/microsoft/phi-4/raw/main/config.json)（`sliding_window: null`）
- [ibm-granite/granite-4.0-h-small](https://huggingface.co/ibm-granite/granite-4.0-h-small/raw/main/config.json)（Mamba 混合）
- [tiiuae/Falcon-H1-34B-Base](https://huggingface.co/tiiuae/Falcon-H1-34B-Base/raw/main/config.json)、[zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5/raw/main/config.json)、[deepseek-ai/DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json)、[moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json)、[MiniMaxAI/MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k/raw/main/config.json)、[allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B/raw/main/config.json)、[HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B/raw/main/config.json)、[arcee-ai/AFM-4.5B](https://huggingface.co/arcee-ai/AFM-4.5B/raw/main/config.json)（均无 SWA）

**官方代码与文档**
- vLLM `--disable-sliding-window` / 滑动窗配置：[vllm/engine/arg_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py)、[vllm/config/model.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)
- vLLM 逐层滑窗（Gemma）：[vllm/model_executor/models/gemma2.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma2.py)
- **vLLM Hybrid KV Cache Manager（混合层 KV 管理）**：[官方文档源码](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/design/hybrid_kv_cache_manager.md) ｜ [渲染页](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)
- SGLang 滑动窗解析与 KV 池：[configs/model_config.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/model_config.py)、[model_executor/model_runner.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py)、[layers/attention/flashattention_backend.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashattention_backend.py)
- FlashAttention 窗口参数：[flash_attn/flash_attn_interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
- transformers Gemma-2 config 派生逻辑：[configuration_gemma2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/configuration_gemma2.py)
- transformers Gemma-3 config 派生逻辑：[configuration_gemma3.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma3/configuration_gemma3.py)
- transformers Llama-4 config（chunked local attention）：[configuration_llama4.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama4/configuration_llama4.py)

---

## 附录 B：给第 1 章 / 第 3 章的引用建议

**第 1 章（技术背景）建议叙事主线：**
1. 用 §1.5 的时间线开篇：Longformer（形）→ Mistral（规模化 + 工程化）→ StreamingLLM（暴露致命缺陷）→ Gemma-2（产业解法：层间混合）→ PHD-SWA（PHD 的重新定位）。
2. **必须区分两个"SWA"**：① 通用 SWA = 替代全注意力的架构选择（省 KV，有质量风险）；② PHD-SWA = 只对 hidden decoding token 生效的 mask 技巧（不省长上下文 KV，不承担长程依赖）。**这是本章最容易写错的地方。**
3. 用 §2 的公式与 Mistral 的"32K ↓8×"作为 KV cache 收益的锚点；同时用 §2.2 的限定（FFN/投影的 O(L·d²) 不可压缩、端到端只有 2×）防止过度承诺。

**第 3 章（技术影响）建议主线：**
1. **产业采用的说服力在于"谁放弃了它"**：Mistral 自家 v0.1 → v0.3 移除滑窗（§3.1）比任何第三方评测都更有力。
2. **"层间混合"是 2024 后的唯一存活形态**，且方向是把全局层做得更少更便宜（Gemma-2 1:1 → Gemma-3 5:1，窗口 4096→1024，上下文 8K→128K）。
3. **用 SWAA 的 Table 1 作为"SWA 边界"的量化底线**（纯 SWA → 检索归零；奇数层 FA vs 偶数层 FA → 20 倍差距）。
4. **系统成本必须写**：vLLM Hybrid KV Cache Manager 的存在本身就是"层间混合不免费"的官方证据（§5.2）。
5. 变种方向（§7）中，方向 2（层位置搜索）、方向 4（窗口自适应）、方向 8（滑窗作用于中间 token）与 PHD 的关联度最高；方向 4 需明确标注为开放问题。

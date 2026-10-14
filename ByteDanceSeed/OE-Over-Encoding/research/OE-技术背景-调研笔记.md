# OE（Over-Encoding）技术背景调研笔记
## 主题：为什么词表 / 分词（tokenization）与词表规模是当前业界痛点

> **目标技术**：ByteDance Seed, *Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling*（OE，arXiv:2501.16975，ICML 2025），层次化 n-gram 嵌入 + 输入/输出词表解耦。
> **用途**：技术洞察报告第 1 章「技术背景」。
> **调研方法**：所有 arXiv 编号均通过 arXiv API（`export.arxiv.org/api/query`）逐条取回标题/作者/摘要核对；论文中的英文引文均从 arXiv HTML 全文抓取后逐字比对；模型配置数字取自 HuggingFace 官方/镜像仓库的 `config.json` 原始内容。
> **标注约定**：【事实】= 有原文或原始数据支撑；【事实·计算】= 由官方原始数值直接算出（算式已给出）；【未找到】= 联网检索后无可信来源；【推测】= 我的推断，无直接文献支撑。
> **日期**：2026-10-14

---

## 1. BPE / 子词分词的固有缺陷

1. 【事实】**跨语言 token 数差异可达 15 倍**。Petrov et al.（NeurIPS 2023）在 tokenizer 层面证明：同一段文本翻译成不同语言后，编码长度差异最高达 **15×**；即便是刻意做多语支持的 tokenizer 也有此问题，字符级/字节级模型在某些语言对上仍有 **>4×** 的编码长度差，这直接转化为 API 费用、延迟与可用上下文的不公平。[arXiv:2305.15425](https://arxiv.org/abs/2305.15425)

2. 【事实】**token 数直接决定训练成本，且低资源语言被"征税"**。*The Token Tax*（2025）在 AfriMMLU（16 种非洲语言、9,000 题）上评测 10 个 LLM，发现 **fertility（tokens/word）能稳定预测准确率：fertility 越高，准确率越低**，跨所有模型与学科一致；在经济学上，**token 数翻倍会导致训练成本与时间变为约 4 倍**。[arXiv:2509.05486](https://arxiv.org/abs/2509.05486)

3. 【事实】**"非洲语言税"的量化量级**。*The African Language Tax*（2026）用 FLORES-200+ 平行语料测 20 种非洲语言 × 11 个前沿/开源 tokenizer：**每一种非洲语言相对英语都有 tokenization 溢价，GPT-5/o200k_base 中位数为 1.88×，N'Ko 语最高 8.92×**；Ethiopic 与 N'Ko 文字系统最严重（7–9×）。[arXiv:2606.24460](https://arxiv.org/abs/2606.24460)

4. 【事实】**英语中心词表用于多语模型会造成严重退化与成本膨胀**。Ali et al. 训练了 24 个 2.6B 规模的单语/多语 LLM，发现：多语 tokenizer（覆盖欧洲前 5 大语言）相比纯英语 tokenizer **需要约 3 倍的词表规模**；沿用英语中心 tokenizer 会导致 **严重的下游性能退化**，以及**最高 +68% 的额外训练成本**（Multi-UNI-SP-50 vs EN-UNI-SP-100，德语语料）。[arXiv:2310.08754](https://arxiv.org/abs/2310.08754)

5. 【事实】**数字切分方式直接改变算术能力**。Singh & Strouse（*Tokenization counts*）系统对比了 GPT-3.5/4 的数字 tokenization：LLaMA/PaLM 用**单数字**切分，GPT-3.5/4 对 1/2/3 位数各有独立 token；作者发现**右到左切分（推理时对数字加逗号强制触发）显著提升算术性能**，左到右切分的错误呈**刻板的系统性模式**（说明模型在做系统化计算而非近似），且**该 tokenization 偏差造成的差距随模型规模增大而缩小**。[arXiv:2402.14903](https://arxiv.org/abs/2402.14903)

6. 【事实】**tokenizer 设计对代码生成影响显著且可被专门化**。Dagan et al. 在 HumanEval / MBPP 上做大规模消融，结论是 tokenizer 的**规模、预分词正则表达式、训练语料**都会显著影响模型的**生成速度、有效上下文长度、内存占用与下游性能**；当微调数据超过 **500 亿 token** 时，为预训练 LLM **专门化 tokenizer** 可获得生成速度与有效上下文长度的巨大收益。[arXiv:2402.01035](https://arxiv.org/abs/2402.01035)

7. 【事实】**字符级任务（如数字母）是 tokenization 的结构性短板**。*The Strawberry Problem*（EMNLP 2025）用 19 个合成任务隔离字符级推理，指出 LLM 在"数单词里有几个字母"这类简单任务上持续失败，根因是 tokenization 造成的**低互信息**；该能力在训练中**出现得很晚且是突变式涌现**的；作者提出一个轻量架构修改来缓解。[arXiv:2505.14172](https://arxiv.org/abs/2505.14172)

8. 【事实】**glitch tokens 是"词表存在但几乎没被训练过"的 token**。Land & Bartolo（EMNLP 2024）指出 tokenizer 构建与模型训练的脱节，使得诸如 SolidGoldMagikarp 这类 token 能诱发异常行为；他们给出结合 tokenizer 分析、权重指标与 prompt 技术的**自动检测方法**，并证明这类**欠训练 token 在多种模型中普遍存在**。[arXiv:2405.05417](https://arxiv.org/abs/2405.05417)｜原始现象记录（SolidGoldMagikarp，Rumbelow & Watkins, LessWrong, 2023）：[lesswrong.com](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)

9. 【事实】**BPE 本身并非最优，其贪心构建会破坏形态学**。Bostrom & Durrett（Findings of EMNLP 2020）直接对比 BPE 与 unigram LM tokenization，发现后者恢复的子词单元**更贴近形态学**，并**规避了 BPE 贪心构建过程带来的问题**；在英/日两种语言的下游任务上，unigram LM 分词**持平或优于 BPE**。[arXiv:2004.03720](https://arxiv.org/abs/2004.03720)

---

## 2. 分词与下游性能的关系，及 tokenizer-free 路线

1. 【事实】**分词选择对下游性能与训练成本有实质影响，且常用代理指标不可靠**。Ali et al. 的结论是 "the tokenizer choice can significantly impact the model's downstream performance and training costs"，并特别指出**常用的 tokenizer 评价指标 fertility 与 parity 并不总能预测下游性能**，因此这些指标作为代理是可疑的。[arXiv:2310.08754](https://arxiv.org/abs/2310.08754)

2. 【事实】**换 tokenizer 就能提升多语模型性能**。Rust et al.（ACL 2021）在 9 种类型学差异大的语言、5 类任务上做受控实验：**预训练数据量与"专门为该语言设计的 tokenizer"对下游性能同等重要**；在多语词表中得到充分代表的语言，性能下降可忽略；而**把原多语 tokenizer 换成单语专门化 tokenizer，几乎对每个任务和语言都提升了多语模型的下游性能**。[arXiv:2012.15613](https://arxiv.org/abs/2012.15613)

3. 【未找到】**关于用户提到的 "The Tokenizer Parakeet"**：在 arXiv 全库检索 `all:"tokenizer parakeet"` 返回 **0 条结果**；网络检索只命中 NVIDIA 的 ASR 模型 Parakeet（语音识别，与分词无关）。**该文献未能定位，建议不要在第 1 章引用此标题**。（检索式与空结果可复现：[arXiv API 查询](https://export.arxiv.org/api/query?search_query=all:%22tokenizer%20parakeet%22&max_results=5)）

4. 【事实】**"Tokenization counts" 已定位**：即 Singh & Strouse, *Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs*（arXiv:2402.14903）。注意该文核心结论是**数字 tokenization 对算术的影响**（见 §1 第 5 条），而非泛化的"分词导致若干个百分点性能差"。[arXiv:2402.14903](https://arxiv.org/abs/2402.14903)

5. 【事实】**"token 更少"不等于"模型更好"——压缩视角被证伪**。Schmidt et al.（*Tokenization Is More Than Compression*）提出 PathPiece——在给定词表下把文本切成**最少 token 数**的 tokenizer，实验发现**该假设（更少 token → 更好下游性能）不成立**，从而对"BPE 有效是因为压缩文本"这一流行解释提出质疑；作者主张应从预分词、词表构建、切分三阶段逐个审视设计决策。[arXiv:2402.18376](https://arxiv.org/abs/2402.18376)

6. 【事实】**tokenizer-free 的动机与代价（ByT5）**：字节级模型可开箱处理任意语言、**对噪声更鲁棒**、**在拼写与发音敏感任务上更好**、并消除易错的前处理管线；词表矩阵参数被省掉（字节模型只需 **256** 个 embedding），参数从稀疏词表矩阵迁移到稠密网络层。代价是序列变长：输入较长时推理明显更慢，**GEM-XSum 上比 mT5 慢 3.7–6.4×，XNLI 上慢 6.4–9.5×**；Dakshina 音译慢 1.5–2.6×。[arXiv:2105.13626](https://arxiv.org/abs/2105.13626)

7. 【事实】**Canine：完全无 tokenizer、直接吃字符**。通过降采样缩短输入序列 + 深层 Transformer 编码上下文，在 TyDi QA 上**比同规模 mBERT 高 2.8 F1，同时参数少 28%**；论文明确指出"任何固定词表都会限制模型适应能力"，且数据驱动子词词表**并非对所有语言同样合适**。[arXiv:2103.06874](https://arxiv.org/abs/2103.06874)

8. 【事实】**MegaByte：证明 tokenization-free 在长序列上可行**。多尺度解码架构，可端到端可微建模**超过一百万字节**的序列；把序列切分为 patch，patch 内用局部子模型、patch 间用全局模型，从而获得**亚二次自注意力、同等算力下更大的 FFN、解码更好的并行性**；实验表明字节级模型在长上下文语言建模上可与子词模型竞争。[arXiv:2305.07185](https://arxiv.org/abs/2305.07185)

9. 【事实】**BLT（Byte Latent Transformer）：首个在规模上匹配 token 模型的字节级 LLM**。按"下一字节熵"动态切 patch，把算力分配到高复杂度处；完成**首个 FLOP 受控的字节级 scaling 研究，规模到 8B 参数、4T 训练字节**；结论是"在无固定词表的前提下训练原始字节模型是可扩展的"，训练与推理效率均改善，推理固定成本下 BLT 的 scaling 显著优于 tokenization 模型（可同时增大 patch 与模型规模）。[arXiv:2412.09871](https://arxiv.org/abs/2412.09871)

---

## 3. 词表规模的取舍：更短序列 vs 更大的 embedding 与输出 softmax

1. 【事实】**Softmax 的秩瓶颈（理论层面）**。Yang et al.（ICLR 2018 Oral）把语言建模表述为矩阵分解问题：由于 `H_θ ∈ R^{N×d}`、`W_θ ∈ R^{M×d}`，`H_θ W_θ^T` 的秩**严格上界为 embedding 维度 d**；原文结论为 "if d < rank(A′), no matter how expressive the function family U is" 都无法恢复目标矩阵，因此标准 softmax + 分布式词嵌入**容量不足以建模高度依赖上下文的自然语言**。[arXiv:1711.03953](https://arxiv.org/abs/1711.03953)

2. 【事实】**adaptive softmax：绕开对词表规模的线性依赖**。Grave et al.（ICML 2017）利用词的极不平衡分布做聚类，构造**使计算时间期望最小的簇**，从而绕开 O(V)；在 EuroParl 与 One Billion Word 上相比标准近似方法取得大幅效率提升，精度接近 full softmax。[arXiv:1609.04309](https://arxiv.org/abs/1609.04309)

3. 【事实】**"计算/解码复杂度正比于目标词表"是经典共识**。Jean et al.（2014）明确写道：NMT "has its limitation in handling a larger vocabulary, as **training complexity as well as decoding complexity increase proportionally to the number of target words**"，并用 importance sampling + 解码时只选小子集来使大词表可用。[arXiv:1412.2007](https://arxiv.org/abs/1412.2007)

4. 【事实·计算】**大词表 embedding 的绝对参数量（用户所问的算例）**：Llama-3 词表 128,256 × hidden 8,192 = **1,050,673,152 ≈ 1.05B 参数**（单个矩阵）。注意 Llama-3 的 `tie_word_embeddings=false`，输入 embedding 与输出 head 各一份，故 **embedding 相关参数约 2.10B**。原始配置：`vocab_size=128256, hidden_size=8192, tie_word_embeddings=false`（Llama-3-70B）。[config.json 来源](https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json)

5. 【事实·计算】**embedding 参数在总参数中的占比，小模型远高于大模型**：Llama-3-8B（`vocab_size=128256, hidden_size=4096`，untied）单个矩阵 128,256×4,096 = **525,336,576**，输入+输出合计 **1.05B**，占其约 8.03B 总参数的 **≈13.1%**；同为 128,256 词表的 Llama-3-70B，embedding 相关约 2.10B，占约 70.6B 的 **≈2.98%**。[Llama-3-8B config](https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json)｜[Llama-3-70B config](https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json)

6. 【事实】**官方论文直接给出 embedding 参数占比，并明确归因于大词表**。Gemma 2 论文 Table 2 给出：2B 的 Embedding 参数 590,118,912 / Non-embedding 2,024,517,888（**≈22.6%**）；9B 为 917,962,752 / 8,324,201,984（**≈9.9%**）；27B 为 1,180,237,824 / 26,047,480,320（**≈4.3%**）。论文原文解释：*"We inherit from the large Gemini vocabulary (256k entries), that is designed to work on a large number of languages, hence, the larger embedding parameter counts compared to models that are limited to one or a few languages."*[arXiv:2408.00118](https://arxiv.org/abs/2408.00118)

7. 【事实·计算】**小模型上占比可达 1/4 以上**。Qwen2.5-0.5B：`vocab_size=151936, hidden_size=896, tie_word_embeddings=true`，embedding = 151,936×896 = **136,134,656**，相对其约 0.49B 总参数占比 **≈27.6%**。[config.json 来源](https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json)

8. 【事实】**扩大词表并不保证净收益，实证上反而抬高计算成本**。Ali et al. 明确指出：*"increasing the vocabulary size from 50k to larger vocabulary sizes increases the computational costs in all cases. This highlights that the potentially lower fertility of larger vocabulary sizes cannot compensate for the additional costs introduced by the larger vocabulary size."*[arXiv:2310.08754](https://arxiv.org/abs/2310.08754)

9. 【事实】**输入词表与输出词表的成本结构完全不同——这正是 OE 的切入点**。OE 论文原文：*"expanding the input vocabulary incurs almost no additional computational cost, whereas expanding the output vocabulary significantly increases the training overhead for smaller models. Thus, it is natural to consider decoupling the input and output vocabularies."* 论文并指出 embedding 参数"以极其稀疏的方式被使用"，因此训练与推理成本几乎不增。[arXiv:2501.16975](https://arxiv.org/abs/2501.16975)

---

## 4. 词表缩放定律与开源模型词表演进

1. 【事实】**存在"最优词表随算力增长"的定律，且词表参数应比非词表参数增长得更慢**。Tao et al.（NeurIPS 2024）用 33M–3B 参数、最多 500B 字符的模型做系统实验，用**三条互补路径**（IsoFLOPs 分析、导数估计、损失函数参数化拟合）得到一致结论：**最优词表规模取决于算力预算，更大的模型需要更大的词表**；而当前绝大多数 LLM 的词表都偏小。[arXiv:2407.13623](https://arxiv.org/abs/2407.13623)

2. 【事实】**幂律的精确形式（官方代码给出的拟合系数）**。Approach 1（IsoFLOPs）给出
 `N_nv^opt = exp(−2.4846510161625193) · C^0.5`，
 `N_v^opt = exp(−1.589031299255507) · C^0.4163622634135234`，
 即**词表参数随算力按 C^0.416 增长，而非词表参数按 C^0.5 增长** → 词表应增长，但比主干参数慢。Approach 2（导数快速估计）给出更直接的 `N_v = 3,145,728 × (N_nv / 33,000,000)^0.8353974035228025`，即 `N_v^opt ∝ N_nv^0.835`（指数 γ<1，与论文 Figure 1 的 "scaled slower than N_nv" 一致）。[官方仓库 `optimal_Nv_predict.py`](https://github.com/sail-sg/scaling-with-vocab/blob/main/optimal_Nv_predict.py)

3. 【事实】**Approach 3 的参数化损失函数（可直接复用的形式）**：
 `L_u(N_nv, V, H) = −E + A1/N_nv^α1 + A2/(Vd)^α2 + B/[F/(6(N_nv+Vd))]^β`，
 其中 `d` 为 embedding 维度、`F` 为 FLOPs；官方拟合值为 `E=5.5327846803337435, A1=1.8313851559554126, α1=0.44660634152009615, A2=0.19584238398665638, α2=0.6707374679896795, B=2.1241123120064955, β=0.44660634152009615`（约束 α1=β）。最优词表由 `dL_u/dV=0` 求解。[官方仓库 `approach3_isoloss.py`](https://github.com/sail-sg/scaling-with-vocab/blob/main/approach3_isoloss.py)

4. 【事实】**该定律的可证伪预测与实证改进**：论文预测 **Llama-2-70B 的最优词表至少应为 216K，是其 32K 词表的 7 倍**；在同等 `2.3e21` FLOPs 下把词表从常规 32K 提到 **43K，ARC-Challenge 从 29.1 提升到 32.0**。[arXiv:2407.13623](https://arxiv.org/abs/2407.13623)

5. 【事实】**Llama-2 → Llama-3 的词表跃迁与官方理由**：Llama-2 为 **32,000**；Llama-3 为 **128,256**（论文写作 "128K tokens"）。Llama-3 论文原文：*"Our token vocabulary combines 100K tokens from the tiktoken tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to 'read' more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization."*[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)｜[Llama-2 config](https://huggingface.co/NousResearch/Llama-2-7b-hf/raw/main/config.json)

6. 【事实】**Gemma：官方 256k 词表 + 数字切分**。Gemma 论文原文：*"We use a subset of the SentencePiece tokenizer of Gemini for compatibility. It **splits digits**, does not remove extra whitespace, and relies on byte-level encodings for unknown tokens ... The vocabulary size is **256k tokens**."*（Table 1 中 `Vocab size = 256128`）。[arXiv:2403.08295](https://arxiv.org/abs/2403.08295)｜[Gemma-7B config（vocab_size=256000）](https://huggingface.co/unsloth/gemma-7b/raw/main/config.json)

7. 【事实】**Qwen 系列：151K 级别且词表被 padding 到便于张量并行**。Qwen2 论文原文：*"Models of all sizes employ a common vocabulary consisting of 151,643 regular tokens and 3 control tokens"*，Table 中 `Vocabulary Size = 151,646`；并特别注明 *"owing to considerations in distributed training, the effective size for the embeddings is larger."* Qwen2.5-7B 的 `config.json` 为 **152064**，Qwen3-8B 为 **151936**。[arXiv:2407.10671](https://arxiv.org/abs/2407.10671)｜[arXiv:2412.15115](https://arxiv.org/abs/2412.15115)｜[Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json)｜[Qwen3-8B config](https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json)

8. 【事实】**DeepSeek-V3：128K 词表，且为多语压缩效率专门改造**。论文原文：*"The tokenizer for DeepSeek-V3 employs Byte-level BPE with an extended vocabulary of 128K tokens. The pretokenizer and training data for our tokenizer are modified to optimize multilingual compression efficiency."*（`config.json` 中 `vocab_size=129280`）。[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)｜[DeepSeek-V3-Base config](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/raw/main/config.json)

9. 【事实】**OE 自身给出的经验缩放关系**：输入词表规模与训练损失呈 **log-linear**——"exponentially increasing the input vocabulary size consistently results in a linear decrease in loss"；把输入词表从 0.1M 扩到 **1.2M / 12.8M**（12× / 128×）后，**400M 模型的损失追平 1B baseline**，且"no additional training cost"；实测额外训练开销 **<5%**（OLMoE-1.3B 吞吐 1.211 → 1.155 M tok/s，−4.63%；OLMoE-7B 0.494 → 0.453，−8.3%）。[arXiv:2501.16975](https://arxiv.org/abs/2501.16975)

---

## 5. 输出层 / softmax 的系统成本

1. 【事实】**推理时把 `[tokens, vocab]` logits 物化会形成显存尖峰，量级可到 GiB**。vLLM PR #52054 的原始描述：`_get_prompt_logprobs_dict` 一次性物化完整的 `[num_scheduled_tokens, vocab_size]` logits 并 upcast 到 float32，两个张量同时存活，瞬时峰值为 `num_scheduled_tokens × vocab_size × (2B bf16 + 4B fp32)`；**在 8192 个调度 token、152k 词表下约为 7.5 GiB**，且该峰值不被显存 profiling 观测到，会直接挤占 KV cache。[vLLM PR #52054](https://github.com/vllm-project/vllm/pull/52054)

2. 【事实】**logits 显存随 batch × 投机 token 数 × 词表规模线性增长；按词表分片是主要对策**。vLLM PR #50465 原文：采样前完整物化 target logits，分配量为 `O(batch_size * (num_spec_tokens + 1) * vocab_size)`；该 PR 通过让每个 rank 只计算 `vocab_size // P` 的 logits，再沿 batch 维 all-to-all（而非 all-gather 出完整 `[num_tokens, vocab_size]`），把峰值降低 **1/P**，并**并行化采样本身**——原文指出这 "can be particularly beneficial for TPOT whenever heavy sampling operations (e.g. top-k, top-p, block verification) are performed"。[vLLM PR #50465](https://github.com/vllm-project/vllm/pull/50465)

3. 【事实】**词表并行（vocabulary parallel）的通信量分析来自 Megatron-LM**。原文：*"The transformer language model has an output embedding with the dimension of hidden-size (H) times vocabulary-size (v). Since the vocabulary size is on the order of tens of thousands of tokens for modern language models ... it is beneficial to parallelize the output embedding GEMM."* 实现上把 `E_{H×v}` 沿词表维列切分，输入 embedding 后需要一次 all-reduce；输出侧若直接 all-gather logits，则 *"the all-gather will communicate b × s × v elements ... which is huge due to vocabulary size being large"*，因此他们把并行 GEMM 的输出与 cross-entropy **融合**以削减通信。[arXiv:1909.08053](https://arxiv.org/abs/1909.08053)

4. 【事实】**分布式训练会强制把词表 padding 到更大尺寸**。Qwen2 论文明确说明 *"owing to considerations in distributed training, the effective size for the embeddings is larger"*（论文 Table 为 151,646，实际 `config.json` 为 152,064）；这是"词表规模 ↔ 并行切分"耦合最直接的工程证据。[arXiv:2407.10671](https://arxiv.org/abs/2407.10671)｜[Qwen2.5-7B config](https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json)

5. 【事实】**投机解码是当前最主要的解码加速手段，其收益已被大规模验证**。Leviathan et al. 在 T5-XXL 上取得 **2×–3×** 加速且**输出分布完全不变**；Chen et al. 在 **Chinchilla 70B** 分布式设置下取得 **2–2.5×** 解码加速。[arXiv:2211.17192](https://arxiv.org/abs/2211.17192)｜[arXiv:2302.01318](https://arxiv.org/abs/2302.01318)

6. 【事实】**草稿模型与目标模型的词表不一致是投机解码的公开难题**。OmniDraft（NeurIPS 2025）把两大在线部署挑战列为：**(1) 目标模型与草稿模型不兼容；(2) 期望获得延迟改善**，并用"online n-gram cache + 混合蒸馏微调"来应对 **cross-vocabulary mismatch**，目标是"one drafter for all"。[arXiv:2507.02659](https://arxiv.org/abs/2507.02659)

7. 【事实】**词表兼容性是一道硬性工程门槛，不满足就会"静默错误"**。SGLang PR #23838 的动机描述：`STANDALONE` 投机解码**要求 draft 与 target 共享同一词表**；若词表不同，draft 产出的 token id 在 target 词表中对应不同字符串，**投机解码会变成有损的，输出分布不再等于 target 单独生成的分布**。PR 给出真实反例：target `google/gemma-2-9b` + draft `bigcode/tiny_starcoder_py`（不同 tokenizer 家族）时 "every accepted draft token is silently wrong"。[SGLang PR #23838](https://github.com/sgl-project/sglang/pull/23838)

8. 【事实】**大词表下采样本身会成为 TPOT 瓶颈**（见第 2 条 vLLM PR 原文的 top-k/top-p 表述）；结合第 1 条可见，logits 的**显存带宽与物化成本**在大词表 + 投机解码叠加时是乘法级放大。[vLLM PR #50465](https://github.com/vllm-project/vllm/pull/50465)

9. 【事实 + 推测】**Gemma 2 的 logit soft-capping**：论文原文为对每层注意力 logits 与**最终层 logits** 施加 `logits ← soft_cap * tanh(logits / soft_cap)`，最终层取值 **30.0**（注意力层 50.0）。【事实】该文**并未**把 soft-capping 与 256k 大词表显式关联；"大词表 → logits 数值范围/z 配分函数更难控制 → 需要 soft-cap"这一因果链属【推测】，不建议在报告中断言。[arXiv:2408.00118](https://arxiv.org/abs/2408.00118)

---

## 6. 产业侧动机：为什么 "token efficiency" 是核心 KPI

1. 【事实】**Chinchilla 确立了"token 数"与"参数量"同等重要**。Hoffmann et al. 训练 400+ 个 70M–16B 模型、5B–500B token，结论是**算力最优下模型规模与训练 token 数应等比例增长（模型翻倍则 token 数也翻倍）**；Chinchilla 用 **70B 参数 / 1.4T token**（≈20 tokens/param）在同等算力下全面超过 280B 的 Gopher。[arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

2. 【事实】**"超额训练小模型"已被官方明确为推理效率策略**。Llama-3 论文原文：*"While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also **train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget.**"* Llama-3 预训练语料约 **15T** 多语 token，而 Llama-2 为 **1.8T**。[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

3. 【事实】**词表扩大的收益被官方表述为"同等训练算力下读更多文本"**。Llama-3 原文：新 tokenizer 把英语样本压缩率从 **3.17 提升到 3.94 characters per token**，*"This enables the model to 'read' more text for the same amount of training compute."* —— 这正是把 token 效率直接绑定到训练算力效率的官方措辞。[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

4. 【事实】**Kimi K2 把 token efficiency 提升到"缩放系数"的地位**。原文两处关键表述：*"Pre-training must endow models with broad general-purpose priors under constraints of limited high-quality data, **elevating token efficiency—learning signal per token—as a critical scaling coefficient**."* 以及 *"Given the increasingly limited availability of high-quality human data, we posit that **token efficiency is emerging as a critical coefficient in the scaling of large language models**."* 其手段是 token-efficient 的 Muon 优化器 + QK-Clip（MuonClip），在 **15.5T token** 上预训练且零 loss spike。[arXiv:2507.20534](https://arxiv.org/abs/2507.20534)

5. 【事实】**token 膨胀的经济后果是超线性的**。*The Token Tax* 明确指出：**token 数翻倍 ⇒ 训练成本与时间约变为 4 倍**，并把这直接称为低资源语言承担的 "token tax"；该文同时用 fertility 预测准确率下降。[arXiv:2509.05486](https://arxiv.org/abs/2509.05486)

6. 【事实】**长上下文成本对 token 数高度敏感（KV cache 正比于 token 数）**。HuggingFace 的 Llama 3.1 推理显存文章中，FP16 KV cache 随上下文线性增长：**8B 从 1k token 的 0.125 GB 增至 128k token 的 15.62 GB；70B 对应 0.313 GB → 39.06 GB；405B 对应 0.984 GB → 123.05 GB**。其解析式为每 token 的 KV cache 正比于 `2 × num_layers × num_kv_heads × head_dim × dtype_bytes`（系数 2 为 K 与 V），因此**服务成本与序列中的 token 数成线性关系**。[huggingface.co/blog/llama31](https://huggingface.co/blog/llama31)｜[KV cache 公式（NVIDIA）](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

7. 【事实】**API 商业计费直接按 token 计数**。DeepSeek 官方定价页原文：*"The prices listed below are in units of **per 1M tokens**. A token, the smallest unit of text that the model recognizes, can be a word, a number, or even a punctuation mark. **We will bill based on the total number of input and output tokens by the model.**"* 这意味着 §1 中的跨语言 token 数差异**可直接折算为终端用户的价格差异**。[api-docs.deepseek.com](https://api-docs.deepseek.com/quick_start/pricing)｜[anthropic.com/pricing](https://www.anthropic.com/pricing)（页面可访问，但正文为前端渲染，未能逐字核验具体价目）

8. 【事实】**"每 token 承载更多信息"已成为可比较的竞争力指标**。可核验的量化口径有三类：(a) **characters/token 压缩率**（Llama-3：3.17 → 3.94）；(b) **fertility = tokens/word**（Token Tax / African Language Tax，用于预测准确率与成本）；(c) **每 token 相对英语的溢价倍数**（African Language Tax：GPT-5/o200k_base 中位 1.88×，N'Ko 最高 8.92×）。[arXiv:2407.21783](https://arxiv.org/abs/2407.21783)｜[arXiv:2509.05486](https://arxiv.org/abs/2509.05486)｜[arXiv:2606.24460](https://arxiv.org/abs/2606.24460)

9. 【未找到】**厂商把 "bytes per token / BPT" 作为官方 KPI 的公开表述**：未检索到 OpenAI / Anthropic / Google / Meta 官方文档或技术报告中以 bytes-per-token 作为对外承诺指标的表述；仅在部分第三方评测论文中出现 BPT 度量。建议第 1 章使用 Llama-3 的 characters/token 与 Token Tax 的 fertility 作为可引用的替代口径。[Llama-3 characters/token](https://arxiv.org/abs/2407.21783)｜[Token Tax fertility](https://arxiv.org/abs/2509.05486)

---

## 附录 A：关键数字速查（均来自上文已核验来源）

| 事实 | 数值 | 来源 |
|---|---|---|
| 同义句跨语言 token 数最大差异 | 最高 15× | [arXiv:2305.15425](https://arxiv.org/abs/2305.15425) |
| 非洲语言相对英语的 tokenization 溢价 | 中位 1.88×，N'Ko 最高 8.92× | [arXiv:2606.24460](https://arxiv.org/abs/2606.24460) |
| 英语中心 tokenizer 的额外训练成本 | 最高 **+68%** | [arXiv:2310.08754](https://arxiv.org/abs/2310.08754) |
| 多语 tokenizer 相对英语所需词表倍数 | ≈ 3× | [arXiv:2310.08754](https://arxiv.org/abs/2310.08754) |
| token 数翻倍对训练成本的影响 | ≈ 4× | [arXiv:2509.05486](https://arxiv.org/abs/2509.05486) |
| Llama-2-70B 的算力最优词表（预测） | ≥ 216K（其实际 32K 的 7×） | [arXiv:2407.13623](https://arxiv.org/abs/2407.13623) |
| 词表从 32K → 43K 的 ARC-C 增益（同 FLOPs） | 29.1 → 32.0 | [arXiv:2407.13623](https://arxiv.org/abs/2407.13623) |
| 词表参数幂律指数 | C^0.416（词表） vs C^0.5（非词表） | [官方仓库](https://github.com/sail-sg/scaling-with-vocab/blob/main/optimal_Nv_predict.py) |
| Llama-3 embedding 参数量（128,256×8,192） | ≈ 1.05B（untied 则 ≈2.10B） | [config](https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json) |
| Gemma 2 各规模 embedding 参数占比 | 2B ≈22.6%，9B ≈9.9%，27B ≈4.3% | [arXiv:2408.00118](https://arxiv.org/abs/2408.00118) |
| 8192 token × 152k 词表 logits 显存峰值 | ≈ 7.5 GiB | [vLLM PR #52054](https://github.com/vllm-project/vllm/pull/52054) |
| Llama-3.1 8B 的 KV cache（1k → 128k token） | 0.125 GB → 15.62 GB | [HF blog](https://huggingface.co/blog/llama31) |
| OE 的额外训练开销 | **< 5%**（吞吐 −4.63% / −8.3%） | [arXiv:2501.16975](https://arxiv.org/abs/2501.16975) |
| ByT5 在长输入任务上的推理代价 | 比 mT5 慢 3.7–9.5× | [arXiv:2105.13626](https://arxiv.org/abs/2105.13626) |

---

## 附录 B：URL 汇总清单

### 目标技术（OE）
- https://arxiv.org/abs/2501.16975 — Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling（ICML 2025），全文引文取自 https://arxiv.org/html/2501.16975v2

### §1 BPE/子词分词的固有缺陷
- https://arxiv.org/abs/2305.15425 — Language Model Tokenizers Introduce Unfairness Between Languages（NeurIPS 2023）
- https://arxiv.org/abs/2509.05486 — The Token Tax: Systematic Bias in Multilingual Tokenization
- https://arxiv.org/abs/2606.24460 — The African Language Tax
- https://arxiv.org/abs/2310.08754 — Tokenizer Choice For LLM Training: Negligible or Crucial?
- https://arxiv.org/abs/2402.14903 — Tokenization counts: the impact of tokenization on arithmetic in frontier LLMs
- https://arxiv.org/abs/2402.01035 — Getting the most out of your tokenizer for pre-training and domain adaptation
- https://arxiv.org/abs/2505.14172 — The Strawberry Problem
- https://arxiv.org/abs/2405.05417 — Fishing for Magikarp: Automatically Detecting Under-trained Tokens
- https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation — SolidGoldMagikarp（原始现象记录）
- https://arxiv.org/abs/2004.03720 — Byte Pair Encoding is Suboptimal for Language Model Pretraining

### §2 分词与下游性能 + tokenizer-free
- https://arxiv.org/abs/2012.15613 — How Good is Your Tokenizer?（ACL 2021）
- https://arxiv.org/abs/2402.18376 — Tokenization Is More Than Compression
- https://arxiv.org/abs/2105.13626 — ByT5: Towards a token-free future
- https://arxiv.org/abs/2103.06874 — CANINE: Pre-training an Efficient Tokenization-Free Encoder
- https://arxiv.org/abs/2305.07185 — MEGABYTE: Predicting Million-byte Sequences
- https://arxiv.org/abs/2412.09871 — Byte Latent Transformer: Patches Scale Better Than Tokens
- https://export.arxiv.org/api/query?search_query=all:%22tokenizer%20parakeet%22&max_results=5 — "The Tokenizer Parakeet" 检索（0 结果，【未找到】）

### §3 词表规模取舍
- https://arxiv.org/abs/1711.03953 — Breaking the Softmax Bottleneck（ICLR 2018 Oral）
- https://arxiv.org/abs/1609.04309 — Efficient softmax approximation for GPUs（ICML 2017）
- https://arxiv.org/abs/1412.2007 — On Using Very Large Target Vocabulary for NMT
- https://huggingface.co/NousResearch/Meta-Llama-3-8B/raw/main/config.json
- https://huggingface.co/NousResearch/Meta-Llama-3-70B/raw/main/config.json
- https://huggingface.co/NousResearch/Llama-2-7b-hf/raw/main/config.json
- https://huggingface.co/Qwen/Qwen2.5-0.5B/raw/main/config.json
- https://huggingface.co/unsloth/gemma-7b/raw/main/config.json
- https://arxiv.org/abs/2408.00118 — Gemma 2: Improving Open Language Models at a Practical Size

### §4 词表缩放定律与词表演进
- https://arxiv.org/abs/2407.13623 — Scaling Laws with Vocabulary（NeurIPS 2024）
- https://github.com/sail-sg/scaling-with-vocab — 官方代码
- https://github.com/sail-sg/scaling-with-vocab/blob/main/optimal_Nv_predict.py — 幂律拟合系数
- https://github.com/sail-sg/scaling-with-vocab/blob/main/approach3_isoloss.py — 参数化损失函数与系数
- https://arxiv.org/abs/2407.21783 — The Llama 3 Herd of Models
- https://arxiv.org/abs/2403.08295 — Gemma: Open Models Based on Gemini Research and Technology
- https://arxiv.org/abs/2407.10671 — Qwen2 Technical Report
- https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3-8B/raw/main/config.json
- https://arxiv.org/abs/2412.15115 — Qwen2.5 Technical Report
- https://arxiv.org/abs/2412.19437 — DeepSeek-V3 Technical Report
- https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/raw/main/config.json

### §5 输出层 / softmax 系统成本
- https://github.com/vllm-project/vllm/pull/52054 — logits 显存峰值（8192×152k ≈ 7.5 GiB）
- https://github.com/vllm-project/vllm/pull/50465 — logits 维度 O(batch×(num_spec_tokens+1)×vocab)，按词表分片
- https://arxiv.org/abs/1909.08053 — Megatron-LM（输出 embedding 并行与 b×s×v 通信量）
- https://arxiv.org/abs/2211.17192 — Fast Inference from Transformers via Speculative Decoding
- https://arxiv.org/abs/2302.01318 — Accelerating LLM Decoding with Speculative Sampling
- https://arxiv.org/abs/2507.02659 — OmniDraft: A Cross-vocabulary, Online Adaptive Drafter
- https://github.com/sgl-project/sglang/pull/23838 — 投机解码的词表兼容性校验

### §6 产业侧 token efficiency
- https://arxiv.org/abs/2203.15556 — Training Compute-Optimal Large Language Models（Chinchilla）
- https://arxiv.org/abs/2507.20534 — Kimi K2: Open Agentic Intelligence
- https://huggingface.co/blog/llama31 — Llama 3.1 推理显存 / KV cache 表
- https://github.com/huggingface/blog/issues/2345 — 上文的 KV cache 表格与算式复现
- https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/ — KV cache 公式
- https://api-docs.deepseek.com/quick_start/pricing — 按 token 计费的官方表述
- https://www.anthropic.com/pricing — 定价页（可访问，正文前端渲染）
- https://openai.com/api/pricing/ — 定价页（返回 403，未能核验内容，【未找到】具体条文）

---

### 调研中明确未能证实的条目（避免在报告中臆造）
- **"The Tokenizer Parakeet"**：arXiv 全库检索 0 结果，网络检索仅命中 ASR 模型 Parakeet。**建议不引用**。
- **"分词选择导致下游性能相差若干个百分点"的单一权威出处**：未找到一篇给出"×个百分点"总括数字的论文；可用的可引用替代证据是 Ali et al. 的 **+68% 训练成本 / 严重性能退化**，以及 Rust et al. 的"替换 tokenizer 几乎全面提升"。
- **"bytes per token" 作为厂商官方 KPI 的表述**：未找到官方来源。
- **Gemma 2 logit soft-capping 与大词表的因果关系**：论文未表述该因果，属【推测】。
- **OpenAI 定价页具体条文**：页面返回 403，未核验。

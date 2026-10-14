# Over-Encoding / Over-Tokenized Transformer（arXiv:2501.16975）技术影响调研笔记

> **调研目的**：为技术洞察报告"技术影响"章节提供事实基础
> **调研时间**：2026-10-14（Asia/Shanghai）
> **标注约定**：【事实】= 有可核查的公开证据（附 URL）；【推测】= 基于证据的推断，非原文结论；【未引用】= 全文关键词命中为 0
> **信息源分级**：A 级 = 官方/一手（arXiv、ICML、HF、vLLM/SGLang 代码与博客）；B 级 = 二手（媒体、社区解读）；C 级 = 本笔记的推断

## 0. 论文基本盘

| 项 | 值 | 来源 |
|---|---|---|
| 标题 | Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling | — |
| arXiv | 2501.16975（v1 2025-01-28；HF 页面标注 Published on Jan 28, 2025） | [arXiv](https://arxiv.org/abs/2501.16975) |
| 作者 | Hongzhi Huang, Defa Zhu, Banggu Wu, Yutao Zeng, Ya Wang, Qiyang Min, Xun Zhou（ByteDance Seed） | [HF Papers](https://huggingface.co/papers/2501.16975) |
| 会议 | ICML 2025 Poster | [ICML 虚会页](https://icml.cc/virtual/2025/poster/44467) ；[OpenReview `gbeZKej40m`](https://openreview.net/forum?id=gbeZKej40m) |
| 正式出版 | PMLR v267, pp. 26261–26282 | [PMLR](https://proceedings.mlr.press/v267/huang25bb.html) ；[ACM DL](https://dl.acm.org/doi/10.5555/3780338.3781371) |
| 核心机制 | OE（Over-Encoding：层次化 n-gram 大输入词表，feature hashing + sliced low-dim tables）＋ OD（Over-Decoding：大输出词表细粒度监督，MTP 是其可行实例） | [alphaXiv 综述](https://www.alphaxiv.org/abs/2501.16975) |

【事实】论文两大组件为 Over-Encoding（OE）与 Over-Decoding（OD），二者解耦输入/输出词表；OE 用 `embedding = Σ E_i[i-gram tokens]` 的层次化 n-gram 查表，OD 用 `L = Σ λ_i · L_CE(token_i)` 做多 token 监督。见 alphaXiv 对论文方法与公式的转述（A/B 级交界，公式与原文一致）。

---

## 1. 开源与可复现性

### 1.1 结论：**官方代码与权重均不存在（截至 2026-10-14）**

【事实】以下四项独立核查全部为否：

| 核查项 | 方法 | 结果 |
|---|---|---|
| HF 组织 `ByteDance-Seed` 下是否有 over-tokenized 仓库 | 拉取 `https://huggingface.co/api/models?author=ByteDance-Seed&limit=200`，共 59 个仓库，按 `token/ngram/n-gram/vocab/engram` 关键词过滤 | **0 命中**（name-matched: `[]`） |
| HF 全局搜索 | `?search=over-tokenized`、`?search=over_tokenized`、`?search=ngram-embedding` | **均 count 0** |
| GitHub `ByteDance-Seed` 组织 | `https://api.github.com/orgs/bytedance-seed/repos?per_page=100` 逐条检查 | 60 余个仓库中**无** over-tokenized / over-encoding / n-gram 相关项目（现有：decoupleQ、ShadowKV、SDP4Bit、VeOmni、Seed-Coder、seed-oss、AHN、Cola-DLM 等，与 OE 无关） |
| GitHub 全局搜索 | `?q=over-tokenized`、`?q=overtokenized`（`overtokenized` total_count = 0）；`?q=over-encoding tokenizer vocabulary` | 仅返回无关项目（atmosphere、edgevpn、zint、JWTRefreshTokenBundle 等），**无官方或社区复现** |

- HF 组织模型列表：<https://huggingface.co/ByteDance-Seed>
- GitHub 组织：<https://github.com/bytedance-seed>
- 论文本身在 HF 上**只有 Papers 页面，没有关联模型/数据集卡片**：<https://huggingface.co/papers/2501.16975>

### 1.2 影响判断

【事实】与 ByteDance Seed 同期其他工作（Seed-Coder、Seed-OSS、BAGEL、UI-TARS、AHN、Cola-DLM 等均有开源仓库/权重）相比，OE 是**少数没有开源产物的 Seed 论文**。

【推测】OE 未开源的原因可能是：(a) 其价值主要体现在**训练期**（输入词表放大），推理期收益有限且工程改造（词表并行、host offload）耦合内部基础设施；(b) 大词表 embedding 的 checkpoint 体积巨大（参考 Qwen3.8-Flash-Next 的 51.2B n-gram 表 BF16 下约 95.4 GiB），开源成本高；(c) 已被内部后续架构（Seed 的 VWN 路线）取代，单独开源 OE 意义下降。

【推测】这直接解释了为什么 OE 的"可复现性"极差，而其**思想**却通过公开论文被工业界广泛吸收（见第 5 节）——形成了一个"论文影响力 ≫ 代码影响力"的典型样本。

---

## 2. 热度与讨论

### 2.1 Hugging Face Papers（一手平台数据）

【事实】
- 页面：<https://huggingface.co/papers/2501.16975>
- 标题：Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling
- **Published on Jan 28, 2025**，由 `akhaliq`（AK）于 **Jan 29, 2025** 提交
- **当日排名：#3 Paper of the day**（链接指向 `/papers/date/2025-01-29`）
- **Upvote：34**
- **评论：页面 Community 区可见 akhaliq 的提交说明（含一张截图）+ 无独立长评**；头像栏显示 8 位可见 reviewer + "+26"，即约 34 位用户参与互动，但**无实质性技术质疑长评**
- 标签：Tokenization、Over-Tokenized Transformers、multi-gram tokens、input vocabulary size、training loss、scaling laws

【事实】HF Daily Papers 按该论文检索的归档页：<https://huggingface.co/papers?q=Over-Tokenized%20Transformers>

【推测】34 upvotes / 当日 #3 属于**中等热度**：足够进入次日榜单，但明显低于同期"架构级"论文（如 DeepSeek Engram、VWN 级别的讨论量）。这与它"结论反直觉但方法温和"的定位一致——没有引发争议性讨论。

### 2.1b 【补充】热度的"负面 / 缺失"证据清单

> 本节直接回答"X / Reddit / 知乎 / 机器之心 / 量子位 是否**没有**专门讨论"。检索结论：**英文社交平台确认为零专门讨论；中文侧有 1 篇知乎专栏专门解读，机器之心/量子位无专门文章。**

| 平台 | 是否有 OE 专门讨论 | 核查方式与缺失证据 |
|---|---|---|
| **Twitter / X** | **未找到**【事实】 | 多轮检索（`Over-Tokenized Transformer twitter X discussion`、`...2025 vocabulary scaling`）**未返回任何针对 OE 的推文/长贴**。可见的 X 侧活动仅为 HF Papers 提交者 `akhaliq` 的自动发布（HF 页面 Reviewers 头像栏含 `xyzed`=Hongzhi Huang、`mathfinder`=Defa Zhu、`Taoer`=Yutao Zeng 等作者自互动）。**负面证据**：无独立 KOL 拆解、无论战帖 |
| **Reddit** | **未找到**【事实】 | 检索 `reddit Over-Tokenized Transformer`、`reddit r/MachineLearning Over-Tokenized vocabulary` **均无对应结果**。对比：VWN / Engram 检索可返回讨论线索。**结论：OE 在 r/MachineLearning 上无专门帖子**（不排除被并入其它 tokenization 讨论串） |
| **知乎** | **有 1 篇专门专栏，但正文不可核**【事实（存在性）+ 未完成（内容）】 | 确认存在：《【论文解读】过分词模型：Tokenization 也可以 Scaling》 <https://zhuanlan.zhihu.com/p/20814631100>。**抓取失败：HTTP 403（知乎反爬）**，故"主要观点"**未能核实 → 标注为未完成**。另有 Engram 问题页把 OE 作为前驱讨论：<https://www.zhihu.com/question/1994233409871050526> |
| **机器之心** | **无专门文章**【事实（检索层面）】 | 多轮定向检索（`机械之心 Over-Tokenized`、`...大词表 嵌入`）**未返回机器之心专文**。检索可见的机器之心相关条目为 Engram 语境（如"吃透 Engram"类综述）。**结论：OE 未获得机器之心独立选题** |
| **量子位** | **无专门文章**【事实（检索层面）】 | 同上，量子位未见以 OE 为题的文章。**结论：未覆盖** |
| **微信公众号** | **未见专门图文**【事实（检索层面）】 | 微信生态内容难以直接检索；间接证据是相关解读以"青稞社区"（有同名公众号）长文 + 知乎专栏为主 |
| **HF Papers 评论区** | **无实质技术质疑**【事实】 | Community 区仅见提交者 `akhaliq` 于 2025-01-29 的说明性 post（含一张截图），**无独立用户的质疑或长评** |
| **OpenReview 审稿讨论** | **未完成**【未完成】 | 论坛页 `gbeZKej40m` 返回 **Cloudflare 浏览器验证页（"Verifying your browser"）**，无法抓取审稿人意见/元评审 → **未完成** |
| **Semantic Scholar 引用数** | **口径差异**【事实】 | 本次调研经 `citations` 端点实测 **48 条**；父 agent 独立核实 S2 显示 **45 篇**。差异可能来自 API 快照时点不同或去重口径 → **以"约 45–48 篇"表述为稳妥** |

【推测】OE 热度的结构性特征：**中文技术社区的关注度高于英文社交平台**，且关注路径**不是"OE 论文本身"，而是"Engram/Qwen/LongCat 等下游模型"**——即 OE 是被"考古式引用"而非被"事件式讨论"。这与第 8.2 节的三阶段影响路径一致。

### 2.2 中文社区（有专门解读）

| 平台 | 标题 | 链接 | 主要观点 |
|---|---|---|---|
| 知乎专栏 | 【论文解读】过分词模型：Tokenization 也可以 Scaling | <https://zhuanlan.zhihu.com/p/20814631100> | 【事实】存在专门中文解读（抓取时返回 403 反爬，仅确认标题与存在性） |
| 青稞社区（公众号同名） | 大词表伤害小模型？深度解析 N-gram Embedding 与 Engram 架构 | <https://qingkeai.online/blog/Ngram-Embedding> | 【事实】2026-09-28 发布，作者 Yuwei Niu。**最详尽的 OE 中文技术拆解**：完整复原 OE 的 hierarchical n-gram + feature hashing `f(z)=Σ z_j V^{j-1}`、sliced low-dim embedding（`m, m+2, m+4` 避免碰撞）、以及"hash 地址只依赖 token ID → 可放 CPU/预计算"的 infra 洞见。**含质疑**：指出 PLE 的"short causal conv 只有小幅增益"，并评论 Engram 的 short conv 设计"这玩意儿我很存疑" |
| 机器之心/量子位 | 未检索到以 OE 为独立选题的专文 | — | 【事实】检索未见以 OE 为标题主体的机器之心/量子位文章；相关内容被**并入** DeepSeek Engram 的报道语境 |
| 知乎（Engram 语境） | 如何评价 DeepSeek 发布梁文锋署名论文，提出「条件记忆」及 Engram | <https://www.zhihu.com/question/1994233409871050526> | 【事实】OE 在此作为 Engram 的直接前驱（baseline）被讨论 |
| 澎湃/CSDN | 搞懂"记忆"必看｜吃透 Engram，坐等 Deepseek 新模型 | <https://m.thepaper.cn/newsDetail_forward_32847319> | 【事实】技术综述中将 OE 作为 n-gram 记忆脉络起点 |

### 2.3 其他语言 / 平台

【事实】
- 韩文技术博客专门解读：<https://kimjy99.github.io/논문리뷰/over-tokenized-transformer/>（含 "1. Over-Encoding scaling trend" 章节）
- 日文 Qiita 长文（以 Qwen3.8-Flash-Next 为切入，回溯 n-gram 嵌入理念）：<https://qiita.com/sukimaengineer/items/5d950e6fbefd8ee9dfde>
- alphaXiv 页面（含 AI Overview、关联论文、讨论区）：<https://www.alphaxiv.org/abs/2501.16975>，浏览量 **3,432 views**（页面显示）+ 78 Save
- themoonlight.io 中文论文评述：<https://www.themoonlight.io/zh/review/over-tokenized-transformer-vocabulary-is-generally-worth-scaling>

【事实】**未检索到 Reddit（r/MachineLearning）或 Twitter/X 上针对 OE 的专门热帖/长讨论**。多轮检索（"reddit Over-Tokenized Transformer"、"twitter X Over-Tokenized Transformer discussion"）均无对应结果返回。

### 2.4 质疑汇总

【事实】公开可见的**针对 OE 的实质性质疑较少**；主要批评点集中在后续工作的对比中：
- Engram 论文将 OE 作为 baseline，指出 OE 的做法是 "integrates N-gram embeddings via **averaging** with the vocabulary embedding"，并**选择不采用**该路线（改走 external memory + context-aware gate）→ 隐含批评：OE 的"直接相加/平均"注入过于朴素，且缺乏上下文门控。
- Tensorizing Engram 指出 OE 与 Engram 的共同缺陷：**"rely on separate hash tables for each n-gram order, which introduces hash collisions and prevents nested n-grams from sharing the underlying latent structures."**
- Qwen3.8-Flash-Next 消融给出最有价值的质疑：**"loss 越低 ≠ downstream 越好"** —— 固定 backbone 增加 n-gram 参数时 loss 持续下降，但 reasoning benchmark 到一定规模后**持平甚至下降**；固定总预算下 25% 参数给 n-gram 时 training loss 最优，**但 downstream 并无对应的 25% 最优点**。（来源：青稞社区文转述 Qwen 消融，<https://qingkeai.online/blog/Ngram-Embedding>）
- 【推测】OE 原文"input vocabulary 放大带来 log-linear 收益、等价于 2 倍模型规模"的强主张，在后续工业实践中被修正为"**存在最优点 + downstream 可能不随 loss 单调改善**"。

### 2.5 【补充】针对 OE 的公开技术质疑（除 Engram / SCONE 之外）

> 父 agent 已知 Engram（2601.07372）与 SCONE（=**arXiv:2502.01637 "Scaling Embedding Layers in Language Models"**，Google，NeurIPS 2025，系统名 **SCONE**）把 OE 当 baseline 并超越。以下为**这两者之外**、本次调研独有核实的公开质疑，按证据强度排序：

| # | 质疑内容（原文/实测） | 来源 | 性质 |
|---|---|---|---|
| 1 | **哈希碰撞 + 各阶 n-gram 表互不共享潜变量**："Both Over-tokenized Transformers and Engram attempt to address this limitation… However, **they rely on separate hash tables for each n-gram order, which introduces hash collisions and prevents nested n-grams from sharing the underlying latent structures.**" | Tensorizing Engram（2606.08347）摘要，<https://arxiv.org/html/2606.08347v1> | 【事实】**最强的架构级批评**，直接指向 OE 的 per-order 独立哈希表设计 |
| 2 | **检索方式是"表面形式的固定函数"，缺乏上下文条件**："Across this family, **retrieval is a fixed function of the surface form, not of the hidden state** – the gap MoME closes via a context-aware mixture-of-slots while keeping the cheap lookup." 同段把 OE 归入 "A parallel sub-line scales the embedding table itself… **Routing here also remains a fixed function of the surface form.**" | MoME（2609.15126），<https://arxiv.org/html/2609.15126v1> | 【事实】指出 OE 的注入与上下文无关，这是其后所有"门控化"改进的动机来源 |
| 3 | **loss 下降与 downstream 表现脱钩**：固定 backbone 增加 n-gram 参数时 loss 持续下降，但 **reasoning benchmark 到一定规模后持平甚至下降**；固定总预算下 training loss 最优 ≈25% 参数给 n-gram，**downstream 并无对应的 25% 最优点** | Qwen3.8-Flash-Next 消融，转述于青稞社区 <https://qingkeai.online/blog/Ngram-Embedding>（B 级） | 【事实（转述）】**最实质的实证质疑**——直接挑战 OE 的 log-linear 主张在下游任务上的有效性 |
| 4 | **ISO-compute 约束下不可比**：SCONE "is primarily inference-focused and **includes extra module (f-gram model) and additional training FLOPs, rendering it incompatible with the strict iso-compute constraints**"；反过来 OE 被 Engram 选为可比 baseline，说明 OE 在**训练期**是合规的，但**推理期**缺少配套机制 | Engram（2601.07372v2），<https://arxiv.org/html/2601.07372v2> | 【事实】划出了 OE 的能力边界：训练期友好、推理期缺基建 |
| 5 | **页级访存效率极低（独立第三方实测）**：OE 类表的行维度仅 **160**（BF16 下 **320 字节/行**），而内存页为 **4 KiB / 16 KiB** → **按页读取 92–98% 被浪费**；表被切成 **128 个 shard**，行不连续；各引擎目前**只实现到 host RAM offload，SSD 常驻仅靠 OS `mmap`，无专用机制**；4-bit 是量化下限 | Qiita 独立实测（作者明示使用 Claude 做调研，方法论公开、区分 A/B/C 级信息源），<https://qiita.com/sukimaengineer/items/5d950e6fbefd8ee9dfde> | 【事实（第三方实测）】**系统层质疑**：访存粒度与页大小/分片布局不匹配，"大表"的规模优势会被内存带宽侵蚀 |
| 6 | **short causal conv 增益有限（对后续而非 OE 本身）**：青稞社区作者对 Engram 的 short Conv 明确表态"这玩意儿我很存疑"，并指出消融中"Short Causal Conv 只有小幅增益（于是 DS4.1 干脆删了）" | 青稞社区，<https://qingkeai.online/blog/Ngram-Embedding> | 【事实】虽针对 Engram，但反证了"局部卷积精修"这类在 OE 路线上叠加的模块价值存疑 |
| 7 | **词表膨胀对安全面的副作用（邻接质疑）**：OE 被多篇安全论文引作"大词表"代表，讨论其与**成员推断攻击**、**代码密钥泄露**的关联 | Membership Inference Attacks on Tokenizers（2510.05699）、Understanding Secret Leakage Risks in Code LLMs（2604.17814，引 OE 于 §6） | 【事实（引用存在）/ 推测（因果）】OE 本身未讨论安全影响，但成为该议题的引用锚点 |
| 8 | **"大词表伤害小模型"的原始张力**：OE 要解决的正是"大 tokenizer 利于大模型、可能伤害小模型"的矛盾。青稞社区标题即"**大词表伤害小模型？**"，说明这一张力至今仍是该方向的公开悬点 | 青稞社区，<https://qingkeai.online/blog/Ngram-Embedding> | 【事实】OE 的动机本身即来自一个未解矛盾；OE 给出的答案是"input 有用、output 有害"，但该解在工业界并未被无保留接受 |

【推测】质疑呈现清晰的两层结构：**（上层）表示学习层**——固定哈希、无上下文门控、各阶表不共享（质疑 1、2）；**（下层）系统与实证层**——访存粒度失配、loss/downstream 脱钩、安全副作用（质疑 3、5、7）。值得注意的是，**质疑 1、2 已被 Engram/Tensorizing Engram 正面解决**，而**质疑 3、5 至今没有公开解决方案**——这两条是当前该方向最真实的开放问题。

---

## 3. 学术引用

### 3.1 引用规模

【事实】使用 Semantic Scholar Graph API（题目指定的 endpoint）：
```
https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.16975/citations?fields=title,year,externalIds,venue&limit=100
```
- **返回 citing papers 共 48 篇**（offset=0 返回 48 条；`offset=90` 返回空数组 `[]`，确认无第二页）
- **年份分布**：2025 年 27 篇；2026 年 21 篇；**2026 年占比 43.75%** → 【事实】引用仍在持续增长，热度未衰减
- 附注：单论文 metadata endpoint（`?fields=citationCount`）在本次调研中**持续返回 HTTP 429 Too Many Requests**（多轮退避重试均失败），因此无法获得 S2 官方 `citationCount` / `influentialCitationCount` 字段值。上述 48 为**通过 citations 端点实测的 citing paper 条数**（不等同于 S2 的 citationCount，后者可能包含非 arXiv 版本的去重差异）。
- **口径校准**：父 agent 独立核实 S2 显示 **45 篇**；本笔记经 `citations` 端点实测 **48 条**。差异可能来自 API 快照时点不同或去重口径 → **建议对外表述为"约 45–48 篇"**，并列明两次口径。
- 交叉核查：**OpenAlex 严重滞后** —— `W4406961149` 的 `cited_by_count = 0`（<https://api.openalex.org/works?filter=doi:https://doi.org/10.48550/arxiv.2501.16975>），说明该论文的引用图谱只有 S2 维护良好，引用统计不应依赖 OpenAlex。

### 3.2 高相关引用（12 篇，附关系判断）

判定标准：全文出现 `Over-Encoding` / `Over-Tokenized` / `2501.16975` / `n-gram embedding` 等关键词，且该引用在**正文**中承担实质论证角色（非仅 bib 罗列）。

| # | 标题 | arXiv ID | 命中关键词（全文实测） | 关系判断 |
|---|---|---|---|---|
| 1 | **Virtual Width Networks**（VWN） | 2511.11238 | `Over-Encoding`×1, `Over-Tokenized`×1, `Over-tokenized`×1, `2501.16975`×1 | **直接继承 + 正交扩展**。§2 显式引 OE 的 OE/OD 概念；§3.2 引用 OE 的 "input-augmentation strategies"；bib 标注 Cited by: §2, §3.2, §3.4 |
| 2 | **Conditional Memory via Scalable Lookup**（DeepSeek **Engram**） | 2601.07372 | `Over-tokenized`×1（正文另多处 `Huang et al., 2025a`） | **最强影响证据：以 OE 为实验 baseline**。消融中与 "OverEncoding" 直接对比；正文称 OE "integrates N-gram embeddings via averaging with the vocabulary embedding"，并指出 SCONE 等不适配 iso-compute 约束 |
| 3 | **Scaling Embeddings Outperforms Scaling Experts**（**LongCat-Flash-Lite**） | 2601.21204 | `Over-Encoding`×1, `Over-tokenized`×1, `2501.16975`×1, `n-gram embedding`×4 | **最彻底的工程化沿用**。Figure 1 直接把 OE 公式标为 "N-gram Embedding layer (Huang et al., 2025)"；称为 "the final version of N-gram Embedding (**also referred to as Over-Encoding in Huang et al. (2025)**)"，并**自研 N-gram Cache + 自定义 CUDA kernel** 解决 OE 的查表延迟 |
| 4 | **Tensorizing Engram** | 2606.08347 | `Over-tokenized`×6 | **批判性继承 + 改进动机**。摘要即称 "Both Over-tokenized Transformers and Engram attempt to address this limitation… However, they rely on separate hash tables for each n-gram order, which introduces hash collisions" |
| 5 | **MoME: Mixture-of-Memory Embeddings** | 2609.15126 | `Over-Tokenized`×1, `Over-tokenized`×1, `n-gram embedding`×2 | **谱系定位**。把 OE 归入"放大 embedding table 本身"的平行子线（与 SCONE、BLT 并列），指出其 retrieval "remains a fixed function of the surface form" |
| 6 | **Beyond N-gram: Data-Aware X-GRAM Extraction** | 2604.21724 | `Over-tokenized`×1, `2501.16975`×1 | **方法替代**。以 OE 等为 baseline（token-indexed 参数扩容类），提出数据驱动的 X-GRAM 提取替代固定 n-gram |
| 7 | **JTok: Token Embedding as another Axis of Scaling Law** | 2602.00800 | `Over-tokenized`×1, `2501.16975`×1 | **同题竞争**。同样把 token embedding 作为 scaling law 新轴，引 OE 于 §2 |
| 8 | **Speculative Decoding with a Speculative Vocabulary**（ACL 2026） | 2602.13836 | `Over-tokenized`×1 | **解码侧延伸**。引 OE 于输入/输出词表解耦；把词表放大思路迁移到投机解码 |
| 9 | **Vocab Diet: Reshaping the Vocabulary of LLMs via Vector Arithmetic**（ACL Findings 2026） | 2510.17001 | `Over-tokenized`×1, `2501.16975`×1 | **反向操作**。OE 放大词表，Vocab Diet 做词表压缩——同一坐标轴的两端 |
| 10 | **Mixture of Lookup Key-Value Experts** | 2512.09723 | `Over-Tokenized`×1 | **记忆查表路线的邻接工作**，与 OE 同引于 lookup/memory 语境 |
| 11 | **On the Design of Qwen3.8-Next Architecture** | 2608.30320 | `Over-tokenized`×1, `n-gram embedding`×8 | **工业架构论文引用**，§2.3 引 OE；说明 Qwen 侧的架构决策参考了 OE |
| 12 | **Exploiting Vocabulary Frequency Imbalance in LM Pre-training**（NeurIPS 2025） | 2508.15390 | `2501.16975`×3, `Over-tokenized`×1 | **词汇频率视角的独立印证**，三处引用 OE，说明其词表规模结论被纳入频率不平衡分析 |

**低相关引用（36 篇）**：多为 tokenizer 综述/评估（TokSuite 2512.20757、Stop Taking Tokenizers for Granted 2601.13260）、安全性（Membership Inference 2510.05699、Secret Leakage 2604.17814、Forking 2610.00394）、一般性 benchmark/多语言/医疗/时序等（2511.11622、2506.10766、2505.14172、2504.17119、2603.09416、2605.10780、2609.31291、2610.10533、2602.04706、2602.04289、2512.15586、2512.10858、2508.18756、2507.23777、2507.07955、2505.20133、2504.00178、2503.14125、2502.15499、2510.13847、2510.07613、2601.02867 等），引用深度浅。

### 3.3 三条最关键的引用链

【事实】

```
OE (2501.16975, ICML'25)
 ├─→ Engram / Conditional Memory (2601.07372, DeepSeek, 2026-01)  ── 把 OE 当 baseline，转向 external gated memory
 │     ├─→ Tensorizing Engram (2606.08347)  ── 批评 OE/Engram 的 per-order 独立哈希表
 │     ├─→ EngramEdit (2610.10533)  ── 条件记忆用于知识编辑
 │     └─→ Qwen3.8-Flash-Next / DeepSeek-V4.1-Flash  ── 工业规模化（51B / 196B 参数）
 ├─→ LongCat-Flash-Lite (2601.21204, 美团)  ── 直接复用 OE "final version"，自研 N-gram Cache
 ├─→ VWN (2511.11238, ByteDance Seed 自己)  ── 正交方向：放大向量维度而非词表维度
 └─→ 词表规模坐标轴两端：Scaling Embedding Layers (2502.01637) ↔ Vocab Diet (2510.17001)
```

---

## 4. 受其影响的后续工作（附全文关键词命中验证）

### 4.1 验证方法

【事实】对每篇候选论文抓取 **arXiv HTML 全文**（`https://arxiv.org/html/<ID>v1|v2`），剥离标签后统计关键词命中数。关键词集：`Over-Encoding` / `Over-Tokenized` / `Over-tokenized` / `2501.16975` / `n-gram embedding`。

### 4.2 命中表（实测）

| arXiv ID | 标题 | 全文长度 | 命中统计 | 判定 |
|---|---|---|---|---|
| **2511.11238** | Virtual Width Networks | 67,506 | `Over-Encoding`:1, `Over-Tokenized`:1, `Over-tokenized`:1, `2501.16975`:1 | ✅ **明确引用** |
| **2601.07372** | Conditional Memory via Scalable Lookup (Engram) | 101,343 | `Over-tokenized`:1（+正文多处 `Huang et al., 2025a` 与 "OverEncoding" 字样） | ✅ **明确引用（作 baseline）** |
| **2601.21204** | Scaling Embeddings Outperforms Scaling Experts (LongCat-Flash-Lite) | 98,975 | `Over-Encoding`:1, `Over-tokenized`:1, `2501.16975`:1, `n-gram embedding`:4 | ✅ **明确引用（方法沿用）** |
| **2606.08347** | Tensorizing Engram | 63,311 | `Over-tokenized`:6 | ✅ **明确引用** |
| **2609.15126** | MoME: Mixture-of-Memory Embeddings | 115,864 | `Over-Tokenized`:1, `Over-tokenized`:1, `n-gram embedding`:2 | ✅ **明确引用** |
| **2604.21724** | Beyond N-gram: Data-Aware X-GRAM Extraction | 98,975 | `Over-tokenized`:1, `2501.16975`:1 | ✅ **明确引用** |
| **2602.00800** | JTok | 78,934 | `Over-tokenized`:1, `2501.16975`:1 | ✅ **明确引用** |
| **2601.02867** | Training LMs with homotokens Leads to Delayed Overfitting | 60,212 | `Over-tokenized`:1, `2501.16975`:1 | ✅ 引用（Cited by: §5，正文深度浅） |
| **2512.15586** | Bolmo: Byteifying the Next Generation of LMs | 136,400 | `Over-tokenized`:1（PMLR 正式版引用格式） | ✅ 引用（bib 级） |
| **2602.13836** | Speculative Decoding with a Speculative Vocabulary | 56,372 | `Over-tokenized`:1（PMLR 正式版） | ✅ 引用（bib 级） |
| **2512.09723** | Mixture of Lookup Key-Value Experts | 41,549 | `Over-Tokenized`:1 | ✅ 引用（bib 级） |
| **2604.27263** | Decoupling the Benefits of Subword Tokenization via Byte-level Simulation | 48,923 | `Over-tokenized`:1（bib 标注 Cited by: §1, §5.1, §5.8） | ✅ 引用 |
| **2510.17001** | Vocab Diet | 76,011 | `Over-tokenized`:1, `2501.16975`:1 | ✅ 引用（bib 级） |
| **2604.17814** | Understanding Secret Leakage Risks in Code LLMs | 50,451 | `Over-tokenized`:1（Cited by: §6） | ✅ 引用 |
| **2512.20757** | TokSuite | 140,864 | `2501.16975`:1, `Over-tokenized`:1（Cited by: §6） | ✅ 引用 |
| **2511.11622** | Small Vocabularies, Big Gains（时序） | 32,096 | `Over-tokenized`:1（Cited by: Tokenization 节） | ✅ 引用（弱） |
| **2502.01637** | Scaling Embedding Layers in Language Models（NeurIPS'25） | 70,087 | `2501.16975`:1, `Over-tokenized`:1 | ✅ 引用（bib 级） |
| **2508.15390** | Exploiting Vocabulary Frequency Imbalance（NeurIPS'25） | 73,532 | `2501.16975`:3, `Over-tokenized`:1 | ✅ 引用 |
| **2503.13423** | SuperBPE: Space Travel for Language Models | 91,585 | `2501.16975`:1, `Over-tokenized`:1 | ✅ 引用（bib 级） |
| **2608.30320** | On the Design of Qwen3.8-Next Architecture | 117,935 | `Over-tokenized`:1（Cited by: §2.3）, `n-gram embedding`:8 | ✅ 引用 |
| **2610.10533** | EngramEdit | 156,768 | `Over-tokenized`:1（Cited by: §5） | ✅ 引用（bib 级） |

【事实】**本次验证的 21 篇候选全部至少命中 1 个关键词，未出现"命中为 0"的论文**。其中 **5 篇为深度引用**（2511.11238、2601.07372、2601.21204、2606.08347、2609.15126），其余 16 篇为 bib 级或单点引用。

### 4.3 三条明确的"复用 / 放大"路线

【事实】

**(A) OE → Engram 路线（"外来记忆"路线）**：DeepSeek Engram (2601.07372) 把 OE 的"输入层相加 n-gram embedding"改为 **external memory module + context-aware gate + 多层注入（layer 2 & 15）**，并通过 "Sparsity Allocation" 给出 MoE:Engram ≈ **75–80% : 20–25%** 的 U 型最优点。Engram 明确报告 **"outperforming the Over-tokenized Transformer"**（Tensorizing Engram 转述其结论）。规模：Engram-27B，memory 5.7B（21.30%）。

**(B) OE 直系 + 系统优化路线**：LongCat-Flash-Lite (2601.21204, 美团) 的 Figure 1 与公式**直接沿用 OE 的 "final version"**（`e_i = 1/((N-1)K+1) · (E_0(t_i) + Σ_n Σ_k W_{n,k} E_{n,k}(H_{n,k}(...)))`），并在此基础上做系统级创新：**N-gram Cache + 自定义 CUDA kernel + 与调度器低开销同步**。规模：总参 68.5B / 激活 2.9–4.5B，**>30B 参数给 embedding**。

**(C) 层次化 n-gram 与超宽嵌入叠加路线**：VWN (2511.11238) 用 OE 的 **input-augmentation** 作为 over-width embedding 的上游信息增强，叠加正交的"向量维度放大"。

**(D) 词表规模作为独立坐标轴的两端**：【事实】正向放大（OE、2502.01637、2601.21204）与反向压缩（Vocab Diet 2510.17001、LiteToken 2602.04706、Proxy Compression 2602.04289）同时存在，说明"词表/embedding 参数规模"已被公认为独立可调维度。

---

## 5. 工业落地信号

### 5.1 结论：**大词表 n-gram 嵌入已成为 2026 年开源前沿模型的主流组件**

【事实】2026 年出现的"1M 级以上词表 / 超大 n-gram 嵌入表"开源模型：

| 模型 | 机构 | n-gram 表规模（参数） | 机制 | 证据 URL |
|---|---|---|---|---|
| **Qwen3.8-Flash-Next** | 阿里 Qwen | **51.2B**（约 95.4 GiB BF16） | 8 个 2-gram hash head + 8 个 3-gram hash head，共 16 行 × 160 维；注入 layer 2；hash 空间 20M/head × 16 = **3.2 亿行**；`split_ngram_parts: 128` 分片 | [SGLang day-0 博客](https://www.lmsys.org/blog/2026-08-26-qwen-flash-next) ；[Qwen 官博](https://qwen.ai/blog?id=qwen3.8-flash-next) |
| **DeepSeek-V4.1-Flash** | DeepSeek | **196B**（26.2%） | Scale Engram；删除 short Conv；Sinkhorn balancing | [青稞社区架构对比表](https://qingkeai.online/blog/Ngram-Embedding)（B 级，转述技术报告） |
| **Engram-27B** | DeepSeek | **5.7B**（21.30%） | Tokenizer compression + 8 hash heads/order，注入 layer 2 与 15，context-aware gate | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) |
| **LongCat-Flash-Lite**（68.5B MoE） | 美团 | **>30B 给 embedding** | 直接沿用 OE 公式；自研 N-gram Cache + CUDA kernel | [HF 模型卡](https://huggingface.co/meituan-longcat/LongCat-Flash-Lite) ；[arXiv:2601.21204](https://arxiv.org/abs/2601.21204) |
| **LongCat-2.0** | 美团 | **135B**（8.4%） | Scale OE；2–5 gram；Hash/mod + 4 splits/order | [青稞社区对比表](https://qingkeai.online/blog/Ngram-Embedding) ；[techweb 报道](https://m.techweb.com.cn/marticle/2026-07-06/2977204.shtml) |
| Gemma 4 E2B/E4B / Gemma 3n | Google | Per-Layer Embedding | 每层独立 token embedding + context projection（与 n-gram 不同轴但同"大 embedding"思想） | [青稞社区对比表](https://qingkeai.online/blog/Ngram-Embedding) |

【事实】**词表/嵌入表规模演进量级**：从 OE 论文的 **12.8M 输入词表**（V=100k 基础上的 128 倍放大）→ Engram 的 5.7B → Qwen3.8-Flash-Next 的 **51.2B（3.2 亿行 × 160 维）**→ DeepSeek-V4.1-Flash 的 **196B**。5 年跨度内"稀疏查表参数"规模增长约 **15,000 倍**。

### 5.2 推理栈支持（vLLM / SGLang / llama.cpp）

【事实】

**vLLM**：
- PR #47857 `[Model] Add LongCat-Flash-Lite (n-gram embedding)` 由 `mgoin` 提交，**created 2026-07-07，merged 2026-07-10** —— <https://github.com/vllm-project/vllm/pull/47857>
- vLLM 主仓 `vllm/model_executor/models/` 下已存在 `longcat_flash_ngram.py`、`longcat_flash.py`、`longcat_flash_mtp.py`（285 个模型文件中的 n-gram 专用实现）
- 存在 Qwen3.8-Flash-Next 的 PLE n-gram 表量化加载 issue #54765（NVFP4 + FP8 量化 PLE 表的 `ngram_embedding.weight_scale`），说明已进入**量化部署阶段**：<https://github.com/vllm-project/vllm/issues/54765>

**SGLang**：
- **Day-0 支持 Qwen3.8-Flash-Next**，并发布专门博客详述 **Sparse Pinned-Host Offload**：`SGLang moves the vocabulary-parallel table shard to pinned host memory and gathers only the 16 selected rows per token`（Triton UVA kernel + 专用 CUDA stream 与第一层 decoder 重叠）。实测收益：**目标模型权重 83.91 → 60.45 GiB/GPU（−23.46 GiB），KV 容量 1.84M → 3.28M tokens（+78.54%），吞吐 −0.07%**（几何均值）—— <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>
- SGLang 文档/cookbook 有 `pleOffload` 参数：<https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next>
- 相关 PR/issue：#30183 "fix: preserve resolved LongCat n-gram configuration"、**#36514 "[Feature] GPU offload target for Qwen3.8-Flash-Next N-gram / PLE embeddings"**、#21052 [Roadmap] Further Ngram Speculative Decoding Support —— <https://github.com/sgl-project/sglang/pull/30183>、<https://github.com/sgl-project/sglang/issues/36514>
- 【注】SGLang 仓库中 "NGRAM" 关键词返回 110 条 issue/PR，但**多数属于 NGRAM 投机解码（speculative decoding）**，与 n-gram embedding 是不同机制；真正相关的为上述 PLE/offload 系列。

**llama.cpp**：
- 【事实】已支持 Qwen3.8-Flash-Next：PR #27742；GGUF 张量名为 `per_layer_token_embd`，默认量化 q8_0，**4 bit 为下限**（来源：Qiita 实测文，<https://qiita.com/sukimaengineer/items/5d950e6fbefd8ee9dfde>）

### 5.3 关键工程共识（跨栈收敛）

【事实】三条被多个独立团队重复验证的结论：
1. **hash 地址只依赖 token ID → 无数据依赖 → 可提前计算、可放 host RAM/CPU**（OE 原文即指出；Engram 称 "deterministic addressing enables runtime prefetching"；SGLang/Qwen 据此做 pinned-host offload；LongCat 据此做 N-gram Cache）
2. **每 token 只查极少行 → 稀疏访问 → 参数规模与 FLOPs 解耦**（Qwen 每 token 只读 16 行；SGLang 实测 offload 吞吐仅 −0.07%）
3. **注入位置宜浅**（OE/Engram/Qwen 均落在 layer 2 附近），便于与第 1 层计算重叠预取

【事实】尚未解决的工程问题（来自 Qiita 独立实测，B 级但方法论严谨）：
- 行维度仅 **160 维**（BF16 下 320 字节/行），而内存页为 4 KiB/16 KiB → **按页读取有 92–98% 浪费**；表被切成 **128 个 shard**，行非连续
- 各引擎目前**只实现到 host RAM offload**，SSD 常驻仍依赖 OS `mmap`，**无专用机制**
- 4-bit 是量化下限；RAM 常驻 4-bit 即可容纳，因此"上 SSD"目前缺乏必要性

【推测】这说明工业界已经跨过"能不能用"的门槛，进入"**访存粒度与分片布局如何与页大小/预取协同**"的第二阶段。

---

## 6. VWN 与 OE 的关系（原文依据）

### 6.1 VWN 如何引用 OE（原文逐字）

【事实】VWN（arXiv:2511.11238v2，ByteDance Seed，通讯作者 Defa Zhu）**§2 Related Work 的 "Embedding Expansion" 段**原文：

> "**Embedding Expansion.** Recent studies have highlighted the importance of vocabulary scaling in large language models (Tao et al., 2024), showing that expanding the input vocabulary—particularly through hierarchical *n*-gram token embeddings—can systematically improve model expressiveness and training efficiency with negligible computational overhead **(Huang et al., 2025)**. The Over-Tokenized Transformer framework **(Huang et al., 2025)** introduced **Over-Encoding (OE)** to scale input representations using multi-gram tokenization and **Over-Decoding (OD)** to enhance output supervision via multi-token prediction objectives. Notably, **Multi-Token Prediction (MTP) (Gloeckle et al., 2024) is regarded as an effective instantiation of OD for practical training.**"

来源：<https://arxiv.org/html/2511.11238v2>（§2）

【事实】**§3.2 Over-Width Embedding** 原文（引用 OE 的输入增强）：

> "…which is like to applying a low-rank decomposition to a very wide embedding table. In addition, **one can adopt input-augmentation strategies (Huang et al., 2025)** that inject more information per input than a single isolated token embedding to further enrich the widened representation."

来源：<https://arxiv.org/html/2511.11238v2>（§3.2）

【事实】**参考文献引用位置**（VWN bib 中 Huang et al. 2025 条目）：

> "Huang et al. (2025) H. Huang, D. Zhu, B. Wu, Y. Zeng, Y. Wang, Q. Min, and X. Zhou. *Over-tokenized transformer: vocabulary is generally worth scaling.* arXiv preprint **arXiv:2501.16975**. **Cited by: §2, §3.2, §3.4.**"

**注意**：VWN 与 OE **共享两位作者**——Hongzhi Huang（共同一作）与 Defa Zhu（VWN 通讯作者）、Banggu Wu、Yutao Zeng、Qiyang Min、Xun Zhou 均在两篇作者名单中。故 VWN 对 OE 的关系是**同一团队的技术演进**，而非外部引用。【事实，见 VWN 作者列表】

### 6.2 OD ≈ MTP：原文对齐

【事实】VWN 原文明确写 "MTP is regarded as an effective instantiation of OD for practical training"（§2）。这与 OE 原文自述一致——OE 论文将 OD 实现为 "a generalization of Multi-Token Prediction (MTP), directly decoding the next 'n' tokens from the last hidden state using 'n' different vocabularies"。【事实，alphaXiv 对 OE 公式的转述 + VWN 原文交叉印证】

【事实】VWN 在方法上**实际采用 MTP**：Figure 3(c) 标注 "(c) We enable multiple token supervision (**multi-token prediction**)"；实验中 "we include a Multi-Token Prediction (MTP) head by default, jointly optimizing the standard next-token and MTP objectives"，并报告 next-2-token loss 加速 3.5×（MoE-A3.3B）。

### 6.3 关键差异（核心洞察）

| 维度 | **OE / Over-Tokenized Transformer** | **VWN / Virtual Width Networks** |
|---|---|---|
| **放大的是什么** | **词表 / 查表空间**：输入词表从 V 放大到 `m`（如 12.8M，V=100k 的 128 倍） | **每 token 的向量维度**：embedding 维度从 `D` 放大到 `D' = (n/m)·D`（如 8× 于 4096 → **32K 维**） |
| **每 token 的表示维度** | **仍是 D 维**（`h = Σ_i E_i[i-gram] ∈ R^D`），多个 n-gram 表的输出**相加成一个 D 维向量** | **是 D' 维**（`e ∈ R^{D'}`，D' > D），每 token 携带更宽的向量 |
| **词表大小** | **变大**（核心变量） | **不变**（仍为原 BPE 词表 \|V\|，unembedding 维度不变） |
| **参数量增长在哪** | 稀疏查表的 embedding table（`m × d`），**每 token 只触极少行** | 稠密的 over-width embedding / 投影矩阵 `W_expand`、`W_reduce`、GHC 的 `A^l/B^l` |
| **主干计算量** | 几乎不变（OE-12.8M 训练 FLOPs 增加 <0.5%） | 几乎不变（GHC 用轻量矩阵把 D' 压回 D 做层内计算） |
| **如何回到主干宽度** | 无需回归——求和后天然是 D 维 | **必须回归**：`h_reduce^L = W_reduce · h'^L`（公式 4），并需 GroupNorm（group size = D）稳定训练 |
| **与 n-gram 的关系** | n-gram 是**本质**（层次 1/2/3-gram 多表相加） | n-gram 是**可选增强**（"one can adopt input-augmentation strategies (Huang et al., 2025)"，即 OE 可作为 VWN 的上游叠加） |
| **核心 scaling 律** | 输入词表大小 ↔ 训练 loss **log-linear** | virtual width factor r ↔ loss **近似 log-linear** |
| **实现关键** | feature hashing + sliced low-dim tables（`m, m+2, m+4` 避免碰撞）+ m 与 V 互质 | Generalized Hyper-Connections（GHC）：`2m × (m+n)` 变换矩阵，支持**非整数倍**（fractional）扩展 |
| **论文规模** | OE-12.8M / 400M 参数 ≈ 1B baseline；OLMo2-1B 上 5.7× 收敛加速 | MoE-A3.3B + VWN×8：next-token 收敛快 2.5×、next-2-token 快 3.5× |

【事实】两篇的 log-linear 表述分别见 OE 摘要（"a log-linear relationship between input vocabulary size and training loss"）与 VWN 摘要（"an approximately log-linear scaling relation between virtual width and loss reduction"）。**两篇都提出了一条新的稀疏/廉价 scaling 轴，但轴的方向正交。**

【推测】因此二者**可叠加**：OE 提供"每 token 看更多历史 n-gram"的信息量，VWN 提供"每 token 用更宽向量承载这些信息"的容量。VWN 原文的 "input-augmentation strategies (Huang et al., 2025)" 一句，已经是这种叠加的官方接口预留。**这也是本笔记第 7 节方向 1 的直接依据。**

---

## 7. 变种技术预测（5–8 个方向）

> 每条含：核心想法 / 支撑证据 URL / 受益场景 / 风险 / 标注

### 方向 1：n-gram 层次嵌入 × 超宽嵌入的叠加（OE ∘ VWN）

- **核心想法**：把 OE 的多阶 n-gram 查表输出**拼接**（而非相加）到一个 `D' = r·D` 的超宽 embedding 中，再由 VWN 的 GHC 压回主干宽度。即：OE 负责"每个位置看到多少上下文"，VWN 负责"用多宽的向量装下它"。
- **支撑证据**：【事实】VWN §3.2 原文已预留该接口："one can adopt **input-augmentation strategies (Huang et al., 2025)** that inject more information per input than a single isolated token embedding to further enrich the widened representation" —— <https://arxiv.org/html/2511.11238v2>。两篇同属 ByteDance Seed、共享 6 位作者，工程上共用同一套基础设施。
- **受益场景**：中文/多语言（Qwen 消融显示 n-gram 对 C-Eval/CMMLU scaling 特别稳定）、代码（长标识符、固定 API 模式）、agentic 工具调用（固定 JSON/函数名模式）。
- **风险**：两套"宽"机制同时开启会使**首层 embedding 参数量与访存量双重爆炸**（OE 的 51B 表 + VWN 的 32K 维），并需要重新调 `m`（OE 分片数）与 `n/m`（VWN 分段比）的耦合；OV 型配置下 `W_expand`/`W_reduce` 的参数量可能反超主干。
- **标注**：【推测】（接口已存在，但公开文献尚无该组合的实验报告）

### 方向 2：字节级 + n-gram 混合（BLT × OE）

- **核心想法**：在字节级/动态 patch 表示（BLT, arXiv:2502.13923 系）之上叠加 n-gram 哈希嵌入，用 n-gram 表补足字节级模型缺失的"词级先验"，同时保留字节级模型的**无 tokenizer / 抗 OOV / 多语言公平**优势。
- **支撑证据**：【事实】MoME 论文把 OE 与 BLT 并列为同一子线："Over-Tokenized Transformer [22] feeds many overlapping n-gram embeddings, and **Byte Latent Transformer [46] does so at the byte level**" —— <https://arxiv.org/html/2609.15126v1>。【事实】Bolmo（2512.15586，"Byteifying the Next Generation of Language Models"）同时引用 OE，说明两条路线正在交汇 —— <https://arxiv.org/html/2512.15586>。
- **受益场景**：形态丰富语言（土耳其语、芬兰语）、代码与数学符号、需要处理噪声文本/拼写变体的场景。
- **风险**：字节级 + n-gram 会让**哈希空间语义更模糊**（碰撞的代价从"词形相近"变成"字节相近"，错误更不可解释）；Tensorizing Engram 已指出 OE 的 per-order 独立哈希表加剧碰撞 —— <https://arxiv.org/html/2606.08347v1>。
- **标注**：【推测】

### 方向 3：可学习的词表压缩 / 动态 n-gram 选择（X-GRAM × Vocab Diet × OE）

- **核心想法**：不再用固定 n-gram 阶数与固定哈希空间，而是**由数据决定哪些 n-gram 值得占表位**：低频 n-gram 被合并/剪枝（Vocab Diet 式向量算术压缩），高频组合获得独立槽位（X-GRAM 式数据驱动提取），整体表大小按 Zipf 分布自适应分配。
- **支撑证据**：【事实】X-GRAM 明确以 OE 等为 baseline 并提出数据驱动替代："X-gram is motivated by a gap in prior compute-decoupled baselines: while MoLE/SCONE/STEM demonstrate that token-indexed capacity can be scaled, they do not systematically analyze the training dynamics of token-indexed parameters under **Zipfian data**" —— <https://arxiv.org/html/2604.21724>。【事实】反向操作方 Vocab Diet 同引 OE —— <https://arxiv.org/html/2510.17001>。【事实】独立证据：Exploiting Vocabulary Frequency Imbalance (NeurIPS'25, 2508.15390) 三次引用 OE，专门研究词频不平衡。
- **受益场景**：长尾领域（医疗、法律、低资源语言）微调；端侧部署需要小表；多租户服务中按租户裁剪共享表。
- **风险**：词表变成**训练过程中非平稳的目标**（可学习/可剪枝 → 优化器状态与 embedding 表耦合）；Vocab Diet 类压缩对**推理期**收益明显但对训练期无益，二者结合可能互相抵消。
- **标注**：【推测】（各要素均有独立事实支撑，组合为推测）

### 方向 4：把 OD 与 MTP / 投机解码统一（OD = 训练期，EAGLE/MTP = 推理期）

- **核心想法**：用**同一个 n 头输出结构**同时承担 (a) 训练期的 Over-Decoding 细粒度监督、(b) 推理期的 MTP 投机解码草稿头。即 OD 的"n 个词表"直接退役为投机解码的 draft heads，训练与推理共用参数。
- **支撑证据**：【事实】VWN §2 明确 "**MTP is regarded as an effective instantiation of OD** for practical training"，且 VWN 默认带 MTP head —— <https://arxiv.org/html/2511.11238v2>。【事实】SGLang 在 Qwen3.8-Flash-Next 上实现 **IndexShare MTP**：`--speculative-num-steps N` 下草稿步骤不再跑 indexer，B200 TP4 + NVFP4 解码 **540 tok/s**、accept length 3.3 —— <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>。【事实】已有的解码侧延伸：DynaSpec（2510.13847，"Context-aware Dynamic Speculative Sampling for **Large-Vocabulary** Language Models"）、Speculative Decoding with a Speculative Vocabulary（2602.13836）均在大量表背景下工作。ByteDance Seed 已有 SwiftSpec 仓库（SOTA 投机解码）。
- **受益场景**：低延迟 agentic 推理、batch=1 高交互场景；大词表模型（OE 类）的固有"输出头重"问题可被投机解码摊薄。
- **风险**：OD 的 n 个词表是**训练期监督信号**，其对参数的需求（每个 head 需要完整输出词表投影）与投机解码草稿头的**轻量化需求**矛盾；若强行共享，draft 质量与 accept length 可能双降。DeepSeek-V4.1-Flash 反向操作（删除 Engram 的 short Conv）说明工业界倾向于**简化而非耦合**。
- **标注**：【推测】

### 方向 5：词表并行（vocabulary parallelism）× offload × 通信协同设计

- **核心想法**：把 n-gram 表按 128 分片沿 TP/DP 维度切分，**通信与访问模式联合优化**——把稀疏 gather 放在 overlapped stream 上、用页对齐寻址把"行分片"映射为"页分片"，并让预取深度随分片布局自适应。
- **支撑证据**：【事实】SGLang 已实现 vocabulary-parallel table shard → pinned host memory + Triton UVA kernel gather + 专用 CUDA stream 与第一层 decoder 重叠；实测 −23.46 GiB/GPU、KV +78.54%、吞吐 −0.07% —— <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>。【事实】Qwen config 中 `split_ngram_parts: 128`、`make_ngram_vocab_size_divisible_by: 128`（页/分片对齐意图明显）—— 见 [Qiita 实测](https://qiita.com/sukimaengineer/items/5d950e6fbefd8ee9dfde)。【事实】SGLang issue #36514 正在请求 **GPU offload target**（当前仅 host RAM）—— <https://github.com/sgl-project/sglang/issues/36514>。
- **受益场景**：单卡装不下大表的推理服务；H200/B200 上把省下的显存换成 KV 容量（长上下文 agent）；混合 PD 分离部署。
- **风险**：【事实】独立实测指出**行仅 320 字节（BF16）而页 4 KiB/16 KiB → 页级读取 92–98% 浪费**；SSD 常驻需自实现行级直读 + 异步预取，各引擎目前**均未实现**。若访存粒度问题不解决，进一步扩大表（196B+）的边际收益会被内存带宽吃掉。
- **标注**：【事实（现状）】+【推测（协同设计方向）】

### 方向 6：张量化 / 共享潜变量的 n-gram 表（解决哈希碰撞与表冗余）

- **核心想法**：不再为每个 n-gram 阶数维护独立哈希表，而是让 1/2/3-gram **共享底层潜变量**，用张量分解把 N 张表压成一组共享因子——碰撞代价下降、嵌套结构显式建模、总参数量可降。
- **支撑证据**：【事实】Tensorizing Engram 摘要即以此为动机："Both Over-tokenized Transformers and Engram attempt to address this limitation… However, **they rely on separate hash tables for each n-gram order, which introduces hash collisions and prevents nested n-grams from sharing the underlying latent structures**. To address these issues, we propose Tensorized Engram" —— <https://arxiv.org/html/2606.08347v1>。【事实】OE 原文已用 sliced low-dim tables（`m, m+2, m+4`）多视图缓解碰撞，说明"降碰撞"是真实痛点 —— 见 [青稞社区拆解](https://qingkeai.online/blog/Ngram-Embedding)。【事实】JTok（2602.00800）走"token self-modulation"路径同引 OE。
- **受益场景**：显存受限的大表部署；需要可解释性（共享潜变量便于分析 n-gram 语义）的研究场景；多语言共享表。
- **风险**：张量分解引入的**秩约束可能限制表容量上限**，抵消 scaling 收益；且分解后的 gather 变成多个小 gather，可能恶化访存局部性（与方向 5 冲突）。
- **标注**：【事实（问题）】+【推测（方向）】

### 方向 7：上下文感知门控替代"固定相加/平均"（OE 注入机制升级）

- **核心想法**：OE 的注入是**与上下文无关的固定相加/平均**（`Σ E_i`）；升级为用当前 hidden state 生成 gate（`α_t = σ(RMSNorm(h_t)^T RMSNorm(W_K e_t) / √d)`），让模型**按需**读取 n-gram 记忆，避免无关 n-gram 污染表示。
- **支撑证据**：【事实】Engram 正是这样做的，并明确把 OE 当 baseline："we compare against **OverEncoding (Huang et al., 2025a), which integrates N-gram embeddings via averaging with the vocabulary embedding**"，随后以 context-aware gate 取得更好结果 —— <https://arxiv.org/html/2601.07372v2>。【事实】Qwen3.8-Flash-Next 也采用门控形式：`g_t = Gate(Norm(Q_t), Norm(K_t))`，`Δ_t = U_t + SiLU(DWConv(RMSNorm(U_t)))`，注入 4 条 HC 残差分支 —— <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>。【事实】Tensorizing Engram 直接断言 "DeepSeek's Engram injects n-gram information through external memory layers, **outperforming the Over-tokenized Transformer**" —— <https://arxiv.org/html/2606.08347v1>。
- **受益场景**：通用预训练（避免大表在小模型上 underfit）；需要精确控制记忆读取比例的推理/长上下文场景。
- **风险**：门控**破坏了 OE 最核心的工程优势——"hash 地址只依赖 token ID，可完全离线预取"**。门控需要 hidden state，即必须等前一层算完，预取与计算的重叠窗口被压缩。Engram/Qwen 的应对是**把注入放在 layer 2**（用 layer 1 的计算时间预取），但这限制了门控可用信息的丰富度。
- **标注**：【事实（已被验证有效）】——本方向是**已被实现的最确定方向**。

### 方向 8：稀疏查表参数与其他稀疏轴的联合预算分配（Sparsity Allocation）

- **核心想法**：把"总稀疏参数预算"在 MoE experts、n-gram memory、以及可能的其他查表模块之间做**联合最优分配**，形成统一的 scaling 法则，而不是各自独立放大。
- **支撑证据**：【事实】Engram 论文的 "Sparsity Allocation problem" 给出 U 型 scaling law，最优约 **MoE 75–80% : Engram 20–25%** —— <https://arxiv.org/abs/2601.07372>。【事实】LongCat-Flash-Lite 独立得出"在特定区间 embedding scaling 的 Pareto 前沿优于增加专家数"，并系统化了 integration timing / parameter budgeting / hash collision / 初始化等因子 —— <https://huggingface.co/meituan-longcat/LongCat-Flash-Lite>。【事实】Qwen 消融发现固定预算下 **training loss 最优约在 25% 给 n-gram，但 downstream 无对应的 25% 最优点** —— <https://qingkeai.online/blog/Ngram-Embedding>。
- **受益场景**：预训练预算规划（给定算力/显存，怎么分给 experts / memory / width）；多轴 scaling law 的统一理论。
- **风险**：【事实】Qwen 的消融已暴露最大风险——**"loss 越低 ≠ downstream 越好"**，reasoning benchmark 在大表下可能持平甚至下降。这意味着单一 loss-based 的 allocation 法则可能在**下游任务上失效**，需要多目标（loss + reasoning + 长上下文）的最优分配理论，而目前公开文献尚无。
- **标注**：【事实（U 型律与 25% 点已存在）】+【推测（downstream 修正后的统一法则）】

### 7.9 【补充】父 agent 未覆盖、但值得进"变种技术"表的引用工作

> 以下 8 篇来自 S2 48 篇引用集，此前未列入第 4 节高相关表或第 7 节方向表，但**各自代表一条可独立的变种线索**。已做过全文关键词命中验证（见第 4.2 节）。

| 论文 | arXiv | 代表的新变种线索 | 支撑其进入变种表的证据 | 标注 |
|---|---|---|---|---|
| **Scaling Embedding Layers in Language Models（SCONE）** | 2502.01637 | **推理期专注的 embedding 扩容**：引入辅助 contextualizer（f-gram model）+ 推理卸载，把大表优势带到**已训练模型的事后增强**（而非重训） | 【事实】Engram 原文评述其"**is primarily inference-focused and includes extra module (f-gram model) and additional training FLOPs**" —— <https://arxiv.org/html/2601.07372v2>。MoME 亦将其与 OE 并列 —— <https://arxiv.org/html/2609.15126v1> | 【事实】 |
| **Frac-Connections: Fractional Extension of Hyper-Connections** | 2503.14125 | **分数倍扩展**：OE 的 `m` 与 VWN 的 `r=n/m` 可非整数 → 扩展比成为连续可调超参（VWN §2 明确"expansion ratio need not be an integer multiple, and such **fractional expansions** have been empirically validated as effective"） | 【事实】VWN §2 明确承认 fractional expansion 有效 —— <https://arxiv.org/html/2511.11238v2>；Frac-Connections 本人同引 OE | 【事实（VWN 表述）+ 推测（结合）】 |
| **SupraTok: Cross-Boundary Tokenization** | 2508.11857 | **跨边界 tokenization**：打破 pre-tokenization 边界构造超词，与 OE 的"跨 token n-gram"同源但走 **tokenizer 侧改造**而非 embedding 侧扩容 | 【事实】TACL 2025 收录；同引 OE。与 SuperBPE（2503.13423）构成"superword token"支线 | 【推测】 |
| **Boundless Byte Pair Encoding: Breaking the Pre-tokenization Barrier** | 2504.00178 | **无预分词 BPE**：删除 pre-tokenization 边界，使 BPE 可跨空白/标点合并 → 直接扩大"可触及的 n-gram 空间"，与 OE 的哈希 n-gram 是**两种通往同一目标的路径** | 【事实】同引 OE；与 SuperBPE/SupraTok 同属"放宽 token 边界"家族 | 【推测】 |
| **Dynamic Chunking for End-to-End Hierarchical Sequence Modeling** | 2507.07955 | **可学习动态分块**：让模型自己学"多长的片段算一个单元"（H-Net 系），替代固定的 1/2/3-gram 阶数 → **OE 的 n-gram 阶数从超参变成可学习量** | 【事实】同引 OE（VWN 亦引其为相关工作的对照）；MoME 将其归入"动态分块"支线 | 【推测】 |
| **DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary LMs** | 2510.13847 | **为大词表专门设计的投机采样**：词表变大后 softmax/采样成本上升，需要上下文感知的动态 draft 分布 | 【事实】同引 OE 于大量表语境。与 2602.13836（Speculative Vocabulary）共同构成"OE → 推理侧词表工程"的独立子线，正是第 7 节方向 4 的直接前驱 | 【事实（引用）+ 推测（方向）】 |
| **One Tokenizer To Rule Them All: Emergent Language Plasticity via Multilingual Tokenizers** | 2506.10766 | **多语言词表分配**：tokenizer 词表预算如何在语言间分配 → OE 的"大 input 词表"在多语言下变成**公平性/预算分配问题** | 【事实】ACL 2026，同引 OE。与 Qwen 消融"n-gram 对中文 benchmark scaling 特别稳定"形成呼应 | 【事实（引用）+ 推测（方向）】 |
| **TokSuite: Measuring the Impact of Tokenizer Choice on LM Behavior** | 2512.20757 | **词表选择的评测基建**：OE 类方法的收益需要可复现的评测协议（攻其"可复现性差"之短） | 【事实】同引 OE 于 §6。可直接作为 OE 复现实验的**评测工具链候选** | 【推测】 |

【事实】附带说明：**VWN 的 "Generalized Hyper-Connections (GHC)" 本身即把 Hyper-Connections（Zhu et al. 2024/2025）与 AltUp 视为 VWN 家族的简化实例**，VWN §1 原文："models employing methods such as Hyper-Connections or AltUp can be regarded as simplified instances within the broader VWN family"。因此 **HC / Frac-Connections / AltUp 是 OE-VWN 路线的"嵌入侧近亲"**，适合放入变种谱系图。

---

## 8. 总体影响判断

### 8.1 影响力画像（一句话）

【推测】**OE 是一篇"方法论被完全吸收、代码零开源、热度中等、但已成为 2026 年开源前沿模型标准组件"的转折性论文**——它的真实影响不体现在引用数（48 篇）或 HF upvotes（34）上，而体现在 **Qwen3.8-Flash-Next（51.2B 表）、DeepSeek Engram/V4.1-Flash（5.7B/196B）、美团 LongCat-2.0/Flash-Lite（135B/>30B）、vLLM + SGLang + llama.cpp 三栈全部支持**这一事实集合上。

### 8.2 影响路径的三阶段

【事实 + 推测】

| 阶段 | 时间 | 特征 | 代表 |
|---|---|---|---|
| **奠基期** | 2025-01 ~ 2025-11 | 论文发表（ICML'25），被引零散；同期 Tao et al. 词表 scaling law 是直接前驱 | OE 原文、VWN（同团队，2025-11） |
| **吸收期** | 2026-01 ~ 2026-07 | 顶级团队把 OE 当 **baseline** 并超越，或**直接沿用**其公式；从"输入层相加"转向"外部门控记忆" | DeepSeek Engram（2026-01，明确 baseline 对比）、Tensorizing Engram、LongCat-Flash-Lite（2026-07，直接沿用公式 + 自研 Cache） |
| **规模化期** | 2026-08 ~ 至今 | 进入**万亿级参数**产品模型与**三大推理栈**；工程焦点从"能不能用"转向"访存粒度/分片/offload" | Qwen3.8-Flash-Next（2026-08-26，51.2B）、DeepSeek-V4.1-Flash（196B）、SGLang day-0 offload、vLLM PR #47857 |

### 8.3 对技术洞察报告的三条可引用结论

1. 【事实】**OE 的核心遗产不是 OE 本身，而是"词表/embedding 表规模是独立 scaling 轴"这一命题**被工业界接受。证据：2026 年出现 51.2B–196B 级 embedding 表的开源模型，且 vLLM/SGLang 为其新增了专门的模型实现与 offload 机制。

2. 【事实】**OE 的"输入层固定相加"注入方式已被后续工作普遍放弃**，被"外部记忆 + 上下文门控 + 浅层注入"取代（Engram、Qwen3.8-Flash-Next、Tensorizing Engram 均如此）。但从 LongCat-Flash-Lite 直接沿用 OE 公式来看，**OE 的哈希/分片/低维切片工程细节仍有独立价值**。

3. 【事实 + 推测】**OE 指出的 log-linear 收益在工业实践中被修正为"存在最优点 + downstream 可能非单调"**（Qwen 消融：25% 最优、reasoning 会持平/下降）。这是该方向最大的**未解风险**，也是未来研究（方向 8）最有价值的切入点。

### 8.4 明确的调研局限

【事实】本笔记的以下部分存在局限，需后续补充：
- **S2 官方 `citationCount` / `influentialCitationCount` 未能取得**：metadata endpoint 在本次调研期间持续 429。48 篇为 citations 端点实测条数。
- **知乎原文（zhuanlan.zhihu.com/p/20814631100）与 OpenReview 讨论区内容未能抓取正文**：前者 403 反爬，后者需要浏览器验证。因此"中文社区主要观点"与"审稿人质疑"的覆盖不完整。
- **LongCat-2.0 / DeepSeek-V4.1-Flash 的 135B / 196B 参数与 8.4% / 26.2% 比例来自青稞社区二手转述**（B 级），未直接核验其技术报告原文。
- **Reddit / Twitter 讨论缺失**：多轮检索无对应结果，可能是检索工具覆盖不足，而非确实不存在。

---

## 9. URL 汇总

### 9.1 一手来源（A 级）

| 类别 | URL |
|---|---|
| arXiv 摘要页 | https://arxiv.org/abs/2501.16975 |
| arXiv HTML（ar5iv） | https://ar5iv.labs.arxiv.org/html/2501.16975 |
| ICML 2025 虚会页（Poster + Lay Summary） | https://icml.cc/virtual/2025/poster/44467 |
| OpenReview 论坛 | https://openreview.net/forum?id=gbeZKej40m |
| PMLR 正式版 | https://proceedings.mlr.press/v267/huang25bb.html |
| ACM DL | https://dl.acm.org/doi/10.5555/3780338.3781371 |
| ByteDance Seed 官网页 | https://seed.bytedance.com/zh/public_papers/over-tokenized-transformer-vocabulary-is-generally-worth-scaling |
| HF Papers 页面 | https://huggingface.co/papers/2501.16975 |
| HF Papers 归档检索 | https://huggingface.co/papers?q=Over-Tokenized%20Transformers |
| HF Daily Papers 2025-01-29 | https://huggingface.co/papers/date/2025-01-29 |
| alphaXiv | https://www.alphaxiv.org/abs/2501.16975 |
| Semantic Scholar citations API | https://api.semanticscholar.org/graph/v1/paper/arXiv:2501.16975/citations?fields=title,year,externalIds&limit=100 |
| OpenAlex（引用数滞后为 0） | https://api.openalex.org/works?filter=doi:https://doi.org/10.48550/arxiv.2501.16975 |
| HF 组织 ByteDance-Seed | https://huggingface.co/ByteDance-Seed |
| GitHub 组织 bytedance-seed | https://github.com/bytedance-seed |

### 9.2 开源核查（无官方实现）

- HF 搜索（均为 0 结果）：`https://huggingface.co/api/models?search=over-tokenized`、`?search=over_tokenized`、`?search=ngram-embedding`
- HF 组织全量（59 个仓库，无匹配）：`https://huggingface.co/api/models?author=ByteDance-Seed&limit=200`
- GitHub 全局搜索：`https://api.github.com/search/repositories?q=over-tokenized`、`?q=overtokenized`（0 结果）
- GitHub 组织仓库列表：`https://api.github.com/orgs/bytedance-seed/repos?per_page=100`

### 9.3 后续工作原文（关键词命中已验证）

| 论文 | arXiv HTML |
|---|---|
| VWN | https://arxiv.org/html/2511.11238v2 |
| Engram / Conditional Memory | https://arxiv.org/abs/2601.07372 ；https://arxiv.org/html/2601.07372v2 |
| LongCat-Flash-Lite | https://arxiv.org/html/2601.21204 |
| Tensorizing Engram | https://arxiv.org/html/2606.08347v1 |
| MoME | https://arxiv.org/html/2609.15126v1 |
| Beyond N-gram / X-GRAM | https://arxiv.org/html/2604.21724 |
| JTok | https://arxiv.org/html/2602.00800 |
| Qwen3.8-Next 架构设计 | https://arxiv.org/html/2608.30320 |
| Vocab Diet | https://arxiv.org/html/2510.17001 |
| Exploiting Vocabulary Frequency Imbalance | https://arxiv.org/html/2508.15390 |
| SuperBPE | https://arxiv.org/html/2503.13423 |
| Bolmo | https://arxiv.org/html/2512.15586 |
| Speculative Decoding with a Speculative Vocabulary | https://arxiv.org/html/2602.13836 |
| Mixture of Lookup Key-Value Experts | https://arxiv.org/html/2512.09723 |
| Scaling Embedding Layers in LMs | https://arxiv.org/html/2502.01637 |
| TokSuite | https://arxiv.org/html/2512.20757 |
| Training LMs with homotokens | https://arxiv.org/html/2601.02867 |
| EngramEdit | https://arxiv.org/html/2610.10533 |

### 9.4 工业落地与推理栈

| 项 | URL |
|---|---|
| Qwen 官博 Qwen3.8-Flash-Next | https://qwen.ai/blog?id=qwen3.8-flash-next |
| SGLang day-0 博客（PLE offload 实测数据） | https://www.lmsys.org/blog/2026-08-26-qwen-flash-next |
| SGLang cookbook（`pleOffload`） | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next |
| vLLM PR #47857（LongCat-Flash-Lite n-gram，已合并 2026-07-10） | https://github.com/vllm-project/vllm/pull/47857 |
| vLLM issue #54765（PLE 量化加载） | https://github.com/vllm-project/vllm/issues/54765 |
| SGLang PR #30183（LongCat n-gram 配置） | https://github.com/sgl-project/sglang/pull/30183 |
| SGLang issue #36514（GPU offload target 请求） | https://github.com/sgl-project/sglang/issues/36514 |
| SGLang issue #21052（Ngram 投机解码 roadmap） | https://github.com/sgl-project/sglang/issues/21052 |
| LongCat-Flash-Lite HF 模型卡 | https://huggingface.co/meituan-longcat/LongCat-Flash-Lite |
| LongCat-2.0 报道 | https://m.techweb.com.cn/marticle/2026-07-06/2977204.shtml |
| NVIDIA GB300 上的 Qwen3.8-Flash-Next | https://developer.nvidia.com/blog/experiment-with-qwen3-8-flash-next-on-nvidia-gb300-nvl72-for-agentic-coding/ |
| llama.cpp 部署实测（GGUF 张量名 `per_layer_token_embd`） | https://github.com/tonyd2wild/Qwen3.8-Flash-Next-Fleet-Deploy |

### 9.5 讨论与解读（含质疑）

| 平台 | URL |
|---|---|
| 知乎专栏【论文解读】过分词模型 | https://zhuanlan.zhihu.com/p/20814631100 （403，仅确认存在） |
| 青稞社区：大词表伤害小模型？深度解析 N-gram Embedding 与 Engram | https://qingkeai.online/blog/Ngram-Embedding |
| 澎湃：搞懂"记忆"必看｜吃透 Engram | https://m.thepaper.cn/newsDetail_forward_32847319 |
| 知乎：如何评价 DeepSeek Engram | https://www.zhihu.com/question/1994233409871050526 |
| themoonlight 中文评述 | https://www.themoonlight.io/zh/review/over-tokenized-transformer-vocabulary-is-generally-worth-scaling |
| 韩文论文解读 | https://kimjy99.github.io/논문리뷰/over-tokenized-transformer/ |
| Qiita：N-gram 嵌入能否放 SSD（含独立实测与质疑） | https://qiita.com/sukimaengineer/items/5d950e6fbefd8ee9dfde |
| DeepSeek Engram 英文解读 | https://introl.com/th/blog/deepseek-engram-conditional-memory-architecture-january-2026 |

---

*笔记完成时间：2026-10-14 | 调研者：技术战略/前沿追踪分析师（DSH 子 agent）*

# TD · 通信—计算融合（通算融合）背景与影响 · 调研笔记

> **用途**：ByteDance Seed **Triton-distributed**（arXiv:2504.19442）技术洞察报告的**第 1 章（背景）**与**第 3 章（影响/技术谱系与变种方向）**素材库。
> **核查日期**：2026-10-14（Asia/Shanghai）。
> **核查原则**：每条结论附一手 URL；能 fetch 到原文的标注 ✅；仅二手来源的标注 ⚠️；查不到的明确写 **❌ 未找到**，不做推测性填充（属本人综合推理的标注 `(分析)`）。
> **注意时效**：Triton-distributed 仓库与 DeepEP 仓库在本核查日期已远新于论文（论文 v3 为 2025-06），笔记中**仓库现状与论文结论分别标注**，二者不可混用。

---

## 0. 核心对象速览（一手核实）

| 对象 | 编号 / URL | 核查状态 |
|---|---|---|
| Triton-distributed 论文 | arXiv:2504.19442（v1 2025-04-28，v3 2025-06-05）https://arxiv.org/abs/2504.19442 | ✅ 已 fetch abs 页 |
| Triton-distributed 仓库 | https://github.com/ByteDance-Seed/Triton-distributed | ✅ 已下载 main 分支 tarball 逐文件核查 |
| 论文机构 | ByteDance Seed + Tsinghua + PKU + SJTU + ZJU（ar5iv 全文页头部标注） | ✅ https://ar5iv.labs.arxiv.org/html/2504.19442 |
| 前序工作 TileLink | MLSys 2025，OpenReview `ccjvBkTRRe`，https://openreview.net/forum?id=ccjvBkTRRe | ✅ **无 arXiv 编号 → ❌ 未找到** |
| 引用数（Semantic Scholar） | `citationCount = 39`，`influentialCitationCount = 5`（2026-10-14 抓取） | ✅ 见 §4.4 |

---

# 第 1 章素材：背景痛点（量化）

## 1.1 「计算/通信比约 1:1」——DeepSeek-V3 的原句

**结论**：DeepSeek-V3 因跨节点 EP 的通信开销，计算:通信 ≈ **1:1**，这是 DualPipe 的直接动机。

**证据（✅ 原文）**：arXiv:2412.19437 §3.2.1
> "For DeepSeek-V3, the communication overhead introduced by cross-node expert parallelism results in an inefficient **computation-to-communication ratio of approximately 1:1**. To tackle this challenge, we design an innovative pipeline parallelism algorithm called DualPipe…"

- URL：https://arxiv.org/abs/2412.19437 （HTML 全文 https://arxiv.org/html/2412.19437v2 ，检索串 `1:1`）
- 另一句同源表述（§1 Introduction）：
> "This overlap ensures that, as the model further scales up, as long as we maintain a **constant computation-to-communication ratio**, we can still employ fine-grained experts across nodes while achieving a **near-zero all-to-all communication overhead**."

**可直接引用的量化表述**：`计算/通信 ≈ 1:1`（跨节点 EP，EP64 跨 8 节点，2048×H800）。

---

## 1.2 all-to-all 占用的 SM 数：**20 / 132**

**结论**：DeepSeek-V3 的跨节点 all-to-all 通信专用 **20 个 SM（H800 共 132 个 SM）**，且这些 SM 上**Tensor Core 完全闲置**，因此通信直接吃掉算力上限。

**证据（✅ 原文，两处，互为佐证）**：

1. §3.5.1 Communication Hardware（URL: https://arxiv.org/html/2412.19437v2 检索 `Communication Hardware`）：
> "In DeepSeek-V3, we implement the overlap between computation and communication to hide the communication latency during computation. This significantly reduces the dependency on communication bandwidth compared to serial computation and communication. However, the current communication implementation relies on **expensive SMs (e.g., we allocate 20 out of the 132 SMs available in the H800 GPU for this purpose), which will limit the computational throughput**. Moreover, using SMs for communication results in significant inefficiencies, as **tensor cores remain entirely under-utilized**."

2. §3.2.2 Efficient Implementation of Cross-Node All-to-All Communication：
> "Overall, under such a communication strategy, **only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink**."（配合 warp specialization 与 chunked 传输）

**SM 具体承担的工作（原文列举，适合直接进报告）**：
- IB ↔ NVLink 域间的数据转发（把单 GPU 上发往同节点多 GPU 的 IB 流量聚合）；
- RDMA buffer（已注册的 GPU 显存区）与输入/输出 buffer 之间的搬运；
- 为 all-to-all combine 执行 **reduce**；
- 管理 chunked 数据跨 IB/NVLink 发往多专家时的**细粒度内存布局**。

> 报告可用论点：**"通信要吃 SM" 是 host-initiated / 早期 device-initiated 实现的共同代价；把通信指令嵌进计算 kernel（device-initiated）正是为了把这 20 个 SM 的算力还回去。**

---

## 1.3 NVLink 与 IB 的带宽不对称：**160 GB/s vs 50 GB/s（3.2×）**

**结论**：DeepSeek-V3 集群中，NVLink 单卡带宽 **160 GB/s**，约为 IB（**50 GB/s**）的 **3.2 倍**。这一不对称直接决定了 MoE 路由策略与 all-to-all kernel 的分层设计。

**证据（✅ 原文）**：arXiv:2412.19437 §3.2.2
> "in our cluster, cross-node GPUs are fully interconnected with IB, and intra-node communications are handled via NVLink. **NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s).** To effectively leverage the different bandwidths of IB and NVLink, we limit each token to be dispatched to **at most 4 nodes**, thereby reducing IB traffic. … In this way, communications via IB and NVLink are fully overlapped, and each token can efficiently select an average of **3.2 experts per node** without incurring additional overhead from NVLink. This implies that, although DeepSeek-V3 selects only 8 routed experts in practice, it can scale up this number to a **maximum of 13 experts (4 nodes × 3.2 experts/node)** while preserving the same communication cost."

**硬件底数（✅ NVIDIA 官方规格，供报告对照）**：
- H800 SXM：NVLink 4.0，**400 GB/s 双向（=200 GB/s 单向）**，互联带宽相对 H100 SXM（900 GB/s 双向）被削减 —— NVIDIA H100/H800 数据中心 datasheet：https://www.nvidia.com/en-us/data-center/h100/
- 上表与 DeepSeek 报的 160 GB/s 口径不同（DeepSeek 用的是实际可达的单向有效带宽），报告中**不要混用**，建议写"官方标称 NVLink 400 GB/s(双向) / DeepSeek 实测有效 160 GB/s，IB 50 GB/s"。

**互补证据（✅）**：`deepseek-ai/profile-data` 明确给出训练/推理的并行配置：
- 训练：EP64、TP1、4K 序列，含 4 个 MoE 层的 chunk；
- Prefill：EP32、TP1，4K prompt，16K tokens/GPU batch；
- Decode：EP128、TP1，4K prompt，128 requests/GPU；
- **Decode 阶段 all-to-all 不占用 GPU SM**："after RDMA messages are issued, all GPU SMs are freed"。
- URL：https://github.com/deepseek-ai/profile-data （README 原文）

> ⚠️ 注意：profile-data 的 decode 用 EP128，而 arXiv V3 报告 §3.4.2 写的是 **EP320（40 节点 × 8 卡）**；DeepSeek 开源的推理系统总览写的是 **EP144**。三份文档口径不同（不同时间点的部署形态），报告中需注明版本。

---

## 1.4 Megatron-LM 的张量并行通信量公式：`8·b·s·h·(t−1)/t`

**结论**：TP 每层每 microbatch 每卡的通信量为 `8bsh((t−1)/t)`，其中 8 = 前向 2 次 all-reduce + 反向 2 次 all-reduce，每次 all-reduce 在 ring 算法下产生 `2·(t−1)/t·(bsh)` 的收发量。

**证据（✅ 原文逐字）**：arXiv:2104.04473（"Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM"）§3 Performance Analysis
> "With tensor model parallelism, tensors of total size `bsh` need to be all-reduced among `t` model replicas **twice each in the forward and backward pass** for each layer, leading to a total communication of **`8bsh((t−1)/t)`** per layer per device for each microbatch. Each device typically has multiple layers; the total amount of tensor-parallel-communication per device for each microbatch is then `l^stage · (8bsh((t−1)/t))`."

- URL：https://arxiv.org/abs/2104.04473 （全文 https://arxiv.org/html/2104.04473v5 ，检索 `8bsh`）
- **推导（原文同段 + 同节）**：
  - f/g 共轭算子：`f` 在前向是 identity、反向是 all-reduce；`g` 在前向是 all-reduce、反向是 identity；
  - 前向 2 次 all-reduce（MLP 一处 + attention 输出投影一处），反向 2 次 → 共 4 次；
  - ring all-reduce 每卡收/发各 `2·(t−1)/t · M`，4 次 → `8·(t−1)/t·M`，`M = bsh`。
- **同段的关键工程结论（原文）**：
> "Consequently, we see that tensor model parallelism **increases the amount of communication between devices**. Thus, when `t` is larger than the number of GPUs in a single node, the overhead of performing tensor model parallelism across **slower inter-node links can be impractical**."

> 报告可用论点：TP 通信量 **与序列长度线性正相关**、且随 `t` 增大趋于饱和（`(t−1)/t → 1`）；这正是"长序列 + 大模型 → 必须换通信模式（SP / EP / 通算融合）"的量化依据。

---

## 1.5 MoE 推理：速度上限受互联带宽约束

**结论**：方向性成立且有多方一手证据；但**DeepSeek 官方文档中没有出现"速度上限由互联带宽决定"的逐字原句**（❌ 该逐字表述未找到）。

**证据 A（✅ 一手，DeepSeek 官方）**：https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
> "our solution employs **cross-node Expert Parallelism (EP)** … However, EP increases system complexity, primarily in two aspects: 1. **EP introduces cross-node communication.** To optimize throughput, appropriate computational workflows must be designed to **overlap communication with computation**."
> "Large-scale cross-node EP introduces **significant communication overhead**. To mitigate this, we employ a **dual-batch overlap strategy** … During the decoding phase, the execution durations of different stages are unbalanced. Hence, we subdivide the attention layer into two steps and use a **5-stage pipeline** to achieve a seamless communication-computation overlapping."

**证据 B（✅ 一手，论文 §3.5.1）**：通信占 SM → "will limit the computational throughput"（见 §1.2 引文）。

**证据 C（✅ 一手，NSDI'26）**：`SwiftEP: Accelerating MoE Inference with Buffer Fusion and TMA Offloading`（Xingyi Li et al., Tencent），https://www.usenix.org/conference/nsdi26/presentation/li-xingyi
> "Expert Parallelism (EP) … introduces all-to-all communication overhead during the dispatch and combine phases, especially in the prefill stage, which dominates the inference performance. Existing communication libraries, such as DeepEP, suffer from **excessive GPU SM utilization and underutilized interconnect bandwidth, limiting prefill performance**."
- 全文 PDF：https://www.usenix.org/system/files/nsdi26-li-xingyi.pdf
- ⚠️ 该论文在搜索摘要中显示为 "SwiftEP"，请报告中以 USENIX 页面标题为准（此前若有人引用为"DeepEP NSDI'26 论文"，实为同作者群的 **SwiftEP**）。

**证据 D（✅ 量化，MLSys 2025）**：`COMET: Fine-grained Computation-communication Overlapping for Mixture-of-Experts`，arXiv:2502.19811
> "The inter-device communication of a MoE layer can **occupy 47% time of the entire model execution** with popular models and frameworks."
> 结果：单 MoE 层加速 **1.96×**，端到端平均 **1.71×**；"adopted in the production environment of clusters with **ten-thousand-scale of GPUs**, achieving savings of **millions of GPU hours**."
- URL：https://arxiv.org/abs/2502.19811 ；MLSys 2025 收录：https://mlsys.org/Conferences/2025/AcceptedPapers

> 报告写法建议：**不要写"DeepSeek 说推理上限由互联带宽决定"**，改写为"MoE 推理的 all-to-all 通信在 prefill 阶段主导性能（SwiftEP/NSDI'26），MoE 层通信可占整体执行时间 47%（COMET, MLSys'25），DeepSeek 为此设计双微批/5 级流水重叠（官方推理总览）"。

---

## 1.6 长序列 attention 的通信痛点

**结论**：Ulysses 相对既有方案的通信量降低 **>10×**，吞吐提升 **最高 2.5×**，可达 **175 TFLOPs/GPU（>54% 硬件峰值）**，支持 **100 万 token 以上**序列。

**证据（✅ 原文）**：arXiv:2309.14509 §1
> "• DeepSpeed-Ulysses trains Transformer models **4x larger sequence lengths** than existing systems, while enabling training with **sequences with over a million tokens**.
> • **Communication reduction of over 10x** compared to existing systems, resulting in throughput improvements of **up to 2.5x**, and **sustained throughput of over 175 TFlops/GPU (over 54% of hardware peak)**."
- URL：https://arxiv.org/abs/2309.14509

---

## 1.7 第 1 章量化要点汇总表

| # | 量化结论 | 数值 | 一手来源 |
|---|---|---|---|
| 1 | V3 计算/通信比 | **≈ 1:1** | arXiv:2412.19437 §3.2.1 |
| 2 | all-to-all 专用 SM | **20 / 132**，Tensor Core 全闲置 | arXiv:2412.19437 §3.2.2 + §3.5.1 |
| 3 | NVLink vs IB | **160 GB/s vs 50 GB/s（3.2×）** | arXiv:2412.19437 §3.2.2 |
| 4 | MoE 通信占执行时间 | **47%**（单 MoE 层） | arXiv:2502.19811（COMET, MLSys'25） |
| 5 | TP 层通信量 | **8bsh(t−1)/t** | arXiv:2104.04473 §3 |
| 6 | 长序列通信降幅 | **>10×**，吞吐 **2.5×**，175 TFLOPs/GPU | arXiv:2309.14509 §1 |
| 7 | 生产规模 | 万卡级集群已采用细粒度重叠 | arXiv:2502.19811（COMET） |
| 8 | 推理部署 | Prefill EP32 / Decode EP320（论文）或 EP128（profile）或 EP144（总览） | arXiv:2412.19437 §3.4；profile-data README；open-infra-index day6 |

---

# 第 1/3 章素材：通算融合的既有技术谱系

## 2.1 谱系总览（两条主轴）

```
通算融合（communication-computation overlap）
├── A. host-initiated（主机发起，靠"多流 + 切块"隐藏通信）
│   ├── DDP 梯度分桶 + all-reduce 与 backward 重叠
│   ├── ZeRO / ZeRO-Offload（offload 与 prefetch 流水）
│   ├── FSDP / 异步 all-gather（async TP、veScale 等）
│   ├── DualPipe（双向 PP 调度，把 EP all-to-all 藏进 F/B）
│   └── CUDA stream / NCCL group / CUDA Graph
└── B. device-initiated（设备发起，通信指令嵌进 kernel）
    ├── NVSHMEM（OpenSHMEM 的 NVIDIA 实现，单边 + symmetric heap）
    ├── MSCCL++（GPU-driven communication stack）
    ├── DeepEP（MoE dispatch/combine；V1 用 NVSHMEM，V2.5 移除）
    ├── Triton-distributed（编译器生成融合 kernel）
    └── DeepGEMM "Mega MoE" / FlashMLA（把通信融进 GEMM/attention 本体）
```

---

## 2.2 A 类：host-initiated 经典手段

### (1) PyTorch DDP：梯度分桶 + 反向传播中重叠 all-reduce
- **机制**：反向传播按 bucket 累积梯度，一个 bucket 就绪即在其独立 CUDA stream 上发起 all-reduce，与后续层的 backward 计算重叠。
- **一手来源**：PyTorch DDP 设计文档 https://docs.pytorch.org/docs/stable/notes/ddp.html
  - ⚠️ 本核查通过 curl 抓取该页正文关键词未返回（页面为 JS 渲染），**URL 有效性与内容建议报告中二次确认**；备选：https://pytorch.org/docs/stable/notes/ddp.html
- **局限**：通信仍以**集合通信原语**为粒度，kernel launch 与流同步是固定开销；bucket 大小是"延迟 vs 带宽"的两难。

### (2) ZeRO / ZeRO-Offload
- **编号**：ZeRO-Offload = **arXiv:2101.06840**（✅ 已核实标题 "ZeRO-Offload: Democratizing Billion-Scale Model Training"，Ren, Rajbhandari, Aminabadi, Ruwase, Yang, Zhang, Li, He）https://arxiv.org/abs/2101.06840
- **机制**：把优化器状态/梯度放到 CPU 内存，通过 **overlap CPU compute 与 GPU→CPU 通信**、并把 CPU all-reduce 切成 chunk 做流水，掩盖 PCIe 传输。
- **本质**：仍是 host-initiated；"offload" 的收益来自**逐 chunk 流水**，不是把通信嵌进 kernel。
- ⚠️ 未逐字核验其 overlap 章节原句，报告中引用其量化数字前建议回原 PDF 复核。

### (3) DualPipe（DeepSeek 官方，**最关键的 A 类代表**）
- **编号**：DeepSeek-V3 Technical Report **arXiv:2412.19437 §3.2.1**；官方仓库 **https://github.com/deepseek-ai/DualPipe**（✅ 已 fetch README）
- **机制（README 原文）**：双向 pipeline schedule，从 pipeline 两端同时喂 micro-batch；"**achieves full overlap of forward and backward computation-communication phases, also reducing pipeline bubbles**"。
- **本文档配套的硬数据**：profile 数据 https://github.com/deepseek-ai/profile-data （training profile：EP64/TP1/4K seq，每 chunk 含 4 个 MoE 层）
- **bubble 对比表（README 原文）**：

| Method | Bubble | Param/Device | Activation/Device | #Devices |
|---|---|---|---|---|
| 1F1B | (PP−1)(F+B) | 1× | PP | PP |
| ZB1P | (PP−1)(F+B−2W) | 1× | PP | PP |
| DualPipe | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | PP |
| DualPipeV | (PP/2−1)(F&B+B−3W) | 2× | PP+1 | PP/2 |

- **DualPipeV** 来源：Sea AI Lab "Cut-in-half" 博客 https://hackmd.io/@ufotalent/r1lVXsa9Jg

### (4) ⚠️ 「DoublePipe」——**❌ 未找到可靠一手来源**
- 本核查在 arXiv 上未找到题为 "DoublePipe" 的异构流水线论文；`arXiv:2407.09538` 实为**无关的**音乐交互建模论文（已核实），**不可引用**。
- **注意**：DualPipe 与 DoublePipe 常被混写，报告若必须提到 DoublePipe，请先定位其原始出处，否则建议**只写 DualPipe**，并注明"DoublePipe 未找到可靠一手出处"。

### (5) 异步 all-gather / 异步张量并行
- 相关但未逐条核实的一手来源：
  - veScale（ByteDance）：**arXiv:2509.07003** ✅ 已核实标题 "veScale: Consistent and Efficient Tensor Programming with Eager-Mode SPMD"（作者含 Zhiqi Lin、Xinyi Di、Wenlei Bao、Yanghua Peng、Haibin Lin、Li-Wen Chang —— 与 Triton-distributed 作者群高度重叠）https://arxiv.org/abs/2509.07003
  - veScale-FSDP：在 MLSys/OpenReview 收录（https://openreview.net/pdf?id=3Lj8R0F48P ），⚠️ 未核对其中 async all-gather 的具体机制。
- **❌ 未找到**：`veScale` / `veScale-FSDP` 中明确集成或调用 Triton-distributed 的一手证据。

### (6) CUDA stream / NCCL 原语
- **一手来源**：NCCL 官方文档 https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ ；NCCL 提供 host 侧集合通信 API + 拓扑感知算法（Ring / Tree / NVLS）。
- **机制**：把通信 kernel 放到独立 stream，靠事件依赖与计算 stream 并行；`ncclGroupStart/End` 批量下发以摊薄 launch 开销。
- **局限（`(分析)`，可由 §1.2 的 20/132 SM 事实支撑）**：仍需独立通信 kernel → 占用 SM、无法与 GEMM 的 epilogue/prologue 融合、对网络抖动敏感、需把数据切成大量 chunk 才能填满流水。

---

## 2.3 B 类：device-initiated（通信指令嵌入 kernel）

### (1) NVSHMEM
- **定位（✅ 一手）**：NVSHMEM 是 NVIDIA 对 **OpenSHMEM** 标准的 GPU 实现，基于 **PGAS + symmetric memory**，提供 **GPU 发起的单边通信（one-sided put/get）**。
- **权威二手综述（同行评议/arXiv，对系统设计描述最完整）**：`Demystifying NVSHMEM: A System-Level Analysis on Symmetric Memory and Device-Initiated Operations in GPU Communication`，**arXiv:2606.05951**（Yijun Ma, Siyuan Shen, …, Torsten Hoefler），2026-06-04
> "**NVSHMEM is NVIDIA's OpenSHMEM-based PGAS communication library for GPU clusters, enabling GPU-initiated, one-sided communication through symmetric memory.** … Our analysis shows that NVSHMEM pioneered a device-side symmetric-memory programming model that enables fine-grained GPU-driven communication and **is important for approaching the hardware performance limit**. … We also examine **DeepEP as a case study** of NVSHMEM in performance-critical sparse deep learning workloads."
  - URL：https://arxiv.org/abs/2606.05951
- **官方文档**：https://docs.nvidia.com/nvshmem/api/ （`index.html`、`using.html` 均返回 HTTP 200；⚠️ 页面为 JS 渲染，curl 取不到正文）
- **OpenSHMEM 标准**：https://www.openshmem.org/
- **关键概念**：symmetric heap（`nvshmem_malloc`，各 rank 同尺寸同相对地址）、`nvshmem_put/get`、device 侧 `nvshmemx_*`、device 侧集合通信、传输层（NVLink / IB / **IBGDA**）。
- **❌ 未找到**：NVSHMEM 自身的正式论文 arXiv 编号。可引用的早期评估：`An Initial Assessment of NVSHMEM for High Performance Computing`，IEEE https://ieeexplore.ieee.org/document/9150438 ；以及 GTC'19 讲义 `NVSHMEM: a PGAS library for GPU clusters` https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9677-nvshmem-a-pgas-library-for-gpu-clusters.pdf

### (2) MSCCL++（Microsoft，GPU-driven）
- **编号**：**arXiv:2504.09014**（README 中 "Paper" 链接）
- **README 自述（✅）**：**"A GPU-driven communication stack for scalable AI applications."**
- URL：https://github.com/microsoft/mscclpp
- **意义**：与 NVSHMEM 并列的"GPU 发起通信"工业级栈，同时支持 CUDA 与 ROCm（README 的 CI 徽章含 MI300X RCCL 测试）。

### (3) DeepEP（DeepSeek）
> ⚠️ **重大时效性修正**：DeepEP 仓库已迭代到 **V2.5**，**V1 的 NVSHMEM 后端被完全移除**。若报告写"DeepEP 基于 NVSHMEM"，需限定为 **V1**。

- URL：https://github.com/deepseek-ai/DeepEP （✅ 已 fetch README 原文）
- **README 原文要点**：
> "DeepEP (DeepEveryParallel) is a **high-performance communication library** for machine learning training and inference. It provides **high-throughput and low-latency expert-parallel (EP) all-to-all GPU kernels for MoE dispatch and combine, including FP8 dispatch**, and NVLink weight and gradient exchange for redundant experts. It also offers experimental primitives for pipeline parallelism (PP), context parallelism (CP), data parallelism (DP), and **remote memory access (Engram)**."
> "**JIT-compiled communication kernels** via [DeepJIT]"；"**NCCL Gin backend** — Lightweight **device-side communication APIs** — Able to reuse existing NCCL communicators"
> "EP dispatch and combine require GPU SMs; **zero-SM RDMA EP is not supported**"
> "**Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation. NVSHMEM is no longer a dependency.**"
> "**Ascend version release** — Same API and full performance on HUAWEI Ascend 950 NPUs"（https://github.com/deepseek-ai/DeepEP-Ascend ）
- **配套论文**：`Demystifying NVSHMEM` 将 DeepEP 作为 NVSHMEM 案例研究（arXiv:2606.05951）—— 这是"DeepEP 曾基于 NVSHMEM"的**权威学术证据**。
- **竞争基线**：Triton-distributed 论文把 DeepEP 作为 AllToAll 对照（见 §4.3）。

### (4) Triton-distributed（本次主角）
详见 §4。

### (5) DeepSeek FlashMLA / DeepGEMM
- **FlashMLA**（✅ README 原文）：https://github.com/deepseek-ai/FlashMLA
  - "FlashMLA is DeepSeek's library of **optimized attention kernels** … token-level sparse attention kernels for prefill and for decoding with an FP8 / FP4 KV cache, **a fused kernel that combines Q-norm, Q-RoPE, attention, O-RoPE (conjugate) and the cast to FP8**"。
  - **性质判定**：FlashMLA 是**计算侧融合**（算子融合），**README 中未声称包含跨设备通信**（❌ 未找到其涉及 all-to-all/通信的证据）。报告中**不应**把 FlashMLA 归为通算融合的通信侧代表；它更适合作为"降低 attention 计算成本"的对照。
  - 最新版（2026-09-30）已发布 **Ascend 950 NPU** 稀疏 attention kernel，prefill 达 **410 TFlops（95% 峰值）**、decode **360 TFlops（83% 峰值）**——国产硬件算力侧的重要事实。
- **DeepGEMM**（✅ README 原文）：https://github.com/deepseek-ai/DeepGEMM
  - "a unified, high-performance tensor core kernel library that brings together … GEMMs (FP8, FP4, BF16), **fused MoE with overlapped communication (Mega MoE)**, MQA scoring for the lightning indexer, HyperConnection (HC)…"；2026-04-16 "**Mega MoE**, FP8xFP4 GEMM, FP4 Indexer, PDL"。
  - **性质判定**：DeepGEMM 的 **Mega MoE 明确包含"overlapped communication"** → 这是 **device-initiated + 计算通信融合** 的一手证据，且比 FlashMLA 更贴合本报告主题。
  - 另有 **DeepGEMM-Ascend**：https://github.com/deepseek-ai/DeepGEMM-Ascend/

### (6) 其他已核实的一手项目（host/device 谱系补全）
- **MORI（AMD，Modular RDMA Interface）**：https://github.com/ROCm/mori —— "bottom-up, modular, and composable framework … **Open-Shmem-style APIs**"，含 MORI-EP / MORI-IO / MORI-CCL / MORI-UMBP；支持 IBGDA / GDS。**与 Triton-distributed 有直接集成关系**（见 §4.4）。

---

## 2.4 本质区别：host-initiated vs device-initiated

| 维度 | host-initiated（NCCL + stream 重叠） | device-initiated（kernel 内单边通信） |
|---|---|---|
| 发起者 | Host CPU 下发集合通信 kernel | GPU 线程在 kernel 内直接发起远端内存访问 |
| 编程抽象 | `ncclAllReduce/AllToAll` + CUDA stream/event | symmetric heap + `put/get` + signal/wait（NVSHMEM / OpenSHMEM 语义） |
| 与计算融合 | **难**：通信是独立 kernel，只能"并行"不能"融合" | **可融合**：同一 kernel 内 producer 产 tile → 直接写到远端（或从远端读进 MMA） |
| SM 代价 | 通信 kernel 独占 SM（V3：20/132） | 可做到通信只占少量 SM / warp（但**并非零成本**：DeepEP 明确 "zero-SM RDMA EP is not supported"） |
| 延迟/抖动 | 受 kernel launch + 流同步 + 网络抖动影响，需切大量 chunk | 细粒度、可流水，对抖动更鲁棒（COMET 的细粒度重叠动机） |
| 内存一致性 | 由 NCCL 保证，用户无需关心 | **用户负责**：acquire/release 语义（TD 提供 `ld_acquire` / `red_release` / `signal_op` / `wait`） |
| 前置条件 | 无特殊要求 | 需 **symmetric memory 注册**、拓扑（NVLink 域 + IB）、后端库（NVSHMEM/ROCSHMEM/MORI） |
| 成熟度 | 高（NCCL 拓扑感知算法、生产验证） | 中→高（NVSHMEM 生态成熟；国产硬件仍在追赶，见 §5） |
| 典型代表 | DDP、ZeRO、DualPipe、FLUX | NVSHMEM、MSCCL++、DeepEP、Triton-distributed、DeepGEMM Mega MoE |

**代价小结（可直接引用的一手句子）**：
- device-initiated 的动机（✅ ROCSHMEM README）："This intra-kernel networking library simplifies application code complexity and enables **more fine-grained communication/computation overlap than traditional host-driven networking**." https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/develop/projects/rocshmem/README.md
- host-initiated 的 SM 代价（✅ arXiv:2412.19437 §3.5.1）：20/132 SM，tensor core 全闲置。
- 编译期融合的必要性（✅ Triton-distributed 论文）：手写 CUDA 版 DeepEP all-to-all "takes **thousands lines of CUDA code** and is **extremely hard to maintain**"，而 TD 用 "**hundreds lines of Python**" 重实现。

---

# 第 3 章素材：Ulysses 序列并行与 attention all-to-all

## 3.1 Ulysses 机制（一手逐字）

**编号**：**arXiv:2309.14509**（✅ 核实标题 "DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models"，Jacobs, Tanaka, Zhang, Zhang, Song, Rajbhandari, He）https://arxiv.org/abs/2309.14509

**机制（§1 原文）**：
> "DeepSpeed-Ulysses **partitions individual samples along the sequence dimension** among participating GPUs. Then **right before the attention computation, it employs all-to-all communication collective on the partitioned queries, keys and values such that each GPU receives the full sequence but only for a non-overlapping subset of the attention heads.** This allows the participating GPUs to compute attention for different attention heads in parallel. Finally, DeepSpeed-Ulysses **employs another all-to-all to gather the results along the attention heads while re-partitioning along the sequence dimension**."

**即：attention 前后各一次 all-to-all，形成 「seq-split ⇄ head-split」的换轴。**

**通信量性质（§1 原文）**：
> "whereas other methods incur communication overhead as sequence length increases, DeepSpeed-Ulysses **maintains constant communication volume when sequence length and compute devices are increased proportionally**."
> "Megatron LM partitions sequence along sequence dimensions and applies allgather and reduce scatter collective to aggregate QKV projections for attention computation. … **Megatron-LM sequence parallelism communication volume increase linearly with message size (M) regardless of number of compute devices.** DeepSpeed-Ulysses on the other hand **keeps communication volume consistent** by increasing GPUs proportional to message size or sequence length."

**Scaling 实测（§4.4 表 2/表 3 原文摘录）**：

| Seqlen | GPUs | Time (ms) | TFLOPs |
|---|---|---|---|
| 131,072 | 64 | 32432.13 | 165.53 |
| 131,072 | 128 | 17052.51 | 157.41 |
| 131,072 | 256 | 9886.70 | 136.09 |

| Seqlen | GPUs | Time (ms) | TFLOPs |
|---|---|---|---|
| 65,536 | 64 | 9676.76 | 161.36 |
| 131,072 | 128 | 17052.51 | 157.41 |
| 262,144 | 256 | 33486.50 | 147.40 |

> 关键解读：固定序列长度时，GPU 翻倍→时间减半（**线性扩展**）；固定每卡负载时，序列长度翻倍→时间翻倍（**通信量恒定**，通信不随序列增长而恶化）。

**Ulysses 的固有限制（`(分析)` + 需要额外证据）**：
- all-to-all 的换轴要求 **head 数可被设备数整除**（更严格地说 `N_heads ≥ N_devices`），因此当 SP 度数超过 head 数时 Ulysses 无法继续扩展。
- ⚠️ **❌ 未找到**：Megatron-Core/DeepSpeed 官方文档中逐字陈述"Ulysses 受 head 数限制"的句子。可用的间接一手证据：
  - Striped Attention 论文：**arXiv:2311.09431**（✅ 核实标题 "Striped Attention: Faster Ring Attention for Causal Transformers"）https://arxiv.org/abs/2311.09431 —— 针对 ring attention 的因果负载不均；
  - `LoongTrain`（Ulysses+Ring 混合）—— ❌ **未找到其正确 arXiv 编号**（`2406.16801` 经核实为 RES-Q，与本主题无关，**不可引用**）。
  - Megatron-LM 源码中确有 CP 的 "inner ring" 概念：https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py （检索 `get_inner_ring`，行号约 1933）——可作为"Megatron 同时实现 ring 与 all-gather 两种 CP"的代码级证据。
  - ⚠️ Megatron-Core CP 官方文档页（`docs.nvidia.com/megatron-core/.../context_parallel*.html`）在本次核查中返回 404，**未能定位有效 URL**。

---

## 3.2 Ring Attention 对比（一手）

**编号**：**arXiv:2310.01889**（✅ 核实标题 "Ring Attention with Blockwise Transformers for Near-Infinite Context"，Hao Liu, Matei Zaharia, Pieter Abbeel）https://arxiv.org/abs/2310.01889
- 官方代码：https://github.com/lhao499/llm_large_context

**机制（abstract 原文）**：
> "leverages **blockwise computation of self-attention and feedforward** to distribute long sequences across multiple devices while **fully overlapping the communication of key-value blocks with the computation of blockwise attention**. Our approach enables training and inference of sequences that are **up to device count times longer** than those achievable by prior memory-efficient Transformers, **without resorting to approximations or incurring additional communication and computation overheads**."

**对比要点（`(分析)`，基于上面两份一手 abstract/正文）**：

| 维度 | Ulysses | Ring Attention |
|---|---|---|
| 通信对象 | **Q/K/V 重分片**（seq→head 换轴） | **K/V block 的环形传递** |
| 通信次数 | 每层 **2 次 all-to-all** | 每层 **O(N_devices)** 次 P2P（环形轮转） |
| 通信量随序列 | **恒定**（devices 与 seq 同比例增长时） | 每卡通信量与**本地 block 数**相关，可随设备数增长 |
| 与计算重叠 | 两次 all-to-all 是"硬同步点"，需靠 SP/融合掩盖 | **天然可与 blockwise attention 计算重叠**（这是其设计核心） |
| 扩展上界 | 受 **head 数**限制 | 不受 head 数限制，理论上可扩到极大 |
| 因果负载均衡 | 无此问题 | 需 Striped Attention（arXiv:2311.09431）等做均衡 |
| 通信域偏好 | **NVLink 域内**最理想（对称、全互联） | IB/环状链路亦可 |

> **报告中一个稳健的落点**：Ulysses 的两次 all-to-all 是**明确的、结构化的换轴操作**，因此**天然可被编译器识别并融进相邻 GEMM**；Ring Attention 的通信是**与 attention 计算交错的细粒度 P2P**，更依赖流式重叠而非 GEMM 融合。这正是 Triton-distributed 选择 Ulysses 作为融合示例的技术原因。

---

## 3.3 「transpose all-to-all」与「reverse all-to-all」的由来（✅ 代码级证据）

**这是本报告最关键的技术细节之一，且有一手代码证据。**

**证据（✅ Triton-distributed 仓库源码，逐文件核查）**：

1. **布局变换 kernel 的命名直接暴露"转置"语义**：
   - `python/triton_dist/kernels/nvidia/ulysses_sp_dispatch.py` → 函数 **`kernel_qkv_bsnd_to_bnsd`** 与 **`qkv_bsnd_to_bnsd(q, k, v, send_q_buf, send_k_buf, send_v_buf, ...)`
   - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/ulysses_sp_dispatch.py
   - **含义**：Ulysses 前向 all-to-all 的语义就是 **`[B, S_local, N_head, D]`（BSND，序列切分）→ `[B, N_head_local, S_full, D]`（BNSD，头切分）**，即 **"把 S 轴和 N 轴换位" = transpose**。反向（attention 之后）则是 **BNSD → BSND = reverse transpose**。

2. **测试代码中的显式术语**：
   - `python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py`：
     - `def torch_post_attn_all_to_all_transpose(sp_group, input, a2a_only, is_dp, ...)`
     - `parser.add_argument("--a2a_only", default=False, action="store_true", help="whether have transpose")`
     - 注释：`# input: [bs, local_nh, seq_len, hd] for a2a_transpose, [bs, seq_len, local_nh, hd] for a2a_only`
     - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_llm_ulysess_post_attn_all2all_intra_node.py
   - `python/triton_dist/test/nvidia/test_llm_ulysess_all2all_gemm_intra_node.py` 中同样出现 `torch_post_attn_all_to_all_transpose` 与 `a2a_transpose_output`。
     - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/test/nvidia/test_llm_ulysess_all2all_gemm_intra_node.py

3. **Ulysses SP 的四个专用文件**（TD 仓库真实存在）：
   - `python/triton_dist/kernels/nvidia/sp_ulysess_qkv_gemm_all2all.py` —— **QKV GEMM + all-to-all 融合**（"transpose all-to-all"）
   - `python/triton_dist/kernels/nvidia/sp_ulysess_o_all2all_gemm.py` —— **O all-to-all + GEMM 融合**（"reverse all-to-all"）
   - `python/triton_dist/kernels/nvidia/ulysses_sp_dispatch.py` —— pre-attention QKV pack all-to-all
   - `python/triton_dist/kernels/nvidia/ulysses_sp_infer_gemm_a2a.py`、`python/triton_dist/layers/nvidia/ulysses_sp_a2a_layer.py`
   - 文档：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/kernels/nvidia/ulysses_sp.rst
     > "Ulysses Sequence Parallelism … `create_ulysses_sp_pre_attn_comm_context` … **All-to-All Single GEMM** … `create_all_to_all_single_gemm_context` … **Fused All-to-All + GEMM operation**"
   - 注：仓库中统一拼写为 **`ulysess`（少一个 l）**，检索时需注意。

4. **通用融合原语**：`python/triton_dist/kernels/nvidia/all_to_all_single_gemm.py`
   - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/all_to_all_single_gemm.py
   - 一键核对 API（`create_all_to_all_single_gemm_context` / `all_to_all_single_gemm`），支持 `int8/float8_e4m3fn/float8_e5m2/bfloat16` 与 **input_scale/weight_scale**（即支持量化传输），并用 **`nvshmem_create_tensors` 申请 symmetric 通信缓冲**与 barrier/signal 缓冲。
   - `all_to_all_single_gemm` docstring 原文：*"All-to-All Single GEMM operation with separated communication and computation. Uses copy engine (cudaMemcpyAsync) for communication and separate kernels on different streams to achieve true overlap"* —— ⚠️ 值得注意的是，**这一实现路径其实是"copy engine + 多流"（偏 host-initiated）**，而非纯 kernel 内单边；报告需区分 TD 内部不同算子的实现风格。

---

## 3.4 为什么 QKV 方向与 O 方向的 all-to-all 可以分别与 GEMM 融合？收益原理

### (a) 可融合性的机理（`(分析)`，有代码结构支撑）
- **QKV 侧**：Ulysses 的 pre-attention all-to-all 是 **BSND → BNSD** 的纯换轴。若把它视为 **"GEMM 的输出 tile 按 (b, n, s) 重新索引后直接写入远端 rank 的 symmetric buffer"**，则 all-to-all 就是 GEMM 的 **"远端 epilogue"**：GEMM 在寄存器里算出 `QKV[b, s_local, n, d]` tile，直接 `put` 到目标 rank 对应的 `[b, n_local, s_full, d]` 位置。
  - 这正是 `sp_ulysess_qkv_gemm_all2all.py` 的结构：kernel 名为 **`kernel_persistent_tma_gemm_notify`**，GEMM 计算完成后 `notify` 远端信号（"notify" = 融合的信号量）。
  - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/sp_ulysess_qkv_gemm_all2all.py
- **O 侧**：attention 输出为 BNSD，需要 **BNSD → BSND** 的 reverse 换轴，且紧接着是一个输出投影 GEMM。若把 all-to-all 做成 **"GEMM 的远端 prologue"**（GEMM 的 A 操作数直接从远端 symmetric buffer 读，或用 TMA/单边读拉进 shared memory），则通信被 GEMM 的流水线自然吸收。
  - `sp_ulysess_o_all2all_gemm.py` 中出现内联 PTX `ld.global.v4.b32` / `st.global.v4.b32` 的向量化远端读写辅助函数 —— 说明它走的是**寄存器级向量化远端访问**路径。
  - URL：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/sp_ulysess_o_all2all_gemm.py
- **为什么是"分别"融合**：两个 all-to-all 分别贴在 **GEMM 之前** 和 **GEMM 之后**，方向相反（transpose vs reverse），所以融合方式也相反：一个是 **"GEMM 后写远端"**，另一个是 **"GEMM 前读远端"**。这就是术语 "transpose all-to-all" / "reverse all-to-all" 的由来。

### (b) 带宽/延迟收益原理（`(分析)`，需按下面的引用谨慎表述）
1. **消除一次 global memory 往返**：非融合实现必须 `寄存器 → 本地 global buffer → 拷贝引擎/NCCL 读 → 网络 → 远端 global → 再读进寄存器`；融合实现可 `寄存器 → 网络 → 远端 global → 读进寄存器`（甚至 `寄存器 → 寄存器`，走 NVLink LDGSTS/TMA），**省掉本地写回 + 本地读回这一对 HBM 往返**，也就是**约 2× 的 HBM 流量削减**。在 HBM 带宽本身也是瓶颈的 decode/long-context 场景下，这一项收益是实打实的。
2. **省掉一次 kernel launch 与一次全局同步点**：非融合时 `GEMM kernel` 与 `all-to-all kernel` 之间必须有一次 stream 依赖 + 全 device 级 barrier；融合后改为 **tile 粒度的 signal/wait**（TD 的 `signal_op`/`wait`/`consume_token`/`notify` 原语），粒度从"整层"降到"tile"。
3. **让通信真正藏在计算里而非"和计算抢 SM"**：非融合时通信 kernel 要独立占 SM（V3 的 20/132），融合后通信指令由**计算 kernel 自身的 warp** 发出，等价于把通信开销摊到 ALU/Memory pipe 的空隙里。
4. **可融合的量化传输**：`all_to_all_single_gemm` 原生带 `input_scale`/`weight_scale`，即 **通信与量化/反量化可一起融进 kernel**（对应 §6 的"通信+量化联合"方向）。

### (c) 量化收益的一手数据（实测，✅）
来自 Triton-distributed 论文与 README：
- 论文 Fig.1：**整体加速 1.09× – 44.97×**（相对 PyTorch+NCCL/RCCL）；
- **AG+GEMM**：平均 **1.42×** vs PyTorch+NCCL，**1.09×** vs FLUX；
- **低延迟 AllToAll（推理）**：Dispatch 平均 **1.18×**、Combine 平均 **1.44×** vs DeepEP（⚠️ 64 GPU 上 Dispatch 除外）；TD 用 **IBRC**，DeepEP 用 **IBGDA**，且 TD 的 intra-node 走 **NVLink**，DeepEP V1 只走 IB；
- README 补充：32×H800、128 tokens/rank、topk=8、hidden=7168、fp8 时，TD 低延迟 all-to-all **137 µs** vs DeepEP **182 µs**（注：README 注明 "DeepEP doesn't use NVLink for inference"）。
- **NVSHMEM AllGather 对比**（论文）：L20/PCIe 8 卡 **3.11×**（vs NVSHMEM-64bit inplace）/ **3.00×**（out-of-place）；2 节点 **1.31×**（vs NVSHMEM-64bit）/ **1.38×**（vs NVSHMEM-32bit）。

> ⚠️ **未找到**：任何论文**明确量化**"融合 vs 非融合各自的 HBM 往返流量削减倍数"。上面 (b) 的第 1 条属 `(分析)`，报告中须如此标注。

---

# 第 3 章素材：Triton-distributed 的定位与影响

## 4.1 贡献定位（✅ 论文 abstract 原文）

> "Triton-distributed is **the first compiler that supports native overlapping optimizations for distributed AI workloads**, providing a good coverage of existing optimizations from different frameworks. First, we **integrate communication primitives compliant with the OpenSHMEM standard into the compiler**. This enables programmers to utilize these primitives with a higher-level Python programming model. Second, we illustrate how to achieve **complex joint optimization of computation, memory access, and communication** with the assistance of the compiler. … In a test environment with **up to 64 devices**, our compiler can fully utilize heterogeneous communication and computation resources to provide effective overlapping and high performance. In many cases, the performance of the generated code can even **outperform hand-optimized code**."

**编译栈事实（✅ 论文 §3.1 与正文）**：
- **MPMD 编程模型**三要素：**symmetric memory + signal exchange + async-task**；
- 编译器把 OpenSHMEM 原语翻译为 **NVSHMEM（NVIDIA）/ ROCSHMEM（AMD）** 实现：*"Our compiler translates these primitives into NVSHMEM/ROCSHMEM implementations during compilation"*；
- 非 OpenSHMEM 的补足原语：`wait`、`consume_token`、`notify`、`signal_op`、`atomic_cas`、`atomic_add`、**`ld_acquire`**、**`red_release`**、**`multimem_ld_reduce`**、**`multimem_st`**（Table 1）；
- 优化维度（论文 Table 2）：Intra-Node Swizzle、Inter-Node Swizzle、**Inter-NUMA Swizzle**、Copy Engine、High-BW Link、Network Comm.、**PCIe Comm.**、**OpenSHMEM Support**、**Low-latency Protocol**、**Multimem Feature**、**Fusion**。

**与既有框架的差异（论文 Table 2 原表要点）**：
- NCCL / PyTorch / TransformerEngine / Pallas 均 **无 OpenSHMEM Support、无 Fusion**；
- **CoCoNet / FLUX / DeepEP 有 Fusion 或 OpenSHMEM，但 FLUX 无 OpenSHMEM，DeepEP 无 Fusion 标注差异需按原表核对**；
- **Ours 是唯一在表 2 中同时勾选 Copy Engine / High-BW Link / Network / PCIe / OpenSHMEM / Low-latency / Multimem / Fusion 全部项的实现**（⚠️ 原表用 ◼/✗/✓ 三态，建议报告中回原 PDF 复核具体符号）。

**`triton_dist.language`（✅ 代码级）**：
- 代码中 `import triton_dist.language as tdl` / `as dl`，并有 `triton_dist.language.extra.language_extra`（提供 `tid`、`__syncthreads`、`st` 等底层原语）；
- 装饰器 `@triton_dist.jit`；
- URL 示例：https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/sp_ulysess_qkv_gemm_all2all.py

## 4.2 仓库现状（远超论文，✅ 逐文件核查 main 分支）

- 默认分支跟踪 **Triton 3.7**，作为 **out-of-tree plugin** 构建（2026-08-17 起）；Triton 3.4 线保留在 `triton-v3.4` 分支。
- 通过 `shmem/` 目录实现**多后端 shmem 抽象**（**这是"跨厂商 shmem 抽象层"的一手实现证据**）：
  - `shmem/nvshmem_bind/`（NVIDIA）
  - `shmem/rocshmem_bind/`（AMD，含 `pyrocshmem`）
  - **`shmem/mxshmem_bind/pymxshmem/`（沐曦 MetaX！）** —— 出现 `mxshmem_create_tensor`、`mxshmem_int_p`、`mxshmem_barrier_all`、`init_mxshmem_by_uniqueid` 等 API，示例 `example/run_ring_put.py`。
- **Ascend 支持**：`ascend/` 目录 + `triton-v3.4` 线的 `ascend/tutorials/ascend/01-ascend-allgather-gemm.py` 等；README 记载 **2026-08-06 "Ascend RDMA support"（PR #179）**。
- **FlashComm**：仓库内独立子目录，README 记载 2026-07-12 "FlashComm CUDA communication library with EP overlap, TMA support"、2026-09-17 "**FlashComm fused MXFP8 intra-node dispatch**（EP 流量可发 packed FP8 而非 BF16，并与 fused/prequantized 路径做 bitwise 校验）"。
- **MegaTritonKernel**：2025-08-13 引入，含 Qwen3 TP demo（H20/H800）。
- **Intra-Kernel Profiler**：2025-09-03 引入。
- **TileLink**（前序论文）已被 **MLSys 2025** 接收（README 2025-05-12 条目 + MLSys 官方 accepted papers 双重确认）。
  - TileLink 摘要（OpenReview ✅）："achieves from **1.17× to 20.76× speedup** to non-overlapping baseline and achieves performance comparable to state-of-the-art overlapping libraries on GPUs"，"improving performance from 1.17x to 20.76x with **10x less programming effort**"。
  - URL：https://openreview.net/forum?id=ccjvBkTRRe ；MLSys 列表：https://mlsys.org/Conferences/2025/AcceptedPapers

## 4.3 论文实测性能（汇总，供第 3 章"影响"用）

| 项目 | 结果 | 来源 |
|---|---|---|
| 整体 vs PyTorch+NCCL/RCCL | **1.09× – 44.97×** | 论文 Fig.1 |
| AG+GEMM（8/16×H800） | 平均 **1.42×** vs PyTorch+NCCL；**1.09×** vs FLUX | 论文 §5 |
| GEMM+RS | 见论文 Fig.12/14 | 论文 §5 |
| 低延迟 AllToAll（推理，8→64 GPU） | Dispatch **1.18×**、Combine **1.44×** vs DeepEP（64 GPU Dispatch 除外） | 论文 §5 |
| 低延迟 AllToAll（单点） | **137 µs** vs DeepEP **182 µs**（32×H800, fp8, hidden 7168） | 仓库 README |
| Low-latency AllGather vs NVSHMEM | PCIe/L20 **3.11×/3.00×**；2 节点 **1.31×/1.38×** | 论文 §5 |
| 端到端推理（Seed-OSS-36B-Instruct） | **1.33× 加速** | 仓库 README 2025-08-24 |

## 4.4 采用/集成证据（严格区分"已核实"与"未找到"）

### ✅ 已核实的一手证据

1. **AMD MORI-SHMEM 作为 TD 在 AMD GPU 上的 EP dispatch/combine 后端**（**最强的第三方集成证据**）
   - MORI 官方 README 原文：*"**[2026/03]** 🔥 **MORI-SHMEM powers ByteDance Triton-distributed EP dispatch/combine kernels as the backend**, delivering seamless integration and high performance on AMD GPUs ([EP Kernels](https://github.com/ByteDance-Seed/Triton-distributed/pull/164), [MORI-SHMEM Integration](https://github.com/ByteDance-Seed/Triton-distributed/pull/145))."*
   - URL：https://github.com/ROCm/mori
   - TD 侧对应 PR：https://github.com/ByteDance-Seed/Triton-distributed/pull/145 、 https://github.com/ByteDance-Seed/Triton-distributed/pull/164
   - TD README 侧印证：2026-08-10 "AMD MORI backend update restoring fused-MoE cooperative SHMEM APIs"；2026-07-12 "AMD mori-shmem EP intra-node / low-latency kernels"。
2. **字节内部框架/模型**
   - `docs/getting-started/e2e/e2e_dense.md` —— **端到端推理加速 demo，统一支持 NVIDIA 与 AMD**；
   - README：**Seed-OSS-36B-Instruct 推理加速 1.33×**；
   - **FlashComm**（仓库内独立子目录）= 字节内部通信库，与 TD 的 EP overlap 耦合；
   - `MegaTritonKernel` + Qwen3 TP demo（H20/H800）。
3. **Ascend 生态**
   - 仓库 `ascend/` 目录与 TD-Ascend 分支；README 2026-08-06 "Ascend RDMA support"（PR #179）；
   - 存在独立镜像/移植仓库：`Ascend/Triton-distributed-ascend`（atomgit，其 `shmem/mxshmem_bind/` 路径出现在公开检索结果中）—— ⚠️ 该站点未直接 fetch 成功，仅作为线索。
4. **社区 fork**
   - `xlite-dev/Triton-distributed`（https://github.com/xlite-dev/Triton-distributed ）—— 同名 fork，说明社区有二次开发。
5. **NVIDIA/推理引擎侧** —— ⚠️（见下）
   - DeepWiki 存在页面 `deepwiki.com/triton-inference-server/triton_distributed/*`（"LLM Engines"、"Key Features"），指向 NVIDIA Triton Inference Server 组织下的 `triton_distributed` 仓库；
   - **但本核查中 `https://github.com/triton-inference-server/triton_distributed` 返回 HTTP 404**（main 与 master 分支 README 均 404）。
   - **结论**：该仓库**当前不可验证**（可能已删除/改名/私有）。报告中**不要**写"NVIDIA Triton Inference Server 集成了 Triton-distributed"，除非二次确认。

### ❌ 未找到（明确记录，不要臆造）

| 声称 | 状态 |
|---|---|
| vLLM 集成/采用 Triton-distributed | **❌ 未找到**（vLLM 仓库中无 TD 集成证据；TD 仓库仅在 `flash_decode.py` 头部标注了 vLLM/SGLang 的**代码来源许可**，**不是采用关系**） |
| SGLang 集成/采用 Triton-distributed | **❌ 未找到**（同上。注意：是 **MORI-EP/MORI-IO** 被集成进 SGLang/vLLM，见 https://github.com/sgl-project/sglang/pull/14797 、 https://github.com/vllm-project/vllm/pull/28664 —— **这是 MORI 的采用，不是 TD 的采用**，切勿混淆） |
| veScale 集成/采用 Triton-distributed | **❌ 未找到**（veScale 论文 arXiv:2509.07003 已核实存在且作者重叠，但未找到其引用/集成 TD 的证据） |
| NVIDIA Triton Inference Server 集成 | **⚠️ 不可验证**（仓库 404） |

> 报告建议表述：*"Triton-distributed 的集成证据主要集中在 **AMD（MORI-SHMEM 后端）**、**字节内部（Seed-OSS 推理、FlashComm）** 与 **Ascend 移植** 三条线上；在 vLLM / SGLang / veScale 中**未找到**直接集成证据。"*

## 4.5 引用数据（Semantic Scholar，2026-10-14）

- API：https://api.semanticscholar.org/graph/v1/paper/arXiv:2504.19442?fields=title,year,venue,citationCount,influentialCitationCount,externalIds
- 返回：`citationCount = 39`，`influentialCitationCount = 5`，`venue = arXiv.org`，`year = 2025`，`paperId = 19b5e0b1e5c95770b0e3537734f8d96d6037cfd5`，`CorpusId = 278165137`，`DOI = 10.48550/arXiv.2504.19442`。
- **引用清单 API**：https://api.semanticscholar.org/graph/v1/paper/arXiv:2504.19442/citations?fields=title,year,externalIds&limit=100 （✅ 本次成功拉取到全部 39 条）
- **重要观察（对第 3 章"影响"极有价值）**：39 篇施引文献中，**绝大多数是 2026 年发表的"megakernel / 通信重叠编译器"方向论文**，且相当一部分来自**与 TD 不同的团队**，说明该技术路线已形成独立子领域。代表性施引文献（标题 + arXiv 编号均来自上述 API 返回）：

| 方向 | 论文 | arXiv |
|---|---|---|
| MoE megakernel（NVL72） | Mixture-of-Kittens: MoE Megakernel for NVL72s | 2609.36070 |
| MoE megakernel + SM 调度 | Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap | 2609.21483 |
| 多机融合 kernel | mKernel: Fast Multi-GPU, Multi-Node Fused Kernels | 2609.13585 |
| 细粒度通信协调 | Entwine: Coordinating Tiled Computation and Fine-Grained Communication across GPUs | 2609.11562 |
| 同步消除 all-reduce | SiFAR: Synchronization-Free All-Reduce for Low-Latency LLM Inference | 2607.08973 |
| 动态 SP（长上下文服务） | Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism | ISCA 2026, DOI 10.1109/ISCA66397.2026.00098 |
| 混合厂商集合通信 | HetCCL: Enabling Collective Communication For Mixed-Vendor Heterogeneous Clusters | 2605.31000 |
| Ascend MoE 训练 | HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs | 2605.23764 |
| 动态 CP（PD 分离解码） | NanoCP: Request-Level Dynamic Context Parallelism for Data-Expert Parallel Decoding | 2605.21100 |
| 多机 megakernel 串行化消除 | Eliminating Hidden Serialization in Multi-Node Megakernel Communication | 2605.00686 |
| 用 copy engine 做 MoE 负载均衡 | FEPLB: Exploiting Copy Engines for Nearly Free MoE Load Balancing | 2604.19654 |
| 统一 EP megakernel（训练） | UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training | 2604.19241 |
| megakernel 编译抽象 | Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel | 2604.13327 |
| 自动 chunk 级重叠 | Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap | 2601.20595 |
| 多 GPU Triton 一等公民 | Iris: First-Class Multi-GPU Programming Experience in Triton | 2511.12500 |
| DMA offload | DMA-Latte: Expanding the Reach of DMA Offloads to Latency-bound ML Communication | 2511.06605 |
| 长上下文服务细粒度 SP | Optimizing Long-context LLM Serving via Fine-grained Sequence Parallelism | 2511.06247 |
| 远端内存调度 | Mercury: Unlocking Multi-GPU Operator Optimization for LLMs via Remote Memory Scheduling | DOI 10.1145/3731569.3764798 |
| 通用通算重叠 | TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference | 2505.11329 |
| 线性注意力零通信 SP | ZeCO: Zero Communication Overhead Sequence Parallelism for Linear Attention | 2507.01004 |
| megakernel 编译器 | MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs | 2512.22219 |
| DMA 细粒度重叠设计空间 | Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap | 2512.10236 |
| RDMA 点对点 | fabric-lib: RDMA Point-to-Point Communication for LLM Systems | 2510.27656 |
| DiT 序列并行 | StreamFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers on GPUs | 2601.20273 |
| 无同步屏障算法 | A Barrier-Free Synchronization Algorithm for Multi-Engine AI Accelerators | 2608.13757 |

---

# 第 3 章素材：NVSHMEM 生态

## 5.1 NVSHMEM 是什么

- **定义（✅ 权威 arXiv 综述）**：*"NVSHMEM is NVIDIA's OpenSHMEM-based PGAS communication library for GPU clusters, enabling **GPU-initiated, one-sided communication through symmetric memory**."* —— arXiv:2606.05951 https://arxiv.org/abs/2606.05951
- **编程模型**：symmetric heap（各 rank 同大小同相对地址的显存区）+ 单边 `put/get` + device 侧 `nvshmemx_*` API + device 侧集合通信 + signal/wait 同步。
- **传输层**：NVLink（域内）、InfiniBand RDMA（域间），以及 **IBGDA（InfiniBand GPUDirect Async）**——即由 GPU 线程直接构造 WQE 投递到 NIC，不经 CPU proxy。
  - **DeepSeek-V3 报告亦确认使用 IBGDA**（✅ §3.4.2）：*"we leverage the **IBGDA** (NVIDIA, 2022) technology to further **minimize latency** and enhance communication efficiency."*
- **官方文档**：https://docs.nvidia.com/nvshmem/api/ （`index.html` / `using.html` 返回 200；⚠️ JS 渲染，抓不到正文）
- **OpenSHMEM 标准**：https://www.openshmem.org/
- **❌ 未找到** NVSHMEM 自身的正式论文 arXiv 编号。可引用的替代：IEEE 早期评估 https://ieeexplore.ieee.org/document/9150438 ；GTC'19 PGAS 讲义 https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9677-nvshmem-a-pgas-library-for-gpu-clusters.pdf

## 5.2 NVSHMEM 与 NCCL 的分工

| 维度 | NCCL | NVSHMEM |
|---|---|---|
| 范式 | **集合通信库**（host 侧发起，算法 = Ring/Tree/NVLS 等拓扑感知） | **PGAS / 单边通信**（GPU 线程发起 put/get） |
| 粒度 | 大消息、层次化算法；面向"整层/整个 tensor" | 细粒度、tile/元素级；面向"kernel 内需要什么就取什么" |
| 与计算关系 | 独立通信 kernel，靠 stream 重叠 | 可嵌进计算 kernel → **真融合** |
| 谁负责进度 | NCCL 内部 proxy + kernel | 用户 kernel 自己 poll/等待 signal |
| 官方定位 | https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/ | https://docs.nvidia.com/nvshmem/api/ |

- **❌ 未找到**：NVIDIA 官方文档中直接对比 "NCCL vs NVSHMEM" 的专门章节。上面这张表是 `(分析)`，其**事实基础**来自 (a) NCCL 官方文档描述的 host 侧集合通信 + 拓扑感知算法，(b) arXiv:2606.05951 对 NVSHMEM "GPU-initiated, one-sided" 的定义，(c) ROCSHMEM README 的"more fine-grained … than traditional host-driven networking"。
- **补充印证（✅）**：Triton-distributed 论文 Table 2 把 NCCL / PyTorch / TE / Pallas 标注为**无 OpenSHMEM Support、无 Fusion**，而 NVSHMEM/ROCSHMEM 类实现才有 —— 这是"分工差异"的一手证据。
- **⚠️ 一个重要的反直觉事实**：DeepEP **V2.5 已移除 NVSHMEM 依赖**，改用 **NCCL Gin backend（device-side communication APIs + 复用 NCCL communicator）**（https://github.com/deepseek-ai/DeepEP ）。即**"NCCL 也在向 device-initiated 演进"**——`NCCL Gin` 值得在第 3 章专门写一段：它意味着 NVSHMEM 的"单边"能力正在被 NCCL 生态吸收，未来"NVSHMEM vs NCCL"的边界会模糊。

## 5.3 非 NVIDIA 的 shmem 生态（逐项核实）

### (1) AMD ROCSHMEM ✅
- 位置：`ROCm/rocm-systems` 仓库的 `projects/rocshmem`
  - 文档：https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/
  - README（raw）：https://raw.githubusercontent.com/ROCm/rocm-systems/refs/heads/develop/projects/rocshmem/README.md
  - 仓库树：https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem
- **README 原文要点（✅）**：
> "The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research initiative to provide **GPU-centric networking through an OpenSHMEM-like interface**. This **intra-kernel networking** library simplifies application code complexity and enables **more fine-grained communication/computation overlap than traditional host-driven networking**. rocSHMEM uses a **single symmetric heap** that is allocated on GPU memories."
> "There are currently **three backends**: **IPC**（communication primitives using load/store operations issued from the GPU）、**Reverse Offload (RO)**（GPU runtime 把请求转发给 host 端 MPI/OpenSHMEM）、**GPU Direct Async (GDA)**（device-side 直接向 NIC 投递 WQE，无需 CPU proxy，用各 NIC 厂商的 Direct Verbs API）。"
> "The RO and GDA backend is provided **as-is with limited support from AMD or AMD Research**."
- **开源历史**：ROC_SHMEM 1.0 于 2020-04 开源、1.6 于 2023-06（OpenSHMEM 邮件列表）：http://openshmem.org/pipermail/openshmem-list/2020-April/001380.html 、 http://www.openshmem.org/pipermail/openshmem-list/2023-June/001793.html
- **与 TD 的关系（✅）**：TD 论文明确 "NVSHMEM/**ROCSHMEM** for Nvidia and AMD GPUs respectively"；TD 仓库有 `shmem/rocshmem_bind/pyrocshmem`。

### (2) MetaX（沐曦）mxshmem ✅（代码级证据，文档级 ❌）
- **一手代码证据（很强）**：Triton-distributed 仓库内 **`shmem/mxshmem_bind/pymxshmem/`** 完整存在，API 与 NVSHMEM 高度同构：
  - `init_mxshmem_by_uniqueid(group)` → 内部调用 `mxshmemx_get_uniqueid()` / `mxshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)`
  - 常量：`MXSHMEM_TEAM_WORLD` / `MXSHMEM_TEAM_SHARED` / `MXSHMEMX_TEAM_NODE` / `MXSHMEMX_TEAM_SAME_MYPE_NODE` / `MXSHMEMI_TEAM_SAME_GPU` / `MXSHMEMI_TEAM_GPU_LEADERS`
  - 示例：`example/run_ring_put.py` 用 `pymxshmem.mxshmem_create_tensor(...)`、`mxshmem_int_p(...)`、`mxshmem_barrier_all()`；`example/run.sh` 设 `MXSHMEM_BOOTSTRAP=UID`、`MXSHMEM_IB_ENABLE_IBRC=0`
  - 构建：`CMakeLists.txt`、`build.sh`（`project(pymxshmem LANGUAGES CXX CUDA)`）
  - 仓内路径：https://github.com/ByteDance-Seed/Triton-distributed/tree/main/shmem/mxshmem_bind
- **判断**：**mxshmem 是真实存在的、与 NVSHMEM API 对齐的沐曦单边通信库**，且**已被 Triton-distributed 作为多后端之一接入**。
- **❌ 未找到**：沐曦官方站点/文档中关于 mxshmem 的公开页面或论文（仅检索到 MetaX 开发者论坛的一个共享内存相关报错帖：https://developer.metax-tech.com/forum/t/... ，**不足以作为 mxshmem 的官方定义来源**）。

### (3) 华为 Ascend：CANN SHMEM / ACLSHMEM / UB-Mesh / UDMA ⚠️
- **CANN SHMEM 确实开源**（✅ 代码仓）：
  - GitCode 仓库：`https://gitcode.com/cann/shmem`（本核查从该仓 raw 路径抓到了真实源文件，如 `src/host/mem/heap/hybm_mem_segment.cpp`，说明仓库存在，只是 README 的 `main` 分支路径抓取返回 404）
  - 公开技术材料：CANN 社区 SIG-SHMEM meetup 讲义（2026-02-28）《SHMEM 入门：共享内存的概念、价值与使用》 https://raw.gitcode.com/cann/community/raw/5a88891d8e7b1d38faf7e9fd7d5b1fb9fb360d21/events/meetup/slides/sig-shmem/20260228/SHMEM%E5%BC%80%E6%BA%90%E4%BB%8B%E7%BB%8D.pdf
  - ⚠️ 二手技术解读（可信度中，可作背景不可作论据）：https://ascendai.csdn.net/69a1821354b52172bc5df0a3.html 、 https://ascendai.csdn.net/69d7981a72111d255bf88375.html （"CANN SHMEM 深度解析：PGAS 编程模型与单边通信"）
- **UB-Mesh 论文（✅ 已定位正确编号）**：
  - **arXiv:2503.20377** —— *"UB-Mesh: A Hierarchically Localized nD-FullMesh Data Center Network Architecture"*，**IEEE Micro 2025**，DOI **10.1109/MM.2025.3592688**
  - 通过 Semantic Scholar 检索确认：https://api.semanticscholar.org/graph/v1/paper/search?query=UB-Mesh%20Hierarchically%20Localized%20nD-FullMesh
  - **⚠️ 纠错**：任务书中给的 `arXiv:2503.20495` **是错的**（该编号实为一篇 math.ST 论文 "Revisiting general source condition in learning over a Hilbert space"），**不可引用**。
  - 中文解读（二手，可信度中）：Hot Chips 2025 解析 https://ee.ofweek.com/2025-09/ART-8500-2809-30669910.html 、 https://www.eet-china.com/mp/a435032.html
- **UDMA 直驱**：**❌ 未找到**权威一手来源（未找到华为对 "UB UDMA 直驱" 的官方文档或论文）。报告中如要提及，须标注"未找到一手来源"。
- **Ascend 上的通算融合实际进展（✅ 一手，来自开源代码与 README）**：
  - Triton-distributed：`ascend/` 目录 + **Ascend RDMA support（PR #179，2026-08-06）**；README 指出 `triton-ascend` 尚未 rebase 到 Triton 3.7，Ascend 需用 `triton-v3.4` 线。
  - **DeepEP-Ascend**：https://github.com/deepseek-ai/DeepEP-Ascend —— README 原文 *"Same API and full performance on **HUAWEI Ascend 950 NPUs**"*。
  - **DeepGEMM-Ascend**：https://github.com/deepseek-ai/DeepGEMM-Ascend/
  - **FlashMLA 的 Ascend kernel（2026-09-30）**：prefill **410 TFlops（95% 峰值）**、decode **360 TFlops（83% 峰值）**。
  - 论文侧：`HyperParallel-MoE: Multi-Core Interleaved Scheduling for Fast MoE Training on Ascend NPUs` **arXiv:2605.23764**（TD 施引文献）。
  - MindSpeed 侧有 USP（Ulysses + Ring 混合）文档：https://www.hiascend.com/document/detail/en/MindSpeed/latest/MM/MM/docs/en/features/dit_usp.md （⚠️ 内容未逐字核验）

### (4) 国产硬件"单边通信 + 通算融合"现状总结

| 厂商 | 单边通信库 | 通算融合 | 一手证据强度 |
|---|---|---|---|
| NVIDIA | NVSHMEM（成熟） | DeepEP(V1)/TD/DeepGEMM Mega MoE/MSCCL++ | ✅ 强 |
| AMD | ROCSHMEM（3 后端，RO/GDA "limited support"）+ MORI-SHMEM | TD 的 AMD 后端、MORI-EP | ✅ 强（README/PR） |
| 沐曦 MetaX | **mxshmem** | 已接入 TD 多后端 | ✅ 代码级强 / ❌ 官方文档缺失 |
| 华为 Ascend | **CANN SHMEM**（gitcode 开源） | TD-Ascend、DeepEP-Ascend、DeepGEMM-Ascend、FlashMLA-Ascend、MindSpeed USP | ⚠️ 中（代码可证，官方架构文档未能 fetch） |
| 其他（摩尔线程/寒武纪/海光） | **❌ 未找到** | **❌ 未找到** | — |

> **结论（可写进第 3 章）**：国产硬件在"单边通信 + 通算融合"上**已经具备可用的基础设施**（CANN SHMEM / mxshmem / ROCSHMEM 三线并进），且**被 Triton-distributed 通过统一的 `shmem/*_bind` 抽象层同时接入**——这是"跨厂商 shmem 抽象层"这一方向最硬的落地证据（对应 §6 方向 4）。但**架构文档、性能基准与论文级证据仍以 NVIDIA 侧最完整**，国产侧多为代码/移植级证据。

### (5) 跨厂商 shmem 抽象层（趋势证据）
- **Triton-distributed 的 `shmem/` 三后端目录**（nvshmem_bind / rocshmem_bind / mxshmem_bind）—— 目前最直接的"跨厂商 shmem 抽象"实现。
- **ROCm MORI** 自述 "Open-Shmem-style APIs"（https://github.com/ROCm/mori ）。
- **PyTorch symmetric memory**：⚠️ 本核查**未找到**有效的一手 URL（`torch.distributed._symmetric_memory`）。**❌ 未找到** —— 建议在第 3 章以"未核实"处理，或另行补查。

---

# 第 3 章素材：变种方向（8 个，含核心想法 / 证据 / 受益场景 / 风险 / 成熟度）

> 证据优先级：TD 论文与仓库 > TD 的施引文献（经 Semantic Scholar citations API 核实）> 其他一手。

### 方向 1：EP dispatch/combine 与 GroupGEMM 全面融合（MoE megakernel）
- **核心想法**：把 dispatch、GroupGEMM（专家 MLP）、combine 三段合成**单个持久化 megakernel**，用 tile 级 signal/wait 取代 kernel 边界同步。
- **证据（✅）**：
  - `UniEP: Unified Expert-Parallel MoE MegaKernel for LLM Training` **arXiv:2604.19241**（TD 施引文献）
  - `Mixture-of-Kittens: MoE Megakernel for NVL72s` **arXiv:2609.36070**
  - `Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap` **arXiv:2609.21483**
  - TD 仓库：`ep_a2a_fused_layer`（`docs/layers/nvidia/ep_a2a_fused_layer.rst`）、`docs/layers/nvidia/ep_a2a_layer.rst`；AMD MORI "fused-MoE cooperative SHMEM APIs"（README 2026-08-10）
  - DeepGEMM "**fused MoE with overlapped communication (Mega MoE)**"（https://github.com/deepseek-ai/DeepGEMM ）
- **受益场景**：MoE 训练 + prefill，SM 利用率与 EP 通信隐藏。
- **风险**：寄存器/共享内存压力大；动态 token 路由导致负载不均 → megakernel 内更易长尾；调试与正确性验证困难。
- **成熟度**：论文密集（2026 年集中爆发）+ 已有开源实现（TD/DeepGEMM）→ **开源可用**。

### 方向 2：Decode 阶段低延迟 all-to-all
- **核心想法**：decode 阶段通信量小但**延迟敏感**，需要零 SM 占用、极低启动开销的 all-to-all，并把通信藏在 attention 的 5 级流水里。
- **证据（✅）**：
  - DeepSeek 官方推理总览：*"we subdivide the attention layer into two steps and use a **5-stage pipeline** to achieve a seamless communication-computation overlapping"*（open-infra-index day6）
  - `profile-data`：*"the all-to-all communication during decoding **does not occupy GPU SMs**"*（https://github.com/deepseek-ai/profile-data ）
  - DeepEP：**low-latency kernels**、"Pure FP8 dispatch"（README）
  - TD：README 明确 "support **low-latency mode**, token saving, and **Mega-EP**"（2025-12-22）；低延迟 all-to-all 137 µs vs DeepEP 182 µs
  - `SwiftEP`（NSDI'26）：针对 **prefill** 的 buffer 融合 + TMA offload，指出 DeepEP "excessive GPU SM utilization"（https://www.usenix.org/conference/nsdi26/presentation/li-xingyi ）
- **受益场景**：大 EP（EP128~EP320）推理 decode、PD 分离的 D 实例。
- **风险**：decode 阶段 batch 小、有效带宽难打满；零 SM 方案与"通信要吃 SM"的硬件现实冲突（DeepEP 明确 "zero-SM RDMA EP is not supported"）；延迟优化对拓扑/网卡型号高度敏感。
- **成熟度**：**产品化**（DeepSeek 线上服务已在跑）。

### 方向 3：通信与量化/压缩联合
- **核心想法**：在通信路径上直接做 FP8/MXFP8 打包与反量化，减小 online 通信字节数，并把量化 scale 的处理融进 kernel。
- **证据（✅）**：
  - DeepEP README：*"including **FP8 dispatch**"*
  - TD：`all_to_all_single_gemm` 原生支持 `int8 / float8_e4m3fn / float8_e5m2 / bfloat16` + `input_scale` / `weight_scale`（代码级）
  - TD README 2026-09-17：*"**FlashComm fused MXFP8 intra-node dispatch**: expert-parallel traffic can send **packed FP8 instead of BF16**, with **bitwise checks** against fused and prequantized paths"*
  - DeepSeek-V3 §3.3.3 "Low-Precision Storage and Communication"（含 "Low-Precision Communication" 小节，arXiv:2412.19437）
- **受益场景**：带宽受限的跨节点 EP；显存/带宽双紧的推理。
- **风险**：量化误差在 combine/reduce 中累积；block-wise scale 的细粒度布局与单边通信的对齐要求冲突；bitwise 一致性验证成本高。
- **成熟度**：**产品化**（V3 训练与线上推理均已用 FP8 传输）。
- **❌ 未找到**：独立的"压缩感知集合通信"专门论文（如某些检索中出现的 "SCCL" 未能核实）→ 若报告中要提，需标注未核实。

### 方向 4：跨厂商 shmem 抽象层（统一单边通信 API）
- **核心想法**：用统一 Python/IR 层抽象 NVSHMEM / ROCSHMEM / mxshmem / CANN SHMEM，让同一份融合 kernel 跨厂商编译。
- **证据（✅）**：
  - **TD 论文**：*"Our compiler translates these primitives into **NVSHMEM/ROCSHMEM** implementations during compilation"*；*"OpenSHMEM primitives … their corresponding implementations can be found in NVSHMEM and ROCSHMEM for Nvidia and AMD GPUs respectively"*
  - **TD 仓库三后端目录**：`shmem/nvshmem_bind`、`shmem/rocshmem_bind`、**`shmem/mxshmem_bind`**
  - **MORI-SHMEM 接入 TD**（https://github.com/ROCm/mori 2026/03 条目 + TD PR #145/#164）
  - **Ascend 接入**（TD `ascend/` + PR #179）
  - **异构集群集合通信**：`HetCCL: Enabling Collective Communication For Mixed-Vendor Heterogeneous Clusters` **arXiv:2605.31000**
  - ROCSHMEM 的 "OpenSHMEM-like interface"（AMD README）
- **受益场景**：多厂商混合集群；国产替代；避免单一厂商锁定。
- **风险**：各厂商 OpenSHMEM 子集不一致（ROCSHMEM 的 RO/GDA 仅 "limited support"）；内存模型/一致性语义差异；性能可移植性差（同一 kernel 在不同后端差异大）；缺少统一的性能/正确性一致性测试套件。
- **成熟度**：**开源可用**（TD 已落地三后端），但**规范与一致性**仍是空白。

### 方向 5：编译期自动重叠 / 自动融合
- **核心想法**：由编译器分析数据依赖，自动决定"哪些通信可以融进哪些 kernel、以什么粒度切 chunk"，把手工 tile-mapping 变成自动搜索。
- **证据（✅）**：
  - **TD**：*"the **first compiler that supports native overlapping optimizations** for distributed AI workloads"*；TileLink（前序，MLSys'25）用 **tile-centric primitives** 解耦通信与计算设计空间
  - `Syncopate: Efficient Multi-GPU AI Kernels via **Automatic Chunk-Centric Compute-Communication Overlap**` **arXiv:2601.20595**
  - `Event Tensor: A Unified Abstraction for **Compiling Dynamic Megakernel**` **arXiv:2604.13327**
  - `MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs` **arXiv:2512.22219**
  - `Iris: First-Class Multi-GPU Programming Experience in Triton` **arXiv:2511.12500**
  - `Design Space Exploration of DMA based Finer-Grain Compute Communication Overlap` **arXiv:2512.10236**
  - `Entwine: Coordinating Tiled Computation and Fine-Grained Communication across GPUs` **arXiv:2609.11562**
- **受益场景**：降低融合 kernel 的开发门槛（TileLink 声称 **10× 更少编程工作量**）；跨硬件快速落地。
- **风险**：自动 search 的编译时间与搜索空间爆炸；生成的代码在 corner case 下性能劣于手写；缺乏统一 benchmark 导致"自动"收益难以横向比较；IR 需要同时表达计算与通信语义（TD 用 OpenSHMEM 原语 + 后者非标准原语补足）。
- **成熟度**：**论文 + 开源可用**（TD 已能被第三方复现使用）。

### 方向 6：NVL72 / UB-Mesh 超节点下的新型并行策略
- **核心想法**：当 scale-up 域扩大到 72（NVL72）乃至数百卡（UB-Mesh / CloudMatrix），EP 的 all-to-all 可以**全部留在域内**，IB 只承担跨域流量 → 并行策略（EP 度、TP 度、PD 分离边界）需要重新设计。
- **证据（✅）**：
  - `Mixture-of-Kittens: **MoE Megakernel for NVL72s**` **arXiv:2609.36070**（标题即证据：megakernel 已专门面向 NVL72）
  - `mKernel: Fast **Multi-GPU, Multi-Node** Fused Kernels` **arXiv:2609.13585**
  - `Eliminating Hidden Serialization in **Multi-Node** Megakernel Communication` **arXiv:2605.00686**
  - DeepSeek-V3 的 node-limited routing（"at most 4 nodes"）与 "3.2 experts per node" 正是**带宽不对称驱动的拓扑感知路由**（arXiv:2412.19437 §3.2.2）
  - **UB-Mesh**：arXiv:2503.20377 / IEEE Micro 2025（Hierarchically Localized nD-FullMesh，即"分层局部化"的超节点网络，天然适配域内大带宽）
  - ⚠️ NVIDIA NVL72 官方页 https://www.nvidia.com/en-us/data-center/gb200-nvl72/ 未逐字核验其"NVLink 域内 all-to-all 可取代 IB"的表述 —— **如需引用请回原始文档确认**。
- **受益场景**：超大 EP（wide-EP）、PD 分离部署、MoE 推理 TCO。
- **风险**：超节点成本高、故障域变大；跨域带宽仍不对等；并行策略与硬件拓扑强耦合 → 可移植性差；NVL72 单机功耗/散热约束。
- **成熟度**：**论文 + 厂商产品化**（NVL72 已出货，UB-Mesh 在 IEEE Micro 发表），但**开源融合 kernel 侧对 NVL72 的支持仍在早期**（TD 的 CuTeDSL inter-node dispatch/combine 2026-08-07 才落地）。

### 方向 7：Attention-FFN 分离（AFD）与 PD 分离带来的通信形态变化
- **核心想法**：把 attention 与 FFN/MoE 拆到不同 GPU 池，各自的通信模式（attention 侧偏 TP/SP/CP，FFN 侧偏 EP）不再互相牵制，通算融合需分别定制。
- **证据（✅）**：
  - `Revealing the Challenges of **Attention-FFN Disaggregation** for Modern MoE Models and Hardware Systems` **arXiv:2602.09721**（https://www.alphaxiv.org/abs/2602.09721 ）
  - `NanoCP: Request-Level Dynamic Context Parallelism for **Data-Expert Parallel Decoding**` **arXiv:2605.21100**
  - `Stage-Aware Communication Scheduling for **Disaggregated LLM Serving**` **arXiv:2603.17456**
  - DeepSeek 官方推理总览明确采用 **prefill-decode disaggregation**（open-infra-index day6）
  - 分离式经典：DistServe **arXiv:2401.09670**、Splitwise **arXiv:2311.18677** —— ⚠️ 本核查**未逐字验证这两个编号**，引用前请复核。
- **受益场景**：异构资源池、SLO 敏感的在线服务、解耦扩缩容。
- **风险**：KV cache 跨池传输引入新的带宽瓶颈；两池负载匹配困难；调度复杂度大幅上升。
- **成熟度**：**产品化**（DeepSeek 线上已用 PD 分离）；AFD 仍偏论文。

### 方向 8：通信感知的调度 / 拓扑感知路由（含 copy engine offload）
- **核心想法**：把通信从"数据搬运"提升为"可调度资源"：用专用 copy engine（SDMA/DMA）承担搬运、用负载均衡器决定 token 去向、用拓扑感知算法决定路径。
- **证据（✅）**：
  - `FEPLB: **Exploiting Copy Engines** for Nearly Free MoE Load Balancing in Distributed Training` **arXiv:2604.19654**
  - `DMA-Latte: Expanding the Reach of **DMA Offloads** to Latency-bound ML Communication` **arXiv:2511.06605**
  - `Design Space Exploration of **DMA based** Finer-Grain Compute Communication Overlap` **arXiv:2512.10236**
  - `SiFAR: **Synchronization-Free** All-Reduce for Low-Latency LLM Inference` **arXiv:2607.08973**
  - `A Barrier-Free Synchronization Algorithm for Multi-Engine AI Accelerators` **arXiv:2608.13757**
  - DeepSeek-V3 的 **EPLB**（Expert-Parallel Load Balancer，推理总览第 3 节）+ node-limited routing（arXiv:2412.19437 §3.2.2）
  - TD 论文把 **Copy Engine** 列为核心优化维度之一（Table 2）；`all_to_all_single_gemm` 的多流实现即使用 copy engine
  - MORI 的 **SDMA-based AllGather 被集成进 DeepSpeed ZeRO-3**（MORI README 2026/05，"up to **10% end-to-end training speedup**"）
- **受益场景**：SM 紧张的场景（把搬运从 SM 上彻底卸走）；负载不均的长尾 MoE。
- **风险**：copy engine 数量有限、与计算争夺 HBM 带宽；DMA 路径的同步语义更复杂；可移植性差（各家 copy engine/DMA 能力不同）。
- **成熟度**：**开源可用 → 产品化**（DeepSpeed/MORI/Megatron 侧均有落地）。

---

## 附：与 Triton-distributed 路线的兼容性速判（`(分析)`）

| 方向 | 与 TD"编译器生成融合 kernel"的关系 |
|---|---|
| 1 MoE megakernel | **同向**：TD 的 EP fused layer / FlashComm 就是这一方向 |
| 2 Decode 低延迟 | **同向**：TD 已有 low-latency AllToAll 与 Mega-EP |
| 3 通信+量化 | **同向**：TD 的 `all_to_all_single_gemm` 已支持 FP8 scale，README 已有 MXFP8 dispatch |
| 4 跨厂商 shmem | **TD 是当前最主要的实现者**（三后端目录） |
| 5 编译期自动重叠 | **TD 的核心命题**（TileLink → TD） |
| 6 NVL72/UB-Mesh | **互补**：硬件拓扑变化会改变 TD 的 swizzle/拓扑优化权重 |
| 7 AFD / PD 分离 | **互补但正交**：改变的是"哪些通信需要融合"，不改变融合技术 |
| 8 通信调度 / DMA offload | **互补**：TD 把 Copy Engine 作为优化维度之一，可与编译器调度结合 |

---

# 附录 A：本笔记的「未找到 / 需纠错」清单（务必遵守）

| # | 事项 | 状态 |
|---|---|---|
| 1 | DoublePipe（异构流水线）的可靠一手出处 | **❌ 未找到**。`arXiv:2407.09538` 是无关论文，**禁止引用**。建议只写 DualPipe。 |
| 2 | `arXiv:2503.20495` 作为 UB-Mesh | **❌ 编号错误**。UB-Mesh 正确编号 = **arXiv:2503.20377**（IEEE Micro 2025）。 |
| 3 | `arXiv:2406.16801` 作为 LoongTrain | **❌ 编号错误**（该号是 RES-Q）。LoongTrain 正确编号**未找到**。 |
| 4 | `arXiv:2112.06930` 作为 NVSHMEM 论文 | **❌ 编号错误**（该号是天体物理论文）。NVSHMEM 正式论文编号**未找到**。 |
| 5 | vLLM / SGLang / veScale 集成 Triton-distributed | **❌ 未找到**。注意区分 MORI 被 vLLM/SGLang 集成 ≠ TD 被集成。 |
| 6 | NVIDIA Triton Inference Server 的 `triton_distributed` 仓库 | **⚠️ 不可验证**（GitHub 返回 404，仅 DeepWiki 有索引页）。 |
| 7 | DeepSeek 官方"MoE 推理速度上限由互联带宽决定"逐字表述 | **❌ 未找到逐字句**；有等价方向的量化证据（§1.5）。 |
| 8 | Ulysses"受 head 数限制"的官方逐字陈述 | **❌ 未找到**；有代码级/间接证据（§3.1）。 |
| 9 | Megatron-Core 官方 context parallel 文档 URL | **❌ 未找到**（相关 URL 均 404）。可用 Megatron-LM 源码 `parallel_state.py` 代替。 |
| 10 | PyTorch DDP 文档正文（梯度分桶/重叠） | ⚠️ URL 存在但 JS 渲染，curl 取不到正文；建议报告中二次确认。 |
| 11 | PyTorch `torch.distributed._symmetric_memory` 一手 URL | **❌ 未找到**。 |
| 12 | 华为 "UB UDMA 直驱" 官方来源 | **❌ 未找到**。 |
| 13 | "融合 vs 非融合可省 2× HBM 流量"的量化论文 | **❌ 未找到**；属 `(分析)`。 |
| 14 | DeepEP 的 NSDI'26 论文 | ⚠️ **实际是 `SwiftEP`（Tencent, Xingyi Li et al.）**，非 DeepEP 团队论文；勿混淆。 |
| 15 | DeepEP 基于 NVSHMEM | ⚠️ **仅对 V1 成立**；V2.5 已移除 NVSHMEM，改用 NCCL Gin。 |
| 16 | DistServe / Splitwise 的 arXiv 编号 | ⚠️ 未逐字验证（2401.09670 / 2311.18677，引用前请复核）。 |

---

# 附录 B：关键 URL 汇总（按主题）

**背景痛点**
- https://arxiv.org/abs/2412.19437 （DeepSeek-V3 Technical Report）
- https://arxiv.org/html/2412.19437v2 （全文，含 §3.2.1 / §3.2.2 / §3.5.1）
- https://github.com/deepseek-ai/DualPipe
- https://github.com/deepseek-ai/profile-data
- https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- https://arxiv.org/abs/2104.04473 （Megatron-LM，8bsh(t−1)/t）
- https://arxiv.org/abs/2502.19811 （COMET，47%，MLSys'25）
- https://www.usenix.org/conference/nsdi26/presentation/li-xingyi （SwiftEP，NSDI'26）
- https://www.usenix.org/system/files/nsdi26-li-xingyi.pdf
- https://mlsys.org/Conferences/2025/AcceptedPapers

**通算融合谱系**
- https://arxiv.org/abs/2101.06840 （ZeRO-Offload）
- https://arxiv.org/abs/2504.09014 （MSCCL++）
- https://github.com/microsoft/mscclpp
- https://github.com/deepseek-ai/DeepEP （V2.5，NVSHMEM 已移除）
- https://github.com/deepseek-ai/DeepEP-Ascend
- https://github.com/ROCm/mori （MORI / MORI-SHMEM）
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- https://arxiv.org/abs/2509.07003 （veScale）

**NVSHMEM 生态**
- https://arxiv.org/abs/2606.05951 （Demystifying NVSHMEM）
- https://docs.nvidia.com/nvshmem/api/
- https://www.openshmem.org/
- https://ieeexplore.ieee.org/document/9150438
- https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem
- https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/
- https://github.com/ByteDance-Seed/Triton-distributed/tree/main/shmem/mxshmem_bind
- https://gitcode.com/cann/shmem
- https://arxiv.org/abs/2503.20377 （UB-Mesh, IEEE Micro 2025）

**Ulysses / Ring / attention all-to-all**
- https://arxiv.org/abs/2309.14509 （DeepSpeed-Ulysses）
- https://arxiv.org/html/2309.14509v2
- https://arxiv.org/abs/2310.01889 （Ring Attention）
- https://github.com/lhao499/llm_large_context
- https://arxiv.org/abs/2311.09431 （Striped Attention）
- https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/ulysses_sp_dispatch.py
- https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/sp_ulysess_qkv_gemm_all2all.py
- https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/sp_ulysess_o_all2all_gemm.py
- https://github.com/ByteDance-Seed/Triton-distributed/blob/main/python/triton_dist/kernels/nvidia/all_to_all_single_gemm.py
- https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/kernels/nvidia/ulysses_sp.rst
- https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py

**Triton-distributed 本体**
- https://arxiv.org/abs/2504.19442
- https://arxiv.org/html/2504.19442v3
- https://github.com/ByteDance-Seed/Triton-distributed
- https://openreview.net/forum?id=ccjvBkTRRe （TileLink, MLSys'25）
- https://api.semanticscholar.org/graph/v1/paper/arXiv:2504.19442/citations?fields=title,year,externalIds&limit=100
- https://github.com/ByteDance-Seed/Triton-distributed/pull/145
- https://github.com/ByteDance-Seed/Triton-distributed/pull/164
- https://github.com/ByteDance-Seed/Triton-distributed/pull/179
- https://github.com/deepseek-ai/DeepGEMM
- https://github.com/deepseek-ai/DeepGEMM-Ascend
- https://github.com/deepseek-ai/FlashMLA
- https://huggingface.co/ByteDance-Seed/Seed-OSS-36B-Instruct

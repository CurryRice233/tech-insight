# Triton-distributed（ByteDance-Seed）机制精读调研笔记

> **对象**：Triton-distributed —— 论文 `arXiv:2504.19442`（"Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler"）；代码 [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)；文档 [triton-distributed.readthedocs.io](https://triton-distributed.readthedocs.io/en/latest/)
> **调研时间**：2026-10-14（北京时间 15:14–15:40）。GitHub/GitCode API 数字为 2026-10-14 15:14（+08:00）抓取快照。
> **标注约定**：**【事实】**= 有可核验 URL / 源码路径支撑（含直接引文）；**【推测】**= 基于命名与结构的推断，无直接文字证据；**【未找到】**= 在一手来源中未检索到，明确记录检索方式。
> **一手来源与方法**：arXiv abs 页 + arXiv HTML v3（`https://arxiv.org/html/2504.19442v3`，用脚本转纯文本后按节读取）；`git clone --depth 1` 上游仓库（默认分支 `main`）逐目录读源码；`git clone -b triton-v3.4` 校验 Ascend 停放目录；ReadTheDocs `primitives.html`/`index.html` 原始 HTML；`https://api.github.com/repos/ByteDance-Seed/Triton-distributed`；GitCode 仓库 clone + `https://api.gitcode.com/api/v5/repos/Ascend/Triton-distributed-ascend/pulls/114`。
> **注意**：本次调查未创建任何临时落盘件（`/tmp` 在每次 bash 调用间被清空，所有解析均在单次调用内完成）。

---

## 1. 项目基本信息

### 1.1 论文

| 项 | 值 | 出处 |
|---|---|---|
| 标题 | *Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler* | [abs 页](https://arxiv.org/abs/2504.19442) |
| arXiv 编号 | **2504.19442**（cs.DC，Distributed, Parallel, and Cluster Computing） | 同上 |
| 版本 | v1 **2025-04-28 03:09:22 UTC**；v2 2025-05-04；v3 **2025-06-05**（当前版本 v3） | abs 页 Submission history |
| DOI | 10.48550/arXiv.2504.19442 | abs 页 |
| 体例 | 技术报告（tech report），**不是**会议论文：abs 页无 venue/comment 字段，正文只有 §1 Introduction → §5 Conclusion + References | 【事实】abs 页 + HTML v3 目录 |

- **作者（22 人，按序）**【事实】：Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang, Tianqi Li, Haojie Duanmu, Renze Chen, Ruifan Xu, Yifan Guo, Ningxin Zheng, Ziheng Jiang, Xinyi Di, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Liqiang Lu, Yun Liang, Jidong Zhai, Xin Liu。来源：[abs 页](https://arxiv.org/abs/2504.19442)、[README.md 的 BibTeX](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/README.md)。
- **团队核实**【事实】：
  - 文档站点页脚：**"© Copyright 2025 ByteDance Seed Infrastructures Team."** —— 出现在 `primitives.html` 与文档首页（原始 HTML 抓取）。→ 用户给出的页脚说法**属实**。
  - 仓库 README 自述："We are **ByteDance Seed team.**"（[README.md](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/README.md)）。
  - 论文 HTML 作者块中可见 `zheng.size@bytedance.com`、`wenlei.bao@bytedance.com`、`liuxin.ai@bytedance.com`（ByteDance 邮箱），HTML 中出现 "Peking University" 字样 1 次。
  - **【未找到】**完整机构列表（arXiv HTML 的 affiliation 标记在抓取中被剥离；未逐作者核实单位，例如 Yun Liang / Jidong Zhai 是否署名北京大学**未证实**）。
- **相关联的会议论文（勿混淆）**【事实】：`docs/primitives.md` 与 `primitives.html` 说高层原语"as described in our **MLSys 2025 paper**"，指向 [MLSys 2025 poster 3248](https://mlsys.org/virtual/2025/poster/3248) = **TileLink**（`arXiv:2503.20313`，README 引用为 `zheng2025tilelink`）。→ **MLSys 2025 收录的是 TileLink，不是本报告 2504.19442**。

### 1.2 仓库、版本与活跃度（GitHub API 快照 2026-10-14 15:14 +08:00）

| 字段 | 值 |
|---|---|
| full_name | `ByteDance-Seed/Triton-distributed` |
| description | "Distributed Compiler and Optimized Parallel Kernels" |
| 创建时间 | **2025-04-02T06:57:03Z** |
| 最近 push | **2026-09-18T07:07:53Z** |
| updated_at | 2026-10-13T09:55:30Z |
| **star** | **1566**（watchers_count 同值） |
| fork | 176 |
| subscribers | 19 |
| open issues | 56 |
| license | MIT（`spdx_id: MIT`）；README 补充：`python/triton_dist/kernels/flash_decode.py` 部分为 Apache-2.0 |
| 语言 / 默认分支 | Python / `main` |
| size | 10702 KB |
| homepage | `https://triton-distributed.readthedocs.io/en/latest/` |

- **最近提交**【事实】（`api.github.com/.../commits?per_page=5`）：`7908e4ea90` 2026-09-18 "Merge pull request #208 from peterjl/sync/triton-3.7-main"；`e66bbf8a33` "fix: drop flaky MoRI EP bench from GitHub AMD CI"；`695ce459b3` "fix: make MoRI ring_put wait for delivery and retry on GitHub unit"；`5bea3ea220` "fix: run GitHub MoRI EP A2A at M=256 so the job cannot hang"；`b195ad1085` "fix: stop AMD e2e from downloading 32B models in an infinite loop"。
- **月度提交量**【事实，基于 `--depth 400` 浅克隆，共 315 条，覆盖全生命周期；`git log --date=format:'%Y-%m' | uniq -c`】：2025-04:6、05:17、06:19、**07:58（峰值）**、08:24、09:21、10:12、11:9、12:32、2026-01:18、02:9、03:6、04:8、05:12、06:14、07:16、08:15、09:19。→ **2026-09 仍有 19 次提交，项目持续活跃**（与 `pushed_at` 2026-09-18 一致）。
- **首次提交 / 版本**【事实 + 注意】：
  - 浅克隆中**最早（根）提交**的 message 为 `Initial commit: Release Triton-distributed source code`，`%ad` 显示 2025-04-30。**但**同一 log 中紧随其后的提交 author date 为 2025-04-08 / 2025-04-21 → author date 不可靠（历史含 rebase/backdate，几乎必然继承了上游 Triton 的旧提交日期）。**确切"首个提交"未进一步核实 → 部分【未找到】**。
  - **Releases / tags**【事实】：`v0.0.2-rc`（2025-09-12）、`v0.0.1-rc`（2025-08-20）、tag `experimental`（2025-07-11）。README 的 pip 安装示例用 `VERSION=v0.0.2`，wheel 名 `triton_dist-3.4.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`。
  - **上游 Triton 版本线**【事实】：README 顶部说明默认分支跟踪 **Triton 3.7**（2026-08-17 起，"builds as an out-of-tree plugin against stock upstream Triton"），旧 **Triton 3.4** 线保留在 [`triton-v3.4`](https://github.com/ByteDance-Seed/Triton-distributed/tree/triton-v3.4) 分支。

### 1.3 支持的硬件后端

- README 原文【事实】："Triton-distributed currently **mainly targets Nvidia GPU and AMD GPU**. It can also be ported to other hardware platforms."
- Roadmap「Backends / Computation」勾选项【事实】：Nvidia **SM90a / SM80 / SM89**、AMD **CDNA3**；「Communication」勾选：**NVLink / IB / PCIe**。

| 后端 | 代码证据（路径） | 说明 |
|---|---|---|
| **NVIDIA** | `python/triton_dist/kernels/nvidia/`、`python/triton_dist/language/extra/cuda/`（`libnvshmem_device.py`、`language_extra.py`、`tma_language.py`）、`lib/Conversion/TritonDistributedToLLVM/NVIDIA/` | 【事实】主力后端；实测 H800、L20（PCIe） |
| **AMD** | `python/triton_dist/kernels/amd/`、`python/triton_dist/language/extra/hip/`（`librocshmem_device.py`、`libmori_shmem_device.py`）、`lib/Conversion/TritonDistributedToLLVM/AMD/`、`shmem/rocshmem_bind/` | 【事实】实测 **MI308X**（论文 §4.3、`docs/amd-perf.md`）；另有 MORI shmem 后端（README news 2026-08-10 "AMD MORI backend update restoring fused-MoE cooperative SHMEM APIs"） |
| **MetaX（沐曦）** | `python/triton_dist/kernels/metax/`（`allgather_gemm.py`、`common_ops.py`、`utils.py`）、`python/triton_dist/language/extra/maca/`（`libmxshmem_device.py`）、`shmem/mxshmem_bind/pymxshmem/`、`lib/Conversion/TritonDistributedToLLVM/METAX/` | 【事实】代码存在（`maca` = MetaX 的类 CUDA 编程模型）；**【未找到】任何 MetaX 性能数字或教程** |
| **Ascend（昇腾）** | 主分支 `ascend/`（"sources (parked)"）、`ascend/python/triton_dist/language/extra/ascend/libaclshmem_device.py`、`ascend/include/.../TritonDistributedToHIVM/`、`ascend/3rdparty/{triton-ascend.patch,AscendNPU-IR.patch}` | 【事实】Triton 3.4 时代的 Ascend 后端被"停放"在主分支；详见第 5 节（另有独立移植仓库） |

---

## 2. 编程模型与原语（`triton_dist.language`）

### 2.1 官方原语清单（逐条，按官方文档分组）

来源：`docs/primitives.md` ≡ [primitives.html](https://triton-distributed.readthedocs.io/en/latest/primitives.html)（"All the primitives are exposed by `triton_dist.language`"）。源码定义在 `python/triton_dist/language/distributed_ops.py`（行号已核对）。

**① Context Querying Primitives**【事实】（`distributed_ops.py:127/133/139`）
```py
rank(axis=-1, _builder=None)          # 当前 rank
num_ranks(axis=-1, _builder=None)     # 参与通信的 rank 数
symm_at(ptr, rank, _builder=None)     # 把本地对称内存指针映射成 rank 上的远端指针
```
**② Signal Control Primitives**【事实】（`distributed_ops.py:87/105/146`）
```py
wait(barrierPtrs, numBarriers, scope: str, semantic: str, _builder=None)   # 实现签名还有 waitValue=1 默认参数
consume_token(value, token, _builder=None)                                 # 与 wait 配对，把 token 变成数据依赖
notify(ptr, rank, signal=1, sig_op="set", comm_scope="inter_node", _builder=None)
```
**③ NVSHMEM-related Primitives**【事实】——"Triton-distributed also expose all the NVSHMEM primitives to Python, allowing users to program communication kernels purely in Python"，全部经 `triton.language.extra.libshmem_device` 暴露：
```py
my_pe() / n_pes() / int_p(dest, value, pe) / remote_ptr(local_ptr, pe)
barrier_all() / barrier_all_block() / barrier_all_warp()
sync_all() / sync_all_block() / sync_all_warp()
quiet() / fence()
getmem_nbi_block(dest, source, bytes, pe) / getmem_block(...) / getmem_nbi_warp(...) / getmem_warp(...) / getmem_nbi(...) / getmem(...)
putmem_block(dest, source, bytes, pe) / putmem_nbi_block(...) / putmem_warp(...) / putmem_nbi_warp(...) / putmem(...) / putmem_nbi(...)
putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe) / putmem_signal(...)
putmem_signal_nbi_block(...) / putmem_signal_block(...) / putmem_signal_nbi_warp(...) / putmem_signal_warp(...)
signal_op(sig_addr, signal, sig_op, pe) / signal_wait_until(sig_addr, cmp_, cmp_val)
```
> 源码里实际导出的集合**远多于**文档（`python/triton_dist/language/extra/libshmem_device.py` 中 100+ 个 `dispatch` 包装），额外包含 `_wave`（AMD wavefront）、`_wg`（NVIDIA warpgroup）、team 相关（`team_my_pe`、`team_n_pes`、`barrier(team)`、`team_sync_block`）、`broadcast*`、`fcollect*`、`putmem_rma*`、`quiet_pe`、`uint64_wait_until_equals` 等（grep 得到）。**【事实】**

**④ High-level primitives**【事实】——文档写"these high-level primitives will be released soon after MLSys 2025"，README Roadmap 中 `- [ ] Release high-level primitives` **仍未勾选**。→ 截至目前高层"tile-centric"原语**未开源**。文档页：[python-api/triton-dist.language.html](https://triton-distributed.readthedocs.io/en/latest/python-api/triton-dist.language.html)、[python-api/triton-dist.semantics.html](https://triton-distributed.readthedocs.io/en/latest/python-api/triton-dist.semantics.html)。

### 2.2 开发者如何"在 Triton kernel 内做通信"

论文 §2.1/§2.2/§2.3 给出三条设计支柱【事实，原文引用】：

1. **MPMD 编程模型**，核心三概念：**Symmetric Memory / Signal Exchange / Async-Task**（§2.1）。
   - *Symmetric Memory*："Each rank allocates a memory buffer in the global scope with the same size. Each memory buffer has a separate address space, and there is **no uniform virtual address (UVA) space** from a global perspective. Remote memory buffers **cannot be accessed directly via pointers**. To perform remote data transfer, specific primitives are required."
   - *Signal Exchange*："A signal is a data object that resides in symmetric memory. There is a fixed set of operations to manipulate signals, including **setting** the value of a signal, **increasing** the value, **checking** the value, and performing a **spin-lock** on a given signal."
   - *Async-Task*："Operations such as data transfer and computation are treated as asynchronous tasks that run in parallel… even on the same rank, the operations are asynchronous. For different hardware backends, there are different ways to implement async-tasks. For GPUs, **multi-streaming and multi-threading** are common choices."
2. **两套原语**（§2.2，Table 1）：OpenSHMEM 语义原语（`my_pe`/`n_pes`/`int_p`/`remote_ptr`/`barrier_all`/`sync_all`/`quiet`/`fence`/`getmem[_nbi]`/`putmem[_nbi]`/`putmem_signal[_nbi]`/`signal_op`/`signal_wait_until`/`broadcast`）+ **非 OpenSHMEM 原语**（`wait`、`consume_token`、`notify`，以及 `atomic_cas`、`atomic_add`、`ld_acquire`、`red_release`、`multimem_ld_reduce`、`multimem_st`）。原文解释：`wait` 与 `consume_token` 配合"construct data dependency between signal operations, and the following MMA operations **for better compiler-based pipelining**"。
3. **典型写法**（§2.3 与 README 的 32-GPU low-latency AllToAll 示例）：kernel 内先用 `libshmem_device.putmem_nbi_block(...)` 把数据推进对端对称内存，再用 `fence()` + `signal_op(...)` 置信号，最后 `signal_wait_until(...)` 自旋等待本 rank 的信号被对端写入（README 示例代码逐行可读，见 [README.md](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/README.md)）。在 token 化的 GEMM 场景中，则用 `wait(...)` 拿到 token、`consume_token(value, token)` 把 token 与后续 `tl.load` 绑定（§2.3："The first primitive `wait` produces a token related to a signal, while the second primitive `consume_token` consumes the token and creates data dependency between the token and the following data load"），从而使不同 tile 只等待自己依赖的通信。
- **编译路径**【事实，原文】："The communication part is implemented with OpenSHMEM primitives, which are further **lowered to LLVM IR along with a bitcode library**. The computation part goes through the original Triton compilation stack to LLVM IR."
- **编译期基础设施**【事实，源码路径】：
  - `include/TritonDistributed/Dialect/Distributed/IR/DistributedOps.td`、`DistributedDialect.td`（新增 Distributed dialect）
  - `include/TritonDistributed/Dialect/SIMT/IR/SIMTOps.td`（SIMT dialect，供 little_kernel/SIMT 编程用）
  - `include/TritonDistributed/Conversion/TritonDistributedToTritonGPU/Passes.td`、`.../TritonDistributedToLLVM/Passes.td`（NVIDIA/AMD/METAX 各自 `DistributedOpToLLVM.cpp`）
  - `plugin/patches/0001-axisinfo-visitor-hook.patch`、`0002-pipeliner-predicate-hook.patch`（对上游 Triton 的 AxisInfo 访问器与软件流水谓词做 hook —— 3.7 起改为 out-of-tree plugin 方式）

---

## 3. `shmem/` 单边通信基座与各后端 device 库

### 3.1 仓库 `shmem/` 目录包含什么（上游 `main` 分支实测）

```
shmem/nvshmem_bind/                 # NVSHMEM 绑定（NVIDIA）
    README.md   (空文件)
    nvshmemi/transfer_device.cu         # 自定义 device 侧实现，编译成 bitcode
    nvshmemi/build_nvshmemi_bc.sh
    runtime/nvshmem_wrapper.cu          # host/runtime 包装
shmem/rocshmem_bind/                # ROCSHMEM 绑定（AMD）
    README.md   (空文件)
    pyrocshmem/ (src/pyrocshmem.cc, python/_pyrocshmem, CMakeLists.txt, setup.py)
    runtime/rocshmem_wrapper.cc
    build.sh / build_rocshmem.sh / scripts/build_rocshmem_device_bc.sh / scripts/build_rshm_ipc_single.sh
shmem/mxshmem_bind/                 # MXSHMEM 绑定（MetaX）
    build.sh
    pymxshmem/ (src/pymxshmem.cc, python/pymxshmem, example, CMakeLists.txt)
```
- **【事实】`shmem/` 下有且仅有 3 个后端目录：`nvshmem_bind`、`rocshmem_bind`、`mxshmem_bind`。** **没有 `aclshmem_bind`**；`3rdparty/shmem` 是一个 gitlink（`.gitmodules` 中 `update = none`，见 `ascend/README.md`），Ascend 的 ACLSHMEM 由该子模块 / 独立仓库提供（见 §5）。
- 相关脚本：`scripts/build_nvshmem_from_src.sh`、`scripts/build_mori_shmem.sh`；FAQ/说明见 `docs/prepare_nvshmem.md`。
- 论文 §2.2 原文：OpenSHMEM 原语"their corresponding implementations can be found in **NVSHMEM and ROCSHMEM** for Nvidia and AMD GPUs respectively"。【事实】

### 3.2 `python/triton_dist/language/extra/` 各文件职责（逐条核实）

| 文件 | 角色 | 证据 |
|---|---|---|
| `extra/libshmem_device.py`（12 KB） | **统一门面 / 后端分发器**。用 `ModuleProxy` + 谓词列表在 import 时选出唯一后端模块，向上暴露 `libshmem_device.xxx` 一个命名空间 | 源码 L28–L41：`_BACKENDS = ((is_cuda, "...extra.cuda.libnvshmem_device"), (is_rocshmem, "...extra.hip.librocshmem_device"), (is_mori_shmem, "...extra.hip.libmori_shmem_device"), (is_maca, "...extra.maca.libmxshmem_device"), (is_ascend, "...extra.ascend.libaclshmem_device"))`【事实】 |
| `extra/cuda/libnvshmem_device.py`（39 KB） | **NVIDIA / NVSHMEM** device 侧原语（含 `multimem`、`_wg` warpgroup 变体） | 文件存在 + `extra/cuda/__init__.py`: `from . import language_extra, libnvshmem_device` |
| `extra/cuda/language_extra.py`（38 KB）、`extra/cuda/tma_language.py`（30 KB） | CUDA 侧 `tid/__syncthreads/ld/st/atomic_*/multimem_*/inline_asm` 与 TMA 描述符语言 | 【事实】被大量 nvidia kernel import |
| `extra/hip/librocshmem_device.py`（14 KB） | **AMD / ROCSHMEM**：`python/triton_dist/test/amd/test_rocshmem_api.py` 为其测试 | 【事实】 |
| `extra/hip/libmori_shmem_device.py`（45 KB） | **AMD / MORI shmem**：为 AMD 上融合 MoE 的协作式 SHMEM API（README news 2026-08-10"AMD MORI backend update restoring fused-MoE cooperative SHMEM APIs"）+ 测试 `test/amd/test_mori_shmem_api.py`、`test_mori_shmem_bw.py` | 【事实（文件+测试+news）】；MORI 的确切全称/来源【未找到】 |
| `extra/maca/libmxshmem_device.py`（9.6 KB，文件头 "Modification Copyright 2025 ByteDance Ltd."） | **MetaX / MXSHMEM** | 【事实】 |
| `extra/ascend/libaclshmem_device.py`（15 KB，文件头 "Copyright (c) Huawei Technologies Co., Ltd. 2026"） | **Ascend / ACLSHMEM** | 【事实】，详见 §5.3 |

- 命名后缀约定【推测，基于函数集合与 AMD/NVIDIA 术语】：`_block` = block/CTA 级、`_warp` = warp 级、`_wave` = AMD wavefront 级、`_wg` = NVIDIA warpgroup 级、无后缀 = 默认（等价 `_wg`）；`_nbi` = non-blocking。

### 3.3 ACLSHMEM（Ascend 的 shmem 实现）——重点核实结论

- **ACLSHMEM 确实存在，且是昇腾侧的 SHMEM 实现**【事实】：
  - 主仓库停放源：`ascend/python/triton_dist/language/extra/ascend/libaclshmem_device.py` + `aclshmem_constants.py` + `algorithm.py`（见 §5.3 路径与 URL）。
  - 独立移植仓库中同名文件位于 `python/triton_dist/language/extra/ascend/libaclshmem_device.py`（从 `gitcode.com/Ascend/Triton-distributed-ascend` clone 实测存在）。
  - 文档明确写："All device APIs are available through `from triton_dist.language.extra import libshmem_device`… For more information about the underlying ACLSHMEM library, refer to the **ACLSHMEM documentation** https://gitcode.com/cann/shmem/blob/master/docs/index.rst"（`docs_ascend/en/api/shmem_device.rst`）。→ **【事实】ACLSHMEM 的库本体在 `gitcode.com/cann/shmem`**。
  - 主机侧 API 示例：`ash.aclshmem_init(attributes)` / `ash.aclshmem_create_tensor(...)` / `ash.aclshmem_free_tensor(...)` / `ash.aclshmem_finalize()` / `ash.set_conf_store_tls(...)`（`tutorials/ascend/reverse-all2all-udma.py`）。
- ACLSHMEM device 侧导出（`grep -nE "^def"`）：`my_pe/n_pes/team_my_pe/team_n_pes/team_translate_pe/int_p/remote_ptr/barrier_all/barrier_all_vec/barrier(team)/barrier_vec(team)/getmem/putmem/getmem_nbi/putmem_nbi/udma_put_nbi/udma_put_signal_nbi/putmem_signal/putmem_signal_nbi/signal_op/signal_wait_until/quiet/udma_quiet/fence`。【事实】
- 常量枚举（`aclshmem_constants.py`）：`ACLSHMEMCmpOp{EQ,NE,GT,GE,LT,LE}`、`ACLSHMEMSignalOp{SET,ADD}`、`ACLSHMEMTeam{INVALID=-1,WORLD=0}`。【事实】
- 结论：**"ACLSHMEM = Ascend 的 shmem 实现"这一说法成立**（函数命名与头文件版权均指向华为 CANN/shmem 项目），但**本笔记未访问 `gitcode.com/cann/shmem` 本体**，其内部实现细节【未找到】。

---

## 4. 通算融合（compute-communication overlapping）算子清单

> 所有 NV kernel 位于 `python/triton_dist/kernels/nvidia/`（下称 `…/nvidia/`），对应文档在 `docs/kernels/nvidia/*.rst`。方向判定依据见 §4.3。

### 4.1 逐条机制与用途

| # | 文件（`…/kernels/nvidia/`） | 规模 | 关键符号（行号） | 机制与用途【事实（符号/命名/代码结构）】 |
|---|---|---|---|---|
| 1 | `sp_ulysess_qkv_gemm_all2all.py` | 963 行 | `kernel_persistent_tma_gemm_notify`(64)、`matmul_kernel_descriptor_persistent`(196)、`matmul_kernel`(341)、`kernel_all2all_pull_intra_node_nvl`(430)、`class SpUlysessQKVGemmAll2AllKernel`(545) | Ulysses SP 的 **QKV 投影 GEMM + All2All**：GEMM 为 producer（persistent，TMA descriptor 版），按 tile 用 `notify` 置信号；A2A 为 consumer，用 **NVLink pull**（`kernel_all2all_pull_intra_node_nvl`）从对端对称内存取 QKV。即"**GEMM→A2A**"顺序、单机内 NVLink。 |
| 2 | `sp_ulysess_o_all2all_gemm.py` | 848 行 | `load_v4_b32`/`store_v4_b32`(43/56, inline asm `ld.global.v4.b32`)、`kernel_all2all_push_intra_node_nvl`(417)、`matmul_descriptor_persistent`(264)、`class SpUlysessOAll2AllGemmKernel`(513)、内部断言 `assert self.a2a_only, "Only support a2a_only mode"` | Ulysses SP 的 **O 投影：All2All + GEMM**（**与 #1 相反的方向**）：先 A2A（**push** 到对端对称内存），再由 consumer GEMM 消费；只支持 `a2a_only=True`。 |
| 3 | `ulysses_sp_dispatch.py` | 707 行 | `kernel_pre_attn_qkv_pack_a2a`(39)、`kernel_qkv_bsnd_to_bnsd`(306)、`qkv_bsnd_to_bnsd`(411)、`class UlyssesSPPreAttnCommContext`(470)、`create_ulysses_sp_pre_attn_comm_context`(546)、`pre_attn_qkv_pack_a2a_op`(606) | **attention 前**的 QKV 打包 A2A（dispatch）：注释给出 shape 语义 `q: [bs, seq//P, q_nheads, k_head_dim] → out_q: [bs, seq, local_q_nheads, k_head_dim]`，即"头并行→序列并行"的转置式重排；用 `libshmem_device` + `membar`。 |
| 4 | `sp_ag_attention_intra_node.py` | 522 行 | `class SPAllGatherAttentionContextIntraNode`(44)、`cp_engine_producer_kv_all_gather`(106)、`_flash_attn_forward_inner`(188)、`kernel_consumer_flash_attn_forward`(257)、`fused_sp_ag_attn_intra_node`(433) | 单机内 **KV AllGather + FlashAttention 融合**：producer 用 **copy engine**（`cudart.cudaMemcpyAsync`，见 L130-135 `_cp_engine_copy_data`）搬运 KV，consumer kernel 做 flash attention，配合 `barrier_all_on_stream`/`BarrierAllContext`。 |
| 5 | `sp_ag_attention_inter_node.py` | 595 行 | `class SPAllGatherAttentionContextInterNode`(45)、`all_gather_push_2d_kernel`(116)、`cp_engine_producer_kv_all_gather`(193)、`kernel_consumer_flash_attn_forward`(329)、`fused_sp_ag_attn_inter_node`(505) | 跨节点 **KV AllGather + FlashAttention**：用 NVSHMEM **push 2D** + `putmem_signal_nbi_block`/`putmem_signal_block`（L152/L182）+ `nvshmem_barrier_all_on_stream`，即"IB 上低延迟 push + 信号"替代 NCCL。 |
| 6 | `all_to_all_single_gemm.py` | 474 行 | `all_to_all_copy_engine`(39)、`kernel_all_to_all_gemm_consumer`(74)、`class AllToAllSingleGemmContext`(191)、`create_all_to_all_single_gemm_context`(236)、`all_to_all_single_gemm`(298)、`gemm_only`(406) | **All2All 走 copy engine（`cudaMemcpyAsync`）+ GEMM 在另一 stream 上**的"分离式重叠"实现（函数 docstring 原文："Uses copy engine (cudaMemcpyAsync) for communication and separate kernels on different streams to achieve true overlap"）。命中文件名的正是本文件。 |
| 7 | `low_latency_all_to_all.py` | 279 行 | `all_to_all_kernel`(36, `@triton_dist.jit`)、`class AllToAllContext`(125)、`create_all_to_all_context`(176)、`fast_all_to_all`(198)、`all_to_all_post_process`(260) | **低延迟 EP All2All（推理向）**：`putmem_nbi_block` 推数据 + `putmem_signal_nbi_block`/`signal_op` 装信号 + `fence()`，`signal_wait_until` 等待；README/tutorial 04 用它做 DeepSeek 推理 all-to-all（支持可选 FP8 scale `WITH_SCALE`）。 |
| 8 | `gemm_allreduce.py` | 840 行 | `class GemmARContext`(44)、`class LLGemmARContext`(80)、`create_[ll_]gemm_ar_context`(103/127)、`consumer_all_reduce_kernel`(140)、`kernel_persistent_gemm[_tma]_notify`(383/480)、`kernel_fused_gemm_allreduce`(565)、`gemm_allreduce_op` | **GEMM + AllReduce 重叠**：GEMM 完成一个 tile 就 notify，AllReduce consumer 在 `ar_stream` 上做（`OverlappingAllReduceMethod`，可选 **Consumer_Multimem** 走 `multimem_ld_reduce_v4`/`multimem_st_v4`）。 |
| 9 | `gemm_reduce_scatter.py` | 768 行 | `class GEMMReduceScatterTensorParallelContext`(52)、`kernel_gemm_rs_producer_persistent[_reduce_st]`(233/144)、`gemm_rs_producer_persistent/non_persistent`(514/563)、`threadblock_swizzle_gemm_reduce_scatter_kernel` | **GEMM + ReduceScatter 重叠**（TP 反向）：producer persistent GEMM + 2D reduce-scatter（复用 `reduce_scatter.py` 的 `ReduceScatter2DContext`/`ring_reduce`），含 `gemm_perf_model`/`comm_perf_model` 的解析选型。 |
| 10 | `allgather_gemm.py` | 800 行 | `copy_kernel`(48)、`copy_and_barrier_all_intra_node_kernel`(85)、`kernel_consumer_gemm_persistent`(201)、`class AllGatherGEMMTensorParallelContext`(450)、`create_ag_gemm_context`(511) | **AllGather + GEMM 重叠**（TP 正向）：AG 用 copy engine（intra）/`cp_engine_producer_all_gather_inter_node`（inter），GEMM 按 tile `wait`/`consume_token` 消费，命中论文 §2.3 的示例结构；含 `ag_gemm_threadblock_swizzle`。 |
| 11 | `ep_all2all_fused.py` | **2071 行** | `tile_kernel_dispatch_token_intra_node`(73)、`tile_kernel_dispatch_token_intra_node_two_stage`(168)、`tile_kernel_gather_combine_token_intra_node`(308)、`tile_kernel_scatter_token_intra_node`(406)、`tile_kernel_topk_reduce_token_intra_node`(485)、`tile_kernel_moe_grouped_gemm_nk_const`(599)、`mega_kernel_dispatch_token_moe_grouped_gemm`(839)、`mega_kernel_moe_grouped_gemm_combine_token`(1020)、`mega_kernel_moe_grouped_gemm_combine_token_transposed_grouped_gemm`(1288) | **EP MoE 的 MegaKernel** 系列：把 dispatch / token 级 gather-scatter / topk reduce / grouped GEMM / combine 融进单个或极少数 kernel，按 tile 粒度流水（`barrier_on_this_grid`、`barrier_all_intra_node_atomic_cas_block`；带 `Profiler`）。文档 `docs/kernels/nvidia/ep_all2all_fused.rst` 给出 token saving ~18%、token sorting ~22%、其他 ~8%、通信量 −30% 等收益。 |
| 12 | `pre_attn_a2a.py` | 287 行 | `kernel_pre_attn_a2a`(83)、`_dispatch_qkv_store`(71) | **warp-specialized TMA kernel**：docstring 明写"Aligned with Flux `PreAttnQKVPackA2AKernel`（TMA-store path）"；Warp0 producer 用 TMA G2S 从**远端 NVSHMEM 输入拉取**，Warp1 consumer 用 TMA S2G 写本地 Q/K/V；`num_warps=2`。数据流 `input (local_seq, total_nh, hd) [remote, via NVSHMEM pull] → Q/K/V (seq_len, local_*_nh, hd)`。 |
| 13 | `post_attn_a2a.py` | 198 行 | `kernel_post_attn_a2a`(64) | 同一 warp-specialized 模板的**反向**：docstring "Aligned with Flux `PostAttnA2AKernel`（`use_tma_store` path）"；`input (seq_len, local_nh, hd) [local] → NVSHMEM push → output (local_seq, nh, hd) [remote, per-rank column slice]`。 |
| 14 | `moe_reduce_ar.py` | 692 行 | `moe_grouped_gemm[_persistent]_kernel`(60/130)、`class MoEReduceARContext`(266)、`moe_gather_ar_grouped_gemm[_persistent]_kernel`(323/405)、`moe_gather_ar_grouped_gemm`(485) | **MoE：gather + GroupedGEMM + top-k reduce + AllReduce** 融合（TP 场景），支持 persistent；配合 `calc_gather_scatter_index_triton`。 |
| 15 | `moe_reduce_rs.py` | 961 行 | `class MoEReduceRSContext`(43)、`moe_gather_rs_grouped_gemm[_persistent]_kernel`(168/250)、`moe_gather_rs_grouped_gemm`(471)、`reduce_topk_reduce_scatter_a2a_intra_node_kernel`(549) | **MoE：gather + GroupedGEMM + top-k reduce + ReduceScatter** 融合（与 #14 的区别即尾部集体操作 AR vs RS），intra-node 用 `barrier_all_intra_node_atomic_cas_block`。 |

（另有同族/相关文件：`ulysses_sp_infer_gemm_a2a.py`（480 行，`kernel_gemm_a2a_producer_gemm_with_quant_persistent`、`ulysses_sp_infer_gemm_a2a_op`，推理向 GEMM→A2A + 量化）、`allgather.py`、`allreduce.py`、`reduce_scatter.py`、`flash_decode.py`、`low_latency_alltoall_v2.py`、`all_to_all_single_2d.py`、`all_to_all_vdev_2d_offset[_inter_node].py`、`group_gemm.py`、`swiglu.py`、`gdn.py`、以及 AMD 镜像集 `kernels/amd/`。）

**高层 layer 封装**【事实】`python/triton_dist/layers/nvidia/`：`ep_a2a_fused_layer.py`、`ep_a2a_layer.py`、`ep_ll_a2a_layer.py`、`ep_moe.py`、`gemm_allreduce_layer.py`、`low_latency_allgather_layer.py`、`pp_block.py`、`pre_attn_a2a_layer.py`、`post_attn_a2a_layer.py`、`sp_flash_decode_layer.py`、`tp_attn.py`、`tp_mlp.py`、`tp_moe.py`、`ulysses_sp_a2a_layer.py`；AMD 侧 `layers/amd/` 另有 `tp_attn.py`、`tp_mlp.py`、`ep_a2a*.py`。

### 4.2 transpose all-to-all vs reverse all-to-all

**昇腾侧给出了最清晰的语义定义（可外推到 NV 侧）**【事实】：

- `transpose-all2all.py` 的参考实现与 docstring（`gitcode.com/Ascend/Triton-distributed-ascend`）：
  ```py
  def torch_transpose_a2a(tensor, dsp_size, group=None):
      """Forward All2All dispatch. tensor: (S, n_head, head_dim) with n_head = H * dsp_size.
      Returns recv: (S * dsp_size, H, head_dim)."""
      # [S, n_head, head_dim] -> [S * dsp_size, n_head / dsp_size, head_dim]
      transposed = torch.cat(torch.chunk(tensor, dsp_size, dim=1), dim=0).contiguous()
  ```
  → **transpose all-to-all = 正向/dispatch**：把 `(seq, 全头)` 变成 `(seq×P, 头/P)`，即"**头维切分 → 序列维切分**"。
- `reverse-all2all.py`：
  ```py
  def torch_reverse_a2a(tensor, dsp_size, group=None):
      send = torch.cat(torch.chunk(tensor, dsp_size, dim=0), dim=1).contiguous()
      send = send.transpose(1, 0).contiguous()
  ```
  → **reverse all-to-all = 反向/combine**：把 `(seq×P, 头)` 变回 `(seq, 头×P)`，即"**序列维切分 → 头维切分**"。其文档 `docs_ascend/en/tutorial/reverse_all2all.md` 标题即 "Reverse All2All Distributed Kernel for **EP MoE**"，并说明 reverse = MoE dispatch/combine 中的 **combine（第二步）**："Reverse All2All is the second half: after expert computation, results must be sent back from the expert's home rank to the token's original rank"、"The key difference from forward All2All is the **dimension swap**"。

**据此对各 kernel 归类**：

| 方向 | NV 侧 kernel | 昇腾侧对应 |
|---|---|---|
| **transpose all-to-all**（正向/dispatch；头→序列；GEMM **然后** A2A） | `sp_ulysess_qkv_gemm_all2all.py`、`ulysses_sp_dispatch.py`、`pre_attn_a2a.py`、`ulysses_sp_infer_gemm_a2a.py`、`ep_all2all_fused.py` 中的 dispatch 侧（`tile_kernel_dispatch_token_intra_node*`、`mega_kernel_dispatch_token_moe_grouped_gemm`） | `tutorials/ascend/transpose-all2all.py`、`transpose-all2all-udma.py`、`qkv-all2all.py`、`moe-dispatch-udma.py` |
| **reverse all-to-all**（反向/combine；序列→头；A2A **然后** GEMM） | `sp_ulysess_o_all2all_gemm.py`、`post_attn_a2a.py`、`all_to_all_single_gemm.py`（A2A→GEMM 顺序） | `tutorials/ascend/reverse-all2all.py`、`reverse-all2all-udma.py`、`moe-combine-udma.py`、`reverse-a2a-norm-matmul*.py` |

- **【事实】** 昇腾文档与文件名明确使用 "transpose"/"reverse" 词汇；上游 NV 内核**没有** `reverse` 一词（`grep -rn -iE "reverse[_ ]?all|reverse a2a|transpose[_ ]?all" python/triton_dist` 在 NV kernels 中**零命中**），NV 侧只有 `*_all2all_*` / `pre_attn` / `post_attn` 命名。
- **【推测】** 上表中 NV 侧的方向归类，依据是：(a) `sp_ulysess_qkv_gemm_all2all` 的 producer 是 GEMM、consumer 是 A2A pull；`sp_ulysess_o_all2all_gemm` 反之先 push-A2A 再 GEMM；(b) `ulysses_sp_dispatch.py` 的 shape 注释明确是 pre-attn 的头→序列方向；(c) `pre_attn_a2a.py`/`post_attn_a2a.py` 的数据流注释恰好互为逆操作。**未找到**上游用 "transpose/reverse" 命名的等价表述。

### 4.3 官方文档入口（用于引用）

- Kernels 索引：[kernels/index.html](https://triton-distributed.readthedocs.io/en/latest/kernels/index.html)；Ulysses SP：[kernels/nvidia/ulysses_sp.html](https://triton-distributed.readthedocs.io/en/latest/kernels/nvidia/ulysses_sp.html)（含 `create_ulysses_sp_pre_attn_comm_context`、`create_all_to_all_single_gemm_context`、`all_to_all_single_gemm`）
- Layers 索引 → Ulysses SP A2A Layer：[layers/nvidia/ulysses_sp_a2a_layer.html](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ulysses_sp_a2a_layer.html)（`docs/layers/nvidia/ulysses_sp_a2a_layer.rst` 仅 12 行，指向 `python/triton_dist/layers/nvidia/ulysses_sp_a2a_layer.py`）
- 低延迟 EP A2A Layer：[layers/nvidia/ep_ll_a2a_layer.html](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ep_ll_a2a_layer.html)；EP A2A fused Layer：[layers/nvidia/ep_a2a_fused_layer.html](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ep_a2a_fused_layer.html)
- AMD 性能页：[amd-perf.html](https://triton-distributed.readthedocs.io/en/latest/amd-perf.html)（`docs/amd-perf.md` 正文**只有两张图**，无数字文本）

---

## 5. Ascend 移植版：`gitcode.com/Ascend/Triton-distributed-ascend`

### 5.1 这是什么【事实，全部来自该仓库 README（raw）】

- 仓库：[gitcode.com/Ascend/Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend)（页面标题："Triton-distributed-ascend:基于 Triton 语言扩展的昇腾分布式计算框架项目"，默认分支 `master`，Apache-2.0）。README raw：`https://raw.gitcode.com/Ascend/Triton-distributed-ascend/raw/master/README.md`。
- 自述："a distributed computing framework **extended from the Triton language and optimized specifically for Ascend AI processors**"；能力："AllReduce, AllGather, ReduceScatter, and All2All"、"Computation-communication overlap"、"**AscendNPU IR-based compiler optimizations**"、"Python-friendly interfaces"。
- 仓库结构（README 原文）：`include/`、`lib/Conversion/TritonDistributedToHIVM/ASCEND`、`python/triton_dist/language/{distributed_ops.py, extra/ascend}`、`3rdparty/triton-ascend`、`docs_ascend/`、`tutorials/`、`shmem/`。
- **硬件**：Atlas A2、Atlas A3、**Ascend 950PR&950DT**。**软件**：**CANN ≥ 9.1.0**、PyTorch ≥ 2.0、torch_npu、CMake ≥ 3.19、GLIBC ≥ 2.28、Python ≥ 3.8。
- 构建三件套：① `TRITON_USE_ASCEND=ON ... pip install ./python`（可选自建 LLVM，pin `llvm-project@f6ded0be...`）；② **AscendNPU-IR**（`gitcode.com/Ascend/AscendNPU-IR`，`./build-tools/build.sh ... --build-shmem-template`）；③ **shmem**（`3rdparty/shmem`，`bash scripts/build.sh -python_extension`，Ascend 950 需 `-soc_type Ascend950`）→ **这就是 ACLSHMEM 的构建入口**。
- 文档站点：<https://triton-distributed-ascend.pages.dev>。
- 相对上游的关系：上游 README 明确"Ascend is not on the 3.7 plugin build path yet… use `triton-v3.4` for a working Ascend build. The 3.4-era Ascend sources are **parked** under [`ascend/`](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/ascend/README.md)"；上游相关 PR：【事实】GitHub issue 搜索命中 `#168 feat: Add ascend backend`（2026-04-24）、`#179 feat: Add Ascend RDMA support`（2026-06-28）、`#118 add allgather_matmul on ascend`（2025-09-17）。
- 活跃度【事实】：`master` HEAD = `f259f6a`，日期 **2026-10-14**，message `!313 merge feat/alltoallv-gmm-fused into master`（GitCode 用 `!NNN` 表示 MR 编号）。→ 与调研同日仍在合并。
- **【未找到】**该仓库的 star/fork 数：`https://api.gitcode.com/api/v5/repos/Ascend/Triton-distributed-ascend` 返回 `{"message": ...}`（非预期 schema），网页为 JS 渲染，未取到。

### 5.2 PR #114 与 UDMA / wait-notify 机制

- **PR #114**【事实，`https://api.gitcode.com/api/v5/repos/Ascend/Triton-distributed-ascend/pulls/114`】：
  - title：**"feat: add wait-notify for reverse-alltoall and transpose-alltoall udma"**
  - state：**merged**；created **2026-08-14T15:51:47+08:00**；merged **2026-08-29T10:00:12+08:00**
  - author：`weilinquan`；head/base 分支均为 `release/3.2.2`；body（全文）：`feat: add wait-notify for reverse-alltoall udma`
  - html_url：<https://gitcode.com/Ascend/Triton-distributed-ascend/merge_requests/114>（GitCode 工作流叫 merge_requests，API 路径是 `pulls`）
  - **【未找到】** changed_files/文件清单（该 API 响应不含 `changed_files`/`additions`/`deletions` 字段；未另抓 diff 页面）。
- **UDMA 是什么**【事实（用法）+ 未找到（全称）】：
  - 主机侧：`import shmem as ash` → `attributes.option_attr.data_op_engine_type = ash.OpEngineType.UDMA` → `ash.aclshmem_init(attributes)`（`tutorials/ascend/reverse-all2all-udma.py`）。即 UDMA 是 ACLSHMEM 初始化时选择的**数据搬运引擎类型**。
  - 文档：`docs_ascend/en/tutorial/reverse_all2all.md`："**UDMA Optimized Version**（Ascend 950PR&950DT products…）：Leverages the new **UDMA link** on Ascend 950PR&950DT products for higher cross-card bandwidth, splitting 'cross-card data exchange' and 'local data reordering' into two separate phases"。
  - device 侧对应原语：`udma_put_nbi(dest, source, num_elements, pe)`、`udma_put_signal_nbi(...)`、`udma_quiet(pe)`（`libaclshmem_device.py` L328/L358/L528）。
  - **【未找到】UDMA 缩写的官方展开**（Unified/Unidirectional Direct Memory Access 之类均为猜测，不做断言）。
- **wait/notify 点对点信号量替代全局 barrier**【事实，直接引文】：
  - 文件 docstring（`tutorials/ascend/reverse-all2all-udma.py`）："Cross-rank UDMA sends and the local self-copy are each synchronized with **point-to-point wait/notify semaphores (signal_mem_ptr) instead of a global `libshmem_device.barrier_all()`**, so a task only stalls on the exact (block, rank) or (block, task) it actually depends on."
  - 文档：**"Unlike barrier-based synchronization (`barrier_all()` / `barrier_all_vec()`), the Reverse All2All kernel uses fine-grained per-task signal synchronization"**；并列出 producer/consumer 的五步/六步序列：生产端 `dl.wait(remote_signal_ptr, 1, "sys", "acquire", waitValue=0)` → `dl.consume_token(remote_ptr, token)` → `tl.store(remote_ptr+off, data)` → `libshmem_device.fence()` → `dl.notify(signal_mem_ptr, target_rank)`；消费端 `dl.wait(..., "sys","acquire")` → `consume_token` → `tl.load` → `tl.store` → `fence()` → `dl.notify(signal_mem_ptr, rank, 0)`（**置 0 释放下一轮**）。
  - 结构：**Vector Core 级 producer/consumer 双缓冲流水**（`role = pid % 2`，`logical_core_id = pid // 2`，`n_role_cores = ncore // 2`），kernel 以 **AI Vector Core 数**启动（`NPUUtils().get_aivector_core_num()`）而非 AICore 数；信号区分两个 region：Region A `[num_blocks_s, rank_size]`（跨卡 UDMA block-ready，只写不重置）、Region B `[buffer_num, tasks_per_block]`（本地自拷贝双缓冲，ready→consumed→free→ready 循环），`SIG_STRIDE = 8`（一个 cache line，避免 false sharing）。
  - 昇腾移植中 `dl.notify` 要求 **int32/uint32 信号指针**，且 `aclshmem_wait` 以 **64 字节**步长遍历 barrier（`64/4 = 16` 元素 → `SIGNAL_SLOT_STRIDE = 16`）——见 `transpose-all2all.py`/`reverse-all2all.py` 顶部注释【事实】。
  - 演进证据：仓库历史中存在 `tutorials/ascend/05-ascend-reverse-all2all-barrier.py`（提交 `b53492f "fix: enforce barrier_all MIX semantics"`）与后来的 wait-notify 版本，说明"barrier → wait/notify"确为一次真实替换【事实（git 历史）+ 推测（因果）】。

### 5.3 `tutorials/ascend/06-ascend-reverse-all2all-udma.py` —— 文件名核实结果

- **【未找到】精确保留该文件名的版本**。对 `git clone https://gitcode.com/Ascend/Triton-distributed-ascend.git`（`--depth 200`）做全历史路径检索：
  - `git log --all --name-status -- 'tutorials/ascend/06-ascend-reverse-all2all-udma.py'` → **空**。
  - 历史上**存在过的**含 `06-ascend` 的路径：`tutorials/ascend/06-ascend-reverse-all2all-udma-autotune.py`、`tutorials/ascend/06-ascend-transpose-all2all-udma.py`、`tutorials/ascend/06-ascend-transpose-all2all.py`、`tutorials/ascend/06-ascend-gemm-ar-notify.py`、`perf/06-ascend-transpose-all2all[-udma].py`。
  - 该功能文件的真实历史名：`tutorials/ascend/04-ascend-reverse-all2all-udma.py`（提交 `fdca83e` 加入）→ **重命名**为 `tutorials/ascend/reverse-all2all-udma.py`（`R099`）；性能脚本迁到 `python/triton_dist/test/ascend/reverse-all2all-udma.py`（提交 `75e50fb` "fix: move perf directory to python/triton_dist/test and add mode"，2026-09-22）。
- **当前实际文件名（`master`）**【事实】：`tutorials/ascend/reverse-all2all-udma.py`、`tutorials/ascend/transpose-all2all-udma.py`、`tutorials/ascend/reverse-all2all-udma-autotune.py`、`tutorials/ascend/reverse-all2all.py`、`tutorials/ascend/transpose-all2all.py`，以及 `qkv-all2all*.py`、`moe-dispatch-udma.py`、`moe-combine-udma.py`、`alltoallv-gmm[-udma].py`、`allgather-gemm[-udma].py`、`gemm-reduce-scatter[-udma].py`、`gemm-allreduce-{oneshot,oneshot-notify,udma}.py`、`reverse-a2a-norm-matmul[-2].py`。
- 上游主仓库（`main` 与 `triton-v3.4` 两分支均已克隆核实）**只有** `ascend/tutorials/ascend/01-ascend-allgather-gemm.py`、`02-ascend-gemm-reduce-scatter.py`、`03-ascend-gemm-allreduce-oneshot.py`（`main` 是 `ascend/tutorials/ascend/`，`triton-v3.4` 是 `tutorials/ascend/`），**没有 04/05/06**。【事实】

---

## 6. 性能数字（逐条给出来源与硬件配置）

### 6.1 论文 §4 Experiments（集群规模 8–64 GPU；硬件 H800 + MI308X）

| 基准 | 硬件 | 基线 | 数字 | 出处 |
|---|---|---|---|---|
| AG+GEMM-intra | 8×H800 | PyTorch+NCCL（cuBLAS）；FLUX（CUTLASS） | **1.42×** vs PyTorch+NCCL、**1.09×** vs FLUX；Triton GEMM 约达 cuBLAS/CUTLASS 的 **95%** | §4.1 / Fig.11 |
| GEMM+RS-intra | 8×H800 | PyTorch+NCCL、FLUX | **1.28×** vs PyTorch+NCCL、**1.30×** vs FLUX（FLUX 把 scatter 融进 GEMM 并在本地 reduction 前做全局同步；本文用独立 stream 异步 scatter） | §4.1 / Fig.12 |
| AG+MoE-intra | 8×H800 | PyTorch+NCCL（Python 循环 GroupGEMM，作者自称弱基线） | 平均 **44.97×**（Table 4 给出 15 组 shape 的绝对 ms，如 AG+MoE-1 256/2048/1408/60/4：Ours 0.33 intra、0.45 inter；PyTorch 23.95/28.84） | §4.1 / Table 4 |
| MoE+RS-intra | 8×H800 | PyTorch+NCCL | 平均 **15.55×**（Table 5，10 组 shape；如 MoE-RS-1：0.51 intra / 3.62 inter vs PyTorch 4.35/12.41） | §4.1 / Table 5 |
| AG+GEMM-inter | 16×H800（2 节点） | PyTorch+NCCL；FLUX（引用其原文数字） | 平均 **1.33×** vs NCCL，达 FLUX 的 **95.60%** | §4.2 / Fig.13 |
| GEMM+RS-inter | 16×H800 | 同上 | 平均 **1.42×** vs NCCL，达 FLUX 的 **96.36%** | §4.2 / Fig.14 |
| AG+MoE-inter / MoE+RS-inter | 16×H800 | PyTorch+NCCL | **26.50×** / **5.16×**；AG+MoE 弱扩展近似线性，**MoE+RS 扩展不佳**（作者归因于缺少专用 RS kernel，"left for future"） | §4.2 / Table 4-5 |
| Distributed Flash-Decode | 8→32×H800（batch=1） | 自身扩展性 | 弱扩展下 32 GPU、每 GPU 32K KV 仍达 **1.7 TB/s** HBM 带宽（H800 峰值约 **3 TB/s**）；全局 KV < 256K 时增加 GPU 无收益；1M 级 KV 越多卡越低延迟 | §4.2 / Fig.15 |
| Low-latency AllGather（PCIe） | 8×L20 单机 + 16×L20（2 节点） | **NVSHMEM `fcollect`**、**NCCL** | 单机带宽提升：vs NVSHMEM-32bit **1.40×**、vs NVSHMEM-64bit **1.48×**、vs NCCL-inplace **3.11×**、vs NCCL-out-of-place **3.00×**；2 节点：vs NVSHMEM-64bit **1.31×**、vs NVSHMEM-32bit **1.38×** | §4.2 / Fig.19 |
| Low-latency AllToAll（推理 EP） | 8→64×H800 | **DeepEP** | Dispatch 平均 **1.18×**、Combine 平均 **1.44×**；作者注明：本文用 **IBRC** + NVLink（intra），DeepEP 用 **IBGDA** 且 intra-node 也走 IB；**64 GPU dispatch 上 DeepEP 更快**；128 GPU 规模 DeepEP 仍最好（IBGDA 扩展性更好），因"current NVSHMEM bitcode library doesn't support IBGDA" | §4.2 / Fig.16 |
| AG+GEMM / GEMM+RS intra | **8×MI308X**（AMD，单机） | AMD PyTorch+RCCL（rocBLAS） | **1.09×** / **1.16×**；Triton GEMM 略低于 rocBLAS 但重叠更好 | §4.3 / Fig.17-18 |

- 论文 Table 3「List of Our Optimized Kernels」列出 AG+GEMM-intra（8 H800 & MI308X）、GEMM+RS-intra（同）、AG+MoE-intra、MoE+RS-intra、FlashDecode+AG-intra、AllToAll-intra、AG+GEMM-inter（16 H800）、GEMM+RS-inter（16）、AG+MoE-inter、MoE+RS-inter、FlashDecode+AG-inter（16 与 32 H800）、AllToAll-inter 等【事实】。
- 论文声明的测试环境："We use both **Nvidia H800 GPUs and AMD MI308X GPUs**… The scale of our cluster ranges from **8 GPUs to 64 GPUs**."（§4 开头）

### 6.2 文档/仓库 README 中的数字

| 指标 | 数字 | 硬件/配置 | 出处 |
|---|---|---|---|
| 低延迟 AllToAll | **137 µs**，对比 **DeepEP 182 µs** | **32×H800**；128 tokens/rank、topk=8、hidden=7168、**fp8**；附注 "DeepEP doesn't use NVLink for inference" | [README.md](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/README.md)（该处与本笔记 §2 引用的示例代码同页） |
| EP Low-Latency AllToAll Layer | "sub-**202µs** end-to-end"；"~202µs for 128 tokens per rank on **8 GPUs**" | 8×H800 | `docs/layers/nvidia/ep_ll_a2a_layer.rst` L11/L16/L34 → [ep_ll_a2a_layer.html](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ep_ll_a2a_layer.html) |
| 同层 32 卡预期 | dispatch **~70 µs**、combine **~67 µs**、**总 ~137 µs** | **32×H800（4 节点）** | 同文件 L293–L306 |
| Low-latency A2A v2 | dispatch **~76 µs**、combine **~126 µs**、**总 ~202 µs** | "Benchmark: **8x H800 GPUs**" | `docs/kernels/nvidia/low_latency_a2a_v2.rst` L22–L31 |
| EP All2All fused | token saving **~18%**、token sorting **~22%**、其他 **~8%**、通信量 **−30%** | 未标注具体卡型（同文档语境为 NV） | `docs/kernels/nvidia/ep_all2all_fused.rst` L172/L496-498 |
| EP A2A fused Layer | 吞吐 **10–20%** 提升、延迟 **15–25%** 改善（大 token 数） | 未标注 | `docs/layers/nvidia/ep_a2a_fused_layer.rst` L403/L422 |
| E2E dense（Seed-OSS-36B-Instruct） | MLP **1.26×**（0.6012→0.4756 s）、Attn Prefill **1.44×**、Attn Decode **1.39×**、E2E Model Prefill **1.35×**（15.78→11.70）、E2E Model Decode **1.33×**（16.54→12.41） | **8×H800** | `docs/getting-started/e2e/e2e_dense.md` L15-38 |
| E2E（同文档，另一组） | MLP **1.34×**、Attn Prefill **1.48×**、Attn Decode **1.39×**、E2E Prefill **1.33×**、E2E Decode **1.32×** | **8×H800**，标题即 "Perf for ByteDance-Seed/Seed-OSS-36B-Instruct on 8xH800" | `docs/e2e.md` L44-52 |
| E2E on AMD | AG_GEMM **1.0025×**、GEMM_RS **1.2627×**、MLP **1.067×**、Attn Prefill **1.0312×**、Attn Decode **1.0310×** | **8×MI308X** | `docs/getting-started/e2e/e2e_dense.md` L40-48 |
| MegaTritonKernel（单步 decode 延迟 ms，bs=1/ctx=512/TP=8） | qwen-8b：torch eager 26.08 → cudagraph 5.49 → triton_dist_AR+cudagraph 4.65 → **mega 3.33**；qwen-32b：49.69 → 10.80 → 9.18 → **7.41** | **8×H800** | `docs/getting-started/megakernel/megakernel.md` |
| 同上 | qwen-8b：28.75 → 5.52 → 4.59 → **3.16**；qwen-32b：52.37 → 13.87 → 11.96 → **8.34** | **8×H20** | 同上 |
| Seed-OSS-36B 推理 | **1.33× speedup** | 未在 news 中标注卡型（与 `docs/e2e.md` 的 8×H800 对应） | README News 08/24/2025 |
| AMD 性能页 | **无数字**，仅两张 PNG 图（`asset/amd-ag-gemm-intranode-perf.png`、`asset/amd-gemm-rs-intranode-perf.png`），标题 MI308X 单机 | **MI308X 单机** | `docs/amd-perf.md` → [amd-perf.html](https://triton-distributed.readthedocs.io/en/latest/amd-perf.html) |
| 昇腾侧 | 有性能采集设施（`python/triton_dist/test/ascend/*.py`、`*-case.csv`、`perf_unified/`）与 `@prof` 装饰器（`ASCEND_PROFILE_WARMUP`/`ASCEND_PROFILE_ITERS`），但**本笔记未提取具体数字** | 昇腾 | `tutorials/ascend/README.md` |

> **【未找到】MetaX 相关性能数字**（论文与文档均无）。

---

## 7. 局限（论文/文档/源码自陈）

- **【事实】论文没有独立的 "Limitations" 章节**：HTML v3 的标题结构只有 `1 Introduction / 2 Programming Model / 3 Overlapping Optimizations / 4 Experiments / 5 Conclusion / References`。以下均为散落在正文、文档、README、源码中的自陈限制。

1. **对 OpenSHMEM/对称内存硬件的强依赖（可移植性门槛）**——论文 §3.6「Optimizations on More Platforms」给出移植的**三个必要条件**原文："First, the hardware should support **symmetric memory allocation and access**, which is expected to conform to the OpenSHMEM standard. Second, the hardware should support **signal exchange**, including signal setting, increasing, checking, and spin-locks. Third, the hardware should support **async-tasks**, allowing us to map specialized tasks spatially to different hardware units." 并承认 "Although we only show optimized kernels for GPUs currently, we can support other platforms similarly"（即：其他平台只是"可以类似支持"，尚未给出结果）。
2. **NVSHMEM 依赖与 IBGDA 缺失（跨节点扩展性上限）**——论文 §4.2 原文："DeepEP uses **IBGDA**, which has better scalability than **IBRC**… current **NVSHMEM bitcode library doesn't support IBGDA**. So we leave this for future work." 直接后果：**64 GPU 的 AllToAll Dispatch 与 128 GPU 规模下 DeepEP 更快**。安装侧亦需 `nvidia-nvshmem-cu12==3.3.9`、`nvshmem4py-cu12==0.1.2`、`cuda.core==0.2.0`（`docs/build.md`）。
3. **MoE+RS 跨节点弱扩展差**——论文 §4.2："for MoE+RS-inter, the scaling is **not good**… this indicates that a **dedicated ReduceScatter kernel is required** for MoE+RS, which is **left for future**."
4. **计算内核本身不如手工库**——论文 §4.1："Triton's generated code can achieve roughly **95% the performance of cuBLAS and CUTLASS**"；§4.3 AMD："The performance of the code generated by Triton is **slightly lower than that of rocBLAS**"。其优势来自重叠而非单算子峰值。
5. **内存占用换性能的权衡**——论文 §4.2 原文承认："we allocate a **much larger memory buffer** than DeepEP and **omit the memory control logic**. As a result, we can achieve better performance compared to DeepEP."（DeepEP 的复杂显存队列控制是为了省显存，代价是额外开销。）
6. **高层原语尚未开源**——`docs/primitives.md`："These high-level primitives will be **released soon after MLSys 2025**"；README Roadmap 中 `- [ ] Release high-level primitives`、`- [ ] Release single-node MoE EP overlapping kernels` 未勾选。用户目前只能用低层原语手写通信。
7. **与生态/框架集成的成本**——需要自建/替换 Triton（`pip uninstall triton` 后装 triton_dist wheel）、CUDA ≥ 12.4、`MAX_JOBS=126` 编译；仓库自述 **AOT 版本 "Not Recommended"**，且"一旦构建 AOT 版本，之后都会按 AOT 构建，必须删除 `python/build` 才能解除"（`docs/build.md`）。
8. **昇腾移植的版本耦合**——上游 README："Ascend is **not on the 3.7 plugin build path yet**（`triton-ascend` has not rebased onto 3.7）；use `triton-v3.4` for a working Ascend build"，且 Ascend 源码被"parked" 在 `ascend/`；移植侧要求 **CANN ≥ 9.1.0**、可选自建 LLVM（pin 特定 revision + 打 patch）、单独构建 AscendNPU-IR 与 shmem（`-soc_type Ascend950`）。→ 集成成本与版本锁定明显。
9. **正确性/易错点（源码与文档层面可证）**：
   - `wait` 必须与 `consume_token` 配对才能建立数据依赖（论文 §2.2 明确指出二者为"specially designed for optimization purposes"的配套原语）——漏用即失去正确性保障。
   - 昇腾侧 `dl.notify` **要求 int32/uint32 信号指针**，且 `aclshmem_wait` 以 **64 字节**步长扫描 barrier（`SIGNAL_SLOT_STRIDE = 16`）；信号槽必须按 cache line（`SIG_STRIDE = 8`）填充以避免 false sharing（`tutorials/ascend/{reverse,transpose}-all2all*.py` 注释）。
   - 双缓冲的释放语义靠 `dl.notify(signal_ptr, rank, 0)` 显式回写 0，写错即死锁/覆盖未消费数据（`docs_ascend/en/tutorial/reverse_all2all.md`）。
   - `EPLowLatencyAllToAllLayer` **要求 `online_quant_fp8=True`，不支持预量化输入**（`docs/layers/nvidia/ep_ll_a2a_layer.rst` L326）。
   - 仓库近期 CI 修复记录本身反映正确性风险面：`fix: make MoRI ring_put wait for delivery and retry on GitHub unit`、`fix: run GitHub MoRI EP A2A at M=256 so the job cannot hang`（2026-09-17 提交）。
10. **【推测】** 名称里 `_intra_node` / `_inter_node` 两套 kernel 并存、通信路径（NVLink / IB / PCIe / copy engine / UDMA）各自特化，意味着**同一算子在换拓扑时要换实现**，这是可移植性与维护面的隐含成本；论文 §3.2–§3.6 对每种拓扑分别优化，未给出统一抽象。

---

## 8. 关键链接汇总

- 论文：[abs](https://arxiv.org/abs/2504.19442) ｜ [HTML v3](https://arxiv.org/html/2504.19442v3) ｜ DOI 10.48550/arXiv.2504.19442
- 前一会议论文（TileLink, MLSys 2025）：[arXiv:2503.20313](https://arxiv.org/abs/2503.20313) ｜ [poster 3248](https://mlsys.org/virtual/2025/poster/3248)
- 仓库：[github.com/ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) ｜ [API](https://api.github.com/repos/ByteDance-Seed/Triton-distributed) ｜ [triton-v3.4 分支](https://github.com/ByteDance-Seed/Triton-distributed/tree/triton-v3.4)
- 文档：[文档首页](https://triton-distributed.readthedocs.io/en/latest/) ｜ [primitives.html](https://triton-distributed.readthedocs.io/en/latest/primitives.html) ｜ [python-api/triton-dist.language](https://triton-distributed.readthedocs.io/en/latest/python-api/triton-dist.language.html) ｜ [kernels/index](https://triton-distributed.readthedocs.io/en/latest/kernels/index.html) ｜ [kernels/nvidia/ulysses_sp](https://triton-distributed.readthedocs.io/en/latest/kernels/nvidia/ulysses_sp.html) ｜ [layers/nvidia/ulysses_sp_a2a_layer](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ulysses_sp_a2a_layer.html) ｜ [layers/nvidia/ep_ll_a2a_layer](https://triton-distributed.readthedocs.io/en/latest/layers/nvidia/ep_ll_a2a_layer.html) ｜ [amd-perf](https://triton-distributed.readthedocs.io/en/latest/amd-perf.html) ｜ [build](https://triton-distributed.readthedocs.io/en/latest/build.html)
- 昇腾移植：[gitcode.com/Ascend/Triton-distributed-ascend](https://gitcode.com/Ascend/Triton-distributed-ascend) ｜ [PR/MR #114](https://gitcode.com/Ascend/Triton-distributed-ascend/merge_requests/114) ｜ [PR #114 API](https://api.gitcode.com/api/v5/repos/Ascend/Triton-distributed-ascend/pulls/114) ｜ [文档站](https://triton-distributed-ascend.pages.dev) ｜ ACLSHMEM 库：[gitcode.com/cann/shmem](https://gitcode.com/cann/shmem) ｜ AscendNPU-IR：[gitcode.com/Ascend/AscendNPU-IR](https://gitcode.com/Ascend/AscendNPU-IR)
- 上游对照：DeepEP [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) ｜ FLUX [bytedance/flux](https://github.com/bytedance/flux/blob/main/README.md) ｜ Distributed-GEMM CUTLASS example [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/tree/main/examples/65_distributed_gemm)

## 9. 本次调研的「未找到 / 存疑」清单（避免被引用为事实）

1. **`tutorials/ascend/06-ascend-reverse-all2all-udma.py` 这个精确文件名在任何历史版本中都不存在**（见 §5.3）；实际对应文件为 `tutorials/ascend/reverse-all2all-udma.py`，历史名为 `04-ascend-reverse-all2all-udma.py`；带 `-autotune` 的 `06-...` 版本曾存在后被改名。
2. **UDMA 缩写全称未找到**（仅知是 `ash.OpEngineType.UDMA`，用于 Ascend 950PR&950DT 的跨卡数据搬运引擎）。
3. **论文的完整机构归属列表未找到**（arXiv HTML 抓取丢失 affiliation 标记）；仅确认 ByteDance 邮箱与页脚团队名。
4. **论文无独立 Limitations 章节**，第 7 节内容均为散落自陈 + 源码/文档证据。
5. **MetaX（沐曦）无任何性能数字**（仅代码路径存在）。
6. **PR #114 的变更文件清单未获取**（GitCode API 响应不含该字段）。
7. **GitCode 仓库的 star/fork 数未获取**（API schema 异常 + 页面 JS 渲染）。
8. **仓库"首个提交"的精确日期存疑**（author date 明显被 rebase/backdate 影响）。
9. **GitHub API 在本次抓取后段触发未认证限流**，故 star/fork 等数字以 2026-10-14 15:14 的单次快照为准；若后续引用需重新抓取。
10. **上游 NV 侧"transpose / reverse all-to-all"的官方命名未找到**（该术语见于昇腾移植仓库与其文档；NV 侧方向归类为推断，见 §4.2）。

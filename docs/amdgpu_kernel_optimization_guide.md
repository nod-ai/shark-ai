# AMDGPU Kernel Optimization Guide

Author: Jakub Kuderski @kuhar

Date: 2024-06-24

Last Update: 2025-04-08

## Introduction

We present a summary of the AMDGPU (micro-)architecture that we found necessary
to understand and account for in [IREE](https://iree.dev) and [Turbine
Kernels](https://github.com/iree-org/iree-turbine) in order to produce
performant kernel code. The information presented strives to be sufficiently
close to reality to be useful in kernel code generation, but **is not**
guaranteed to be 100% correct and accurate.

In addition, this document interleaves actionable optimization tips that we
derived from our understanding of the architecture.

> [!NOTE]
> This is not a reference manual or an official AMDGPU architecture guide.

### Resources

For official documentation, see:
* [MI300 ISA
  Manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
* [CDNA3
  Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
* [ROCm Optimization Guide for LLM
  Inference](https://rocm.docs.amd.com/en/latest/how-to/llm-fine-tuning-optimization/index.html)
* [MI300 Compute and Memory Partitioning Modes](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html)

For third-party documentation, see:
* [Introduction to
  AMDGPU](https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)
  from the Oak Ridge National Lab
* [CDNA3 Compute Architecture
  Overview](https://chipsandcheese.com/2023/12/17/amds-cdna-3-compute-architecture/)
  from Chips and Cheese
* [MI300X Benchmarks](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/) (bandwidth, latency, speed-of-light numbers) from Chips and Cheese

## Glossary

GPU programming uses naming specific to compute and graphics APIs, and is often
vendor specific. The table below juxtaposes the few most common ones:

| Vulkan | CUDA | AMDGPU | *Executes On* |
| --- | --- | --- | --- |
| Invocation / Thread | Thread | Thread | SIMD Lane |
| Subgroup | Warp | Wave(front) | SIMD |
| Workgroup | (Thread) Block | (Thread) Block / Workgroup | Compute Unit |
| N/A | (Thread) Block Cluster | N/A | Shader Engine |
| (Work)group counts | Grid | Grid | GPU |
| Workgroup Memory | Shared Memory | Local Data Store | LDS & Crossbar (Compute Unit) |

The rest of the document uses the Vulkan and AMDGPU naming for consistency.

## GFX9 Architecture Overview

The GFX9 line of hardware covers both consumer Radeon GPUs (GCN) and newer
datacenter MI accelerators like MI300 (CDNA3).

### MI300X Compute Topology

The MI300 GPU uses a chiplet design. For example, the MI300X variant consists
of 8 chiplets called XCDs, sat on 4 pairs of AIDs/IODs. One XCD contains 4
Shader Engines with the total of 38 Compute Units (CUs). One CU contains 4
16-lane-wide SIMDs. In total, MI300X has 304 CUs.

> [!TIP]
> The number of workgroups launched should be a multiple of the number of CUs to
> continuously utilize the whole GPU.

A kernel is dispatched to one or more CUs.

![MI300 Topology](./assets/mi300_topology.png)

![MI300 IOD](./assets/mi300_iod.png)

> Source: [The CDNA3
> Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)

Each XCD has its own L2 cache and is connected to LLC cache (a.k.a. MALL:
memory-attached last-level, a.k.a 'Infinity Cache') via its IOD.

### MI300X Memory Bandwidth

The GPU comes with 8 stacks of HBM3 memory, with each IOD is connected to two
HBM stacks. The memory interface width is 8192 bits or 1024 bytes. HBM3 supports
Quad Data Rate (QDR), which brings the theoretical peak bandwidth to:

```
1300 MHz * 1024 B * 4 = 5.2 TB/s
```

To achieve bandwidths close to the peak, the kernel code must, at minimum, engage
all 4 IODs.

### MI300X Cache Hierarchy

There are 3 levels of cache:

| Name | Size | Cache Line Size | Associativity | Execution Unit | Comments |
| --- | --- | --- | --- | --- | --- |
| L1D | 32 kB | 128 B | 64-way, 4 sets | Compute Unit | Write-through |
| L1I | 64 kB | 128 B | 8-way set-associative | Compute Unit | Instruction cache |
| L2 | 4 MB (16 channels * 256 kB) | 128 B | 16-way set-associative, 128 sets per channel | XCD | Writeback / Write-allocate, Coherent within XCD |
| LLC | 32 MB (16 channels * 2 MB), 256 MB total | 64 B | 16-way set-associative, 2048 sets per channel | IOD | Non-coherent, MALL |

L2 cache is flushed between kernel launches. Memory accesses that miss L2 are
coalesced and go to the data fabric.

> [!TIP]
> Due to power consumption, we want to minimize the number of data fabric
> transactions.

### Execution Model

When a kernel is launched, its workgroups get distributed across the GPU.
A workgroup executes on a single CU and never gets migrated to another CU.

Each subgroup / wave within the workgroup gets assigned to a single SIMD unit.
One SIMD has 10 waveslots used to 'context-switch' between the assigned subgroups
(up to 10). However, only up to 16 subgroups are allowed within a single
workgroup.

On GFX9, the subgroup size is 64, which, for most instructions, necessitates
multiple clock cycles on 16-lane SIMD. For example, something like an `add`
would execute in 4 cycles, for each set of 16 threads within the subgroup.

> [!TIP]
> To fully utilize all 4 SIMDs within the CU, use a workgroup size of 256 (or a
> multiple of 256). To conserve power, you can use a workgroup size 128 so that
> 2 SIMDs remain idle.

![GFX9 Compute Unit](./assets/gfx9_compute_unit.png)

> Source: [Introduction to AMDGPU](https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)

### Register Usage

GFX9 features large register files. Registers are DWORD-sized (4 B), and are
split into 3 general groups:
* SGPRs: Scalar registers (uniform value within subgroup threads). Up to 104
  SGPRs per workgroup on MI300.
* VGPRs: General-purpose vector registers (each thread holds a different value).
  Up to 256 VGPRs per thread on MI300.
* AGPRs: Matrix accumulation vector registers (each thread holds a different
  value). Up to 256 AGPRs per thread on MI300.

On CDNA2 and latter architectures, VGPRs and AGPRs share the same register file:
512 registers * 64 threads per SIMD.

> [!TIP]
> Register usage affects occupancy. A kernel utilizing all 256 VGPRs can
> launch only one or two subgroups per SIMD, depending on the number of AGPRs
> used.

When the kernel runs out of VGPRs, it may spill: first to AGPRs (through
the `v_accvgpr_*` instructions), later to scratch (through the `scratch_score_*`
instructions). The latter comes at a significant performance penalty.

> [!TIP]
> You can check the register usage by looking at the very end of the kernel
> ISA dump (`.s` or `.rocmasm` file). Make sure to check there are no spilled
> registers, which leads to poor performance. For example:
> ```
>    .group_segment_fixed_size: 0
>    .kernarg_segment_align: 8
>    .kernarg_segment_size: 24
>    .max_flat_workgroup_size: 256
>    .name:           main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32
>    .private_segment_fixed_size: 0
>    .sgpr_count:     19
>    .sgpr_spill_count: 0
>    .symbol:         main_0_dispatch_0_contract_2048x10240x1280_f16xf16xf32.kd
>    .uses_dynamic_stack: false
>    .vgpr_count:     102
>    .vgpr_spill_count: 0
>    .wavefront_size: 64
> amdhsa.target:   'amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-'
> amdhsa.version:
>  - 1
>  - 2
> ```

> [!TIP]
> You can hint the amdgpu register allocator by setting the
> `amgpu-waves-per-eu` llvm function attribute:
> https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-attributes.

> [!TIP]
> In HIP, to utilize more than the default maximum number of registers (128)
> you need to specify the workgroup size with the
> `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`
> attribute.

### Workgroup Memory (LDS)

On GFX9, workgroup / shared memory is not the same as L1 cache and its size
cannot be configured. An MI300 CU has 64 kB of workgroup memory (the same as the
VGPR register file size!).

LDS is split into 32 banks of DWORD-sized (4 B) entries. For example, a 128 B
contiguous chunk of memory spans all banks. The bank index of an accessed byte
is calculated with `(address / 4) % 32`.

When LDS is accessed, the first clock cycles are spent on sending the addresses.
It accepts up to 16 addresses per SIMD per cycle (up to 32 addresses per CU per
cycle). Next, the data is sent/received in multiple phases, depending on the
exact instruction used. Therefore, not all threads access LDS at the same time.

> [!TIP]
> LDS access is 'fast' in only two cases: when threads access the same
> address and the value gets broadcast, or when threads access a unique bank.
> Two or more threads accessing different addresses that map to the same bank
> create an **LDS bank conflict**.

> [!TIP]
> Make sure that workgroup memory accesses use `ds_` instructions instead
> of `flat_` instructions. The latter allow for both global and local addresses
> which makes them slower.

#### Avoiding LDS Bank Conflicts

With the number of LDS banks (32) not matching the subgroup size (64) nor the
SIMD size (16), it is not immediately obvious when bank conflicts arise.

LDS is able to access all 32 banks at once. Depending on the exact LDS
instruction used (read/write of `b32` vs. `b64` vs. `b128`), a different number
of threads within the same subgroup / wave access LDS banks concurrently. For
`b32`, 32 adjacent threads read/write from LDS (32 dwords), while for `b128` the
access covers *all* VGPRs of a group of 8 threads (also 32 dwords total).

For `ds_read_b32`, the access happens in two phases with the following groups
of 32 threads accessing LDS: `T0`-`T31`, then `T32`-`T64`.

For `ds_read_b64`, the access happens in four phases of 16 threads each:
`T0`-`T15`, `T16`-`T31`, `T32`-`T47`, then `T48`-`T64`.

For `ds_read2_b64`, the access happens in eight phases:
  * First `b64` in four phases: `T0`-`T15`, `T16`-`T31`, `T32`-`T47`,
    `T48`-`T64`.
  * Then the second `b64` in the next 4 phases, in the same groups of thread.

For `ds_read_b128`, the access happens in eight phases:
  1. `T0`-`T3` and `T20`-`T23`
  2. `T32`-`T35` and `T52`-`T55`
  3. `T4`-`T7` and `T16`-`T19`
  4. `T36`-`T39` and `T48`-`T51`
  5. `T8`-`T11` and `T28`-`T31`
  6. `T40`-`T43` and `T60`-`T63`
  7. `T12`-`T15` and `T24`-`T27`
  8. `T44`-`T47` and `T56`-`T59`

For `ds_write_b128`, the access happens in eight phases:
  1. `T0`-`T7`
  2. `T8`-`T15`
  3. `T16`-`23`
  4. `T24`-`T31`
  5. `T32`-`T39`
  6. `T40`-`T47`
  7. `T48`-`T55`
  8. `T56`-`T63`

When more than one thread within a group currently being handled attempts to
access the same bank, a bank conflict occurs. The conflict may be over one or
more banks, depending on the addresses accessed. The higher the number of
threads that participate in a conflict over the same bank, the higher the LDS
access latency.

Bank conflicts are resolved by picking the first group of threads (by thread ID)
that do not conflict, and then this is repeated for leftover threads. In the
worst case where all threads access the same bank, this can turn into a *waterfall
loop* (only one thread gets to access LDS per cycle).

> [!TIP]
> It is best to use wide 16, 8, or 4 byte-wide LDS instructions (e.g.,
> `ds_read_b128`, or `ds_read2_b64` for two 4 B values at unique
> addresses, `ds_read_b64`, `ds_read_b32`).

### Global Memory

To achieve peak kernel performance on MI300, it's crucial to access the global
memory efficiently and minimize the number of data fabric transactions.

The optimal memory access size is 8 B or 128 bits, using the
`global_load_dwordx4` and `global_store_dwordx4`. Further, make sure that the
memory access is subgroup-contiguous, such that the whole subgroup accesses 512 B
at once.

A sequence of up to 4 adjacent `global_load_dwordx4` instructions (implicitly)
forms a *clause* that translates to a single data fabric transaction.

> [!TIP]
> To achieve peak L1 bandwidth, make sure that your memory access engages all
> four L1 cache sets. That is, at the level of the workgroup, you should be
> loading 4 cache lines (128 B) that each map to a different cache set.

> [!TIP]
> For data that is 'streamed' and does not need to be cached, consider
> using *non-temporal* loads/stores. This disables coherency and invalidates
> cache entries.

> [!TIP]
> For allocations of 4 GB or less, you can implement predicated loads using the
> `buffer` instructions.

## Data-Parallel Primitives and Warp-level Reduction

For cross-lane data sharing, the most straightforward way is LDS. Some lanes
write data to some locations on LDS and other lanes read data from LDS. Besides,
there are several instructions can be used to share data cross lanes within a
wavefront/warp.

Here's a brief introduction of these instructions. Please check out [this
blog](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/) for
details.

### ds_permute/ds_bpermute

`ds_permute`/`ds_bpermute` instructions use LDS hardware for data sharing but
don't actually write to an LDS location. But it still needs `s_waitcnt`
instruction to determine when data is returned to `dest` VGPR.

Example:
```nasm
ds_bpermute_b32 dest, addr, src [offset:addr_offset]
```

### ds_swizzle

Compared to `ds_bpermute`, the `ds_swizzle` instruction doesn't require an
additional VGPR for offset since it's encoded in the instruction.

`ds_swizzle` is likely to have less address generation instructions required
than `ds_bpermute`.

The cons are:
1. It only supports limited patterns.
2. Similar to `ds_bpermute`, `s_waitcnt` is required to wait for the `dest` VGPR.

Example:
```nasm
ds_swizzle_b32 dest, src offset:ds_pattern
```

### Data-Parallel Primitives, DPP

DPP is a 32-bit instruction modifier appended to the normal VALU instructions.
It allows VALU instructions to access data in neighboring lanes directly, which
means it doesn't need LDS hardware anymore, hence `s_waitcnt` instructions are
**not required**.

Unfortunately, it also supported limited patterns like `ds_swizzle`. And there
are some instructions that can't be modified by DPP.

Example:
```nasm
; Normal VALU instruction.
v_add_f32

; Instruction modified by DPP.
v_add_f32_dpp
```

It's worth mentioning that DPP has different names and syntaxes on different
architectures:
* CDNA: DPP
* RDNA: DPP8/DPP16

For details, please check the [MI300 ISA Reference
Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
and the [RDNA3 ISA Reference
Guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf).

### How to use them in MLIR

Each instruction has a corresponding Op in MLIR (except for `ds_permute`, this
one is not implemented at the time of writing):
* `ds_bpermute`: `rocdl.ds_bpermute`
* `ds_swizzle`: `rocdl.ds_swizzle`
* DPP: `rocdl.update.dpp`, `amdgpu.dpp` (a thin wrapper around
  `rocdl.update.dpp` with more comprehensive user interface, e.g., replace magic
  numbers with enums)

The first 2 are straightforward, while DPP follows a different fashion.

Since DPP is an instruction modifier instead of an instruction itself, there are
tremendous number of combinations of VALU instructions and DPP. To solve that,
`rocdl.update.dpp` and `amdgpu.dpp` are designed to be a wrapper of
`v_mov_b32_dpp` instruction. And it depends on LLVM compiler to fuse it with the
subsequent VALU instruction **with best efforts**.

For example, `v_mov_b32_dpp` + `v_add_f32_e32` might be fused into `v_add_f32_dpp`.

There are plenty of constraints stopping an instruction from being merged. For
example, if either the `bank_mask` or the `row_mask` is not `0xf`, it can't be
fused. You can check the
[GCNDPPCombine::combineDPPMov](https://github.com/llvm/llvm-project/blob/ab51eccf88f5321e7c60591c5546b254b6afab99/llvm/lib/Target/AMDGPU/GCNDPPCombine.cpp#L522)
function to see how it works.

### Comparison

To summarize, there's no free lunch: instruction's expressivity comes at the
expense of performance.

The relative performance of cross-lane instructions is as follows:

DPP > `ds_swizzle` >= `ds_permute` > `ds_bpermute`

while the generality ranking is the reverse:

DPP < `ds_swizzle` < `ds_permute` < `ds_bpermute`

This table presents the approximate instruction latency, collected
experimentally on Fused Softmax kernel with
[rocprofv2](https://github.com/ROCm/rocprofiler?tab=readme-ov-file#plugin-support)
on the MI300 GPU:

| Instructions           | MLIR Op                      | Hardware     | latency/#cycles |
| ---------------------- | ---------------------------- | ------------ | --------------- |
| ds_permute/ds_bpermute | rocdl.ds_bpermute            | LDS hardware | ~50*            |
| ds_swizzle             | rocdl.ds_swizzle             | LDS hardware | ~50*            |
| DPP                    | rocdl.update.dpp, amdgpu.dpp | VALU         | 4~12            |

*: For `ds_permute`/`ds_bpermute` and `ds_swizzle`, the latency includes the
instruction itself and its corresponding `s_waitcnt` instruction.

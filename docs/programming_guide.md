# amdshark Tank Programming Guide

*NOTE: This document should be considered forward looking. While proof of
 concepts of everything presented here are available, it more provides a
 roadmap for completion of the project.*

amdshark Tank provides development tooling to create inference optimized models
using [PyTorch](https://github.com/pytorch/pytorch) and
[IREE](https://github.com/iree-org/iree). While the basic programming model
is provided by those projects, amdshark Tank brings an opinionated approach to
*how* models, layers, and datasets are managed. This helps us bridge the gap
from the more dominant training-time use cases for which these things are
typically developed, focusing instead on the organization of the data,
parameters, and servability concerns. This is by no means the only way to
build inference solutions on top of these base tools, but we built it after
hitting the same issues over and over again.

We were inspired by the great work that the
[llama.cpp](https://github.com/ggerganov/llama.cpp) team did in terms of
systematizing access to LLMs, focused on inference, and showing us all that
when you focus on the task specifically, you can get further faster. With
that said, our toolchain and compilers are built for PyTorch and Python, and
we wanted to see what it would take to apply a similar development
methodology but with the interactive tools we're comfortable with. In practice,
we simply use the the technology and models developed for llama.cpp when
appropriate, and our tooling can source directly from GGUF files for supported
model families. For other model families, we follow a similar development
methodology that encourages specialization and systematization.

## Additional reading:

* [IREE's out of the box AOT PyTorch Support](https://iree.dev/guides/ml-frameworks/pytorch/#ahead-of-time-aot-export)
* GGUF [standardized K/V hyperparams](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-key-value-pairs)
  and [standardized tensor structured](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#standardized-tensor-names)
  for LLMs, which we adopt wholsale.
* [llama.cpp quantization schemes](https://github.com/ggerganov/llama.cpp/blob/master/ggml-common.h#L144),
  which we support natively (presently a subset of them).

# Core Concepts

amdshark Tank's core ideas come from a combination of history with various Python
based inference setups. They differ from PyTorch's canonical `nn.Module` based
usage in a few key ways:

* A `Dataset` is a first order concept, consisting of a set of systematized
  parameters and a dict of `InferenceTensors`. Datasets can be loaded from either
  GGUF files or IREE's native IRPA file format. They can be saved to IRPA files.
* An `InferenceTensor` is a *logical* tensor with a shape and an at-rest dtype.
  Each `InferenceTensor` can be manifested as a specific type of physical
  representation:

    - `PrimitiveInferenceTensor`: Simply backed by a PyTorch tensor (typically
        from a memory mapped array in a `Dataset` on storage but can be arbitrary).

    - Packed `QuantizedTensor`: These tensors are backed by a single at-rest
        PyTorch tensor with a specific manner of packing scheme, logically
        represented by a `Layout`. In practice, each GGUF quantization scheme has
        a distinct type of packed `QuantizedTensor` implementation. It is an
        open world, and arbitrary implementations are easily created.

    - Planar `QuantizedTensor`: These tensors are backed by an arbitrary
        dictionary of tensors (i.e. "planes"), logically represented by a `Layout`.
        Typically, packed `QuantizedTensors` can be converted to planar form.
        As a tensor compiler, IREE operates best on the planar form for generic
        kernels, since it is easiest for it to process directly and repack into
        more architecture specific forms.

* A `Layout` operates on a planar arrangement, providing the reference math
  to quantize/dequantize, specifically preserving any latent block structure
  to the underlying data. Custom kernels are typically keyed on the `Layout`
  type for specialization.
* `InferenceOps` are defined for all "hero ops" of modern ML models. These ops
  take as arguments combinations of plain PyTorch tensors and
  `InferenceTensors`. They are pluggable and have a dispatch mechanism for
  delegating to an appropriately optimized kernel for each combination.
  Generic optimized implementations are provided for all planar layouts, and
  specific implementations can be coded for arbitrary packed forms if it is
  profitable to do so.
* `Theta` objects are used to contain a dictionary of `InferenceTensors` in
  a `Dataset`. Theta dictionaries are hierarchical and support various
  transformations and slicing of the parameter set. As a parallel hierarchy
  to the `nn.Module`, most layers in amdshark Tank descend from `ThetaLayer`,
  meaning that they are initialized with a slice or subset of the root theta.
  Typically such layers are then implemented in terms of `InferenceOps`, which
  provide the optimized workhorse implementations based on the actual types
  of `InferenceTensors` in a given theta slice.


# Development Model

All layers in amdshark Tank are completely defined for eager execution -- even
those which are operating on exotic inference tensors. In reference mode, all
math is implemented in terms of native PyTorch operations, which is always
available and allows rapid bootstrapping prior to the development of more
optimized type/machine specific kernels. In practice, bootstrapping a new
model is always done with interactive, eager execution, using the usual tools
to verify numerics and algorithmic correctness.

By default, `InferenceOps` with optimized, portable IREE kernels are activated,
even in eager mode, and any set of kernels can be interactively authored and
evaluated in this way. While this precludes some advanced fusions that the
compiler could do on the whole graph, it is a large productivity boost for the
vast majority of modeling techniques. Runtime logs can be dumped to a directory,
saving all intermediate IR, specializations, and invocations so that kernel
and optimization teams can work in tandem with model bringup.

For actual deployment outside of a development environment, we export the entire
program, compile it with IREE and plug it into the `shortfin` serving engine
(or other harness as needed).

In all usage and compilation modes, amdshark Tank models and layers make explicit
use of various key features:

* Dynamism: With Torch Dynamo's excellent and detailed support for dynamic
  shapes, we can express precisely specialized layers and programs with the
  task specific appropriate level of dynamism (i.e. batch and/or sequence
  length vs all characteristics). Due to PyTorch's strong modeling of these
  concepts, it lets us specialize the underlying kernels with strong
  constraints.
* Mutability: PyTorch is mutable, and modern serving solutions have to manage
  increasing amounts of mutable state in the form of caches and other
  constructs. Unlike in many prior ML workloads, cache management for modern
  genai can only be done efficiently with in-place and/or indirection at
  scale. Dynamo and IREE's implementation preserves mutability through to the
  compiler stack and runtime which lets us express these kinds of dataflows
  naturally.
* Custom Ops and Fusion: Efficient inference requires specialization of
  increasingly important and exotic layouts and shapes of fusions. While
  compilers are good at certain types of these optimizations, for the true
  hero ops of any architecture, we prefer a development model which makes it
  cheap to specialize such things versus relying on the compiler to get
  everything right from a high level compute graph. In practice, this means that
  we write custom ops for a lot of things, and we have invested in approaches
  that make this cheap and scalable. In many cases, our custom ops are simply
  bypassing layers of the framework and targeting lower level forms of the
  compiler directly, where there is no ambiguity as to the structure. In other
  cases, we write the implementations in a low-level Pythonic kernel language.
  In still others, nothing beats a hand coded kernel, and we use those.
  amdshark Tank makes it easy and interactive to do any of this.

By applying this development model and making it cheap to specialize, we aim to
hit the sweet spot in our models whereby modeling code is re-used when
appropriate but never at the expense of optimized performance.

# Development Activities

## Developing Custom Ops

See:

* Existing ops in `amdsharktank.ops`
* Test cases in `tests/ops`

TODO: Complete this section.

* Differentiate between high-level (i.e. linalg, etc) custom ops and low-level
(i.e. TK, Triton, binary kernels).
* Explain/document the `CustomOp` facility in Turbine and how to use it.
* Explain how `CustomOps` are eagerly dispatched or graph compiled via the
  same implementation.

## Layer Development

See: `amdsharktank.layers` for existing core layers

TODO: Discuss common layers, etc.

## Model Development

See: `amdsharktank.models` for specific model family implementations.

TODO: Discuss model configs, registry, and serving protocols to implement (for
various kinds of serving scenarios).

### LLM Models

We have some conventions and common infrastructure for LLM implementations,
including:

* K/V cache management (either direct or page table based).
* Generic export and runner scripts (see current: `export_paged_llm_v1.py`,
  `paged_llm_v1.py`).
* Systematized hyperparameters and configs that cover the whole family.

TODO: The scripts and configs are not organized very well and need to be
generalized a bit vs hard-coded for the LLAMA model we did as a POC.

### Debugging Numerical Imprecisions

Often numerical errors occur only when multiple components are combined into a large
model. Usually we have some ground-truth reference. This can be an external model.
E.g. a model from Hugging Face's
[transformers](https://huggingface.co/docs/transformers) package.
It may be another variant of the same model in amdshark Tank. For example a variant in
higher numerical precision.

We can employ binary search to find the source of the numerical error,
which means that theoretically it should not be too slow.
This usually involves manual modification of the model.
For example iteratively adding and removing early returns from a deeply nested call
stack to bisect the model code. This process is error prone and can't be replicated
easily.
This debugging can be facilitated by employing tensor tracing of some intermediate
values and then comparing them to the reference.

#### Tensor Tracing with Module Patching

One approach is to not modify the model code at all, but to patch it externally such
that it traces arguments and results of some method calls of child submodules.

```Python
import torch
from amdsharktank.utils import debugging
from amdsharktank.utils.patching import TraceTensorModulePatch

# Create the model.
model: torch.nn.Module = MyModel(...)

# Patch the model.
patcher = TraceTensorModulePatch()
patcher.patch_child_modules(model)

# Enable tracing and set the sink to save into safetensors format.
debugging.flags.enable_tensor_trace = True
debugging.flags.trace_path = "my/trace/dump/path"
debugging.set_trace_tensor_callback(debugging.trace_tensor_to_safetensors_callback)

# Run the model.
model(x, y, z)
```

In the above example the results of all `forward` methods in `model` and its submodules
will be trace. The tensor trace will be recorded into safetensors files under directory
`my/trace/dump/path`.

The `patch_child_modules` method supports more sophisticated filtering of what module
methods to be patched. See its description for more information.

An alternative approach for tracing is instead of using `TraceTensorModulePatch` to use
`SaveModuleResultTensorsPatch`. The difference is that `SaveModuleResultTensorsPatch`
keeps the trace in memory and dumps it at the end into 1 safetensors file.
The plan is to retire `SaveModuleResultTensorsPatch` and add a standard trace sink that
is able to record into a single file.

```Python
patcher = SaveModuleResultTensorsPatch()
patcher.patch_child_modules(model)
model(x, y, z)
patcher.save_file("tensor-trace.safetensors")
```

#### Tensor Tracing with IREE

IREE also supports tensor tracing. In a similar way as in eager, models can be exported
with tracing operations embedded.


```Python
from amdsharktank.utils import debugging
from amdsharktank.utils.iree import get_iree_devices, load_iree_module
from amdsharktank.utils.patching import TraceTensorModulePatch
import iree.runtime
import torch

# Create the model.
model: torch.nn.Module = MyModel(...)

# Patch the model.
patcher = TraceTensorModulePatch()
patcher.patch_child_modules(model)

# Enable tracing.
debugging.flags.enable_tensor_trace = True

# Export to MLIR and compile the model with IREE.
...
iree_module_path = ...

# Configure the IREE module debug sink and load the module.
iree_devices = get_iree_devices(driver="local-task", device_count=1)
iree_buffere_view_trace_callback = (
    make_hal_buffer_view_trace_default_callback(iree_devices[0])
)
debug_sink = iree.runtime.HalModuleDebugSink(
    iree_buffere_view_trace_callback
)
iree_module, iree_vm_context, _ = load_iree_module(
    module_path=str(iree_module_path),
    devices=iree_devices,
    debug_sink=debug_sink,
)

# Configure amdshark Tank tracing.
debugging.flags.trace_path = "my/trace/dump/path"
debugging.set_trace_tensor_callback(debugging.trace_tensor_to_safetensors_callback)

# Run your IREE model.
...
```

For a complete example see `TestTraceTensors.testTraceTensorWithIree` in
[ops_test.py](../amdsharktank/tests/ops/ops_test.py).

There is a known limitation that tensor tracing with IREE does not work in a
multi-device context.

#### Comparing Tensor Traces

Traces can be compared using the tool
[amdsharktank.tools.compare_safetensors](../amdsharktank/amdsharktank/tools/compare_safetensors.py).

```
python amdsharktank.tools.compare_safetensors \
  --dir=trace/compare/dump \
  my-actual-trace.safetensors \
  my-expected-trace.safetensors
```

The tool supports also remapping of corresponding tensor key values via
`--keys_map_path`.
This is useful when the actual and reference models have different structures.
E.g.
```
# actual -> expected corresponding tensor key
model.layer.0.attention.forward.0: model.block.0.attention.forward.0
```

## Tools

### Dataset Tool

Datasets can be manipulated via some common command line tools documented here.
For more advanced use, you are encouraged to `load()`, transform, and `save()`
as needed.

TODO: Document tools.
TODO: Rename `dump_gguf.py` to `dataset_tool.py` and generalize.
TODO: Document APIs for interactively manipulating datasets, converting
parameters, quantizing, extending, etc.
TODO: Discuss the role of composable datasets for adding features like
quantized activations to a model.

## Quantization and Tensor Types

### Generic Layouts

See: `amdsharktank.types.layouts`

#### BlockScaledLayout

See `amdsharktank.types.layouts.BlockScaledLayout`.

Block-quantized representation which consists of a scale (`d`)
and offset (`m`) per block in a higher precision type. The offset, if
present, is pre-scaled.

The dequantization formula:

```python
result = d.to(dtype) * qs.to(dtype) + m.to(dtype)
```

The inner-most dims will retain block structure. For example, if the
block size is 32 and the original shape was NxK, then the component
shapes would be:

* `d`: `[N, K // 32, 1]`
* `m`: `[N, K // 32, 1]`
* `qs`: `[N, K // 32, 32]`

Note that the offset (`m`) is optional.


#### BlockScaledI4Layout

See `amdsharktank.types.layouts.BlockScaledI4Layout`.

A BlockScaledLayout where the `qs` are internally packed 2 values per byte.

Per convention, the `qs` property returns a tensor as either uint8 or
int8 (depending on `signed=`) that can be used directly for arithmetic.
The underlying bit-packed tensor can be accessed via `qs_bit_packed` and
it is laid out in little endian bit order, linearly across the block
dimension. There are an arbitrary ways to organize such things, and
if more specificity is needed, a dedicated layout class should be used. In
general, for these "generic" layouts, we choose defaults that mate well
with how the compiler infra and prevailing targets are built and trust that
optimizations that care will choose a specific packing.


#### SuperBlockOffsetScaled_4_6_Layout

See: `amdsharktank.types.layouts.SuperBlockOffsetScaled_4_6_Layout`

Super block scaled q4 matmul with transposed RHS and 6 bit sub-block
scale/offset.

Arguments:

* `a`: [B, M, K]
* `d`: [N, SUP_COUNT, 1]
* `dmin`: [N, SUP_COUNT, 1]
* `sb_scales_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
* `sb_scales_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
* `sb_min_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
* `sb_mins_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
* `qs`: [N, SUP_COUNT, SUB_COUNT, BS // 2]

Where: `K == SUP_COUNT * SUB_COUNT * BS`

Given this and hi/lo combined into a single value, the dequantization
formula is:

```python
d_scaled = (d * sb_scales).unsqueeze(-1)
dmin_scaled = (dmin * sb_mins).unsqueeze(-1)
return d_scaled * qs - dmin_scaled
```

### GGML Layouts

#### Q8_0

Corresponds to GGML Q8_0 quantization (8 bit, symmetric).

```c
#define QK8_0 32
typedef struct {
    ggml_fp16_t d;         // delta
    int8_t  qs[QK8_0];     // quants
} block_q8_0;
```

This is generically planarized to a `BlockScaledLayout` unless if a specific
packed, optimized kernel is available.

#### Q4_1

Correspnds to GGML Q4_1 quantization (4bit qs with FP scale/offset).

```c
#define QK4_1 32
typedef struct {
    ggml_fp16_t d;          // delta
    ggml_fp16_t m;          // min
    uint8_t qs[QK4_1 / 2];  // nibbles / quants
} block_q4_1;
```

This is generically planarized to a `BlockScaledI4Layout` unless if a specific
packed, optimized kernel is available.

#### Q4_K

Corresponds to GGML Q4_K quantization (4 bit qs with super/sub-blocks, where
the super-block scale/offset is FP and the sub-block scale/offset is 6bit
unsigned integers).

```c
#define QK_K 256
#define K_SCALE_SIZE 12
typedef struct {
    union {
        struct {
            ggml_half d;    // super-block scale for quantized scales
            ggml_half dmin; // super-block scale for quantized mins
        } GGML_COMMON_AGGR;
        ggml_half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
```

This is generically planarized to the `SuperBlockOffsetScaled_4_6_Layout`
unless if a specific packed, optimized kernel is available:

This uses the same 6bit scales and mins packing scheme and super-block
structure as some other "K" quantizations. We planarize itthe inner block
scales and mins to 4 arrays with POT bit depths.

* 8 * i4 : uint8 ms_low[4]
* 8 * i2 : uint8 ms_hi[2]
* 8 * i4 : uint8 ds_low[4]
* 8 * i2 : uint8 ds_hi[2]

This gives us the characteristic of linear addressing on the components,
which the compiler can do more with than a heavily interleaved format.

Arguments:

* `a`: [B, M, K]
* `d`: [N, SUP_COUNT, 1]
* `dmin`: [N, SUP_COUNT, 1]
* `sb_scales_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
* `sb_scales_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
* `sb_min_hi`: [N, SUP_COUNT, SUB_COUNT // 4]
* `sb_mins_lo`: [N, SUP_COUNT, SUB_COUNT // 2]
* `qs`: [N, SUP_COUNT, SUB_COUNT, BS // 2]

Where: `K == SUP_COUNT * SUB_COUNT * BS`

Given this and hi/lo combined into a single value, the dequantization
formula is:

```python
d_scaled = (d * sb_scales).unsqueeze(-1)
dmin_scaled = (dmin * sb_mins).unsqueeze(-1)
return d_scaled * qs - dmin_scaled
```

#### Q5_K

TODO

#### Q6_K

TODO

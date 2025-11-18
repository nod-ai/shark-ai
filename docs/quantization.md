---
title: Direct quantization with amdsharktank
author: Stella Laurenzo
date: June 30, 2024
---

# Direct Quantization with amdshark Tank

As a toolkit for building and adapting PyTorch based models for deployment,
amdshark Tank provides rich quantization support. By targeting the
[IREE compiler](https://github.com/iree-org/iree) for optimizations, we can
strike a balance with our quantization setup that:

* Is completely eager friendly for development, debugging, and light use.
* Optimizes directly using the advanced fusion and code generation capabilities
  native to the IREE compiler.
* Is configured via calibration parameters from a variety of quantization
  simulators.
* Is implemented in PyTorch in terms of the underlying quantization math
  without using opaque approaches like QDQ, fakequant, or black box quantization
  op sets.
* Can be extended with fine-grained scheme/target specific kernel fusions at
  need while relying on the compiler to get it right for most things without
  additional fuss.

While not a completely novel concept (there are several full or partial priors),
this approach does deviate from what has become the classical norm of
the last ten years which were born primarily out of mobile scenarios and from
a position of quantization being something that some minor subset of use cases
call for. As such, these prior indirect approaches focused on being able to be
bolted on, through layers of infrastructure, to model development practices
that were not considering them. It should be noted that the IREE ecosystem
(primarily built on top of
[torch-mlir and its ONNX support](https://github.com/llvm/torch-mlir/blob/main/docs/importers/onnx_importer.md))
supports these indirect schemes -- effectively using compiler transformations
under the covers to do opaque model transformations that mirror a subset of
what is exposed directly to the user in the rest of this document.

As an alternative, when developing amdshark Tank and bringing up the initial
models, we wanted something more flexible, easier to debug/extend, and
less laden with needing to lowest common denominator something for everyone
in order to fit into fixed-function op sets that are very expensive to change.
We call the result *Direct Quantization* since it is formulated directly in
terms of the math that underlies the layers of infrastructure that exists in the
classical approaches.

This is not a principled, all or nothing, approach. It is simply that after many
years of staring at opaque walls of *slightly different* numbers from layers
of infrastructure, we preferred to write new implementations in a way that
was more inspectable and open to evolution.

## API Levels

The direct quantization features are exposed through a few different levels of
Python API, extending from the user/nn.Module level down through types/ops.
It is expected that for model-adaptation scenarios, users may choose to just
do the traditional thing and replace key nn.Modules; whereas in custom
model development, it may be beneficial to reach deeper. It is all just a small
amount of Python code implementing direct math and packing schemes.

1. `nn.Module`: Provides nn.Module subclasses for common quantized sequences.
   While requiring a bit of external configuration data, these should be
   drop-in replacements for subsets of the functionality available in stock
   PyTorch modules like `Linear` and `Conv2D`.
2. Types/Ops: The `nn.Module` implementations we provide are built in terms
   of amdshark Tank custom
   [`InferenceTensor`](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/types/tensors.py#L153)
   and [polymorphic functional ops library](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/signatures.py).
3. Op specializations for optimized subsets of op type signatures and features
   (for example, [an optimized affine quantized linear specialization for
   supported combinations of `TensorScaledLayout` arguments](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/qlinear_impls.py)).

(TODO: good place for a diagram)


### `nn.Module` Implementations

Available modules that support direct quantization (TODO: refactor to use
torch "Module" terminology and naming schemes consistently):

* [`LinearLayer`](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/layers/linear.py)
* [convolution layers](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/layers/conv.py)

Note that most amdsharktank modules extend
[`ThetaLayer`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/layers/base.py#L325),
which calls for a bit of explanation. Traditional PyTorch Modules directly
instantiate their backing parameters in their constructor. For dataset-heavy
and polymorphic implementations like we commonly see in quantization and
distribution, however, it can be beneficial to separate these concerns.

The `ThetaLayer` simply takes a
[`Theta` object](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/layers/base.py#L328),
which is a tree-structured bag of native `torch.Tensor` or `InferenceTensor`
instances, and it adopts the tensors in the bag as its own vs creating them.
For those familiar with the concept, this is a form of dependency-injection
for tensors that allows us to easily separate the question of how you prepare
the data to operate on ("Theta") from the computation ("Module"), allowing
us to build tooling specifically geared towards data transformation based on
`Theta` trees. `Theta` objects support lossless round-trip to disk of all of
their metadata and custom types, making them ideal for use in data
transformation pipelines. The on-disk form can also be directly mmap'd by
the IREE runtime for compiled deployment with or without Python.

If replacing a native PyTorch module with a amdsharktank module, one would simply
take the original parareters, put them in a `Theta` object and pass it to the
new Module. Tooling will eventually be provided to automate this.

For models that were coded directly in amdsharktank, we usually start from a
root `Theta` loaded from disk (or transformed on-the-fly from something like a
GGUF file) and then construct each model layer by mapping a node in the `Theta`
tree to a specific Module instance.

### Types

We've already met the `Theta` object above, which holds a tree of something
called an
[`InferenceTensor`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/tensors.py#L192).
Now we describe what this is. Note that presently, `InferenceTensor` is not a
`torch.Tensor` but its own `ABC` type that:

* Is a composite of named, global `torch.Tensor` sub-parts.
* Has a shape
* Can be named
* Supports serialization/deserialization to IREE parameter archives and metadata
  structs.

The composition is key since by having a 1:N mapping of logical tensors to
physical tensors with enough structure to faithfully persist/round-trip in a way
that an eventual inference engine can mount directly, we leave open direct
type-mappings of many forms of algebra that show up in modern ML models (i.e.
quantization, sharding/replication, sparsity, statistics/simulation, etc).

Note that these logical tensors are power-user features: unless if specifically
instructed, they do not escape the more user-oriented `nn.Modules`, but they
are used heavily to implement those modules and manage data transformation
pipelines.

#### InferenceTensor subtypes

There is a growing list of `InferenceTensor` sub-types, many of which are
related to quantization:

* [`PrimitiveTensor`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/tensors.py#L628):
  A simple composition of a single `torch.Tensor`. This is often used
  interchangeably with a `torch.Tensor` but is present for completeness of
  the type hierarchy and to be able to type select on.
* [`QuantizedTensor`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/tensors.py#L740):
  Abstract base class of all quantized tensors, providing two primary operations:

  * `unpack`: Accesses the backing `QuantizedLayout` of the tensor, which is
    used for all concrete manipulation of the contents.
  * `to_planar`: Converts any `QuantizedTensor` to a canonical "planar form"
    (i.e. if the specific type was implemented in terms of a compressed/packed
    layout, this explodes it into a canonical representation of individual
    tensors which can be algebraically implemented individually/generically).

* [`PlanarQuantizedTensor`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/tensors.py#L778):
  Concrete implementation for all non-packed quantized tensors that can be
  losslessly represented by a layout based on individual tensor components.
  All `QuantizedTensor` instances can be converted to a `PlanarQuantizedTensor`.

* [`QuantizerTensor`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/quantizers.py#L52):
  (note the "r" in the name) An abstract `InferenceTensor` that exposes a
  `quantize(torch.Tensor | InferenceTensor) -> QuantizedTensor` operation used
  to transform an arbitrary tensor to a quantized form. There are a handful
  of implementations:

  * `StaticScaledQuantizer`: Performs per-axis or per-tensor affine quantization
  to a specified dtype using scale/offset tensors. Produces a `PlanarQuantizedTensor`
  with a `TensorScaledLayout`.
  * `DynamicScaledQuantizer`: Similar to `StaticScaledQuantizer` but derives its
  scale dynamically based on the contents of the tensor passed to `quantize`.

#### `QuantizedLayout`

Previously we saw that the `QuantizedTensor` and `QuantizerTensor` types
manipulate tensor contents via `QuantizedLayout`, but we haven't yet defined
that. The *Tensor types are structural and exist to give identity, but the
`QuantizedLayout` is where the "magic happens".

[`QuantizedLayout`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/tensors.py#L83)
is an `ABC`, supporting:

* Serialization/interop with parameter archives.
* Arbitrary static metadata.
* References a number of named `plane` tensors that fully represent the contained
  contents, using some form of packing/compression/composition-algebra.
* Defines a `def dequant(dtype: Optional[torch.dtype] = None) -> torch.Tensor`
  abstract method to produce a fully linearized dequantization of the contents.
* Optionally defines a `dequant_blocked()` method which dequantizes, preserving
  any latent blocking structure in the layout.

There are a number of implementations, as every quantization scheme typically
needs at least one concrete `QuantizedLayout`. Simple schemes like affine
quantization can be fully defined in terms of a single
[`TensorScaledLayout`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/layouts.py#L43).
Whereas packed schemes like we find in inference engines like GGML and XNNPACK
optimally require both a packed layout and a planar layout.

In such cases, the packed layout typically depends on custom kernel
implementations that operate in a fixed function way on a very specifically
formatted representation in memory (often optimized for a specific architecture
like a small family of CPU SKUs). The planar layout is chosen to represent
the underlying packed data in a way that can be algebraically manipulated or
repacked as needed. The planar layout can stand-in for "generic" implementations
that will work on any device using (typically) just the standard features of the
compiler to implement.

At the time of writing, amdsharktank has custom quantized layouts for a handful
of formats found in the wild:

* `BlockScaledI4Layout`: Simple planar I4 layout supporting block structure
  and a per-block scale.
* `SuperBlockOffsetScaled_4_6_Layout`: Planar form of the GGUF "Q4_K" format
  (which is actually a mixed 4/6 bit format where the quantized values are
  represented in a two layer blocking structure with both super-block FP
  scales and 6-bit sub-block scales/offsets).

There is also interop support for mmap'ing from GGUF files and
interpreting/transforming using their natively defined forms.

### Functional Ops

Previously, we found a rich type system defining all manner of layouts and
quantization schemes, but what can be done with it? That is where the
amdsharktank functional op library comes in. These
[logical ops](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/signatures.py)
provide the building blocks to implement built-in and custom `nn.Module`
implementations operating on `InferenceTensor` (and torch.Tensor) types.

The ops are all implemented in a pluggable way that can be specialized by
registering new type signatures or filter lambdas. This allows us to
define them generically in whatever way is needed to produce an optimal
implementation at any needed level of granularity:

* Generic/default implementation that "unboxes" to torch.Tensor to perform
  the operation with Torch-native ops (including on-the-fly dequant).
* Layout aware default implementations which understand specific block
  structures and preserve it when computing (when combined with a
  fusing compiler, this alone provides decent fallback implementations for a
  variety of "weight compression" oriented techniques). See
  [some examples](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/custom_impls.py#L51).
* Pure-Torch decompositions for algebraic techniques like affine quantization
  (when combined with a fusing compiler, this alone is sufficient for
  optimization). See
  [qlinear](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/qlinear_impls.py) and
  [qconv](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/ops/qconv_impls.py)
  implementations of actual affine quantized decompositions.
* Completely custom packed/optimized implementation. These can be written to
  activate on any level of detail of the type hierarchy. The implementation
  can be anything from a torch-level decomposition to custom-linalg-based
  template expansion, to a custom machine-oriented kernel defined in something
  like Triton, TK, C++, etc.

We generally presume that we are deploying through a fusing compiler and prefer
implementations higher up the list because they are more general/portable
(and for certain common classes of problems are actually optimal). But
complete customization is possible/encouraged by plugging in specializations
that use any kind of hand coded kernel.

In all cases, there is no difference between what runs eagerly and what
compiles, with the exception that when running eagerly, there is no fusion,
so performance will be degraded. This characteristic is critical for
implementation quantization schemes as it eliminates the layers of interpolation
and supposition that usually has to line up to ensure good numeric results
in classical implementations. Here, you just run the Python and use whatever
tools you normally do to print/check/debug. Then you compile it. If you used
completely custom kernels, they will be inlined/scheduled into your overall
program or launched individually during eager execution.

### Custom Kernels

Underlying everything is IREE's custom kernel system. This language agnostic
scheme lets us program IREE's internal IR directly in implementations of
normal PyTorch ops. This can be literally anything that IREE supports (which
is everything). We're just starting to exploit some of this as the PyTorch
level. Some examples:

* Something as simple as a humble runtime
[tensor trace/print](https://github.com/iree-org/iree-turbine/blob/main/iree.turbine/ops/iree.py#L52)
* [Simple linalg based template expansion](https://github.com/iree-org/iree-turbine/blob/main/iree.turbine/ops/_jinja_test_ops.py#L28)
  (see backing example [jinja template](https://github.com/iree-org/iree-turbine/blob/main/iree.turbine/ops/templates/test_add_jinja.mlir)).
* Optimal linalg-based [8-bit block scaled mmt for weight compression](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/kernels/mmt_block_scaled_q8.py)
  (see backing [jinja template](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/kernels/templates/mmt_block_scaled_q8_3d.mlir)).
* DSL based [like this fused attention kernel](https://github.com/iree-org/iree-turbine/blob/main/tests/kernel/fused_attention_test.py#L20)
  (note that in this case, the DSL exports to the unerlying IR-based registration
  mechanism used in the previous examples).
* Triton based (today via binary/CUDA/HIP kernel injection, soon via inlined-IR
  integration like the DSL example above).

Since all of these types of custom kernels are just defined with simple Python
tooling, they are really fast to iterate on. The linalg based kernels specifically
tend to be highly portable, and we don't hesitate to write one of those when
we need something specific that PyTorch doesn't provide out of the box
(i.e. [proper mixed-precision integer conv](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/kernels/conv_2d_nchw_fchw.py)
([template](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/kernels/templates/conv_2d_nchw_fchw.mlir))).

## Dataset transformation

All of the above is well and good, but for most quantization users, the key
question becomes "where do I get the parameters"? While we hope that in the
future, quantization simulators will directly use our APIs to emit datasets
that can be directly consumed by amdsharktank, there will always be cases where
you have to adapt from some parameter format or another.

We take a practical approach to this, writing implementation specific converters
where needed, and taking advantage of industry-standard consolidation points
where available (like GGUF) in order to cover a wider surface area.

Behind both is the notion of a [`Dataset`](https://github.com/nod-ai/amdshark-ai/blob/74521ab40c915878a46fa94c8ba4e88106b15ff8/amdsharktank/amdsharktank/types/theta.py#L366),
which combines some set of hyper-parameters with a root `Theta` object
(typically representing the layer-tree of frozen tensors). Datasets can be
losslessly persisted to IREE IRPA files, which can then be loaded by either
amdsharktank Python code or the native IREE runtime for standalone or PyTorch
integrated deployment.

We also provide support for creating a `Dataset` object from GGUF files, which
is a defacto rally point for LLM development these days. Once represented as
a `Dataset` object, generic tools can be used to translate, serialize,
transform, shard, etc.

See some examples:

* [models/punet/tools/import_hf_dataset.py](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/tools/import_hf_dataset.py) :
  Creating a `Dataset` object from an HF diffusers safetensors file and config.json.
* [models/punet/tools/import_brevitas_dataset.py](https://github.com/nod-ai/amdshark-ai/blob/main/amdsharktank/amdsharktank/models/punet/tools/import_brevitas_dataset.py) :
  Creates a quantized `Dataset` by combining:

  * HF diffusers `config.json`
  * HF compatible safetensors file, fine-tuned by Brevitas
  * Brevitas provided `quant_params.json` file providing per-layer calibration
    and configuration parameters ([example](https://huggingface.co/amd-amdshark/sdxl-quant-models/blob/main/unet/int8/quant_params.json)).

Through a long history of quantizing models, we have learned that such practical,
data oriented conversion tools are the foundation of getting work done. While
the APIs exist and can be eventually adopted to have more direct and seamless
interop, the ability to roll a data converter with good transformation-oriented
APIs in a few minutes is a lifesaver in this fragmented space, and it ensures
that with just a bit of Python code, you can typically port from any source
if needed.

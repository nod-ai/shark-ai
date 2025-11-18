# amdshark-ai: amdshark Modeling and Serving Libraries

![GitHub License](https://img.shields.io/github/license/nod-ai/amdshark-ai)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## amdshark Users

If you're looking to use amdshark check out our [User Guide](docs/user_guide.md). For developers continue to read on.

<!-- TODO: high level overview, features when components are used together -->

## Sub-projects

### [`shortfin/`](./shortfin/)

<!-- TODO: features list here? -->

[![PyPI version](https://badge.fury.io/py/shortfin.svg)](https://badge.fury.io/py/shortfin) [![CI - shortfin](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-libshortfin.yml/badge.svg?event=push)](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-libshortfin.yml?query=event%3Apush)

The shortfin sub-project is amdshark's high performance inference library and
serving engine.

* API documentation for shortfin is available on
  [readthedocs](https://shortfin.readthedocs.io/en/latest/).

### [`amdsharktank/`](./amdsharktank/)

[![PyPI version](https://badge.fury.io/py/amdsharktank.svg)](https://badge.fury.io/py/amdsharktank) [![CI - amdsharktank](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharktank.yml/badge.svg?event=push)](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharktank.yml?query=event%3Apush)

The amdshark Tank sub-project contains a collection of model recipes and
conversion tools to produce inference-optimized programs.

<!-- TODO: features list here? -->

* See the [amdshark Tank Programming Guide](./docs/programming_guide.md) for
  information about core concepts, the development model, dataset management,
  and more.
* See [Direct Quantization with amdshark Tank](./docs/quantization.md)
  for information about quantization support.

### [`amdsharktuner/`](./amdsharktuner/)

[![PyPI version](https://badge.fury.io/py/amdsharktuner.svg)](https://badge.fury.io/py/amdsharktuner) [![CI - amdsharktuner](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharktuner.yml/badge.svg?event=push)](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharktuner.yml?query=event%3Apush)

The amdshark Tuner sub-project assists with tuning program performance by searching for
optimal parameter configurations to use during model compilation. Check out [the readme](amdsharktuner/README.md) for more details.

### [`amdsharkfuser/`](./amdsharkfuser/)

[![CI - amdsharkfuser](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharkfuser.yml/badge.svg?event=push)](https://github.com/nod-ai/amdshark-ai/actions/workflows/ci-amdsharkfuser.yml?query=event%3Apush)

The amdshark Fuser sub-project is home to Fusilli - a C++ Graph API and Frontend to the IREE compiler and runtime for JIT compilation and execution of training and inference graphs. It exposes cuDNN-like primitives backed by IREE code-generated kernels. Check out [the readme](amdsharkfuser/README.md) for more details.

## Support matrix

<!-- TODO: version requirements for Python, ROCm, Linux, etc.  -->

### Models

Model name | Model recipes | Serving apps | Guide |
---------- | ------------- | ------------ | ----- |
SDXL       | [`amdsharktank/amdsharktank/models/punet/`](https://github.com/nod-ai/amdshark-ai/tree/main/amdsharktank/amdsharktank/models/punet) | [`shortfin/python/shortfin_apps/sd/`](https://github.com/nod-ai/amdshark-ai/tree/main/shortfin/python/shortfin_apps/sd) | [shortfin/python/shortfin_apps/sd/README.md](shortfin/python/shortfin_apps/sd/README.md)
llama      | [`amdsharktank/amdsharktank/models/llama/`](https://github.com/nod-ai/amdshark-ai/tree/main/amdsharktank/amdsharktank/models/llama) | [`shortfin/python/shortfin_apps/llm/`](https://github.com/nod-ai/amdshark-ai/tree/main/shortfin/python/shortfin_apps/llm) | [docs/shortfin/llm/user/llama_serving.md](docs/shortfin/llm/user/llama_serving.md)
Flux       | [`amdsharktank/amdsharktank/models/flux/`](https://github.com/nod-ai/amdshark-ai/tree/main/amdsharktank/amdsharktank/models/flux) | [`shortfin/python/shortfin_apps/flux/`](https://github.com/nod-ai/amdshark-ai/tree/main/shortfin/python/shortfin_apps/flux) | [`shortfin/python/shortfin_apps/flux/README.md`](https://github.com/nod-ai/amdshark-ai/blob/main/shortfin/python/shortfin_apps/flux/README.md)

## amdshark Developers

If you're looking to develop amdshark, check out our [Developer Guide](docs/developer_guide.md).

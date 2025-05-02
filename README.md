# shark-ai: SHARK Modeling and Serving Libraries

![GitHub License](https://img.shields.io/github/license/nod-ai/shark-ai)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

## SHARK Users

If you're looking to use SHARK check out our [User Guide](docs/user_guide.md). For developers continue to read on.

<!-- TODO: high level overview, features when components are used together -->

## Sub-projects

### [`shortfin/`](./shortfin/)

<!-- TODO: features list here? -->

[![PyPI version](https://badge.fury.io/py/shortfin.svg)](https://badge.fury.io/py/shortfin) [![CI - shortfin](https://github.com/nod-ai/shark-ai/actions/workflows/ci-libshortfin.yml/badge.svg?event=push)](https://github.com/nod-ai/shark-ai/actions/workflows/ci-libshortfin.yml?query=event%3Apush)

The shortfin sub-project is SHARK's high performance inference library and
serving engine.

* API documentation for shortfin is available on
  [readthedocs](https://shortfin.readthedocs.io/en/latest/).

### [`sharktank/`](./sharktank/)

[![PyPI version](https://badge.fury.io/py/sharktank.svg)](https://badge.fury.io/py/sharktank) [![CI - sharktank](https://github.com/nod-ai/shark-ai/actions/workflows/ci-sharktank.yml/badge.svg?event=push)](https://github.com/nod-ai/shark-ai/actions/workflows/ci-sharktank.yml?query=event%3Apush)

The SHARK Tank sub-project contains a collection of model recipes and
conversion tools to produce inference-optimized programs.

> [!WARNING]
> SHARK Tank is still under development. Experienced users may want to try it
> out, but we currently recommend most users download pre-exported or
> pre-compiled model files for serving with shortfin.

<!-- TODO: features list here? -->

* See the [SHARK Tank Programming Guide](./docs/programming_guide.md) for
  information about core concepts, the development model, dataset management,
  and more.
* See [Direct Quantization with SHARK Tank](./docs/quantization.md)
  for information about quantization support.

### [`sharktuner/`](./sharktuner/)

[![CI - sharktuner](https://github.com/nod-ai/shark-ai/actions/workflows/ci-tuner.yml/badge.svg?event=push)](https://github.com/nod-ai/shark-ai/actions/workflows/ci-tuner.yml?query=event%3Apush)

The Tuner sub-project assists with tuning program performance by searching for
optimal parameter configurations to use during model compilation.

> [!WARNING]
> SHARK Tuner is still in early development. Interested users may want
> to try it out, but the tuner is not ready for general use yet. Check out
> [the readme](sharktuner/README.md) for more details.

## Support matrix

<!-- TODO: version requirements for Python, ROCm, Linux, etc.  -->

### Models

Model name | Model recipes | Serving apps | Guide |
---------- | ------------- | ------------ | ----- |
SDXL       | [`sharktank/sharktank/models/punet/`](https://github.com/nod-ai/shark-ai/tree/main/sharktank/sharktank/models/punet) | [`shortfin/python/shortfin_apps/sd/`](https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps/sd) | [shortfin/python/shortfin_apps/sd/README.md](shortfin/python/shortfin_apps/sd/README.md)
llama      | [`sharktank/sharktank/models/llama/`](https://github.com/nod-ai/shark-ai/tree/main/sharktank/sharktank/models/llama) | [`shortfin/python/shortfin_apps/llm/`](https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps/llm) | [docs/shortfin/llm/user/llama_serving.md](docs/shortfin/llm/user/llama_serving.md)
Flux       | [`sharktank/sharktank/models/flux/`](https://github.com/nod-ai/shark-ai/tree/main/sharktank/sharktank/models/flux) | [`shortfin/python/shortfin_apps/flux/`](https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps/flux) | [`shortfin/python/shortfin_apps/flux/README.md`](https://github.com/nod-ai/shark-ai/blob/main/shortfin/python/shortfin_apps/flux/README.md)

## SHARK Developers

If you're looking to develop SHARK, check out our [Developer Guide](docs/developer_guide.md).

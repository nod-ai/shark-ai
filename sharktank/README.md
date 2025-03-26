# SHARK Tank

**WARNING: This is an early preview that is in progress. It is not ready for
general use.**

Light weight inference optimized layers and models for popular genai
applications.

This sub-project is a work in progress. It is intended to be a repository of
layers, model recipes, and conversion tools from popular LLM quantization
tooling.

## Project Status

[![CI - sharktank nightly](https://github.com/nod-ai/shark-ai/actions/workflows/ci-sharktank-nightly.yml/badge.svg?branch=main)](https://github.com/nod-ai/shark-ai/actions/workflows/ci-sharktank-nightly.yml)

## Examples

The repository will ultimately grow a curated set of models and tools for
constructing them, but for the moment, it largely contains some CLI exmaples.
These are all under active development and should not yet be expected to work.


### Perform batched inference in PyTorch on a paged llama derived LLM:

Note: Use `--device='cuda:0'` to run this inference on an AMD GPU.

```shell
python -m sharktank.examples.paged_llm_v1 \
  --hf-dataset=open_llama_3b_v2_f16_gguf \
  "Prompt 1" \
  "Prompt 2" ...
```

### Export an IREE compilable batched LLM for serving:

```shell
python -m sharktank.examples.export_paged_llm_v1 \
  --hf-dataset=open_llama_3b_v2_f16_gguf \
  --output-mlir=/tmp/open_llama_3b_v2_f16.mlir \
  --output-config=/tmp/open_llama_3b_v2_f16.json
```

### Dump parsed information about a model from a gguf file:

```shell
python -m sharktank.tools.dump_gguf --hf-dataset=open_llama_3b_v2_f16_gguf
```

## Package Python Release Builds

* To build wheels for Linux:

    ```bash
    ./build_tools/build_linux_package.sh
    ```

    That should produce
    `build_tools/wheelhouse/sharktank-{X.Y.Z}.dev0-py3-none-any.whl`, which can
    then be installed with

    ```bash
    python3 -m pip install build_tools/wheelhouse/sharktank-{X.Y.Z}.dev0-py3-none-any.whl
    ```

* To build a wheel for your host OS/arch manually:

    ```bash
    # Build sharktank.*.whl into the dist/ directory
    #   e.g. `sharktank-3.0.0.dev0-py3-none-any.whl`
    python3 -m pip wheel -v -w dist .

    # Install the built wheel.
    python3 -m pip install dist/*.whl
    ```

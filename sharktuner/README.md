# SHARK Tuner
`libtuner.py` is the core Python script that provides the fundamental functions
for the tuning loop. It imports `candidate_gen.py` for candidate generation. To
implement the full tuning loop, `libtuner.py` requires a separate Python script
that uses the provided `TuningClient` API from `libtuner.py`.

## Prerequisites
### [Optional] Using virtual environments:

```shell
cd sharktuner
python -m venv .venv
source .venv/bin/activate
```

### Install python dependencies:

Development:
```bash
pip install -r requirements-dev.txt
pip install -r requirements-test.txt
```

### IREE's Python bindings setup:

#### Using the local IREE's Python bindings:
   - Building with CMake:
      ```bash
      # Configure (also including other options)
      cmake -G Ninja -B ../iree-build/ \
         -DIREE_BUILD_PYTHON_BINDINGS=ON  \
         -DPython3_EXECUTABLE="$(which python3)" \
      .

      # Build
      cmake --build ../iree-build/
      ```

> [!IMPORTANT]
> Make sure to enable the ROCM and HIP in your cmake configuration.
> See [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings) for the details.

   - Extend `PYTHONPATH` with IREE's `bindings/python` paths
      ```shell
      source ../iree-build/.env && export PYTHONPATH
      export PATH="$(realpath ../iree-build/tools):$PATH"
      ```

  For more information, refer to the [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings).

#### Using nightly IREE's Python bindings:

```shell
pip install --upgrade -r ../requirements-iree-unpinned.txt
```

## Examples

For a concrete example, check the [`model_tuner` directory](./model_tuner/) for a sample tuner implemented with `libtuner`.
The [`dispatch example`](model_tuner/README.md) should be a good starting point for most users.

# IREE dispatch auto-tuning scripts
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

```shell
pip install -r requirements-tuner.txt
pip install -r requirements-dev.txt
```

### IREE's Python bindings setup:

#### Using the local IREE's Python bindings:
   - Building with CMake

     Configure

     ```shell
     cmake -G Ninja -B ../iree-build/ -S . \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_C_COMPILER=clang \
     -DCMAKE_CXX_COMPILER=clang++ \
     -DIREE_HAL_DRIVER_HIP=ON -DIREE_TARGET_BACKEND_ROCM=ON \
     -DIREE_BUILD_PYTHON_BINDINGS=ON \
     -DPython3_EXECUTABLE="$(which python3)"
     ```

     Build

     ```shell
     cmake --build ../iree-build/
     ```

     [!IMPORTANT]
     Make sure to enable the ROCM and HIP in your cmake configuration.
     See [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings) for the details.

   - Set environment

      ```shell
      source ../iree-build/.env && export PYTHONPATH
      export PATH="$(realpath ../iree-build/tools):$PATH"
      ```

  For more information, refer to the [IREE documentation](https://iree.dev/building-from-source/getting-started/#python-bindings).


#### Using nightly IREE's Python bindings:

```shell
pip install -r ../requirements-iree-unpinned.txt
```

## Examples

Check the `model_tuner` directory for a sample tuner implemented with `libtuner`.
The [`dispatch example`](model_tuner)
should be a good starting point for most users.

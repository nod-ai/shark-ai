# IREE dispatch auto-tuning scripts
`libtuner.py` is the core Python script that provides the fundamental functions
for the tuning loop. It imports `candidate_gen.py` for candidate generation. To
implement the full tuning loop, `libtuner.py` requires a separate Python script
that uses the provided `TuningClient` API from `libtuner.py`.

## Prerequisites
[Optional] Using virtual environments:
```shell
cd tuner
python -m venv .venv
source .venv/bin/activate
```
Install python dependencies:
```shell
pip install -r requirements-tuner.txt
pip install -r requirements-dev.txt
```
Using the IREE's Python bindings:
   - Building with CMake
     ```shell
     -DIREE_BUILD_PYTHON_BINDINGS=ON \
     -DPython3_EXECUTABLE="$(which python)"
     ```
   - Set environment
      ```shell
      source ../iree-build/.env && export PYTHONPATH
      export PATH="$(realpath ../iree-build/tools):$PATH"
      ```
For more information, refer to the [IREE
documentation](https://iree.dev/building-from-source/getting-started/#python-bindings).

## Examples

Check the `examples` directory for sample tuners implemented with `libtuner`.
The [`dispatch example`](https://github.com/nod-ai/shark-ai/tree/main/tuner/examples/simple)
should be a good starting point for most users.

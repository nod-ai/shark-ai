# shortfin - SHARK inference library and serving engine

The shortfin project is SHARK's open source, high performance inference library
and serving engine. Shortfin consists of these major components:

* The "libshortfin" inference library written in C/C++ and built on
  [IREE](https://github.com/iree-org/iree)
* Python bindings for the underlying inference library
* Example applications in
  ['shortfin_apps'](https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps)
  built using the python bindings

## Prerequisites

* Python 3.11+

## RDNA4 Demo

In this technology preview demo, we will set up Stable Diffusion XL (SDXL) running locally on
RDNA4 (AMD Radeon RX 9070 XT and 9070) and RDNA3 (AMD Radeon 7900 XTX and Radeon Pro W7900).
SDXL is a popular text to image ML model.

Our demo benefits from improved AI accelerator hardware introduced in RDNA4 that improves
performance on data types already supported by RDNA3 like fp16, and enabled support for a new data
type: fp8 (OCP).

### Installation

#### Prerequisites

* Ubuntu 24.04 or 22.04 (not tested on other systems)
* Python 3.11 or 3.12
* RDNA4 (gfx1201) or RDNA3 (gfx1100) AMD GPU

Create a new directory for the demo:

```shell
mkdir demo
cd demo
```

#### Install ROCm Community Edition

Simply download the matching tarball, extract it, and add it to your environment variables:

```shell
mkdir rocm
cd rocm
# Check GitHub releases for other distributions.
wget https://therock-artifacts.s3.us-east-2.amazonaws.com/therock-dist-gfx1201-20250305.tar.gz
tar -xzf therock-dist-gfx1201-20250305.tar.gz
export PATH="$PWD/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/lib:$LD_LIBRARY_PATH"
cd ..
```

You can find other ROCm Community releases in the [TheRock repository](https://github.com/ROCm/TheRock).

Confirm that your GPU is detected:
```console
~/demo
➜  which rocm-smi
/home/jakub/demo/rocm/bin/rocm-smi
~/demo
➜  rocm-smi
======================================== ROCm System Management Interface ========================================
================================================== Concise Info ==================================================
Device  Node  IDs              Temp    Power  Partitions          SCLK  MCLK   Fan    Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)                                                 
==================================================================================================================
0       2     0x7550,   37870  32.0°C  3.0W   N/A, N/A, 0         0Mhz  96Mhz  0%     auto  0.0W    5%     0%    
1       1     0x7448,   7019   31.0°C  7.0W   N/A, N/A, 0         0Mhz  96Mhz  20.0%  auto  241.0W  0%     0%    
==================================================================================================================
============================================== End of ROCm SMI Log ===============================================
```

Above, `rocm-smi` lists two GPUs: a Radeon RX 9070 and a Radeon Pro W7900.

#### Install SHARK AI and Shortfin

First, create and activate a Python Virtual Environment:

```shell
python -m venv venv
source venv/bin/activate
```

Clone the shark-ai repository and install Shortfin and its dependencies:

```shell
git clone https://github.com/nod-ai/shark-ai
cd shark-ai
git switch shared/rdna4
```

Install pip requirements:
```shell
pip install -r requirements-iree-pinned.txt
pip install -r pytorch-cpu-requirements.txt
pip install -r requirements.txt

cd shortfin
pip install -e .
```

### Start Shortfin and run SDXL

Start the Shortfin server with the correct target (`gfx1100` for RDNA3, `gfx1201` for RDNA4).
You can override the network port used using the `--port <PORT-NUM>` flag.

#### FP8: RDNA4 only

Note that the first run will download all the artifacts necessary (the model code and the weights).
This may take a while. The subsequent runs will use the artifacts cached in `~/.cache/shark/genfiles/sdxl`.

```shell
python -m python.shortfin_apps.sd.server --device=amdgpu --target=gfx1201 --build_preference=precompiled \
  --device=hip --device_ids 0 --model_config=sdxl_config_fp8.json
```

You should see the server running:

```console
[2025-03-05 21:05:00] Application startup complete.
[2025-03-05 21:05:00] Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Open another terminal and start the client in the interactive mode:

```shell
cd demo
source venv/bin/activate
cd shark-ai/shortfin

python -m python.shortfin_apps.sd.simple_client --interactive
```

The client will ask you for the input prompt and save the generated image:

```console
➜  python -m python.shortfin_apps.sd.simple_client --interactive
Waiting for server.
Successfully connected to server.
Enter a prompt:  Shark jumping out of water at sunset. Vaporwave style.
Sending request with prompt:  ['Shark jumping out of water at sunset. Vaporwave style.']
Sending request batch # 0
Saving response as image...
Saved to gen_imgs/shortfin_sd_output_2025-03-05_21-14-23_0.png
Responses processed.
```

While the server will print the total inference time to generate the image:

```console
[2025-03-05 21:14:08] 127.0.0.1:40956 - "GET /health HTTP/1.1" 200
[2025-03-05 21:14:19.545] [info] [metrics.py:51] Completed denoise (UNet) single step average (batch size 1) in 0ms
[2025-03-05 21:14:23.752] [info] [metrics.py:51] Completed inference process in 4209ms
[2025-03-05 21:14:23] 127.0.0.1:57240 - "POST /generate HTTP/1.1" 200
```

![The generated image of a shark](./sample_image_shark.png)

The end-to-end image generation time is around 3.4 seconds on RX 9070 XT and 3.7 seconds on RX 9070.

You can exit the server and the client by pressing `Ctrl + C`.


#### Int8: Both RDNA3 and RDNA4

Use the following command to start the server:

```shell
python -m python.shortfin_apps.sd.server --device=amdgpu --target=gfx1201 --build_preference=precompiled \
  --device=hip --device_ids 0 --model_config=sdxl_config_i8.json
```

Use `--target=gfx1100` when running on RDNA3.
Open a new terminal and follow the steps from the section above to run the client.

End-to-end generation time:
* On RDNA4: around 3.9 seconds on RX 9070 XT and 4.2 seconds on RX 9070.
* On RDNA3: around 8.1 seconds on RX 7900 XTX and 8.9 seconds on Pro W9700.

The RX 9070 XT card about twice as fast as RX 7900 XTX while RX 9070 is about twice as fast as Pro W7900!

### Preliminary performance results

In addition to the total image generation time from the section above, we benchamrked a
portion of the SDXL model called 'UNet', comparing the fp16 and fp8 implementation across
RDNA3 and RDNA4. 'UNet' is typically executed 20 times when generating an image with SDXL.
This was done using the
[sdxl-scripts repository](https://github.com/nod-ai/sdxl-scripts/tree/shared/rdna4).

GPU Name | fp16 time [ms] | fp8 time [ms]
-- | -- | --
RX 9070 XT | 217 | 140
RX 9070 | 263 | 151
RX 7900 XTX | 292 |  N/A
Pro W7900 | 318 |  N/A

> [!NOTE]
> Disclaimer: The results above are for information purpose only. The evaluation was performed
> on engineering sample hardware and may differ from retail parts.

On RDNA4, UNet compiled with fp8 data types is about 50-75% faster than fp16. Despite having fewer
Compute Units than 7900-series RDNA3 cards, 9070 and 9070 XT are noticeably faster with fp16, and
almost twice as fast with fp8.

## Simple user installation

Install the latest stable version:

```bash
pip install shortfin
```

## Developer guides

### Quick start: install local packages and run tests

After cloning this repository, from the `shortfin/` directory:

```bash
pip install -e .
```

Install test requirements:

```bash
pip install -r requirements-tests.txt
```

Run tests:

```bash
pytest -s tests/
```

### Simple dev setup

We recommend this development setup for core contributors:

1. Check out this repository as a sibling to [IREE](https://github.com/iree-org/iree)
   if you already have an IREE source checkout. Otherwise, a pinned version will
   be downloaded for you
2. Ensure that `python --version` reads 3.11 or higher (3.12 preferred).
3. Run `./dev_me.py` to build and install the `shortfin` Python package with both
   a tracing-enabled and default build. Run it again to do an incremental build
   and delete the `build/` directory to start over
4. Run tests with `python -m pytest -s tests/`
5. Test optional features:
   * `pip install iree-base-compiler` to run a small suite of model tests intended
     to exercise the runtime (or use a [source build of IREE](https://iree.dev/building-from-source/getting-started/#using-the-python-bindings)).
   * `pip install onnx` to run some more model tests that depend on downloading
     ONNX models
   * Run tests on devices other than the CPU with flags like:
     `--system amdgpu --compile-flags="--iree-hal-target-backends=rocm --iree-hip-target=gfx1100"`
   * Use the tracy instrumented runtime to collect execution traces:
     `export SHORTFIN_PY_RUNTIME=tracy`

Refer to the advanced build options below for other scenarios.

### Advanced build options

1. Native C++ build
2. Local Python release build
3. Package Python release build
4. Python dev build

Prerequisites

* A modern C/C++ compiler, such as clang 18 or gcc 12
* A modern Python, such as Python 3.12

#### Native C++ builds

```bash
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
cmake --build build --target all
```

If Python bindings are enabled in this mode (`-DSHORTFIN_BUILD_PYTHON_BINDINGS=ON`),
then `pip install -e build/` will install from the build dir (and support
build/continue).

#### Package Python release builds

* To build wheels for Linux using a manylinux Docker container:

    ```bash
    sudo ./build_tools/build_linux_package.sh
    ```

* To build a wheel for your host OS/arch manually:

    ```bash
    # Build shortfin.*.whl into the dist/ directory
    #   e.g. `shortfin-0.9-cp312-cp312-linux_x86_64.whl`
    python3 -m pip wheel -v -w dist .

    # Install the built wheel.
    python3 -m pip install dist/*.whl
    ```

#### Python dev builds

```bash
# Install build system pre-reqs (since we are building in dev mode, this
# is not done for us). See source of truth in pyproject.toml:
pip install setuptools wheel

# Optionally install cmake and ninja if you don't have them or need a newer
# version. If doing heavy development in Python, it is strongly recommended
# to install these natively on your system as it will make it easier to
# switch Python interpreters and build options (and the launcher in debug/asan
# builds of Python is much slower). Note CMakeLists.txt for minimum CMake
# version, which is usually quite recent.
pip install cmake ninja

SHORTFIN_DEV_MODE=ON pip install --no-build-isolation -v -e .
```

Note that the `--no-build-isolation` flag is useful in development setups
because it does not create an intermediate venv that will keep later
invocations of cmake/ninja from working at the command line. If just doing
a one-shot build, it can be ommitted.

Once built the first time, `cmake`, `ninja`, and `ctest` commands can be run
directly from `build/cmake` and changes will apply directly to the next
process launch.

Several optional environment variables can be used with setup.py:

* `SHORTFIN_CMAKE_BUILD_TYPE=Debug` : Sets the CMAKE_BUILD_TYPE. Defaults to
  `Debug` for dev mode and `Release` otherwise.
* `SHORTFIN_ENABLE_ASAN=ON` : Enables an ASAN build. Requires a Python runtime
  setup that is ASAN clean (either by env vars to preload libraries or set
  suppressions or a dev build of Python with ASAN enabled).
* `SHORTFIN_IREE_SOURCE_DIR=$(pwd)/../../iree`
* `SHORTFIN_RUN_CTESTS=ON` : Runs `ctest` as part of the build. Useful for CI
  as it uses the version of ctest installed in the pip venv.

### Running tests

The project uses a combination of ctest for native C++ tests and pytest. Much
of the functionality is only tested via the Python tests, using the
`_shortfin.lib` internal implementation directly. In order to run these tests,
you must have installed the Python package as per the above steps.

Which style of test is used is pragmatic and geared at achieving good test
coverage with a minimum of duplication. Since it is often much more expensive
to build native tests of complicated flows, many things are only tested via
Python. This does not preclude having other language bindings later, but it
does mean that the C++ core of the library must always be built with the
Python bindings to test the most behavior. Given the target of the project,
this is not considered to be a significant issue.

#### Python tests

Run platform independent tests only:

```bash
pytest tests/
```

Run tests including for a specific platform (in this example, a gfx1100 AMDGPU):

(note that not all tests are system aware yet and some may only run on the CPU)

```bash
pytest tests/ --system amdgpu \
    --compile-flags="--iree-hal-target-backends=rocm --iree-hip-target=gfx1100"
```

## Production library building

In order to build a production library, additional build steps are typically
recommended:

* Compile all deps with the same compiler/linker for LTO compatibility
* Provide library dependencies manually and compile them with LTO
* Compile dependencies with `-fvisibility=hidden`
* Enable LTO builds of libshortfin
* Set flags to enable symbol versioning

## Miscellaneous build topics

### Free-threaded Python

Support for free-threaded Python builds (aka. "nogil") is in progress. It
is currently being tested via CPython 3.13 with the `--disable-gil` option set.
There are multiple ways to acquire such an environment:

* Generally, see the documentation at
  <https://py-free-threading.github.io/installing_cpython/>
* If using `pyenv`:

    ```bash
    # Install a free-threaded 3.13 version.
    pyenv install 3.13t

    # Test (should print "False").
    pyenv shell 3.13t
    python -c 'import sys; print(sys._is_gil_enabled())'
    ```

# Fusilli

Fusilli is a C++ Graph API and Frontend to the IREE compiler & runtime for JIT compilation & execution of training and inference graphs. It exposes cuDNN-like primitives backed by IREE kernel code-generation.

:construction: **This project is under active development and APIs may change** :construction:

A side note on naming: 'amdsharkFuser' is the name of the project (may change as the project evolves), but the library itself is called 'Fusilli'. Both are inspired from the term 'Fusions' - a bread-and-butter compiler optimization to improve performance.

![Fusilli](docs/fusilli.png)

## Developer Guide

### Setup

Although optional, we recommend docker as the canonical development setup for a no-fuss quick start, hermetic and reproducible builds, and consistency with CI. Follow [these steps](https://github.com/sjain-stanford/docker.git) to launch an interactive docker container with the required dependencies pre-installed (and skip to the `Build and Test` section below).

If you prefer a custom setup instead, the following dependencies need to be brought in to build/test Fusilli:

**Build Requirements:** cmake, ninja-build, clang, IREE

**Test Requirements:** catch2, lit, FileCheck, iree-opt, iree-compile

Fusilli interfaces with the IREE compiler through the CLI and with IREE runtime through its C-API. In the future we may want an alternate C-API integration for the compiler as well but for now running it as a tool with process isolation is useful for general developer ergonomics. The IREE compiler is a heavy dependency to build (due to MLIR/LLVM), so we recommend using a prebuilt release either from a python nightly package or shared library distribution. The IREE runtime on the other hand is much more lightweight and is designed to be built from source and statically linked in. IREE does not export a shared runtime library to allow for maximum flexibility with low-level and toolchain specific (LTO style) optimizations.

Easiest way to get [`lit`](https://llvm.org/docs/CommandGuide/lit.html), and the `iree-*` CLI tools is through `pip install`. [`FileCheck`](https://llvm.org/docs/CommandGuide/FileCheck.html) comes packaged with clang / llvm distributions. Everything else should be available via `apt` based install.

### Build and Test

Build and test Fusilli as follows:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_BUILD_TYPE=<Debug|Release|RelWithDebInfo> \
    -DIREERuntime_DIR=</path/to/iree/build/lib/cmake/IREE>
cmake --build build --target all
ctest --test-dir build
```

To skip building tests and samples, specify the cmake flag `-DFUSILLI_BUILD_TESTS=OFF`. When building on a CPU-only system, specify `-DFUSILLI_SYSTEMS_AMDGPU=OFF` to disable the AMDGPU build.

To run clang-tidy during compilation, specify the cmake flag `-DFUSILLI_ENABLE_CLANG_TIDY=ON`.

To re-run failed tests verbosely:
```shell
ctest --test-dir build --rerun-failed --output-on-failure --verbose
```

To run tests in parallel (concurrently):
```shell
ctest --test-dir build --output-on-failure -j $(nproc)
```

Tests and samples are also built as standalone binary targets (in the `build/bin` directory) to make debugging isolated failures easier.

### Benchmarks

The easiest way to benchmark on AMD GPU systems is using the `rocprofv3` tool (included in the docker image). Here's a sample command to dump a `*.pftrace` file that may be opened using [Perfetto](https://ui.perfetto.dev/) for further analysis.

```shell
rocprofv3 --output-format pftrace -r --  build/bin/benchmarks/fusilli_benchmark_driver <ARGS>
```

For example:
```shell
rocprofv3 --output-format pftrace -r --  build/bin/benchmarks/fusilli_benchmark_driver --iter 10 conv --bf16 -n 16 -c 288 --in_d 2 -H 48 -W 32 -k 288 --fil_d 2 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 2 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --in_layout "NDHWC" --out_layout "NDHWC" --fil_layout "NDHWC" --spatial_dim 3
```

To skip building benchmarks, specify the cmake flag `-DFUSILLI_BUILD_BENCHMARKS=OFF`.

### Code Coverage (using gcov + lcov)

This works with gcc builds (code coverage with clang instrumentation is future work).

To generate code coverage metrics:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DFUSILLI_CODE_COVERAGE=ON \
    -DIREERuntime_DIR=</path/to/iree/build/lib/cmake/IREE>
cmake --build build --target all
ctest --test-dir build -T test -T coverage
```

This generates the `*.gcda` and `*.gcno` files with coverage info. At this point one may use an IDE to visualize the coverage info inlayed with the source code. If using VSCode's gcov-viewer extension: Hit `Cmd+Shift+P` -> Gcov Viewer: Reload (Import gcda files) to load coverage info and `Cmd+Shift+P` -> Gcov Viewer: Reset (Delete gcda files) to reset it.

To generate an HTML (interactive) coverage report:
```shell
lcov --capture --directory build --output-file build/coverage.info
# Exclude external sources from being reported in code coverage
# For example:
#   /usr/include/c++/13/*
#   /usr/include/x86_64-linux-gnu/c++/*
#   /usr/local/include/catch2/*
lcov --remove build/coverage.info '/usr/*' '*/iree/*' --output-file build/coverage.info
genhtml build/coverage.info --output-directory coverage_report
```

### Lint

Run clang-format:
```shell
find . -path ./build -prune -o \( -type f \( -name "*.cpp" -o -name "*.h" \) -print \) | xargs clang-format -i
```

Alternatively, run pre-commit which runs clang-format along with a few other lint checks:
```shell
pre-commit run --all-files
```

### Logging

Fusilli records execution flow through the logging interface. This is disabled by default but can be enabled for debugging.

To configure logging behavior using environment variables:

|   Set output stream \ Enable logging            | `FUSILLI_LOG_INFO` = 0 | `FUSILLI_LOG_INFO` = 1
| ----------------------------------------------- | ---------------------- | ----------------------
| `FUSILLI_LOG_FILE` not set                      | no logging             | no logging
| `FUSILLI_LOG_FILE` set to `stdout` or `stderr`  | no logging             | logging to cout / cerr
| `FUSILLI_LOG_FILE` set to `/path/to/file.txt`   | no logging             | logging to file.txt

Tests and samples that are built with the cmake flag `-DFUSILLI_ENABLE_LOGGING=ON` have their env variables automatically configured for logging to cout.

Alternatively, one may call the logging API directly as needed:
- Calling `fusilli::isLoggingEnabled() = <true|false>` has the same effect as setting `FUSILLI_LOG_INFO = 1|0`.
- Calling `fusilli::getStream() = <stream_name>` has the same effect as setting the output stream using `FUSILLI_LOG_FILE`.

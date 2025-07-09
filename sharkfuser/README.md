# SharkFuser (aka Fusili)

Fusili is a C++ Graph API and Frontend to expose cuDNN-like primitives backed by IREE code-generated kernels.

![Fusili](docs/fusili.png)


## Developer Guide:

Build and test:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
cmake --build build --target all
ctest --test-dir build
```

Code coverage:
```shell
ctest --test-dir build -T test -T coverage
```

Re-run failed tests verbosely:
```shell
ctest --test-dir build --rerun-failed --output-on-failure
```

Run clang-format:
```shell
find . -path ./build -prune -o \( -type f \( -name "*.cpp" -o -name "*.h" \) -print \) | xargs clang-format -i
```

Debugging:
Enable logging (TODO)

Create a fusili graph:
TODO



Project Roadmap:
- [x] Build/test infra, logging
- [ ] Graph, tensor, node datastructures and builder API
- [ ] conv_fprop MLIR ASM emitter
- [ ] IREE compiler integration
- [ ] IREE runtime integration
- [ ] `g->execute()` (calls IREE compiler/runtime C API)
- [ ] conv_fprop integration testing
- [ ] Kernel cache
- [ ] Elementwise ops (relu?)
- [ ] Op fusion templates
- [ ] Python bindings
- [ ] Shape inference for static dims
- [ ] Serialization
- [ ] hipDNN integration

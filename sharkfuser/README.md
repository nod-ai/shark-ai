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

Re-run failed tests verbosely:
```shell
ctest --test-dir build --rerun-failed --output-on-failure
```

Run clang-format:
```shell
find . -path ./build -prune -o \( -type f \( -name "*.cpp" -o -name "*.h" \) -print \) | xargs clang-format -i
```

Enable logging:



Project Roadmap:
- [x] Build files, source code layout
- [ ] `fusili::Graph` datastructures (tensor, node, graph, types)
- [ ] MLIR ASM emitter
- [ ] IREE build integration (enable building compiler/runtime deps)
- [ ] `g->execute()` (calls IREE compiler C API then IREE runtime C API)
- [ ] Kernel Cache (skip re-compilations upon cache hit)
- [ ] Op Fusion Templates
- [ ] Serialization?

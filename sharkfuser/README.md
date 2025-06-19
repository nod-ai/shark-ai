# SharkFuser (aka Fusili)

Fusili is a C++ Graph API and Frontend to expose cuDNN-like primitives backed by IREE code-generated kernels.

Proposed features (TBD): JIT compilation with caching, fusions

Project Roadmap:
1. MLIR ASM emitter
2. IREE compiler JIT (c API)
3. IREE runtime (c API)
4. Caching
5. Fusions

To build and test:
```shell
cmake -GNinja -S. -Bbuild \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_LINKER_TYPE=LLD
cmake --build build --target all
ctest --test-dir build
```

To run clang-format:
```shell
find . -path ./build -prune -o \( -type f \( -name "*.cpp" -o -name "*.h" \) -print \) | xargs clang-format -i
```

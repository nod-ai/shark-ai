# How to run Llama 3.1 Benchmarking Tests
In order to run Llama 3.1 8B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    -m "expensive" \
    --run-quick-test \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0
```

In order to filter by test, use the -k option. If you
wanted to only run the Llama 3.1 70B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    -m "expensive" \
    --run-nightly-llama-tests \
    -k 'testBenchmark70B_f16_TP8_Decomposed' \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0
```

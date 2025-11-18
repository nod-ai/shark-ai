# How to run Llama 3.1 Benchmarking Tests
In order to run Llama 3.1 8B F16 Decomposed test:
```
pytest amdsharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    --run-quick-test \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0 \
    --llama3-8b-f16-model-path="/amdshark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa"
```

In order to filter by test, use the -k option. If you
wanted to only run the Llama 3.1 70B F16 Decomposed test:
```
pytest amdsharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    -m "expensive" \
    --run-nightly-test \
    -k 'testBenchmark70B_f16_TP8_Decomposed' \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0 \
    --llama3-70b-f16-tp8-model-path="/amdshark-dev/70b/instruct/weights/tp8/llama3.1_70b_instruct_fp16_tp8.irpa"
```

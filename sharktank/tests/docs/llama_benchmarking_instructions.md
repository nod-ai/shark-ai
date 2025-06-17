# How to run Llama 3.1 Benchmarking Tests
In order to run Llama 3.1 8B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    --run-quick-test \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0 \
    --llama3-8b-f16-model-path="/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa" \
    --llama3-8b-f8-model-path="/shark-dev/8b/fp8/native_fp8_e4m3fnuz_llama3_8b.irpa" \
    --llama3-8b-f8-attnf8-model-path="/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa" \
    --llama3-70b-f16-model-path="/shark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa" \
    --llama3-70b-f8-model-path="/shark-dev/70b/fp8/llama70b_fp8.irpa" \
    --llama3-405b-f16-model-path="/shark-dev/data/llama3.1/weights/405b/fp16/llama3.1_405b_fp16.irpa" \
    --llama3-405b-f8-model-path="/shark-dev/405b/f8/llama3.1_405b_fp8.irpa"
```

In order to filter by test, use the -k option. If you
wanted to only run the Llama 3.1 70B F16 Decomposed test:
```
pytest sharktank/tests/models/llama/benchmark_amdgpu_test.py \
    -v -s \
    -m "expensive" \
    --run-nightly-tests \
    -k 'testBenchmark70B_f16_TP8_Decomposed' \
    --iree-hip-target=gfx942 \
    --iree-device=hip://0 \
    --llama3-8b-f16-model-path="/shark-dev/8b/instruct/weights/llama3.1_8b_instruct_fp16.irpa" \
    --llama3-8b-f8-model-path="/shark-dev/8b/fp8/native_fp8_e4m3fnuz_llama3_8b.irpa" \
    --llama3-8b-f8-attnf8-model-path="/shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa" \
    --llama3-70b-f16-model-path="/shark-dev/70b/instruct/weights/llama3.1_70b_instruct_fp16.irpa" \
    --llama3-70b-f8-model-path="/shark-dev/70b/fp8/llama70b_fp8.irpa" \
    --llama3-405b-f16-model-path="/shark-dev/data/llama3.1/weights/405b/fp16/llama3.1_405b_fp16.irpa" \
    --llama3-405b-f8-model-path="/shark-dev/405b/f8/llama3.1_405b_fp8.irpa"
```

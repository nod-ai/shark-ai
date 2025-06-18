# cd shark-ai and setup python env
./scripts/setenv.sh --nightly

# run llama-8B-FP8
./scripts/export_and_compile.sh --irpa /shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa --bs-prefill 4 --bs-decode 4 --dtype fp8

./scripts/run_iree_benchmark.sh --parameters /shark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa --model llama-8B-FP8 --bs-prefill 4 --bs-decode 4

cd shortfin
../scripts/run_offline_serving.sh --mode chat
cd ..





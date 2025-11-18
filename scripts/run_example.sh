# # cd amdshark-ai and setup python env
# ./scripts/setenv.sh --nightly

# # run llama-8B-FP8
./scripts/export_and_compile.sh --irpa /amdshark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa --bs-prefill 4 --bs-decode 4 --dtype fp8
./scripts/run_iree_benchmark.sh --parameters /amdshark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa --model llama-8B-FP8 --bs-prefill 4 --bs-decode 4 | tee $(pwd)/output_artifacts/iree_benchnark.log

cd shortfin
bash ../scripts/run_offline_serving.sh \
  --irpa /amdshark-dev/8b/fp8/attnf8/native_fp8_e4m3fnuz_llama3_8b.irpa \
  --tokenizer_json /amdshark-dev/8b/instruct/tokenizer.json \
  --vmfb ../output_artifacts/output.vmfb \
  --model_config ../output_artifacts/config_attn.json \
  --mode chat | tee ../output_artifacts/offline_serving_chat.log

# run llama-70B-FP16-tp8
./scripts/export_and_compile.sh --irpa /amdshark-dev/70b/instruct/weights/tp8/llama3_70b_instruct_fp16_tp8.irpa --bs-prefill 4 --bs-decode 4 --tensor-parallelism-size 8
./scripts/run_iree_benchmark.sh --parameters /amdshark-dev/70b/instruct/weights/tp8/llama3_70b_instruct_fp16_tp8.irpa --model llama-70B-FP16-tp8 --bs-prefill 4 --bs-decode 4

# cd shortfin
# ../scripts/run_offline_serving.sh --mode chat
# cd ..

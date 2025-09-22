import pytest
import subprocess
import os
from pathlib import Path
import json
from pathlib import Path

OUTPUT_DIR = Path(os.getcwd()) / "output_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)
CONFIGS_DIR = Path(__file__).parent/"configs"

def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default="llama-8b-fp8",
        help="Model name (e.g. llama-70b-fp16, llama-70b-fp8, llama-8b-fp16, llama-8b-fp8, mistral)",
    )

############# Run The Command and Generate Logs ##############
def run_cmd(cmd, log_file):
    log_path = OUTPUT_DIR / log_file
    with open(log_path, "w") as f:
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in process.stdout:
            decoded = line.decode()
            f.write(decoded)
            print(decoded, end="")
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
    return log_path


########### Return Model Config for a Specific Model ############
@pytest.fixture(scope="session")
def model_config(pytestconfig):
    model_name = pytestconfig.getoption("model")
    config_path = CONFIGS_DIR / f"{model_name}.json"

    if not config_path.exists():
        raise ValueError(f"Unknown Model : {model_name} | Existing Models in Sharktank : llama-70b-fp16, llama-70b-fp8, llama-8b-fp16, llama-8b-fp8, mistral")
    with open(config_path, "r") as f:
        return json.load(f)


############### Export The MLIR Through Sharktank ################
@pytest.fixture(scope="session")
def export_fixture(model_config):

    gen_mlir_path = OUTPUT_DIR / "output.mlir"
    gen_config_path = Path(model_config['output_dir']) / "config_attn.json"


    #check if the mlir already exists
    if os.path.exists(gen_mlir_path) and os.path.exists(gen_config_path):
        print("File exists. Skipping Export... Moving to Compile...")
        return gen_mlir_path
    else:
        print("Continuing With Export...")

    return run_cmd(
        f"python scripts/run_export.py --irpa {model_config['irpa']} "
        f"--attention-kernel {model_config['attention_kernel']} "
        f"--dtype {model_config['dtype']} --bs-prefill {model_config['bs_prefill']} --bs-decode {model_config['bs_decode']} "
        f"--device-block-count {model_config['device_block_count']} "
        f"--extra-export-flags-list '{json.dumps(model_config['extra_export_flags_list'])}' "
        f"--output-dir {OUTPUT_DIR}",
        "export.log"
    )


######## Compile MLIR(Generated from SharkTank) Through IREE #########
@pytest.fixture(scope="session")
#def compile_fixture(export_fixture):
def compile_fixture(export_fixture, model_config):
    gen_vmfb_path = OUTPUT_DIR / "output.vmfb"

    #check if the vmfb already exists
    if os.path.exists(gen_vmfb_path):
        print("File exists. Skipping Compile...")
        return gen_vmfb_path
    else:
        print("Continuing With Continuing...")

    return run_cmd(
        "python scripts/run_compile.py "
        f"--output_dir {model_config['output_dir']} "
        f"--extra-compile-flags-list '{json.dumps(model_config['extra_compile_flags_list'])}' --dtype {model_config['dtype']} ",
        f"--iree-hip-target {model_config['iree_hip_target']}"
        "compilation.log"
    )


############## VMFB Check Through Prompt-Response ###############
@pytest.fixture(scope="session")
def validate_vmfb_fixture(model_config, compile_fixture):
    return run_cmd(
        f"python scripts/validate_numerics.py --irpa {model_config['irpa']} "
        f"--vmfb {OUTPUT_DIR}/output.vmfb "
        f"--config $(pwd)/output_artifacts/config_attn.json "
        f"--tokenizer {model_config['tokenizer']} "
        f"--tokenizer_config {model_config['tokenizer_config']} "
        f"--steps 64 --kv-cache-dtype {model_config['kv_dtype']} ",
        "validate_vmfb.log"
    )

@pytest.fixture(scope="session")
def benchmark_fixture(model_config, compile_fixture):
    benchmarks = str(model_config['benchmarks']).replace("'", '"')
    return run_cmd(
        f"python scripts/run_iree_benchmark.py "
        f"--benchmarks '{benchmarks}' --benchmark_repetition {model_config['benchmark_repetitions']} "
        f"--parameters {model_config['irpa']} --model {model_config['benchmark_model']} --vmfb {OUTPUT_DIR}/output.vmfb && "
        f"python scripts/utils_and_time_check.py --combine-json {OUTPUT_DIR}/benchmark_module "
        f"--output-json {OUTPUT_DIR}/consolidated_benchmark.json --benchmark-model {model_config['benchmark_model']} "
        f"--prefill-gold {model_config['prefill_gold']} --decode-gold {model_config['decode_gold']} --isl 2048 --append-isl",
        "iree_benchmark.log"
    )


################ Shortfin Online Serving Check ################
@pytest.fixture(scope="session")
def online_serving_fixture(model_config, compile_vmfb_fixture):
    # OUTPUT_DIR = Path.cwd().parent / "output_artifacts"
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    return run_cmd(
        f"cd shortfin && python ../scripts/run_online_serving.py "
        f"--irpa {model_config['irpa']} "
        f"--tokenizer_json {model_config['tokenizer']} "
        f"--vmfb {OUTPUT_DIR}/output.vmfb "
        f"--model_config {OUTPUT_DIR}/config_attn.json "
        f"--port 8900",
        "serving.log"
    )

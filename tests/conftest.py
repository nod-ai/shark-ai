import pytest
import subprocess
import os
from pathlib import Path
from configs import MODELS

OUTPUT_DIR = Path(os.getcwd()) / "output_artifacts"
OUTPUT_DIR.mkdir(exist_ok=True)


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

@pytest.fixture(scope="session")
def model_config(pytestconfig):
    model_name = pytestconfig.getoption("model")
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return MODELS[model_name]


@pytest.fixture(scope="session")
def export_fixture(model_config):
    return run_cmd(
        f"python scripts/run_export.py --irpa {model_config['irpa']} "
        f"--dtype {model_config['dtype']} --bs-prefill 4 --bs-decode 4",
        "export.log"
    )


@pytest.fixture(scope="session")
def compile_fixture(export_fixture):
    return run_cmd(
        "python scripts/run_compile.py",
        "compilation.log"
    )


@pytest.fixture(scope="session")
def validate_vmfb_fixture(model_config, compile_fixture):
    return run_cmd(
        f"python scripts/validate_numerics.py --irpa {model_config['irpa']} "
        f"--vmfb $(pwd)/output_artifacts/output.vmfb "
        f"--config $(pwd)/output_artifacts/config_attn.json "
        f"--tokenizer {model_config['tokenizer']} "
        f"--tokenizer_config {model_config['tokenizer_config']} "
        f"--steps 64 --kv-cache-dtype {model_config['kv_dtype']}",
        "validate_vmfb.log"
    )


@pytest.fixture(scope="session")
def benchmark_fixture(model_config, validate_vmfb_fixture):
    return run_cmd(
        f"python scripts/run_iree_benchmark.py --bs-prefill 4 --bs-decode 4 "
        f"--parameters {model_config['irpa']} --model {model_config['benchmark_model']} && "
        f"python scripts/utils.py --combine-json $(pwd)/output_artifacts/benchmark_module "
        f"--output-json $(pwd)/output_artifacts/consolidated_benchmark.json --append-isl",
        "iree_benchmark.log"
    )


@pytest.fixture(scope="session")
def serving_fixture(model_config, validate_vmfb_fixture):
    os.environ["ROCR_VISIBLE_DEVICES"] = "0"
    return run_cmd(
        f"cd shortfin && python ../scripts/run_online_serving.py "
        f"--irpa {model_config['irpa']} "
        f"--tokenizer_json {model_config['tokenizer']} "
        f"--vmfb ../output_artifacts/output.vmfb "
        f"--model_config ../output_artifacts/config_attn.json "
        f"--port 8900",
        "serving.log"
    )

def pytest_addoption(parser):
    parser.addoption("--model", action="store", default="llama-8b-fp8",
                    help="Model name (e.g. llama-70b-fp16, llama-70b-fp8, llama-8b-fp16, llama-8b-fp8, mistral)")

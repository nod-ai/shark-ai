import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output_artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULTS = {
    "irpa": "/shark-dev/llama3.1/405b/fp4/fp4_2025_07_10_fn.irpa",
    "tokenizer": "/shark-dev/llama3.1/405b/fp4/tokenizer.json",
    "tokenizer_config": "/shark-dev/llama3.1/405b/fp4/tokenizer_config.json",
    "vmfb": str(OUTPUT_DIR / "output.vmfb"),
    "config": str(OUTPUT_DIR / "config_attn.json"),
    "model": "llama-405b-fp4",
    "steps": 64,
    "kv_cache_dtype": "float8_e4m3fn",
}
OUTPUT_FILE = OUTPUT_DIR / "numeric_validation.log"

def parse_args():
    parser = argparse.ArgumentParser(description="Numeric validation script")
    parser.add_argument("--irpa", default=DEFAULTS["irpa"])
    parser.add_argument("--vmfb", default=DEFAULTS["vmfb"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--tokenizer", default=DEFAULTS["tokenizer"])
    parser.add_argument("--tokenizer_config", default=DEFAULTS["tokenizer_config"])
    parser.add_argument("--config", default=DEFAULTS["config"])
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--kv-cache-dtype", dest="kv_cache_dtype", default=DEFAULTS["kv_cache_dtype"])
    return parser.parse_args()

PROMPT_RESPONSES = {
    "<|begin_of_text|>Name the capital of the United States.<|eot_id|>":
        "The capital of the United States is Washington, D.C.",

    "Fire is hot. Yes or No ?":
        "Yes",

    """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Hey!! Expect the response to be printed as comma separated values.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Give me the first 10 prime numbers<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>""":
        "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
}

def run_llm_vmfb(prompt, response, steps, args, counter):
    print(f"\nExecuting prompt {counter}")
    cmd = [
        sys.executable, "-m", "sharktank.tools.run_llm_vmfb",
        "--prompt", prompt,
        "--irpa", args.irpa,
        "--vmfb", args.vmfb,
        "--config", args.config,
        "--tokenizer", args.tokenizer,
        "--tokenizer_config", args.tokenizer_config,
        "--steps", str(steps),
        "--kv-cache-dtype", args.kv_cache_dtype,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
    except Exception as e:
        output = str(e)
        return 1

    with open(OUTPUT_FILE, "a") as f:
        f.write("\n=======================================================\n")
        f.write(f"Prompt {counter}:\n{prompt}\n\nResponse:\n{output}\n\n")

    if response in output:
        print(f"Response matches for prompt {counter}")
        return 0
    else:
        print(f"Response did not match for prompt {counter}")
        return 1


def main():
    args = parse_args()
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    result = 0
    counter = 1

    # PROMPT 1
    steps = 20
    result |= run_llm_vmfb(list(PROMPT_RESPONSES.keys())[0],
                           list(PROMPT_RESPONSES.values())[0],
                           steps, args, counter)
    counter += 1

    # PROMPT 2
    steps = 5
    result |= run_llm_vmfb(list(PROMPT_RESPONSES.keys())[1],
                           list(PROMPT_RESPONSES.values())[1],
                           steps, args, counter)
    counter += 1

    # PROMPT 3
    steps = 100
    result |= run_llm_vmfb(list(PROMPT_RESPONSES.keys())[2],
                           list(PROMPT_RESPONSES.values())[2],
                           steps, args, counter)

    sys.exit(result)


if __name__ == "__main__":
    main()

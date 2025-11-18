# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
import requests

import iree.compiler as ireec
import iree.runtime

from amdsharktank.utils.e2e_test_utils import (
    BenchmarkUtils,
    OnlineServingUtils,
    VERY_LARGE,
)

STAGES = ["export", "compile", "validate_vmfb", "benchmark", "online_serving", "all"]
MODEL_CHOICES = [
    "llama-70b-fp16",
    "llama-70b-fp8",
    "llama-8b-fp16",
    "llama-8b-fp8",
    "mistral",
    "gpt-oss-20b-bfp16",
    "llama-405b-fp4-with-topk",
    "llama-405b-fp4-without-topk",
    "llama-405b-fp4-pp2-with-topk",
    "llama-405b-fp4-pp2-without-topk",
    "llama-405b-fp4-pp8-with-topk",
    "llama-405b-fp4-pp8-without-topk",
]


def run_cmd(cmd, OUTPUT_DIR, append=True):
    LOG_FILE = OUTPUT_DIR / "e2e_testing_log_file.log"
    mode = "a" if append else "w"
    with open(LOG_FILE, mode) as f:
        process = subprocess.Popen(
            cmd,
            shell=isinstance(cmd, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for line in process.stdout:
            decoded = line.decode()
            f.write(decoded)
            logging.info(decoded.strip())
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
    return LOG_FILE


def run_stage(
    stage,
    model_name,
    irpa,
    tokenizer,
    tokenizer_config,
    cfg,
    gpu_model,
    OUTPUT_DIR,
):
    print(f"\n Running stage: {stage} for model: {model_name}")
    print(f"    IRPA: {irpa}")
    print(f"    Tokenizer: {tokenizer}")
    print(f"    Tokenizer Config: {tokenizer_config}")

    gen_mlir_path = OUTPUT_DIR / "output.mlir"
    gen_config_path = OUTPUT_DIR / "config_attn.json"
    gen_vmfb_path = OUTPUT_DIR / "output.vmfb"

    LOG_FILE = OUTPUT_DIR / "e2e_testing_log_file.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )

    # device ids are passed in the file amdsharktank/tests/e2e/configs/models.json for each model
    device_ids = cfg.get("device_ids", [])
    if not isinstance(device_ids, list):
        raise ValueError(f"device_ids must be a list, but got {type(device_ids)}")
    if len(device_ids) == 0:
        logging.info(
            "No Device Passed. Pass it in the amdsharktank/tests/e2e/configs/models.json file as a value of the key device_ids."
        )
        sys.exit(1)
    else:
        logging.info("========= Devices IDs Passed =========")
        logging.info(str(device_ids))

    # === Export Stage ===
    if stage in [
        "export",
        "compile",
        "validate_vmfb",
        "benchmark",
        "online_serving",
        "all",
    ]:
        if os.path.exists(gen_mlir_path) and os.path.exists(gen_config_path):
            logging.info("File exists. Skipping Export..")
        else:
            logging.info("Exporting IR Through amdsharktank")

            # if required to add any extra flag for export command please add in amdsharktank/tests/e2e/configs/models.json file for each model extra_export_flags_list.
            export_cmd = [
                sys.executable,
                "-m",
                "amdsharktank.examples.export_paged_llm_v1",
                f"--irpa-file={cfg['irpa']}",
                f"--output-mlir={gen_mlir_path}",
                f"--output-config={gen_config_path}",
                f"--bs-prefill={cfg['bs_prefill']}",
                f"--bs-decode={cfg['bs_decode']}",
                f"--device-block-count={cfg['device_block_count']}",
            ]

            extra_flags = cfg.get("extra_export_flags_list", [])
            if not isinstance(extra_flags, list):
                raise ValueError(
                    f"extra_export_flags_list must be a list, got {type(extra_flags)}"
                )

            if len(extra_flags) == 0:
                logging.info("No Extra Export Flag Passed.")
            else:
                logging.info("Appending Extra Export Flags...")
                logging.info(str(extra_flags))
                export_cmd += extra_flags

            logging.info(
                "=============================================================================== Using Export Command ==============================================================================="
            )
            logging.info("")
            logging.info(f"Using Export Command: {' '.join(export_cmd)}")
            logging.info("")
            logging.info(
                "===================================================================================================================================================================================="
            )
            start = time.time()
            run_cmd(export_cmd, OUTPUT_DIR, append=True)
            logging.info(
                "============================================================================================== Export Done =============================================================================================="
            )
            logging.info(f"Time taken by Export: {int(time.time()-start)} seconds")

    # === Compile Stage ===
    if stage in ["compile", "validate_vmfb", "benchmark", "online_serving", "all"]:
        if os.path.exists(gen_vmfb_path):
            logging.info("File exists. Skipping Compile...")
        else:
            logging.info("Continuing with Compile...")
            logging.info("Compiling IR ....")

            input_file = str(gen_mlir_path)
            output_file = str(gen_vmfb_path)
            # Flags are added below from https://github.com/iree-org/iree/blob/main/tests/external/iree-test-suites/torch_models/llama_8b_fp16/modules/llama_gfx942.json.
            # Flags not present here are added in the amdsharktank/tests/e2e/configs/models.json file for each model in the extra_compile_flags_list.
            # If required to add any extra flag, please add in amdsharktank/tests/e2e/configs/models.json file.
            extra_args = [f"--iree-hal-target-device=hip[{d}]" for d in device_ids]
            extra_flags = cfg.get("extra_compile_flags_list", [])
            if not isinstance(extra_flags, list):
                raise ValueError(
                    f"extra_compile_flags_list must be a list, got {type(extra_flags)}"
                )
            if len(extra_flags) == 0:
                logging.info("No Extra Compile Flag Passed.")
            else:
                logging.info("Appending Extra Compile Flags...")
                logging.info(str(extra_flags))
                extra_args += extra_flags

            print()
            logging.info(
                "=============================================================== Using Compile Command ==============================================================="
            )
            logging.info("")
            logging.info(
                f"Using ireec.compile_file with flags(extra_args): {extra_args}"
            )
            logging.info("")
            logging.info(
                "======================================================================================================================================================"
            )
            print()

            start = time.time()
            ireec.compile_file(
                input_file,
                output_file=output_file,
                # target_backends=["rocm"],
                extra_args=extra_args,
            )
            logging.info(
                f"Time taken for compiling: {int(time.time() - start)} seconds"
            )
            logging.info(
                "============================================================================================== Compile Done =============================================================================================="
            )

    # === Validate Stage ===
    if stage in ["validate_vmfb", "all"]:
        PROMPT_RESPONSES = {
            "<|begin_of_text|>Name the capital of the United States.<|eot_id|>": "The capital of the United States is Washington, D.C.",
            "Fire is hot. Yes or No ?": "Yes",
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Hey!! Expect the response to be printed as comma separated values.<|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            Give me the first 10 prime numbers<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>""": "2, 3, 5, 7, 11, 13, 17, 19, 23, 29",
        }

        result = 0
        counter = 1

        for steps, prompt, response in [
            (20, list(PROMPT_RESPONSES.keys())[0], list(PROMPT_RESPONSES.values())[0]),
            (5, list(PROMPT_RESPONSES.keys())[1], list(PROMPT_RESPONSES.values())[1]),
            (100, list(PROMPT_RESPONSES.keys())[2], list(PROMPT_RESPONSES.values())[2]),
        ]:
            logging.info(f"\nExecuting prompt {counter}")
            cmd = [
                sys.executable,
                "-m",
                "amdsharktank.tools.run_llm_vmfb",
                "--prompt",
                prompt,
                "--irpa",
                irpa,
                "--vmfb",
                gen_vmfb_path,
                "--config",
                gen_config_path,
                "--tokenizer",
                tokenizer,
                "--tokenizer_config",
                tokenizer_config,
                "--steps",
                str(steps),
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                output = proc.stdout + proc.stderr
            except Exception as e:
                output = str(e)

            logging.info("\n=======================================================")
            logging.info(f"Prompt {counter}:\n{prompt}\n\nResponse:\n{output}\n\n")

            if response in output:
                logging.info(f"Response matches for prompt {counter}")
            else:
                logging.info(f"Response did not match for prompt {counter}")
                result |= 1

            counter += 1

        logging.info(
            "============================================================================================== Validate VMFB Done =============================================================================================="
        )

    # === IREE Benchmark ===
    if stage in ["benchmark", "all"]:
        try:
            extra_flags = cfg.get("extra_benchmark_flags_list", [])
            if not isinstance(extra_flags, list):
                raise ValueError(
                    f"Invalid value for --extra-benchmark-flags-list: {cfg['extra_benchmark_flags_list']}"
                )
        except Exception as e:
            raise ValueError(
                f"Invalid value for --extra-benchmark-flags-list: {cfg['extra_benchmark_flags_list']}"
            ) from e

        if not extra_flags:
            logging.info("No Extra Benchmark Flag Passed.")
        else:
            logging.info("Appending Extra Benchmark Flags...")
            logging.info(str(extra_flags))

        benchmark_dir = OUTPUT_DIR / "benchmark_module"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        for benchmark in cfg["benchmarks"]:
            func = benchmark["name"]
            inputs = benchmark["inputs"]
            isl = benchmark.get("seq_len")
            out_file = benchmark_dir / f"{model_name}_{func}_isl_{isl}.json"

            ## TODO :: iree.runtime.benchmark_module does not handle passing multiple devices currently
            # if needed to add any extra flag for benchmark, please add in amdsharktank/tools/tests/e2e/configs/models.json in the extra_benchmark_flags_list
            # kwargs = {
            #     "module": str(gen_vmfb_path),
            #     "entry_function": func,
            #     "inputs": inputs,
            #     "timeout": None,
            #     "benchmark_repetitions": int(cfg["benchmark_repetitions"]),
            #     "benchmark_out_format": "json",
            #     "benchmark_out": str(out_file),
            #     "parameters": f"model={irpa}",
            #     "device": f"hip://{0}",
            #     **{flag.lstrip("-").replace("-", "_"): True for flag in extra_flags},
            # }

            repetations = int(cfg["benchmark_repetitions"])
            benchmark_inputs = [f"--input={input_value}" for input_value in inputs]
            benchmark_devices = [f"--device=hip://{id}" for id in device_ids]
            iree_benchmark_command = [
                "iree-benchmark-module",
                f"--module={str(gen_vmfb_path)}",
                f"--parameters=model={irpa}",
                f"--function={func}",
                f"--benchmark_repetitions={repetations}",
                f"--benchmark_out_format=json",
                f"--benchmark_out={str(out_file)}",
            ]

            final_command = iree_benchmark_command + extra_flags
            final_command.extend(benchmark_inputs)
            final_command.extend(benchmark_devices)

            logging.info(
                f"\n[===================  Benchmark CMD] iree-benchmark-module(**{final_command}) ====================\n"
            )
            ## TODO:: iree.runtime.benchmark_module does not handle passing multiple devices currently
            # results = iree.runtime.benchmark_module(**kwargs)
            try:
                proc = subprocess.run(
                    final_command, capture_output=True, text=True, check=False
                )
                output = proc.stdout + proc.stderr
            except Exception as e:
                output = str(e)

            logging.info(f"Benchmark results written to {out_file}")
            logging.info(output)

            logging.info("Benchmark done")

        BenchmarkUtils.append_isl_to_json(f"{OUTPUT_DIR}/benchmark_module", None)
        BenchmarkUtils.combine_json(
            f"{OUTPUT_DIR}/benchmark_module",
            f"{OUTPUT_DIR}/consolidated_benchmark.json",
        )

        ISL = cfg["isl"]
        metrics = BenchmarkUtils.extract_prefill_decode_pairs_for_isl(
            f"{OUTPUT_DIR}/consolidated_benchmark.json",
            ISL,
            cfg["benchmark_model"],
            cfg["prefill_bs_for_time_check"],
            cfg["decode_bs_for_time_check"],
        )

        metrics.sort(key=lambda x: x["prefill_batch_size"])
        prefill_status_result = "FAILED"
        decode_status_result = "FAILED"
        if gpu_model == "MI300X":
            prefill_gold = cfg["prefill_gold_mi300x"]
            decode_gold = cfg["decode_gold_mi300x"]
        elif gpu_model in ["MI325X", "MI325"]:
            prefill_gold = cfg["prefill_gold_mi325x"]
            decode_gold = cfg["decode_gold_mi325x"]
        elif gpu_model in ["MI350X", "MI350", "MI355", "MI355X"]:
            prefill_gold = cfg["prefill_gold_mi350x"]
            decode_gold = cfg["decode_gold_mi350x"]
        else:
            logging.INFO("GPU Model Not Found. Available Models are MI300X and MI325.")

        for data in metrics:
            prefill_status_result = (
                "-"
                if metrics[0] == VERY_LARGE
                else BenchmarkUtils.prefill_status(
                    data["Today's Prefill Time(ms)"], prefill_gold
                )
            )
            decode_status_result = (
                "-"
                if metrics[0] == VERY_LARGE
                else BenchmarkUtils.decode_status(
                    data["Today's Decode Time(ms)"], decode_gold
                )
            )

            current_prefill_bs = data["prefill_batch_size"]
            current_prefill = data["Today's Prefill Time(ms)"]
            current_decode_bs = data["decode_batch_size"]
            current_decode = data["Today's Decode Time(ms)"]

            logging.info(
                "\n==================================================================================  TIME SUMMARY  ==================================================================================\n"
            )
            logging.info(f"ISL: {cfg['isl']}")
            logging.info(f"Prefill Batch Size: {current_prefill_bs}")
            logging.info(f"Decode Batch Size: {current_decode_bs}")
            logging.info(
                f"GOLD PREFILL_TIME: {prefill_gold} | CURRENT PREFILL_TIME: {current_prefill}"
            )
            logging.info(
                f"GOLD DECODE_TIME : {decode_gold}   | CURRENT DECODE_TIME : {current_decode}"
            )
            logging.info(
                "\n=======================================================================================  END  =======================================+++++===========================================\n"
            )

        if prefill_status_result == "PASS" and decode_status_result == "PASS":
            logging.info(
                "[SUCCESS] Both prefill and decode status are within 3% and 6% of tolerance w.r.t the Gold Number"
            )

        elif prefill_status_result == "FAIL" and decode_status_result == "PASS":
            logging.error(
                "ERROR: [FAILED] Prefill Number Not within 3% tolerance of Gold number."
            )

        elif prefill_status_result == "PASS" and decode_status_result == "FAIL":
            logging.error(
                "ERROR: [FAILED] Decode Number Not within 6% tolerance of Gold Number."
            )

        elif prefill_status_result == "-" or decode_status_result == "-":
            raise RuntimeError(
                "ERROR: Unable To Fetch The Prefill or Decode Value. Check for Correct Isl, Prefill bs and Decode bs value."
            )
        else:
            logging.error(
                "ERROR: [FAILED] Both decode and prefill not within range of their respective 3% and 6% tolerance."
            )

        logging.info(
            "============================================================================================== Benchmark Done =============================================================================================="
        )

    # === Online Serving ===
    if stage in ["online_serving", "all"]:
        logging.info("Running server ...")

        original_dir = os.getcwd()

        os.chdir("shortfin")
        device_ids = map(str, device_ids)
        try:
            server_cmd = [
                sys.executable,
                "-m",
                "shortfin_apps.llm.server",
                f"--tokenizer_json={tokenizer}",
                f"--model_config={gen_config_path}",
                f"--vmfb={gen_vmfb_path}",
                f"--parameters={irpa}",
                "--device=hip",
                "--port",
                str(cfg["port_for_serving"]),
            ]
            server_cmd.append("--device_ids")
            server_cmd.extend(device_ids)

            server_proc = subprocess.Popen(server_cmd)
        finally:
            os.chdir(original_dir)

        if not OnlineServingUtils.wait_for_server(cfg["port_for_serving"]):
            logging.error("ERROR: Failed to start the server")
            server_proc.kill()
            sys.exit(1)

        logging.info(
            f"Server with PID {server_proc.pid} is ready to accept requests on port {cfg['port_for_serving']}..."
        )

        logging.info("Running Client ...")
        start_time = time.time()

        try:
            response = requests.post(
                f"http://localhost:{cfg['port_for_serving']}/generate",
                headers={"Content-Type": "application/json"},
                json={
                    "text": "<|begin_of_text|>Name the capital of the United States.<|eot_id|>",
                    "sampling_params": {"max_completion_tokens": 50},
                },
                timeout=30,
            )
            logging.info(f"Client Response: {response.text}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Client request failed: {e}")
            server_proc.kill()
            sys.exit(1)

        end_time = time.time()
        time_taken = int(end_time - start_time)
        logging.info(f"Time Taken for Getting Response: {time_taken} seconds")

        time.sleep(10)
        os.kill(server_proc.pid, signal.SIGKILL)

        content = response.text

        expected1 = '"responses": [{"text": "assistant\\nThe capital of the United States is Washington, D.C."}]'
        expected2 = '"responses": [{"text": "Washington D.C."}]'
        expected3 = '"responses": [{"text": "assistant\\n\\nThe capital of the United States is Washington, D.C."}]'
        expected4 = '"responses": [{"text": "assistant\\n\\nThe capital of the United States is Washington, D.C. (short for District of Columbia)."}]'

        if (
            expected1 in content
            or expected2 in content
            or expected3 in content
            or expected4 in content
        ):
            logging.info("[SUCCESS] Online Response Matches Expected Output.")
        elif re.search(
            r'"text": ".*washington(,?\s*d\.?c\.?)?"', content, flags=re.IGNORECASE
        ):
            logging.warning("[CHECK REQUIRED] Partially Correct Response Detected.")
            logging.info(content)
            sys.exit(1)
        else:
            logging.error("[FAILURE] Gibberish or Invalid Response Detected.")
            logging.info(content)
            sys.exit(1)

        logging.info(
            "============================================================================================== Online Serving Done =============================================================================================="
        )


def main():
    parser = argparse.ArgumentParser(description="Model Test Runner")
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model name (e.g., llama-8b-fp8)",
    )
    parser.add_argument(
        "--stage",
        default="all",
        required=False,
        choices=STAGES,
        help="Stage to run. Default all",
    )
    parser.add_argument("--irpa", help="Path to IRPA file")
    parser.add_argument("--tokenizer", help="Path to tokenizer.json")
    parser.add_argument("--tokenizer_config", help="Path to tokenizer_config.json")
    parser.add_argument(
        "--config-path",
        default="amdsharktank/tests/e2e/configs/models.json",
        help="Path For Models Config File.",
    )

    try:
        result = subprocess.run(
            ["amd-smi", "static", "-g", "all", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        data = json.loads(result.stdout)
        product_names = [
            gpu["board"]["product_name"]
            for gpu in data
            if "board" in gpu and "product_name" in gpu["board"]
        ]
        gpu_models = list(
            {
                re.search(r"(MI\d+\w*)", name, re.I).group(1).upper()
                for name in product_names
                if re.search(r"(MI\d+\w*)", name, re.I)
            }
        )

        gpu_model_name = gpu_models[0] if gpu_models else "UNKNOWN"
        print("Detected AMD GPU model:", gpu_model_name)
    except Exception as e:
        print("Error detecting AMD GPU:", e)

    parser.add_argument("--gpu-model", help="Runner Machine Name. Eg. mi300x, mi325")

    args = parser.parse_args()

    print(f"Using Config File: {args.config_path}")
    with open(f"{args.config_path}", "r") as f:
        MODELS = json.load(f)

    output_dir = Path(os.getcwd()) / "output_artifacts"
    OUTPUT_DIR = output_dir / f"output_{args.model}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.model not in MODELS:
        print(MODELS)
        print(
            f" Model '{args.model}' not found in config. Models Available are {MODELS}."
        )
        sys.exit(1)

    cfg = MODELS[args.model]

    irpa = args.irpa or cfg["irpa"]
    tokenizer = args.tokenizer or cfg["tokenizer"]
    tokenizer_config = args.tokenizer_config or cfg["tokenizer_config"]
    gpu_model = args.gpu_model or gpu_model_name

    run_stage(
        args.stage,
        args.model,
        irpa,
        tokenizer,
        tokenizer_config,
        cfg,
        gpu_model,
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()

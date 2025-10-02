# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import ast
import glob
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

import iree.compiler as ireec
import iree.compiler.tools as ireec_tools
import iree.runtime

import sys
from pathlib import Path

with open("sharktank/sharktank/tools/models.json", "r") as f:
    MODELS = json.load(f)


STAGES = ["export", "compile", "validate_vmfb", "benchmark", "online_serving"]
MODEL_CHOICES = [
    "llama-70b-fp16",
    "llama-70b-fp8",
    "llama-8b-fp16",
    "llama-8b-fp8",
    "mistral",
]
VERY_LARGE = 1e9


def wait_for_server(port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            time.sleep(2)
    return False


def combine_json(dir, outfile):
    dir = Path(dir)
    files = glob.glob(str(dir.absolute()) + "/*.json")
    merged_data = [json.load(open(path, "r")) for path in files]
    with open(outfile, "w") as outs:
        json.dump(merged_data, outs, indent=2)


def append_isl_to_json(dir, isl=None):
    dir = Path(dir)
    files = glob.glob(str(dir.absolute()) + "/*.json")
    for f in files:
        length = isl
        if not length:
            length = Path(f).stem.rsplit("isl_")[-1]
        try:
            length = int(length)
        except Exception as e:
            print(f"Invalid ITL encountered, Exception {e}")

        with open(f, "r") as src:
            data = json.load(src)
            if "context" in data:
                context = data["context"]
                context["ISL"] = length

                with open(f, "w") as src:
                    json.dump(data, src, indent=2)


def extract_prefill_decode_pairs_for_isl(
    json_path, target_isl, model, prefill_batch_size, decode_batch_size
):
    with open(json_path, "r") as f:
        data = json.load(f)

    results = []
    prefill_map = {}
    decode_map = {}
    for entry in data:
        context = entry.get("context", {})
        isl = context.get("ISL")
        if isl != target_isl:
            continue

        for bench in entry.get("benchmarks", []):
            name = bench.get("name", "")
            run_type = bench.get("run_type", "")
            if run_type != "aggregate" or "mean" not in name:
                continue

            bs_match = re.search(r"bs(\d+)", name)
            if not bs_match:
                continue
            bs = int(bs_match.group(1))

            if "prefill" in name:
                prefill_map[bs] = round(bench.get("real_time", VERY_LARGE), 3)
            elif "decode" in name:
                decode_map[bs] = round(bench.get("real_time", VERY_LARGE), 3)

    for prefill_bs, prefill_time in sorted(prefill_map.items()):

        if prefill_bs != prefill_batch_size:
            continue
        decode_bs = decode_batch_size
        decode_time = decode_map.get(decode_bs, VERY_LARGE)

        results.append(
            {
                "prefill_batch_size": prefill_bs,
                "Today's Prefill Time(ms)": prefill_time,
                "decode_batch_size": decode_bs,
                "Today's Decode Time(ms)": decode_time,
                "ISL": isl,
            }
        )
    return results


def prefill_status(current, historical):
    if current == "-":
        return "FAIL"
    if historical == "-":
        return "FAIL"
    return "PASS" if current <= 1.03 * float(historical) else "FAIL"  # 3% tolerance


def decode_status(current, historical):
    if current == "-":
        return "FAIL"
    if historical == "-":
        return "FAIL"
    return "PASS" if current <= 1.06 * float(historical) else "FAIL"  # 6% tolerance


def run_cmd(cmd, append=True):
    OUTPUT_DIR = Path(os.getcwd()) / "output_artifacts"
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
            logging.info(decoded.strip())  # also send to logging
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
    return LOG_FILE


def run_stage(stage, model_name, irpa, tokenizer, tokenizer_config, cfg):
    print(f"\n Running stage: {stage} for model: {model_name}")
    print(f"    IRPA: {irpa}")
    print(f"    Tokenizer: {tokenizer}")
    print(f"    Tokenizer Config: {tokenizer_config}")

    OUTPUT_DIR = Path(os.getcwd()) / "output_artifacts"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    # === Export Stage ===
    if stage in ["export", "compile", "validate_vmfb", "benchmark", "online_serving"]:
        if os.path.exists(gen_mlir_path) and os.path.exists(gen_config_path):
            logging.info("File exists. Skipping Export..")
        else:
            logging.info("Exporting IR Through Sharktank")

            export_cmd = [
                sys.executable,
                "-m",
                "sharktank.examples.export_paged_llm_v1",
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

            logging.info("=============================================================================== Using Export Command ===============================================================================")
            logging.info("")
            logging.info(f"Using Export Command: {' '.join(export_cmd)}")
            logging.info("")
            logging.info("====================================================================================================================================================================================")
            run_cmd(export_cmd, append=True)
            logging.info(
                "============================================================================================== Export Done =============================================================================================="
            )

    # === Compile Stage ===
    if stage in ["compile", "validate_vmfb", "benchmark", "online_serving"]:
        if os.path.exists(gen_vmfb_path):
            logging.info("File exists. Skipping Compile...")
        else:
            logging.info("Continuing with Compile...")
            logging.info("Compiling IR ....")

            input_file = str(gen_mlir_path)
            output_file = str(gen_vmfb_path)
            extra_args = [
                "--iree-hal-target-device=hip",
                "--iree-opt-level=O3",
                "--iree-hal-indirect-command-buffers=true",
                "--iree-stream-resource-memory-model=discrete",
                "--iree-hip-enable-tensor-ukernels",
                "--iree-hal-memoization=true",
                "--iree-codegen-enable-default-tuning-specs=true",
                "--iree-stream-affinity-solver-max-iterations=1024",
                f"--iree-hip-target={cfg['iree_hip_target']}",
            ]

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
            logging.info("=============================================================== Using Compile Command ===============================================================")
            logging.info("")
            logging.info(
                f"Using ireec.compile_file with flags(extra_args): {extra_args}"
            )
            logging.info("")
            logging.info("======================================================================================================================================================")
            print()

            start = time.time()
            ireec.compile_file(
                input_file,
                output_file=output_file,
                target_backends=["rocm"],
                extra_args=extra_args,
            )
            logging.info(
                f"Time taken for compiling: {int(time.time() - start)} seconds"
            )
            logging.info(
                "============================================================================================== Compile Done =============================================================================================="
            )


    # === Validate Stage ===
    if stage in ["validate_vmfb"]:
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
                "sharktank.tools.run_llm_vmfb",
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
                "--steps", str(steps),
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
        sys.exit(result)

    # === IREE Benchmark ===
    if stage in ["benchmark"]:
        try:
            extra_flags = cfg.get("extra_compile_flags_list", [])
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

            logging.info(f"\nRunning benchmark for function={func}, seq_len={isl}\n")

            results = iree.runtime.benchmark_module(
                module=str(gen_vmfb_path),
                entry_function=func,
                inputs=inputs,
                timeout=None,
                benchmark_repetitions=int(cfg["benchmark_repetitions"]),
                benchmark_out_format="json",
                benchmark_out=str(out_file),
                parameters=f"model={irpa}",
                device="hip://1",
                **{flag.lstrip("-").replace("-", "_"): True for flag in extra_flags},
            )

            logging.info(f"Benchmark results written to {out_file}")
            for r in results:
                logging.info(str(r))

            logging.info("Benchmark done")

        append_isl_to_json(f"{OUTPUT_DIR}/benchmark_module", None)
        combine_json(
            f"{OUTPUT_DIR}/benchmark_module",
            f"{OUTPUT_DIR}/consolidated_benchmark.json",
        )

        ISL = cfg["isl"]
        metrics = extract_prefill_decode_pairs_for_isl(
            f"{OUTPUT_DIR}/consolidated_benchmark.json",
            ISL,
            cfg["benchmark_model"],
            cfg["prefill_bs_for_time_check"],
            cfg["decode_bs_for_time_check"],
        )

        metrics.sort(key=lambda x: x["prefill_batch_size"])
        prefill_status_result = "FAILED"
        decode_status_result = "FAILED"

        for data in metrics:
            prefill_status_result = (
                "-"
                if metrics[0] == VERY_LARGE
                else prefill_status(
                    data["Today's Prefill Time(ms)"], cfg["prefill_gold"]
                )
            )
            decode_status_result = (
                "-"
                if metrics[0] == VERY_LARGE
                else decode_status(data["Today's Decode Time(ms)"], cfg["decode_gold"])
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
                f"GOLD PREFILL_TIME: {cfg['prefill_gold']} | CURRENT PREFILL_TIME: {current_prefill}"
            )
            logging.info(
                f"GOLD DECODE_TIME : {cfg['decode_gold']}   | CURRENT DECODE_TIME : {current_decode}"
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
                "[FAILED] Prefill Number Not within 3% tolerance of Gold number."
            )
            sys.exit(1)
        elif prefill_status_result == "PASS" and decode_status_result == "FAIL":
            logging.error(
                "[FAILED] Decode Number Not within 6% tolerance of Gold Number."
            )
            sys.exit(1)
        elif prefill_status_result == "-" or decode_status_result == "-":
            raise RuntimeError(
                "Unable To Fetch The Prefill or Decode Value. Check for Correct Isl, Prefill bs and Decode bs value."
            )
        else:
            logging.error(
                "[FAILED] Both decode and prefill not within range of their respective 3% and 6% tolerance."
            )
            sys.exit(1)

        logging.info(
            "============================================================================================== Benchmark Done =============================================================================================="
        )

    # === Online Serving ===
    if stage in ["online_serving"]:
        logging.info("Running server ...")

        server_cmd = [
            sys.executable,
            "-m",
            "shortfin_apps.llm.server",
            f"--tokenizer_json={tokenizer}",
            f"--model_config={gen_config_path}",
            f"--vmfb={gen_vmfb_path}",
            f"--parameters={irpa}",
            "--device=hip",
            "--device_ids",
            "0",
            "--port",
            str(cfg["port_for_serving"]),
        ]
        server_proc = subprocess.Popen(server_cmd)

        if not wait_for_server(cfg["port_for_serving"]):
            logging.error("Failed to start the server")
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

        if expected1 in content or expected2 in content or expected3 in content or expected4 in content:
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
    parser = argparse.ArgumentParser(description="Model Test Runner")  # add choices
    parser.add_argument(
        "--model",
        required=True,
        choices=MODEL_CHOICES,
        help="Model name (e.g., llama-8b-fp8)",
    )
    parser.add_argument("--stage", required=True, choices=STAGES, help="Stage to run")
    parser.add_argument("--irpa", help="Path to IRPA file")
    parser.add_argument("--tokenizer", help="Path to tokenizer.json")
    parser.add_argument("--tokenizer_config", help="Path to tokenizer_config.json")

    args = parser.parse_args()

    if args.model not in MODELS:
        print(
            f" Model '{args.model}' not found in config. Models Available are llama-70b-fp16, llama-70b-fp8, llama-8b-fp16, llama-8b-fp8, mistral."
        )
        sys.exit(1)

    cfg = MODELS[args.model]

    irpa = args.irpa or cfg["irpa"]
    tokenizer = args.tokenizer or cfg["tokenizer"]
    tokenizer_config = args.tokenizer_config or cfg["tokenizer_config"]

    run_stage(args.stage, args.model, irpa, tokenizer, tokenizer_config, cfg)


if __name__ == "__main__":
    main()

# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import glob
import json
import re
import time
from pathlib import Path

import requests

from pathlib import Path


STAGES = ["export", "compile", "validate_vmfb", "benchmark", "online_serving", "all"]
MODEL_CHOICES = [
    "llama-70b-fp16",
    "llama-70b-fp8",
    "llama-8b-fp16",
    "llama-8b-fp8",
    "mistral",
]

VERY_LARGE = 1e9


class OnlineServingUtils:
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


class BenchmarkUtils:
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

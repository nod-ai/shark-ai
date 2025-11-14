# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import os


def normalize_ascii(obj):
    if isinstance(obj, str):
        return obj.replace("–", "-").replace("—", "-")
    elif isinstance(obj, list):
        return [normalize_ascii(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: normalize_ascii(v) for k, v in obj.items()}
    return obj


def parse_log(log_file_path):
    gold_prefill_time = current_prefill_time = None
    gold_decode_time = current_decode_time = None
    with open(log_file_path, "r") as file:
        for line in file:
            if "GOLD PREFILL_TIME" in line:
                gold_prefill_time = float(
                    line.split(":")[3].strip().split(" ")[0].replace(" ", "")
                )
                current_prefill_time = float(
                    line.split(":")[4].strip().split(" ")[0].replace(" ", "")
                )
            if "GOLD DECODE_TIME" in line:
                gold_decode_time = float(
                    line.split(":")[3].strip().split(" ")[0].replace(" ", "")
                )
                current_decode_time = float(
                    line.split(":")[4].strip().split(" ")[0].replace(" ", "")
                )

    if None in (gold_prefill_time, current_prefill_time, gold_decode_time, current_decode_time):
        return None
    return (
        gold_prefill_time,
        current_prefill_time,
        gold_decode_time,
        current_decode_time,
    )


def update_json_for_conditions(json_file_path, log_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    updated = False
    for model, details in data.items():
        log_file_path = os.path.join(
            log_path, f"output_{model}/e2e_testing_log_file.log"
        )

        if not os.path.exists(log_file_path):
            print(f"Skipping {model} — log file not found.")
            continue

        parsed = parse_log(log_file_path)
        if parsed is None:
            print(f"Skipping {model} — GOLD PREFILL/DECODE values missing in log.")
            continue

        gold_prefill, current_prefill, gold_decode, current_decode = parsed

        gold_prefill_mi325x = float(details.get("prefill_gold_mi325x", None))
        gold_decode_mi325x = float(details.get("decode_gold_mi325x", None))
        if gold_prefill_mi325x and gold_decode_mi325x:
            if current_prefill < gold_prefill_mi325x * (1 - 0.03):
                print(
                    f"Updating PREFILL gold for {model}: {gold_prefill_mi325x} -> {current_prefill}"
                )
                details["prefill_gold_mi325x"] = round(current_prefill, 3)
                updated = True
            if current_decode < gold_decode_mi325x * (1 - 0.06):
                print(
                    f"Updating DECODE gold for {model}: {gold_decode_mi325x} -> {current_decode}"
                )
                details["decode_gold_mi325x"] = round(current_decode, 3)
                updated = True

    if updated:
        with open(json_file_path, "w") as f:
            json.dump(normalize_ascii(data), f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("[IMPROVEMENT SEEN] Gold values updated in the JSON file. Creating a Pr..")
    else:
        print("No updates made — all models within tolerance.")


if __name__ == "__main__":
    json_file_path = "sharktank/tests/e2e/configs/models.json"
    log_path = "output_artifacts"
    update_json_for_conditions(json_file_path, log_path)

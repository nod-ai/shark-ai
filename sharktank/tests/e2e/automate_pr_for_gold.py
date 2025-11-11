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
    with open(log_file_path, "r") as file:
        for line in file:
            if "GOLD PREFILL_TIME" in line:
                gold_prefill_time = float(
                    line.split(":")[3].strip().split(" ")[0].replace(" ", "")
                )
                current_prefill_time = float(
                    line.split(":")[4].strip().split(" ")[0].replace(" ", "")
                )
                print(gold_prefill_time, " ", current_prefill_time)
            if "GOLD DECODE_TIME" in line:
                gold_decode_time = float(
                    line.split(":")[3].strip().split(" ")[0].replace(" ", "")
                )
                current_decode_time = float(
                    line.split(":")[4].strip().split(" ")[0].replace(" ", "")
                )
                print(gold_decode_time, " ", current_decode_time)
    return (
        gold_prefill_time,
        current_prefill_time,
        gold_decode_time,
        current_decode_time,
    )


def update_json_for_conditions(json_file_path, log_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    for model, details in data.items():
        log_file_path = os.path.join(
            log_path, f"output_{model}/e2e_testing_log_file.log"
        )

        if not os.path.exists(log_file_path):
            continue

        gold_prefill, current_prefill, gold_decode, current_decode = parse_log(
            log_file_path
        )
        print(f"prefill gold: {gold_prefill}")
        print(f"prefill current: {current_prefill}")
        print(f"decode gold: {gold_decode}")
        print(f"decode current: {current_decode}")

        gold_prefill_mi325x = float(details.get("prefill_gold_mi325x", None))
        gold_decode_mi325x = float(details.get("decode_gold_mi325x", None))
        print(f"json p gold{gold_prefill_mi325x}")
        print(f"json d gold{gold_decode_mi325x}")
        if gold_prefill_mi325x and gold_decode_mi325x:
            if current_prefill < gold_prefill_mi325x * (1 - 0.03):
                print(f"Updating prefill gold value for model: {model}")
                details["prefill_gold_mi325x"] = current_prefill
            if current_decode < gold_decode_mi325x * (1 - 0.06):
                print(f"Updating decode gold value for model: {model}")
                details["decode_gold_mi325x"] = current_decode

    with open(json_file_path, "w") as f:
        json.dump(normalize_ascii(data), f, indent=2, ensure_ascii=False)
        print("Gold values updated in the JSON file.")


if __name__ == "__main__":
    json_file_path = "sharktank/tests/e2e/configs/models.json"
    log_path = "output_artifacts"
    update_json_for_conditions(json_file_path, log_path)

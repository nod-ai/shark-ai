"""
Combines the IREE Benchmark json files for Prefill and Decode into a Single
File -> consolidated_benchmark.json

Tests the Results Through Prefill and Decode Time
By Comparing The Time with Respective prefill_gold and decode_gold present in model config.
Tolerance 3% for Prefill and 6% for Decode
"""
import json
import argparse
from pathlib import Path
import numpy as np
import sys
import re
import glob


def combine_json(dir, outfile):
    files = glob.glob(str(dir.absolute()) + "/*.json")
    merged_data = [json.load(open(path, "r")) for path in files]
    with open(outfile, "w") as outs:
        json.dump(merged_data, outs, indent=2)


def append_isl_to_json(dir, isl=None):
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


# ############ Test Above Results Through Prefill-Decode Time ################
def extract_prefill_decode_pairs_for_isl(json_path, target_isl, model):
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

        if prefill_bs != args.prefill_bs_for_time_check:
            continue
        decode_bs = args.decode_bs_for_time_check
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--combine-json",
        type=Path,
        help="Combine all json files into single file",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output json file name",
    )
    parser.add_argument(
        "--append-isl",
        action="store_true",
        help="Append isl to the json",
    )
    parser.add_argument(
        "--isl",
        type=int,
        default=None,
        help="Input sequence length to append to the json",
    )
    parser.add_argument(
        "--prefill-gold",
        default=None,
        help="prefill gold number stored",
    )
    parser.add_argument(
        "--decode-gold",
        default=None,
        help="decode gold number stored",
    )
    parser.add_argument(
        "--benchmark-model",
        default=None,
        help="Benchmark Model Name",
    )
    parser.add_argument(
        "--prefill-bs-for-time-check",
        type=int,
        required=True,
        help="prefill bs for time check",
    )
    parser.add_argument(
        "--decode-bs-for-time-check",
        type=int,
        required=True,
        help="decode bs for time check",
    )
    args = parser.parse_args()

    if args.append_isl:
        append_isl_to_json(args.combine_json, None)
    combine_json(args.combine_json, args.output_json)

    ##### Test for Prefill and Decode Time #####
    VERY_LARGE = 1e9
    ISL = args.isl
    metrics = extract_prefill_decode_pairs_for_isl(
        args.output_json, ISL, args.benchmark_model
    )

    metrics.sort(key=lambda x: x["prefill_batch_size"])
    prefill_status_result = "FAILED"
    decode_status_result = "FAILED"

    for data in metrics:
        prefill_status_result = (
            "-"
            if metrics[0] == VERY_LARGE
            else prefill_status(data["Today's Prefill Time(ms)"], args.prefill_gold)
        )
        decode_status_result = (
            "-"
            if metrics[0] == VERY_LARGE
            else decode_status(data["Today's Decode Time(ms)"], args.decode_gold)
        )

        current_prefill_bs = data["prefill_batch_size"]
        current_prefill = data["Today's Prefill Time(ms)"]
        current_decode_bs = data["decode_batch_size"]
        current_decode = data["Today's Decode Time(ms)"]

        print("\n==================================  TIME SUMMARY  ===================================\n")
        print(f"ISL: {args.isl}")
        print(f"Prefill Batch Size: {current_prefill_bs}")
        print(f"Decode Batch Size: {current_decode_bs}")
        print(
            f"GOLD PREFILL_TIME: {args.prefill_gold} | CURRENT PREFILL_TIME: {current_prefill}"
        )
        print(
            f"GOLD DECODE_TIME : {args.decode_gold}   | CURRENT DECODE_TIME : {current_decode}"
        )
        print("\n=======================================  END  =======================================\n")

    if prefill_status_result == "PASS" and decode_status_result == "PASS":
        print(
            "[SUCCESS] Both prefill and decode status are within 3% and 6% of tolerance w.r.t the Gold Number"
        )
    elif prefill_status_result == "FAIL" and decode_status_result == "PASS":
        print("[FAIL] Prefill Number Not within 3% tolerance of Gold number.")
        sys.exit(1)
    elif prefill_status_result == "PASS" and decode_status_result == "FAIL":
        print("[FAIL] Decode Number Not within 6% tolerance of Gold Number.")
        sys.exit(1)
    else:
        print(
            "[FAIL] Both decode and prefill not within range of their respective 3% and 6% tolerance."
        )
        sys.exit(1)

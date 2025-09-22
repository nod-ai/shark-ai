import json
import argparse
from pathlib import Path
import numpy as np
import sys
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
    args = parser.parse_args()

    if args.append_isl:
        append_isl_to_json(args.combine_json, args.isl)
    combine_json(args.combine_json, args.output_json)

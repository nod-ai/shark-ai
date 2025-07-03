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
    args = parser.parse_args()
    combine_json(args.combine_json, args.output_json)

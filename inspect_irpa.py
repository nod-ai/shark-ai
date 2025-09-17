#!/usr/bin/env python3

import sys
from sharktank.types import Dataset

if len(sys.argv) != 2:
    print("Usage: python inspect_irpa.py <path_to_irpa_file>")
    sys.exit(1)

irpa_path = sys.argv[1]

try:
    dataset = Dataset.load(irpa_path)

    print("=== Dataset Properties Structure ===")
    print("Keys:", list(dataset.properties.keys()))
    for key, value in dataset.properties.items():
        if isinstance(value, dict):
            print(f"{key}: (dict with keys: {list(value.keys())})")
            if key == "hparams":
                print("  hparams contents:")
                for hkey, hvalue in value.items():
                    print(f"    {hkey}: {hvalue}")
        else:
            print(f"{key}: {value}")

    print("\n=== Theta Structure ===")
    print("Theta keys (first 10):", list(dataset.root_theta.keys())[:10])

except Exception as e:
    print(f"Error loading IRPA file: {e}")
    sys.exit(1)

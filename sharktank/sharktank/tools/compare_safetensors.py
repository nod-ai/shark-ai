# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compares two safetensors files of expected vs actual results."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from safetensors import safe_open
import torch

from sharktank.utils import cli


class Reporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.index = open(self.output_dir / "index.html", "wt")
        self.index.write("<html>\n")
        self.index.write("<body>\n")

    def close(self):
        self.index.write("</body>\n")
        self.index.write("</html>\n")
        self.index.close()

    def compare(self, name: str, expected: torch.Tensor, actual: torch.Tensor):
        f = self.index

        def print_line(line: str, color: str = ""):
            style = ""
            if color:
                style = f"color: {color}"
            f.write(f"<div style='font-family: monospace; white-space: pre; {style}'>")
            f.write(line)
            f.write("</div>\n")

        def print_stats(label, t):
            t = t.to(dtype=torch.float32)
            std, mean = torch.std_mean(t)
            print_line(
                f"    {label}: "
                f"MIN={torch.min(t)}, "
                f"MAX={torch.max(t)}, "
                f"MEAN={mean}, STD={std}"
            )

        exp_flat = expected.flatten().to(torch.float32)
        act_flat = actual.flatten().to(torch.float32)
        diff_flat = exp_flat - act_flat
        diff_min = torch.min(diff_flat)
        diff_max = torch.max(diff_flat)
        diff_mean, diff_stddev = torch.std_mean(diff_flat)
        exp_max = torch.max(exp_flat)
        exp_min = torch.min(exp_flat)
        bound = torch.abs(exp_max - exp_min) / 2.0

        f.write("<hr>\n")
        f.write(f"<h3>{name}</h3>\n")
        non_finite_count = (
            torch.nonzero(torch.logical_not(torch.isfinite(diff_flat)))
            .flatten()
            .shape[0]
        )
        if non_finite_count > 0:
            print_line(f"Samples have {non_finite_count} non finite values!", "red")

        print_stats(" REF", expected)
        print_stats(" ACT", actual)
        print_stats("DIFF", diff_flat)

        diff_mean_percent = 100.0 * torch.abs(diff_mean) / bound
        diff_outlier_percent = (
            100.0 * max(float(torch.abs(diff_min)), float(torch.abs(diff_max))) / bound
        )
        diff_stddev_percent = 100.0 * torch.abs(diff_stddev) / bound
        print_line(
            f"       DIFF MEAN PERCENT: {diff_mean_percent:.5f}%",
            color="red" if diff_mean_percent > 1.0 else "",
        )
        print_line(
            f"     DIFF STDDEV PERCENT: {diff_stddev_percent:.5f}%",
            color="red" if diff_stddev_percent > 1.0 else "",
        )
        print_line(
            f"    DIFF OUTLIER PERCENT: {diff_outlier_percent:.5f}%",
            color="red" if diff_outlier_percent > 1.0 else "",
        )

        print(f"Generating plots {name}")
        dpi = 144
        file_name_root = f"{self.counter}_{name}_"
        plot_filename = f"{file_name_root}_diff.png"
        self.counter += 1
        x = torch.arange(0, diff_flat.shape[0])
        y = diff_flat

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set(xlabel="dim", ylabel="diff", title="Difference")
        ax.axhline(float(bound), linestyle=":", color="red", alpha=1.0)
        ax.axhline(float(-bound), linestyle=":", color="red", alpha=1.0)
        ax.axhline(float(diff_min), linestyle=":", color="blue", alpha=1.0)
        ax.axhline(float(diff_max), linestyle=":", color="blue", alpha=1.0)
        ax.grid()
        fig.set_figheight(2.5)
        fig.set_figwidth(8)
        fig.savefig(str(self.output_dir / plot_filename), dpi=dpi)
        plt.close(fig)

        f.write(f"<img src='{plot_filename}'>\n")

        f.flush()


def main(argv):

    parser = cli.create_parser()
    parser.add_argument("--dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "expected_path", type=Path, help="Path to expected safetensors file"
    )
    parser.add_argument(
        "actual_path", type=Path, help="Path to actual safetensors file"
    )
    args = cli.parse(parser, args=argv)

    reporter = Reporter(args.dir)
    try:
        with safe_open(args.expected_path, framework="pt") as exp_f, safe_open(
            args.actual_path, framework="pt"
        ) as act_f:
            exp_keys = exp_f.keys()
            act_keys = act_f.keys()
            for name in exp_keys:
                if name not in act_keys:
                    continue
                reporter.compare(name, exp_f.get_tensor(name), act_f.get_tensor(name))
    finally:
        reporter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

from skimage import metrics
from PIL import Image

import argparse
from pathlib import Path
import numpy as np
import sys
import json
from math import isclose


def compare_images(args):
    gen_images = args.gen_images
    ref_images = args.ref_images

    status = True

    if len(gen_images) != len(ref_images):
        if len(ref_images) == 1:
            ref_images = ref_images * len(gen_images)
        else:
            print("Number of reference images are not equal to the generated images")
            return 1

    for img1, img2 in zip(gen_images, ref_images):
        try:
            gen_image = Image.open(img1)
            ref_image = Image.open(img2)
            if not args.use_original_sizes:
                gen_image = gen_image.resize(ref_image.size)
            gen_image_numpy = np.array(gen_image)
            ref_image_numpy = np.array(ref_image)
            ssim_value = metrics.structural_similarity(
                ref_image_numpy, gen_image_numpy, data_range=255, channel_axis=2
            )
            if ssim_value < args.ssim_threshold:
                print(f"Images {img1} and {img2} are not similar, SSIM {ssim_value}")
                status = False

        except Exception as e:
            print(f"Exception : '{e}' while comparing {img1} and {img2}")
            return 1

    return status == False


def compare_iree_benchmark(iree_benchmark_file, golden_ref_file, model):
    status = True
    try:
        with open(iree_benchmark_file, "r") as f1:
            iree_results = json.load(f1)

        with open(golden_ref_file, "r") as f2:
            golden_ref = json.load(f2)

        golden_values = None
        tolerance_percentage = None  # In percentage

        for ref in golden_ref:
            if ref["model"] == model:
                golden_values = ref["values"]
                tolerance_percentage = ref["tolerance_percentage"]

        if not golden_values:
            return 1

        for result in iree_results:
            ISL = result["context"]["ISL"]
            for benchmark in result["benchmarks"]:
                if "aggregate_name" in benchmark:
                    if benchmark["aggregate_name"] == "mean":
                        time = benchmark["real_time"]
                        function = benchmark["name"].split("/")[0]
                        for value in golden_values:
                            if value["ISL"] == ISL and value["name"] == function:
                                golden_time = value["time"]
                                # Default tolerance is 10% of golden value.
                                rel_tol = golden_time * value["tolerance_percentage"]
                                is_close = isclose(time, golden_time, rel_tol=rel_tol)
                                if not is_close:
                                    print(
                                        f"Exceeded tolerance limit for {model} with ISL {ISL} on {function}"
                                    )
                                    print(
                                        f"\nCurrent time : {time}, Golden time {golden_time}"
                                    )
                                    status = False
                                continue
                        continue

    except Exception as e:
        print(f"Exception : '{e}' occured while comparing iree benchmark files")
        return 1

    return status == False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-images",
        type=Path,
        nargs="+",
        help="Absolute path to reference images for comparison",
    )
    parser.add_argument(
        "--gen-images",
        type=Path,
        nargs="+",
        help="Absolute path to generated images to compare",
    )
    parser.add_argument(
        "--ssim-threshold",
        type=float,
        default=0.9,
        help="SSIM threshold value to identify the similarity",
    )
    parser.add_argument(
        "--use-original-sizes",
        action="store_true",
        help="Compare images with original sizes without resizing",
    )
    parser.add_argument(
        "--compare-npy",
        type=str,
        nargs="+",
        default=None,
        help="List of absolute path for numpy files to compare",
    )
    parser.add_argument(
        "--compare-iree-benchmark",
        type=Path,
        default=None,
        help="Absolute path to json with IREE Benchmark values",
    )
    parser.add_argument(
        "--golden-ref",
        type=Path,
        default=None,
        help="Absolute path to json with Golden Reference Values for IREE Benchmark",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Name of the model to compare for IREE Benchmark",
    )
    args = parser.parse_args()
    if args.compare_iree_benchmark:
        sys.exit(
            compare_iree_benchmark(
                args.compare_iree_benchmark, args.golden_ref, args.model
            )
        )
    sys.exit(compare_images(args))

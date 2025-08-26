from skimage import metrics
from PIL import Image

import argparse
from pathlib import Path
import numpy as np
import sys


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


def compare_npy(inputs):
    assert len(inputs) > 1
    status = True
    try:
        base = inputs[0]
        base_array = np.load(base)
        for i in range(1, len(inputs)):
            status = status and np.allclose(base_array, np.load(inputs[i]))

            if not status:
                print(f"Output mismatch for {base} and {inputs[i]}")

    except Exception as e:
        print(f"Exception : '{e}' occured while comparing npy files")
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
    args = parser.parse_args()
    if args.compare_npy:
        sys.exit(compare_npy(args.compare_npy))
    sys.exit(compare_images(args))

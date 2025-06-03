import argparse
import glob
import os
from collections import defaultdict

import nibabel as nib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Ensemble average")
    parser.add_argument(
        "--path_predictions",
        type=str,
        required=True,
        nargs="+",
        help="Path to predictions",
    )
    parser.add_argument("--path_output", type=str, required=True, help="Path to output")
    parser.add_argument("--path_output_float", type=str, required=False, help="Path to output")

    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for segmentation")
    args = parser.parse_args()

    predictions = defaultdict(list)
    affine_headers = {}

    for directory in args.path_predictions:
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")

        paths = sorted(glob.glob(f"{directory}/*.nii.gz"))

        for path in paths:
            name = os.path.basename(path)

            image = nib.load(path)
            x = image.get_fdata()
            print(f"{path} {x.shape} {np.min(x)} {np.max(x)}", flush=True)

            predictions[name].append(x)
            affine_headers[name] = (image.affine, image.header)

    # create output directory
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    if args.path_output_float is not None:
        if not os.path.exists(args.path_output_float):
            os.makedirs(args.path_output_float)

    for name, preds in predictions.items():
        x = np.mean(preds, axis=0)
        print(f"{name} {len(preds)} {x.shape} {np.min(x)} {np.max(x)}", flush=True)

        if args.path_output_float is not None:
            path = os.path.join(args.path_output_float, name)
            nib.save(
                nib.Nifti1Image(x, affine_headers[name][0], affine_headers[name][1]),
                path,
            )

        x = (np.sign(x - args.threshold) + 1) / 2
        x = x.astype(np.uint8)

        path = os.path.join(args.path_output, name)
        nib.save(nib.Nifti1Image(x, affine_headers[name][0], affine_headers[name][1]), path)


if __name__ == "__main__":
    main()

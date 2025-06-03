import argparse
import glob
import os
import pickle
import sys
from functools import partial

import haiku as hk
import nibabel as nib
import numpy as np

import jax


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument("--path_config", type=str, required=True, help="Path to config.pkl")
    parser.add_argument("--path_weights", type=str, required=True, help="Path to weights.pkl")
    parser.add_argument("--path_scripts", type=str, required=True, help="Path to python scripts")
    parser.add_argument("--path_images", type=str, required=True, help="Path to images")
    parser.add_argument("--path_labels", type=str, required=False, help="Path to labels")
    parser.add_argument("--path_output", type=str, required=True, help="Path to output")

    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for segmentation")
    args = parser.parse_args()

    sys.path.insert(0, args.path_scripts)
    import model  # noqa: F401
    from functions import (
        eval_model,
        preprocess_miccai22_image,
        round_zooms,
    )  # noqa: F401

    # Load model args
    with open(args.path_config, "rb") as f:
        train_config = pickle.load(f)
    print(train_config, flush=True)

    @partial(jax.jit, static_argnums=(2,))
    def apply(w, x, zooms):
        return hk.without_apply_rng(hk.transform(model.create_model(train_config.model))).apply(w, x, zooms)

    with open(args.path_weights, "rb") as f:
        w = pickle.load(f)
        w = jax.device_put(w)

    images = sorted(glob.glob(f"{args.path_images}/*.nii.gz"))

    print(images, flush=True)
    if args.path_labels is not None:
        labels = sorted(glob.glob(f"{args.path_labels}/*.nii.gz"))
    else:
        labels = [None] * len(images)

    if len(images) != len(labels):
        raise ValueError("Number of images and labels do not match")

    # create output directory
    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    DSCs = []

    for image_path, label_path in zip(images, labels):
        print(f"Evaluating run {image_path}", flush=True)

        image = nib.load(image_path)
        img = preprocess_miccai22_image(image.get_fdata())
        pred = eval_model(
            img,
            lambda x: apply(w, x, round_zooms(image.header.get_zooms())),
            overlap=2.0,
            verbose=True,
        )
        print(flush=True)

        nib.save(
            nib.Nifti1Image(pred, image.affine, image.header),
            f"{args.path_output}/{os.path.basename(image_path)}",
        )

        if label_path is not None:
            if os.path.basename(image_path) != os.path.basename(label_path):
                print(
                    f"Image and label names do not match: {image_path} {label_path}",
                    flush=True,
                )

            label = nib.load(label_path)
            y_gt = label.get_fdata()
            y_pred = (np.sign(pred - args.threshold) + 1) / 2
            four_classes = 2 * y_gt + y_pred
            nib.save(
                nib.Nifti1Image(four_classes, label.affine, label.header),
                f"{args.path_output}/confusion{os.path.basename(image_path)}",
            )

            tp = np.sum(y_pred * y_gt)
            # tn = np.sum((1 - y_pred) * (1 - y_gt))
            fp = np.sum(y_pred * (1 - y_gt))
            fn = np.sum((1 - y_pred) * y_gt)

            DSC = 2 * tp / (2 * tp + fp + fn)
            print(f"DSC (dice score): {DSC}", flush=True)
            DSCs.append(DSC)

    print(f"DSC (dice score): {', '.join(map(repr, DSCs))}")


if __name__ == "__main__":
    main()

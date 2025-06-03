import glob
import os
import pickle
import time
from functools import partial

import haiku as hk
import jax
import nibabel as nib
import numpy as np

import model
from config import get_config
from functions import eval_model, preprocess_miccai22_image, round_zooms, format_time

train_config = get_config()
print(train_config, flush=True)


@partial(jax.jit, static_argnums=(2,))
def apply(w, x, zooms):
    f = hk.without_apply_rng(hk.transform(model.create_model(train_config.model))).apply
    return jax.vmap(f, (None, 0, None), 0)(w, x, zooms)


images = sorted(glob.glob("/input/*.nii.gz"))
weights = sorted(glob.glob("/home/*.pkl"))
t0 = time.perf_counter()

for i, image_path in enumerate(images):
    sum_pred = 0
    for j, weight_path in enumerate(weights):
        print(f"{image_path} {weight_path}", flush=True)
        with open(weight_path, "rb") as f:
            w = pickle.load(f)
            w = jax.device_put(w)

        image = nib.load(image_path)
        img = preprocess_miccai22_image(image.get_fdata())
        pred = eval_model(
            img,
            lambda x: apply(w, x, round_zooms(image.header.get_zooms())),
            overlap=1.1,
            verbose=True,
        )
        sum_pred += pred

        s = time.perf_counter() - t0
        eta = s / (i * len(weights) + j + 1) * (len(images) * len(weights) - (i * len(weights) + j + 1))

        print(f"SPENT {format_time(s)}  ETA {format_time(eta)}", flush=True)

    x = np.array(sum_pred / len(weights))

    print(f"{image_path} n={len(weights)} {x.shape} {np.min(x)} {np.max(x)}", flush=True)

    threshold = 0.0
    x = (np.sign(x - threshold) + 1) / 2
    x = x.astype(np.uint8)

    name = os.path.basename(image_path)
    path = os.path.join("/predict", name)
    nib.save(nib.Nifti1Image(x, image.affine, image.header), path)
    print(f"------> {path}", flush=True)
    print("", flush=True)

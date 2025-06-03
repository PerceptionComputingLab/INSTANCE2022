import math
import pickle
import time
from collections import namedtuple
from functools import partial
from typing import Callable, List, Tuple

import nibabel as nib
import numpy as np

import jax
import jax.numpy as jnp
import wandb
from diffeomorphism import deform, scalar_field
from jax.scipy.special import logsumexp


@jax.jit
def noise_mri(rng, img):
    assert img.shape[0] == img.shape[1]
    noise = jax.vmap(lambda r: scalar_field(img.shape[0], 128, r), 0, 2)(jax.random.split(rng, img.shape[2] * img.shape[3]))
    return img + 1e-2 * jnp.reshape(noise, img.shape)


@jax.jit
def deform_mri(rng, img, temperature):
    r"""Apply a diffeomorphism to an MRI image.
    Apply the same deformation to all slices of the image.

    Args:
        rng: A random number generator.
        img: The MRI image ``[x, y, ...]``

    Returns:
        The deformed image.
    """
    assert img.shape[0] == img.shape[1]
    temperature = jax.random.uniform(rng, (), minval=0.0, maxval=temperature)
    f = lambda i: deform(i, temperature, 5, rng, "nearest")
    for _ in range(img.ndim - 2):
        f = jax.vmap(f, -1, -1)
    return f(img)


def preprocess_miccai22_image(image: np.ndarray):
    bone = 0.005 * np.clip(image, a_min=0.0, a_max=1000.0).astype(np.float32)
    range1 = 0.03 * np.clip(image, a_min=0.0, a_max=80.0).astype(np.float32)
    range2 = 0.013 * np.clip(image, a_min=-50.0, a_max=220.0).astype(np.float32)
    image = np.stack([bone, range1, range2], axis=-1)
    return image


def load_miccai22(path: str, i: int) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """i goes from 1 to 100"""
    image = nib.load(f"{path}/data/{i:03d}.nii.gz")
    label = nib.load(f"{path}/label/{i:03d}.nii.gz")
    assert np.allclose(image.affine, label.affine)

    zooms = image.header.get_zooms()
    image = image.get_fdata()
    label = label.get_fdata()

    image = preprocess_miccai22_image(image)

    label = 2.0 * label - 1.0
    label = label.astype(np.float32)
    return image, label, zooms


def round_mantissa(x, n):
    """Round number

    Args:
        x: number to round
        n: number of mantissa digits to keep

    Returns:
        rounded number

    Example:
        >>> round_mantissa(0.5 + 0.25 + 0.125, 0)
        1.0

        >>> round_mantissa(0.5 + 0.25 + 0.125, 2)
        0.875
    """
    if x == 0:
        return 0
    s = 1 if x >= 0 else -1
    x = abs(x)
    a = math.floor(math.log2(x))
    x = x / 2**a
    assert 1.0 <= x < 2.0, x
    x = round(x * 2**n) / 2**n
    x = x * 2**a
    return s * x


def random_slice(rng: jnp.ndarray, size: int, target_size: int) -> slice:
    if size <= target_size:
        return 0
    start = jax.random.randint(rng, (), 0, size - target_size)
    return start


@partial(jax.jit, static_argnums=(2,))
def random_sample(
    rng: jnp.ndarray, x: jnp.ndarray, target_sizes: Tuple[int, int, int]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    rng = jax.random.split(rng, 4)
    starts = jax.tree_util.tree_map(random_slice, tuple(rng[:3]), x.shape[:3], target_sizes)
    starts = starts + (0,) * (x.ndim - 3)
    target_sizes = target_sizes + x.shape[3:]
    return jax.lax.dynamic_slice(x, starts, target_sizes), rng[3]


def format_time(seconds: float) -> str:
    if seconds < 1:
        return f"{1000 * seconds:03.0f}ms"
    if seconds < 60:
        return f"{seconds:05.2f}s"
    minutes = math.floor(seconds / 60)
    seconds = seconds - 60 * minutes
    if minutes < 60:
        return f"{minutes:02.0f}min{seconds:02.0f}s"
    hours = math.floor(minutes / 60)
    minutes = minutes - 60 * hours
    return f"{hours:.0f}h{minutes:02.0f}min"


def unpad(x, pads):
    return x[pads[0] : -pads[0], pads[1] : -pads[1], pads[2] : -pads[2]]


def cross_entropy(p: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    r"""Safe cross entropy loss function.

    Args:
        p: predictions (logits)
        y: labels (-1 or 1)

    Returns:
        loss: cross entropy loss ``log(1 + exp(-p y))``
    """
    assert p.shape == y.shape
    zero = jnp.zeros_like(p)
    return logsumexp(jnp.stack([zero, -p * y]), axis=0)


@jax.jit
def confusion_matrix(y: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    r"""Confusion matrix.

    Args:
        y: labels (-1 or 1)
        p: predictions (logits)

    Returns:
        confusion matrix
        [tn, fp]
        [fn, tp]
    """
    assert y.shape == p.shape
    y = y > 0.0
    p = p > 0.0
    tp = jnp.sum(y * p)
    tn = jnp.sum((1 - y) * (1 - p))
    fp = jnp.sum((1 - y) * p)
    fn = jnp.sum(y * (1 - p))
    return jnp.array([[tn, fp], [fn, tp]])


TrainState = namedtuple(
    "TrainState",
    [
        "time0",
        "train_set",
        "test_set",
        "t4",
        "losses",
        "best_sorted_dices",
        "rng",
        "w_eval",
    ],
)


def init_train_loop(config, data_path, old_state, step, w, opt_state) -> TrainState:
    print("Prepare for the training loop...", flush=True)

    if config.seed_shuffle_data is None:
        indices = list(range(1, 101))
    else:
        indices = jax.random.permutation(jax.random.PRNGKey(config.seed_shuffle_data), 100) + 1
        indices = [int(i) for i in indices]

    train_set = [(i,) + load_miccai22(data_path, i) for i in indices[: config.trainset]]

    test_set = []
    test_sample_size = np.array([200, 200, 25])
    for i in indices[::-1][: config.testset]:
        img, lab, zooms = load_miccai22(data_path, i)  # test data
        center_of_mass = np.stack(np.nonzero(lab == 1.0), axis=-1).mean(0).astype(np.int)
        start = np.maximum(center_of_mass - test_sample_size // 2, 0)
        end = np.minimum(start + test_sample_size, np.array(img.shape[:3]))
        start = np.maximum(end - test_sample_size, 0)

        img = img[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        lab = lab[start[0] : end[0], start[1] : end[1], start[2] : end[2]]
        test_set.append((i, img, lab, zooms))

    return TrainState(
        time0=getattr(old_state, "time0", time.perf_counter()),
        train_set=train_set,
        test_set=test_set,
        t4=time.perf_counter(),
        losses=getattr(old_state, "losses", np.ones((len(train_set),))),
        best_sorted_dices=getattr(old_state, "best_sorted_dices", np.zeros((len(test_set),))),
        rng=getattr(old_state, "rng", jax.random.PRNGKey(config.seed_train)),
        w_eval=getattr(old_state, "w_eval", w),
    )


sample_size = (144, 144, 13)  # physical size ~= 65mm ^ 3
sample_padding = (22, 22, 2)  # 10mm of border removed
train_padding = sample_padding


def round_zooms(zooms: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return jax.tree_util.tree_map(lambda x: round_mantissa(x, 4), zooms)


@partial(jax.jit, static_argnums=(1,))
def random_split(rng: jnp.ndarray, n: int) -> jnp.ndarray:
    return jax.random.split(rng, n)


def train_loop(config, state: TrainState, step, w, opt_state, update, apply_model) -> TrainState:
    t0 = time.perf_counter()
    t_extra = t0 - state.t4

    if step == 120:
        jax.profiler.start_trace(wandb.run.dir)

    i, img, lab, zooms = state.train_set[step % len(state.train_set)]
    img, lab = jax.device_put((img, lab))

    # data augmentation
    rng = random_split(state.rng, 8)

    if jax.random.uniform(rng[0]) < config.augmentation.noise:
        img = noise_mri(rng[1], img)

    if jax.random.uniform(rng[2]) < config.augmentation.deformation:
        img = deform_mri(rng[3], img, config.augmentation.deformation_temperature)
        lab = deform_mri(rng[3], lab, config.augmentation.deformation_temperature)
        lab = jnp.round(lab)

    # regroup zooms and sizes by rounding and taking subsets of the volume
    zooms = round_zooms(zooms)
    if jax.random.uniform(rng[4]) < 0.5:
        # avoid patch without label
        r = rng[5]
        while True:
            x, _ = random_sample(r, img, sample_size)
            y, r = random_sample(r, lab, sample_size)
            if np.any(unpad(y, train_padding) == 1):
                img, lab = x, y
                break
    else:
        # avoid patch full of air
        r = rng[6]
        while True:
            x, _ = random_sample(r, img, sample_size)
            y, r = random_sample(r, lab, sample_size)
            if np.any(x > 0.0):
                img, lab = x, y
                break
    del x, y

    rng = rng[7]

    t1 = time.perf_counter()

    lr = config.optimizer.lr * max(
        config.optimizer.lr_div_factor ** math.floor(step / config.optimizer.lr_div_step),
        config.optimizer.lr_div_factor_min,
    )
    w, opt_state, train_loss, train_pred = update(w, opt_state, img, lab, zooms, lr, train_padding)
    train_loss.block_until_ready()

    w_eval = jax.tree_util.tree_map(
        lambda x, y: (1.0 - config.weight_avg) * x + config.weight_avg * y,
        state.w_eval,
        w,
    )

    t2 = time.perf_counter()
    c = np.array(confusion_matrix(unpad(lab, sample_padding), unpad(train_pred, sample_padding)))
    with np.errstate(invalid="ignore"):
        train_dice = 2 * c[1, 1] / (2 * c[1, 1] + c[1, 0] + c[0, 1])

    min_median_max = np.min(train_pred), np.median(train_pred), np.max(train_pred)

    state.losses[step % len(state.train_set)] = train_loss

    time_str = time.strftime("%H:%M", time.localtime())
    print(
        (
            f"{wandb.run.dir.split('/')[-2]} "
            f"[{time_str}] [{step:04d}:{format_time(time.perf_counter() - state.time0)}] "
            f"train[ loss={np.mean(state.losses):.4f} LR={lr:.1e} dice={100 * train_dice:02.0f} ] "
            f"time[ S{format_time(t1 - t0)}+U{format_time(t2 - t1)}+EX{format_time(t_extra)} ]"
        ),
        flush=True,
    )

    if step % 500 == 0:
        with open(f"{wandb.run.dir}/w.pkl", "wb") as f:
            pickle.dump(w, f)
        with open(f"{wandb.run.dir}/w_eval.pkl", "wb") as f:
            pickle.dump(w_eval, f)

    if step == 120:
        jax.profiler.stop_trace()

    if step % 100 == 0 and len(state.test_set) > 0:
        c = np.zeros((len(state.test_set), 2, 2))
        for j, (i, img, lab, zooms) in enumerate(state.test_set):
            zooms = round_zooms(zooms)
            test_pred = eval_model(img, lambda x: apply_model(w_eval, x, zooms))
            c[j] = np.array(confusion_matrix(lab, test_pred))

        with np.errstate(invalid="ignore"):
            dice = 2 * c[:, 1, 1] / (2 * c[:, 1, 1] + c[:, 1, 0] + c[:, 0, 1])

        wandb.log(
            {f"dice_{i}": d for (i, _, _, _), d in zip(state.test_set, dice)},
            commit=False,
            step=step,
        )

        sorted_dices = np.sort(dice)
        for i, (old_dice, new_dice) in enumerate(zip(state.best_sorted_dices, sorted_dices)):
            if new_dice > old_dice:
                state.best_sorted_dices[i] = new_dice
                wandb.log({f"best{i}_min_dice": new_dice}, commit=False, step=step)
                with open(f"{wandb.run.dir}/best{i}_w.pkl", "wb") as f:
                    pickle.dump(w_eval, f)

        dice_txt = ",".join(f"{100 * d:02.0f}" for d in dice)
        best_dice_txt = ",".join(f"{100 * d:02.0f}" for d in state.best_sorted_dices)

        t4 = time.perf_counter()

        time_str = time.strftime("%H:%M", time.localtime())
        print(
            (
                f"{wandb.run.dir.split('/')[-2]} "
                f"[{time_str}] [{step:04d}:{format_time(time.perf_counter() - state.time0)}] "
                f"test[ dice={dice_txt} best_sorted_dices={best_dice_txt} ] "
                f"time[ E{format_time(t4 - t2)} ]"
            ),
            flush=True,
        )

        epoch_avg_confusion = np.mean(c, axis=0)
        epoch_avg_confusion = epoch_avg_confusion / np.sum(epoch_avg_confusion)

        wandb.log(
            {
                "true_negatives": epoch_avg_confusion[0, 0],
                "true_positives": epoch_avg_confusion[1, 1],
                "false_negatives": epoch_avg_confusion[1, 0],
                "false_positives": epoch_avg_confusion[0, 1],
                "confusion_matrices": c,
                "time_eval": t4 - t2,
            },
            commit=False,
            step=step,
        )

        del dice, dice_txt, c, test_pred, epoch_avg_confusion
    else:
        t4 = t2

    wandb.log(
        {
            "_runtime": time.perf_counter() - state.time0,
            "train_loss": train_loss,
            "avg_train_loss": np.mean(state.losses),
            "min_pred": min_median_max[0],
            "median_pred": min_median_max[1],
            "max_pred": min_median_max[2],
            "time_update": t2 - t1,
        },
        commit=True,
        step=step,
    )

    state = TrainState(
        time0=state.time0,
        train_set=state.train_set,
        test_set=state.test_set,
        t4=t4,
        losses=state.losses,
        best_sorted_dices=state.best_sorted_dices,
        rng=rng,
        w_eval=w_eval,
    )
    return (state, w, opt_state)


def patch_slices(total: int, size: int, pad: int, overlap: float) -> List[int]:
    r"""
    Generate a list of patch indices such that the center of the patches (unpaded patches) cover the full image.

    Args:
        total: The total size of the image.
        size: The size of the patch.
        pad: The padding of the patch.
        overlap: The overlap of the patches.
    """
    step = max(1, round((size - 2 * pad) / overlap))
    naive = list(range(0, total - size, step)) + [total - size]
    return np.round(np.linspace(0, total - size, len(naive))).astype(int)


def eval_model(
    img: jnp.ndarray,
    apply: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    overlap: float = 1.0,
    size: Tuple[int, int, int] = None,
    padding: Tuple[int, int, int] = None,
    verbose: bool = False,
) -> np.ndarray:
    assert img.ndim == 3 + 1

    smaller_axis = np.argmin(img.shape[:3])
    img = jnp.moveaxis(img, smaller_axis, 2)

    if size is None:
        size = sample_size
    if padding is None:
        padding = sample_padding

    assert img.shape[2] < img.shape[0]
    assert img.shape[2] < img.shape[1]

    pos = np.stack(
        np.meshgrid(
            np.linspace(-1.3, 1.3, size[0] - 2 * padding[0]),
            np.linspace(-1.3, 1.3, size[1] - 2 * padding[1]),
            np.linspace(-1.3, 1.3, size[2] - 2 * padding[2]),
            indexing="ij",
        ),
        axis=-1,
    )
    gaussian = np.exp(-np.linalg.norm(pos, axis=-1) ** 2)

    ii = patch_slices(img.shape[0], size[0], padding[0], overlap)
    jj = patch_slices(img.shape[1], size[1], padding[1], overlap)
    kk = patch_slices(img.shape[2], size[2], padding[2], overlap)

    xs = []
    for i in ii:
        for j in jj:
            for k in kk:
                x = img[i : i + size[0], j : j + size[1], k : k + size[2]]
                xs.append(x)

    def slice(x, start, end):
        x = x[start:end]
        if x.shape[0] < end - start:
            x = jnp.concatenate([x, jnp.zeros((end - start - x.shape[0],) + x.shape[1:])], axis=0)
        return x

    xs = jnp.stack(xs, axis=0)
    n_parallel = 4
    ps = jnp.concatenate([apply(slice(xs, i, i + n_parallel)) for i in range(0, len(xs), n_parallel)])
    ps = ps[: len(xs)]
    ps = jax.vmap(unpad, (0, None), 0)(ps, padding)

    sum = np.zeros_like(img[:, :, :, 0])
    num = np.zeros_like(img[:, :, :, 0])

    index = 0
    for i in ii:
        for j in jj:
            for k in kk:
                p = ps[index]
                index += 1

                sum[
                    i + padding[0] : i + size[0] - padding[0],
                    j + padding[1] : j + size[1] - padding[1],
                    k + padding[2] : k + size[2] - padding[2],
                ] += (
                    p * gaussian
                )
                num[
                    i + padding[0] : i + size[0] - padding[0],
                    j + padding[1] : j + size[1] - padding[1],
                    k + padding[2] : k + size[2] - padding[2],
                ] += gaussian

    negative_value = -10.0
    sum[num == 0] = negative_value
    num[num == 0] = 1.0

    output = sum / num
    output = jnp.moveaxis(output, 2, smaller_axis)
    return output

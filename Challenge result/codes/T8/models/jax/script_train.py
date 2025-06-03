import hashlib
import importlib
import pickle
import shutil
import sys
import time
from functools import partial

import haiku as hk
import numpy as np
import optax
from absl import app, flags
from ml_collections import config_flags

import jax
import jax.numpy as jnp
import wandb
from jax.config import config

_CONFIG = config_flags.DEFINE_config_file("config")

FLAGS = flags.FLAGS
flags.DEFINE_string("pretrained", None, "Path to weights")
flags.DEFINE_string("name", None, "Name of the run", required=True)
flags.DEFINE_string("data", "../../train_2", "Path to train data")
flags.DEFINE_string("logdir", ".", "Path to log directory")

config.update("jax_debug_nans", True)
config.update("jax_debug_infs", True)
np.set_printoptions(precision=3, suppress=True)


def hash_file(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def flatten_dictionary(dictionary):
    """Flatten a dictionary

    Args:
        dictionary (dict): Dictionary to flatten, it must have `items` method

    Returns:
        dict: Flattened dictionary with keys as strings separated by dots

    Example:
    >>> flatten_dictionary({'a': {'b': 1, 'c': 2}, 'd': 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    output = {}
    for key, value in dictionary.items():
        if not hasattr(value, "items"):
            output[key] = value
        else:
            sub_output = flatten_dictionary(value)
            for sub_key, sub_value in sub_output.items():
                output[f"{key}.{sub_key}"] = sub_value
    return output


def main(_):
    config = _CONFIG.value
    print(config, flush=True)

    wandb.init(
        project="miccai22",
        entity="instance2022",
        name=FLAGS.name,
        dir=FLAGS.logdir,
        config=flatten_dictionary(config),
    )
    shutil.copy(__file__, f"{wandb.run.dir}/main.py")
    shutil.copy(f"./models/{config.model.name}.py", f"{wandb.run.dir}/model.py")
    shutil.copy("./functions.py", f"{wandb.run.dir}/functions.py")
    shutil.copy("./script_evaluate.py", f"{wandb.run.dir}/evaluate.py")
    shutil.copy("./diffeomorphism.py", f"{wandb.run.dir}/diffeomorphism.py")
    sys.path.insert(0, wandb.run.dir)
    import functions
    import model

    with open(f"{wandb.run.dir}/config.pkl", "wb") as f:
        pickle.dump(config, f)

    # Load data
    print("Loading data...", flush=True)
    img, lab, zooms = functions.load_miccai22(FLAGS.data, 1)

    # Create model
    model = hk.without_apply_rng(hk.transform(model.create_model(config.model)))

    if FLAGS.pretrained is not None:
        print("Loading pretrained parameters...", flush=True)
        w = pickle.load(open(FLAGS.pretrained, "rb"))
    else:
        print("Initializing model...", flush=True)
        t = time.perf_counter()
        w = model.init(jax.random.PRNGKey(config.seed_init), img[:100, :100, :25], zooms)
        print(
            f"Initialized model in {functions.format_time(time.perf_counter() - t)}",
            flush=True,
        )

    def opt(lr):
        if config.optimizer.algorithm == "adam":
            return optax.adam(lr)
        if config.optimizer.algorithm == "sgd":
            return optax.sgd(lr, 0.9)

    @partial(jax.jit, static_argnums=(2,))
    def apply_model(w, x, zooms):
        return model.apply(w, x, zooms)

    @partial(jax.jit, static_argnums=(4, 6))
    def update(w, opt_state, x, y, zooms, lr, pads):
        r"""Update the model parameters.

        Args:
            w: Model parameters.
            opt_state: Optimizer state.
            x: Input data ``(x, y, z)``.
            y: Ground truth data ``(x, y, z)`` of the form (-1.0, 1.0).
            zooms: The zoom factors ``(x, y, z)``.
            lr: Learning rate.

        Returns:
            (w, opt_state, loss, pred):
                w: Updated model parameters.
                opt_state: Updated optimizer state.
                loss: Loss value.
                pred: Predicted data ``(x, y, z)``.
        """
        assert x.ndim == 3 + 1
        assert y.ndim == 3

        un = lambda x: functions.unpad(x, pads)

        def h(w, x, y):
            p = model.apply(w, x, zooms)
            return jnp.mean(functions.cross_entropy(un(p), un(y))), p

        grad_fn = jax.value_and_grad(h, has_aux=True)
        (loss, pred), grads = grad_fn(w, x, y)

        updates, opt_state = opt(lr).update(grads, opt_state)
        w = optax.apply_updates(w, updates)
        return w, opt_state, loss, pred

    opt_state = opt(config.optimizer.lr).init(w)

    hash = hash_file(f"{wandb.run.dir}/functions.py")
    state = functions.init_train_loop(config, FLAGS.data, None, 0, w, opt_state)
    print("Start main loop...", flush=True)

    for step in range(config.train_steps):

        # Reload the loop function if the code has changed
        new_hash = hash_file(f"{wandb.run.dir}/functions.py")
        if new_hash != hash:
            hash = new_hash
            importlib.reload(functions)
            state = functions.init_train_loop(config, FLAGS.data, state, step, w, opt_state)
            print("Continue main loop...", flush=True)

        state, w, opt_state = functions.train_loop(config, state, step, w, opt_state, update, apply_model)


if __name__ == "__main__":
    app.run(main)

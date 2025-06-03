import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(0, 1))
def scalar_field_modes(size, num_modes):
    """
    Generate energies and modes for 2D scalar field.

    Args:
        size: size of the scalar field
        num_modes: number of modes to compute

    Returns:
        sqrt(1 / Energy per mode) and the modes
    """
    with jax.core.eval_context():
        k = jnp.arange(1, num_modes + 1)
        i, j = jnp.meshgrid(k, k, indexing="ij")
        r = jnp.sqrt(i**2 + j**2)
        e = (r < num_modes + 0.5) / r

        x = jnp.linspace(0, 1, size)
        s = jnp.sin(jnp.pi * x[:, None] * k[None, :])
        return e, s


@partial(jax.jit, static_argnums=(0, 1))
def scalar_field(size, num_modes, rng_key):
    """
    random scalar field `size` x `size` made of the first `num_modes` modes

    Args:
        size: size of the scalar field
        num_modes: number of modes
        rng_key: random number generator key

    Returns:
        scalar field
    """
    with jax.core.eval_context():
        e, s = scalar_field_modes(size, num_modes)
    c = jax.random.normal(rng_key, (num_modes, num_modes)) * e
    return jnp.einsum("ij,xi,yj->yx", c, s, s)


@partial(jax.jit, static_argnums=(3,))
def remap(image, dx, dy, interpolation="linear"):
    """
    Remap image(s) using displacement field dx, dy

    Args:
        image: image(s) [..., y, x]
        dx: displacement field in x [y, x]
        dy: displacement field in y [y, x]
    """
    size1, size2 = image.shape[-2:]
    assert dx.shape == (size1, size2) and dy.shape == (
        size1,
        size2,
    ), "Image(s) and displacement fields shapes should match."

    with jax.core.eval_context():
        y, x = jnp.meshgrid(jnp.arange(size1), jnp.arange(size2), indexing="ij")

    xn = jnp.clip(x - dx, 0, size2 - 1)
    yn = jnp.clip(y - dy, 0, size1 - 1)

    if interpolation == "linear":
        xf = jnp.floor(xn).astype(jnp.int32)
        yf = jnp.floor(yn).astype(jnp.int32)
        xc = jnp.ceil(xn).astype(jnp.int32)
        yc = jnp.ceil(yn).astype(jnp.int32)

        xv = xn - xf
        yv = yn - yf

        return (
            (1 - yv) * (1 - xv) * image[..., yf, xf]
            + (1 - yv) * xv * image[..., yf, xc]
            + yv * (1 - xv) * image[..., yc, xf]
            + yv * xv * image[..., yc, xc]
        )
    if interpolation == "nearest":
        xf = jnp.round(xn).astype(jnp.int32)
        yf = jnp.round(yn).astype(jnp.int32)
        return image[..., yf, xf]


@partial(jax.jit, static_argnums=(2, 4))
def deform(image, temperature, num_modes, rng_key, interpolation="linear"):
    """
    Args:
        image: image(s) [..., y, x]
        T: temperature
        num_modes: number of modes
        rng_key: random number generator key
        interpolation: interpolation method, "linear" or "nearest"

    Returns:
        deformed image(s)
    """
    size = image.shape[-1]
    assert image.shape[-2] == size, "Image(s) should be square."

    # Sample dx, dy
    # u, v are defined in [0, 1]^2
    # dx, dx are defined in [0, size]^2
    key_u, key_v = jax.random.split(rng_key)
    u = scalar_field(size, num_modes, key_u)  # [size,size]
    v = scalar_field(size, num_modes, key_v)  # [size,size]
    dx = jnp.sqrt(temperature) * size * u
    dy = jnp.sqrt(temperature) * size * v

    # Apply tau
    return remap(image, dx, dy, interpolation)


def displacement_from_temperature(temperature, num_modes, size):
    return 0.5 * size * jnp.sqrt(jnp.pi * temperature * jnp.log(num_modes))


def temperature_from_displacement(displacement, num_modes, size):
    return (2 * displacement) ** 2 / (jnp.pi * size**2 * jnp.log(num_modes))


def temperature_range(size, num_modes):
    """
    Define the range of allowed temperature
    for given image size and num_modes.
    """
    T1 = temperature_from_displacement(0.5, num_modes, size)
    T2 = 4 / (jnp.pi**3 * num_modes**2 * jnp.log(num_modes))
    return T1, T2

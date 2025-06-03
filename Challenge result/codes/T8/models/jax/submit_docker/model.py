import dataclasses
from typing import Tuple

import e3nn_jax as e3nn
import haiku as hk
import numpy as np
from e3nn_jax.experimental.voxel_convolution import Convolution
from e3nn_jax.experimental.voxel_pooling import zoom

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class Voxels:
    zooms: Tuple[float, float, float]
    data: e3nn.IrrepsArray


jax.tree_util.register_pytree_node(
    Voxels,
    lambda voxel: ((voxel.data,), voxel.zooms),
    lambda zooms, tuple: Voxels(zooms=zooms, data=tuple[0]),
)


def downsample(input: Voxels, min_zoom: float) -> Voxels:
    z_in = np.array(input.zooms)
    s_in = np.array(input.data.shape[-4:-1])

    s_out = np.floor(s_in * z_in / np.maximum(z_in, min_zoom)).astype(int)
    z_out = z_in * s_in / s_out
    z_out = tuple(float(z) for z in z_out)
    s_out = tuple(int(s) for s in s_out)

    return Voxels(
        zooms=z_out,
        data=e3nn.IrrepsArray(
            input.data.irreps,
            jax.vmap(lambda x: zoom(x, output_size=s_out), -1, -1)(input.data.array),
        ),
    )


def upsample(input: Voxels, zooms: Tuple[float, float, float], size: Tuple[int, int, int]) -> Voxels:
    output = Voxels(
        zooms=zooms,
        data=e3nn.IrrepsArray(
            input.data.irreps,
            jax.vmap(lambda x: zoom(x, output_size=size), -1, -1)(input.data.array),
        ),
    )
    assert np.allclose(
        output.zooms,
        np.array(input.zooms) * np.array(input.data.shape[-4:-1]) / np.array(output.data.shape[-4:-1]),
    )
    return output


def n_vmap(n, fun):
    for _ in range(n):
        fun = jax.vmap(fun)
    return fun


def print_stats(name, x):
    return
    if isinstance(x, Voxels):
        x = x.data
    assert isinstance(x, e3nn.IrrepsArray)
    print(
        name,
        x.shape,
        " ".join(f"{mulir}:{jnp.mean(a):.1f}+{jnp.mean(a**2)**0.5:.2f}" for mulir, a in zip(x.irreps, x.list)),
        flush=True,
    )


class MixChannels(hk.Module):
    def __init__(self, output_size: int, output_irreps: e3nn.Irreps):
        super().__init__()
        self.output_size = output_size
        self.output_irreps = e3nn.Irreps(output_irreps)

    def __call__(self, input: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
        assert len(input.shape) == 1 + 1
        input = input.repeat_mul_by_last_axis()
        output = e3nn.Linear([(self.output_size * mul, ir) for mul, ir in self.output_irreps])(input)
        return output.factor_mul_to_last_axis(self.output_size)


def g(x: e3nn.IrrepsArray) -> e3nn.IrrepsArray:
    return e3nn.gate(
        x,
        even_act=jax.nn.gelu,
        odd_act=jax.nn.tanh,
        even_gate_act=jax.nn.sigmoid,
        odd_gate_act=jax.nn.tanh,
    )


def bn(x: e3nn.IrrepsArray, eps: float) -> e3nn.IrrepsArray:
    if eps >= 1:
        return x

    f = e3nn.BatchNorm(instance=True, eps=eps)
    if x.ndim == 1 + 3 + 1:
        return f(x)
    if x.ndim == 1 + 3 + 1 + 1:
        return jax.vmap(f, 4, 4)(x)


def create_model(config):
    assert config.equivariance in ["E3", "SE3"]

    e3nn.config("path_normalization", config.path_normalization)
    e3nn.config("gradient_normalization", config.gradient_normalization)

    def f(input: jnp.ndarray, zooms: Tuple[float, float, float]) -> jnp.ndarray:
        r"""Unet with irreps regrouped into chunks of varying size through the network.

        Args:
            input (jnp.ndarray): input data of shape ``(x, y, z, channels)``
            zooms (Tuple[float, float, float]): the zooms of the input data

        Returns:
            jnp.ndarray: output data of shape ``(x, y, z)``
        """
        irreps_sh = e3nn.Irreps.spherical_harmonics(4, p=-1 if config.equivariance == "E3" else 1)
        irreps_sh = e3nn.Irreps([mul_ir for mul_ir in irreps_sh if config.num_radial_basis_sh[mul_ir.ir.l] > 0])
        kw = dict(
            irreps_sh=irreps_sh,
            num_radial_basis={l: n for l, n in enumerate(config.num_radial_basis_sh)},
            relative_starts={l: s for l, s in enumerate(config.relative_start_sh)},
        )

        def cbg(vox: Voxels, mul: float, *, radius: float, filter=None) -> Voxels:
            mul = round(mul)
            assert len(vox.data.shape) == 1 + 3 + 1 + 1  # (batch, x, y, z, channel, irreps)

            if config.equivariance == "E3":
                irreps_a = e3nn.Irreps("4x0e + 4x0o")
                irreps_b = e3nn.Irreps("2x1e + 2x1o + 2e + 2o")
            if config.equivariance == "SE3":
                irreps_a = e3nn.Irreps("4x0e")
                irreps_b = e3nn.Irreps("2x1e + 2e")

            if filter is not None:
                irreps_a = irreps_a.filter(filter)
                irreps_b = irreps_b.filter(filter)
            irreps = e3nn.Irreps(f"{irreps_a} + {irreps_b.num_irreps}x0e + {irreps_b}")

            x = vox.data

            # Linear
            x = n_vmap(1 + 3, MixChannels(mul, irreps))(x)
            x = bn(x, config.instance_norm_eps)
            x = g(x)

            # Convolution
            x = jax.vmap(Convolution(irreps, diameter=2.0 * radius, steps=vox.zooms, **kw), 4, 4)(x)
            x = bn(x, config.instance_norm_eps)
            x = g(x)

            # Linear
            x = n_vmap(1 + 3, MixChannels(mul, irreps))(x)
            x = bn(x, config.instance_norm_eps)
            x = g(x)

            return Voxels(zooms=vox.zooms, data=x)

        def down(vox, min_zoom):
            assert len(vox.data.shape) == 1 + 3 + 1 + 1  # (batch, x, y, z, channel, irreps)
            return jax.vmap(lambda x: downsample(x, min_zoom), 4, 4)(vox)

        def cat(vox1: Voxels, vox2: Voxels) -> Voxels:
            assert len(vox1.data.shape) == 1 + 3 + 1 + 1
            assert len(vox2.data.shape) == 1 + 3 + 1 + 1
            assert vox1.zooms == vox2.zooms
            assert vox1.data.irreps == vox2.data.irreps
            return Voxels(
                zooms=vox1.zooms,
                data=e3nn.IrrepsArray.cat([vox1.data, vox2.data], axis=-2),
            )

        def upcat(low_vox: Voxels, high_vox: Voxels) -> Voxels:
            assert len(low_vox.data.shape) == 1 + 3 + 1 + 1
            vox = jax.vmap(lambda x: upsample(x, high_vox.zooms, high_vox.data.shape[1:4]), 4, 4)(low_vox)
            return cat(vox, high_vox)

        def group_conv(vox: Voxels, *, irreps: e3nn.Irreps, mul: int, radius: float) -> Voxels:
            assert len(vox.data.shape) == 1 + 3 + 1 + 1
            x = vox.data
            x = n_vmap(1 + 3, MixChannels(mul, x.irreps))(x)
            x = jax.vmap(Convolution(irreps, diameter=2.0 * radius, steps=vox.zooms, **kw), 4, 4)(x)
            return Voxels(zooms=vox.zooms, data=x)

        mul = config.width  # default is 5

        assert input.ndim == 3 + 1
        x = Voxels(
            zooms=zooms, data=e3nn.IrrepsArray("0e", input[None, :, :, :, :, None])
        )  # Voxel of shape (batch, x, y, z, channel, irreps)

        min_zoom = config.min_zoom
        cvrad = config.conv_diameter / 2.0

        # Block A
        print_stats("Block A", x)
        x = group_conv(
            x,
            irreps="0e + 1o" if config.equivariance == "E3" else "0e + 1e",
            mul=round(mul),
            radius=cvrad * min_zoom,
        )
        x = cbg(
            x,
            mul,
            filter=["0e", "0o", "1e", "1o"],
            radius=cvrad * min_zoom,
        )
        min_zoom *= config.downsampling
        x_a = x
        x = down(x, min_zoom=min_zoom)

        if config.dummy:
            x = cbg(x, mul, filter=["0e", "0o", "1e", "1o"], radius=cvrad * min_zoom)
            x = upcat(x, x_a)
            min_zoom /= config.downsampling
            x = group_conv(x, irreps="0e", mul=round(mul), radius=cvrad * min_zoom)
            x = x.data.repeat_irreps_by_last_axis()  # [batch, x, y, z, irreps]
            x = bn(x, config.instance_norm_eps)
            x = g(x)
            x = n_vmap(1 + 3, e3nn.Linear("0e"))(x)
            return x.array[0, :, :, :, 0]

        # Block B
        print_stats("Block B", x)
        x = cbg(x, 3 * mul, radius=cvrad * min_zoom)
        x = cbg(x, 3 * mul, radius=cvrad * min_zoom)
        min_zoom *= config.downsampling
        x_b = x
        x = down(x, min_zoom=min_zoom)

        # Block C
        print_stats("Block C", x)
        x = cbg(x, 6 * mul, radius=cvrad * min_zoom)
        x = cbg(x, 6 * mul, radius=cvrad * min_zoom)
        min_zoom *= config.downsampling
        x_c = x
        x = down(x, min_zoom=min_zoom)

        # Block D
        print_stats("Block D", x)
        x = cbg(x, 10 * mul, radius=cvrad * min_zoom)
        x = cbg(x, 10 * mul, radius=cvrad * min_zoom)
        x = cbg(x, 10 * mul, radius=cvrad * min_zoom)

        # Block E
        print_stats("Block E", x)
        x = upcat(x, x_c)
        min_zoom /= config.downsampling
        x = cbg(x, 6 * mul, radius=cvrad * min_zoom)
        x = cbg(x, 6 * mul, radius=cvrad * min_zoom)

        # Block F
        print_stats("Block F", x)
        x = upcat(x, x_b)
        min_zoom /= config.downsampling
        x = cbg(x, 3 * mul, radius=cvrad * min_zoom)
        x = cbg(x, mul, filter=["0e", "0o", "1e", "1o"], radius=cvrad * min_zoom)

        # Block G
        print_stats("Block G", x)
        x = upcat(x, x_a)
        min_zoom /= config.downsampling
        x = cbg(
            x,
            mul,
            filter=["0e", "1o", "2e"] if config.equivariance == "E3" else ["0e", "1e", "2e"],
            radius=cvrad * min_zoom,
        )

        x = group_conv(x, irreps="0e", mul=round(2 * mul), radius=cvrad * min_zoom)

        x = x.data.repeat_irreps_by_last_axis()  # [batch, x, y, z, irreps]

        print_stats("MLP", x)
        for h in [round(16 * mul), round(16 * mul), 1]:
            x = bn(x, config.instance_norm_eps)
            x = g(x)
            x = n_vmap(1 + 3, e3nn.Linear(f"{h}x0e"))(x)

        print_stats("output", x)

        return x.array[0, :, :, :, 0]

    return f

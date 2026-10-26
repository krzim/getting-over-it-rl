"""Running mean / variance as SB3's RunningMeanStd (count starts at 1e-4, parallel-variance merge), used the way
VecNormalize uses it: observations normalised and clipped to +-clip_obs; rewards divided by the std of a running
discounted return and clipped to +-clip_reward (no mean subtraction). Under pmap the batch moments are merged over
the device axis first, so every device holds the same statistics."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

EPS = 1e-8


class RMS(NamedTuple):
    mean: jax.Array
    var: jax.Array
    count: jax.Array


def rms_init(shape=()):
    return RMS(jnp.zeros(shape, jnp.float32), jnp.ones(shape, jnp.float32), jnp.asarray(1e-4, jnp.float32))


def rms_update(rms: RMS, x, axis_name=None):
    """x: (batch, *shape). axis_name: a pmap axis whose devices hold equal-size batches."""
    mean = jnp.mean(x, axis=0)
    sq = jnp.mean(x * x, axis=0)
    n = jnp.asarray(x.shape[0], jnp.float32)
    if axis_name is not None:
        mean, sq = jax.lax.pmean(mean, axis_name), jax.lax.pmean(sq, axis_name)
        n = jax.lax.psum(n, axis_name)
    var = jnp.maximum(sq - mean * mean, 0.0)
    delta = mean - rms.mean
    tot = rms.count + n
    new_mean = rms.mean + delta * n / tot
    m2 = rms.var * rms.count + var * n + delta * delta * rms.count * n / tot
    return RMS(new_mean, m2 / tot, tot)


def normalize_obs(rms: RMS, x, clip=10.0):
    return jnp.clip((x - rms.mean) / jnp.sqrt(rms.var + EPS), -clip, clip)


def scale_reward(rms: RMS, r, clip=10.0):
    return jnp.clip(r / jnp.sqrt(rms.var + EPS), -clip, clip)

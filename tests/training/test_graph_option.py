"""GOI_GPU_SPEED option "graph" (goi_jax/box2d/gpu/graph.py) changes no bit: the helpers on their own, and one PPO
device call (rollout with auto-resets, GAE, minibatch epochs) with the option off and on, every leaf compared.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/training/test_graph_option.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goi_jax.box2d.gpu import graph as G

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
ROUTE = Path(os.environ.get("GOI_ROUTE", ROOT / "dumps" / "obs" / "route.json"))


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape
    if a.dtype.kind == "f":
        return np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))
    return np.array_equal(a, b)


def test_opaque_set_rows_set_at_bit_exact(monkeypatch):
    f = np.array([0.0, -0.0, 1.5, np.inf, -np.inf, np.nan, 1e-45, -3.4e38], np.float32)
    f[5] = np.frombuffer(np.uint32(0x7FC00123).tobytes(), np.float32)[0]   # a NaN with a payload
    leaves = (jnp.asarray(f), jnp.arange(-3, 5, dtype=jnp.int32), jnp.asarray([True, False]), jnp.uint8(7))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(8, 3)).astype(np.float32))
    # a new function per use: jit's trace cache does not see G.ENABLED
    loop = lambda: jax.jit(lambda x_: jax.lax.fori_loop(0, 8, lambda k, o: G.set_at(o, k, jnp.where(k % 3 == 0, -k, k)), x_))  # noqa
    xi = jnp.full(8, 7, jnp.int32)
    plain = [G.set_rows(x, 2, 5.0), G.set_rows(x, [1, 4, 6], 0), G.set_rows(x, [0, 7], jnp.ones((2, 3)) * 2),
             G.set_rows(x[:, 0], 3, -0.0), loop()(xi), jax.jit(lambda x_: G.set_at(x_, jnp.int32(5), 1.5))(x)]
    monkeypatch.setattr(G, "ENABLED", True)
    z = jax.jit(G.runtime_zero)(jnp.int32(12345))
    assert int(z) == 0
    out = jax.jit(lambda z_, t: G.opaque(z_, t))(z, leaves)
    for a, b in zip(leaves, out):
        assert _same(a, b)
    on = [jax.jit(lambda x_: G.set_rows(x_, 2, 5.0))(x), G.set_rows(x, [1, 4, 6], 0),
          G.set_rows(x, [0, 7], jnp.ones((2, 3)) * 2), G.set_rows(x[:, 0], 3, -0.0), loop()(xi),
          jax.jit(lambda x_: G.set_at(x_, jnp.int32(5), 1.5))(x)]
    for a, b in zip(plain, on):
        assert _same(a, b)


@pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")
def test_ppo_call_identical_with_graph_option(monkeypatch):
    """One device call (2 updates, 4 envs x 4 steps, auto-resets by time limit) with the option off and on."""
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "runs" / ".jax-cache"))
    cfg = TrainConfig(name="pytest-graph", parity_dir=str(DATA), route=str(ROUTE), num_envs=4, num_steps=4,
                      minibatch_size=8, num_minibatches=0, epochs=2, updates_per_call=2, total_steps=64,
                      episode_steps=3)
    env = St.make_env(cfg)
    outs = []
    for on in (False, True):
        monkeypatch.setattr(G, "ENABLED", on)
        tr = ppo.Trainer(env, cfg, 1)          # a new trainer: its jitted call is traced with the option as set
        runner, metrics = tr.chunk(tr.init(jax.random.PRNGKey(0)))
        outs.append(jax.device_get((runner, metrics)))
    la = jax.tree_util.tree_leaves_with_path(outs[0])
    lb = jax.tree.leaves(outs[1])
    assert len(la) == len(lb)
    bad = [jax.tree_util.keystr(p) for (p, a), b in zip(la, lb) if not _same(a, b)]
    assert not bad, bad

"""goi_jax/game/rays_gpu.py (observe.raycast's ray tests as a Pallas kernel, interpret mode on CPU) against the XLA
raycast, bit for bit, on bank states moved to places of the level with many rows per grid cell, with lanes grouped
by row count (SORT_LANES) and several programs (blocks of 4).

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/game/test_rays_gpu.py -q
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goi_jax.game import observe as O

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def env():
    if not (ROOT / "dumps/parity").exists():
        pytest.skip("needs dumps/parity")
    from goi_jax.env import GoiEnv
    return GoiEnv.from_recordings(root=ROOT / "dumps/parity", route=ROOT / "dumps/obs/route.json",
                                  compact_fixtures=True, obs_config=O.ObsConfig(8, 15.0, 0, 4.0))


@pytest.mark.parametrize("sort", [False, True])
def test_rays_kernel_matches_xla(env, sort, monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("interpret-mode parity runs on CPU")
    from goi_jax.box2d.gpu import kernel as GK
    from goi_jax.game import rays_gpu
    p = env.default_params
    om = p.obs
    g = om.grid
    cnt = np.asarray(g.count)
    rng = np.random.default_rng(1)
    # origins: cells with the most rows, some with a few, some empty
    order = np.argsort(-cnt)
    cells = np.concatenate([order[:5], rng.choice(np.nonzero((cnt > 0) & (cnt < 30))[0], 4),
                            rng.choice(np.nonzero(cnt == 0)[0], 1)])
    shape = np.asarray(g.shape)
    lo, cell = np.asarray(g.lo), float(g.cell)
    xy = np.stack([lo[0] + (cells % shape[0] + rng.uniform(0.2, 0.8, len(cells))) * cell,
                   lo[1] + (cells // shape[0] + rng.uniform(0.2, 0.8, len(cells))) * cell], -1).astype(np.float32)
    n = len(cells)
    st = jax.tree.map(lambda x: x[np.arange(n) % x.shape[0]], p.starts).world
    dirs = O.ray_directions(8)
    monkeypatch.setattr(GK, "SORT_LANES", sort)
    monkeypatch.setattr(rays_gpu, "BLOCK", 4)

    def run(kernel):
        monkeypatch.setattr(O, "RAYS_KERNEL", kernel)
        f = jax.jit(jax.vmap(lambda s, o: O.raycast(om, s, o, dirs, np.float32(15.0))))
        return np.asarray(f(st, jnp.asarray(xy)))
    a, b = run(False), run(True)
    assert np.array_equal(a.view(np.int32), b.view(np.int32)), np.argwhere(a != b)[:10]
    assert (a < 15).any()  # some rays hit

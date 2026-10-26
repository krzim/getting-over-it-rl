"""Go-Explore start-state search (goi_jax/training/search.py) and the searched start group (starts.py).

The archive / pick / group tests are instant; the end-to-end test needs the recordings and compiles the env step on
CPU (~2 min, faster on reruns through runs/.jax-cache):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/training/test_search.py -q
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
ROUTE = Path(os.environ.get("GOI_ROUTE", ROOT / "dumps" / "obs" / "route.json"))
needs_data = pytest.mark.skipif(not ((DATA / "index.json").exists() and ROUTE.exists()),
                                reason=f"no recordings in {DATA} or no route row at {ROUTE}")


def _sc(**kw):
    from goi_jax.training import search as S
    return S.SearchConfig(**dict(dict(bin_size=2.0, angle_cells=2, max_progress=20.0, tol=0.5), **kw))


def test_archive_update_rules():
    import jax.numpy as jnp
    from goi_jax.training import search as S
    sc = _sc()
    ar = S.archive_init(sc, {"x": jnp.zeros(3)})
    st = lambda v: {"x": jnp.asarray(v, jnp.float32)[:, None] * jnp.ones((1, 3))}  # noqa: E731
    # two candidates for cell 2, one for cell 5, one not a candidate
    cell = jnp.array([2, 2, 5, 7])
    prog = jnp.array([2.2, 2.9, 5.0, 7.0])
    ar, new = S.archive_update(ar, sc, st([1, 2, 3, 4]), cell, prog, jnp.array([5, 9, 1, 1]), jnp.array([0, 0, 1, 1]),
                               jnp.array([True, True, True, False]), 3)
    assert int(new) == 2
    f = np.asarray(ar.filled)
    assert f[2] and f[5] and not f[7]
    assert float(ar.states["x"][2, 0]) == 2.0 and float(ar.progress[2]) == np.float32(2.9)   # the furthest wins
    assert int(ar.found[2]) == 3 and int(ar.time[2]) == 9
    # within tol and fewer actions: replaces; within tol and more actions: does not; much further: replaces
    ar, new = S.archive_update(ar, sc, st([10]), jnp.array([2]), jnp.array([2.6]), jnp.array([4]), jnp.array([0]),
                               jnp.array([True]))
    assert int(new) == 0 and float(ar.states["x"][2, 0]) == 10.0 and int(ar.time[2]) == 4
    ar, _ = S.archive_update(ar, sc, st([11]), jnp.array([2]), jnp.array([2.7]), jnp.array([8]), jnp.array([0]),
                             jnp.array([True]))
    assert float(ar.states["x"][2, 0]) == 10.0
    ar, _ = S.archive_update(ar, sc, st([12]), jnp.array([2]), jnp.array([3.5]), jnp.array([50]), jnp.array([0]),
                             jnp.array([True]))
    assert float(ar.states["x"][2, 0]) == 12.0 and int(ar.found[2]) == 3   # first-filled round kept


def test_cell_of_and_sections():
    import jax.numpy as jnp
    from goi_jax.training import search as S
    from goi_jax.training import starts as St
    sc = _sc(max_progress=300.0)
    c = np.asarray(S.cell_of(sc, jnp.array([0.1, 3.9, 299.0, 1000.0]), jnp.array([1.0, -1.0, 1.0, 1.0]),
                             jnp.array([0.0, 0.1, -0.1, 0.0])))
    assert list(c // 2) == [0, 1, 149, 149]          # bins (clipped)
    assert c[1] % 2 == 1 and c[2] % 2 == 0            # hammer pointing up-left / down-right
    sec = S.cell_section(sc)
    names = [n for _, n in St.SECTIONS]
    assert names[sec[0]] == "start" and names[sec[2 * 128]] == "garden"   # bin 128 = 256 m


def test_pick_prefers_frontier_and_rare():
    import jax.numpy as jnp
    from goi_jax.training import search as S
    sc = _sc(angle_cells=1, frontier_bonus=3.0, pick="cell")
    ar = S.archive_init(sc, {"x": jnp.zeros(1)})
    ar = ar._replace(filled=ar.filled.at[jnp.array([0, 1, 2])].set(True),
                     progress=ar.progress.at[jnp.array([0, 1, 2])].set(jnp.array([0.5, 2.5, 4.5])),
                     chosen=ar.chosen.at[1].set(8.0))
    lg = np.asarray(S.pick_logits(ar, sc, jnp.asarray(S.cell_section(sc)), -np.inf, np.inf))
    assert np.isneginf(lg[3:]).all()
    w = np.exp(lg[:3])
    np.testing.assert_allclose(w, [1.0, 1 / 3, 4.0], rtol=1e-5)   # cell 2 is the frontier, cell 1 often chosen
    lg = np.asarray(S.pick_logits(ar, sc, jnp.asarray(S.cell_section(sc)), 2.0, 4.0))
    assert np.isfinite(lg[1]) and np.isneginf(lg[[0, 2]]).all()   # window


def test_split_group():
    from goi_jax.training import starts as St
    assert St.split_group("all") == ("all", False)
    assert St.split_group("all+searched") == ("all", True)
    assert St.split_group("searched") == (None, True)
    with pytest.raises(ValueError):
        St.split_group("all+route")


@needs_data
def test_search_end_to_end_and_bank(tmp_path, monkeypatch):
    """A tiny search from the level start on CPU: the archive only holds valid states, its bank file round-trips and
    starts.make_env loads it as start_group start+searched."""
    import jax
    from goi_jax.training import search as S
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "runs" / ".jax-cache"))
    cfg = TrainConfig(parity_dir=str(DATA), route=str(ROUTE), start_group="start", compact_fixtures=True)
    env = St.make_env(cfg)
    sc = S.SearchConfig(branches=4, horizon=4, rounds=2, min_held=1, max_speed=50.0)
    se = S.Searcher(env, sc)
    bank = env.default_params.starts
    ar = se.seed(S.archive_init(sc, jax.tree.map(lambda a: a[0], bank)), jax.tree.map(np.asarray, bank))
    n0 = int(np.asarray(ar.filled).sum())
    assert n0 >= 1
    ar, st = se.run(ar, jax.random.PRNGKey(0), 0, S.no_policy(env.obs_size))
    st = jax.device_get(st)
    assert int(st["invalid"]) == 0 and np.isfinite(float(st["best"]))
    f = np.asarray(ar.filled)
    assert f.sum() >= n0 and np.asarray(ar.chosen)[f].sum() == 2 * sc.branches
    ok, _ = jax.vmap(lambda s: S.state_ok(s, env.sim.pot, np.inf))(jax.tree.map(lambda a: a[np.nonzero(f)[0]], ar.states))
    assert np.asarray(ok).all()
    path = tmp_path / "bank.pkl"
    b = S.save_bank(path, ar, sc, {"rounds": 2})
    ar2 = S.archive_from_bank(S.load_bank(path), sc)
    np.testing.assert_array_equal(np.asarray(ar2.filled), f)
    np.testing.assert_array_equal(np.asarray(ar2.states.world.bodies.xf_p)[f], np.asarray(ar.states.world.bodies.xf_p)[f])
    monkeypatch.setenv("GOI_SEARCHED_BANK", str(path))
    env2 = St.make_env(dataclasses.replace(cfg, start_group="start+searched"))
    n_bank = len(jax.tree.leaves(bank)[0])
    assert len(env2.bank_where) == n_bank + len(b["cell"])
    assert env2.bank_where[-1][0].startswith("searched:")

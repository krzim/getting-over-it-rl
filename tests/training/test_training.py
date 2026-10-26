"""The on-device PPO trainer (goi_jax/training) and the policy runner (goi_jax/play.py), on CPU with a tiny batch.

The network / optimiser / normaliser tests are instant; the trainer and play tests need the recordings (dumps/ is
gitignored) and compile the env step (~2 min each on CPU, faster on reruns through runs/.jax-cache):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/training -q
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


def test_nets_shapes_and_log_prob():
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    p = N.init_params(jax.random.PRNGKey(0), 105, 2, (256, 256))
    assert p["pi"][0][0].shape == (105, 256) and p["mu"][0].shape == (256, 2) and p["v"][0].shape == (256, 1)
    obs = jax.random.normal(jax.random.PRNGKey(1), (7, 105))
    mean, log_std, v = N.apply(p, obs)
    assert mean.shape == (7, 2) and v.shape == (7,) and log_std.shape == (2,)
    assert float(jnp.abs(mean).max()) < 0.5   # action head gain 0.01
    a = jnp.ones((7, 2))
    want = np.sum(-0.5 * (np.asarray(a - mean) / np.exp(np.asarray(log_std))) ** 2 - np.asarray(log_std)
                  - 0.5 * np.log(2 * np.pi), -1)
    np.testing.assert_allclose(np.asarray(N.log_prob(mean, log_std, a)), want, rtol=1e-5)
    bm, _, _ = N.apply(p, obs * 1e4, bounded_mean=True)
    assert float(jnp.abs(bm).max()) <= 1.0


def test_adam_minimises_quadratic():
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    params = {"x": jnp.array([3.0, -2.0])}
    st = N.adam_init(params)
    f = lambda q: jnp.sum(q["x"] ** 2)  # noqa: E731
    for _ in range(300):
        g = jax.grad(f)(params)
        params, st, _ = N.adam_step(params, g, st, 0.05, max_grad_norm=10.0)
    assert float(f(params)) < 1e-2


def test_running_stats_match_numpy():
    import jax.numpy as jnp
    from goi_jax.training import normalize as Nz
    rng = np.random.default_rng(0)
    xs = [rng.normal(3.0, 2.0, (50, 4)).astype(np.float32) for _ in range(5)]
    rms = Nz.rms_init((4,))
    for x in xs:
        rms = Nz.rms_update(rms, jnp.asarray(x))
    allx = np.concatenate(xs)
    np.testing.assert_allclose(np.asarray(rms.mean), allx.mean(0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(rms.var), allx.var(0), rtol=1e-3)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Two PPO updates (one device call) with 2 envs x 4 steps; checkpointed."""
    import jax
    from goi_jax.training import checkpoint as C
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "runs" / ".jax-cache"))
    cfg = TrainConfig(name="pytest", parity_dir=str(DATA), route=str(ROUTE), num_envs=2, num_steps=4,
                      minibatch_size=4, num_minibatches=0, epochs=2, updates_per_call=2, total_steps=16,
                      episode_steps=3)
    env = St.make_env(cfg)
    tr = ppo.Trainer(env, cfg, 1)
    runner = tr.init(jax.random.PRNGKey(0))
    runner, metrics = tr.chunk(runner)
    metrics = jax.device_get(metrics)
    path = tmp_path_factory.mktemp("ckpt") / "update_0000002.pkl"
    C.save(path, C.pack(cfg, tr.first(runner), 2, 16, 1.0))
    return dict(cfg=cfg, env=env, tr=tr, runner=runner, metrics=metrics, path=path)


@needs_data
def test_two_updates_finite(trained):
    m = trained["metrics"]
    assert m["update"].shape == (1, 2)
    assert list(m["update"][0]) == [1, 2]
    for k in ("pg_loss", "value_loss", "approx_kl", "entropy", "explained_var", "grad_norm"):
        assert np.isfinite(m[k]).all(), k
    assert (m["grad_steps"] > 0).all()
    # episode_steps = 3 with 4 steps per update: every env finishes one episode by time limit in each update
    assert m["episodes"][0, 0] >= 2 and m["ends_timeout"][0, 0] >= 2
    assert np.isfinite(m["win_best"]).all() and m["win_episodes"][0, -1] >= 4
    r = trained["runner"]
    assert np.isfinite(np.asarray(r.obs)).all()
    assert int(np.asarray(r.update)[0]) == 2


@needs_data
def test_checkpoint_round_trip(trained):
    import jax
    import jax.numpy as jnp
    from goi_jax.training import checkpoint as C
    from goi_jax.training import nets as N
    from goi_jax.training.config import TrainConfig
    ck = C.load(trained["path"])
    first = trained["tr"].first(trained["runner"])
    for a, b in zip(jax.tree.leaves(ck["params"]), jax.tree.leaves(first.params)):
        assert np.array_equal(a, b)
    assert np.array_equal(ck["obs_rms"].mean, first.obs_rms.mean)
    assert ck["update"] == 2 and ck["archive"] is None
    latest = C.load(Path(trained["path"]).parent / "latest.pkl")
    assert latest["archive"] is not None
    assert dataclasses.asdict(TrainConfig.from_dict(ck["config"])) == dataclasses.asdict(trained["cfg"])
    # resume: a Trainer re-initialised from the checkpoint holds the same learner state
    tr = trained["tr"]
    from goi_jax.training import starts as St
    r2 = tr.init(jax.random.PRNGKey(1), ck["params"], ck["opt"], ck["obs_rms"], ck["ret_rms"],
                 St.Archive(*latest["archive"]), ck["update"], ck["record"])
    o = jnp.zeros((1, trained["env"].obs_size))
    a1 = N.apply(jax.tree.map(lambda x: x[0], r2.params), o)[0]
    a2 = N.apply(jax.tree.map(jnp.asarray, ck["params"]), o)[0]
    assert np.array_equal(np.asarray(a1), np.asarray(a2))


@pytest.mark.parametrize("sections", [0, 5])
def test_policy_npz_round_trip(tmp_path, sections):
    """save_npz -> load: the same params, observation normaliser and config, read with allow_pickle=False."""
    import jax
    from goi_jax.training import checkpoint as C
    from goi_jax.training import nets as N
    from goi_jax.training import normalize as Nz
    from goi_jax.training.config import TrainConfig
    params = C.to_numpy(N.init_params(jax.random.PRNGKey(0), 105, sections=sections))
    rms = Nz.RMS(np.arange(105, dtype=np.float32), np.full(105, 2.0, np.float32), np.float32(7.0))
    ck = dict(config=TrainConfig(name="npz").to_dict(), params=params, obs_rms=rms)
    C.save_npz(tmp_path / "policy.npz", ck)
    back = C.load(tmp_path / "policy.npz")
    assert jax.tree.structure(back["params"]) == jax.tree.structure(params)
    for a, b in zip(jax.tree.leaves(back["params"]), jax.tree.leaves(params)):
        assert np.array_equal(a, b)
    for a, b in zip(back["obs_rms"], rms):
        assert np.array_equal(a, b)
    assert back["config"] == ck["config"] and back["opt"] is None


@needs_data
def test_play_dry_run_sim(trained, tmp_path):
    from goi_jax import play
    out = tmp_path / "play.json"
    rc = play.main([str(trained["path"]), "--backend", "sim", "--steps", "3", "--along", "80", "--out", str(out)])
    assert rc == 0
    import json
    rep = json.loads(out.read_text())
    ep = rep["episodes"][0]
    assert ep["steps"] == 3 and np.isfinite(ep["best"]) and 60 < ep["start"] < 120
    assert len(rep["curves"][ep["name"]]) == 4


@needs_data
def test_host_episode_matches_env(trained):
    """play.HostEpisode (the real-game bookkeeping) gives the env's observation tail and reward on a sim rollout."""
    import jax
    import jax.numpy as jnp
    from goi_jax import play
    cfg = dataclasses.replace(trained["cfg"], episode_steps=3000)
    sim = play.SimRunner(cfg)
    obs, st = sim.start(sim.bank_state(0))
    ep = play.HostEpisode(cfg, obs[:sim.env.cfg.size])
    np.testing.assert_array_equal(ep.obs(), obs)
    rng = np.random.default_rng(0)
    for _ in range(3):
        a = rng.uniform(-1, 1, 2).astype(np.float32)
        obs, st, r, term, trunc, snake = sim.step(st, a)
        r_h, _ = ep.step(a, obs[:sim.env.cfg.size], snake)
        np.testing.assert_array_equal(ep.obs(), obs)
        assert abs(r_h - r) < 1e-6

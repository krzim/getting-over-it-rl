"""Learning-speed options of goi_jax/training (TrainConfig flags, all off by default): log_std floor and reset,
entropy bonus, network widening / deepening warm starts, section-aware input, return statistics' gamma.

The network tests are instant; the trainer smoke tests (marked needs_data) compile the env step on CPU:
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/training/test_learning_options.py -q
"""
from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
ROUTE = Path(os.environ.get("GOI_ROUTE", ROOT / "dumps" / "obs" / "route.json"))
needs_data = pytest.mark.skipif(not ((DATA / "index.json").exists() and ROUTE.exists()),
                                reason=f"no recordings in {DATA} or no route row at {ROUTE}")
OBS = 105


def _trained_like(key, hidden=(256, 256)):
    """Random params with non-trivial weights everywhere (a fresh init has a 0.01-gain head)."""
    import jax
    from goi_jax.training import nets as N
    p = N.init_params(key, OBS, 2, hidden)
    leaves, tree = jax.tree.flatten(p)
    ks = jax.random.split(jax.random.fold_in(key, 1), len(leaves))
    leaves = [x + 0.3 * jax.random.normal(k, x.shape) for x, k in zip(leaves, ks)]
    return jax.tree.unflatten(tree, leaves)


def _obs(n=64, seed=3):
    import jax
    return 2.0 * jax.random.normal(jax.random.PRNGKey(seed), (n, OBS))


def test_widening_is_exact():
    import jax
    from goi_jax.training import nets as N
    p = _trained_like(jax.random.PRNGKey(0))
    w = N.widen_params(jax.random.PRNGKey(1), p, OBS, (512, 512))
    assert N.hidden_of(w) == (512, 512)
    x = _obs()
    m0, s0, v0 = N.apply(p, x)
    m1, s1, v1 = N.apply(w, x)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m0), atol=1e-5)
    np.testing.assert_allclose(np.asarray(v1), np.asarray(v0), atol=1e-4)
    assert np.array_equal(np.asarray(s0), np.asarray(s1))
    # every weight still gets a gradient (no dead block)
    g = jax.grad(lambda q: jnp_sum(N.apply(q, x)))(w)
    assert np.abs(np.asarray(g["pi"][0][0])[:OBS, :256]).sum() > 0
    assert np.abs(np.asarray(g["mu"][0])[256:]).sum() > 0          # new units' outgoing weights learn


def jnp_sum(out):
    m, s, v = out
    return m.sum() + v.sum()


def test_deepening_is_close():
    import jax
    from goi_jax.training import nets as N
    p = _trained_like(jax.random.PRNGKey(0))
    w = N.widen_params(jax.random.PRNGKey(1), p, OBS, (512, 512, 512), deepen_scale=0.2)
    assert N.hidden_of(w) == (512, 512, 512)
    x = _obs()
    m0, _, v0 = N.apply(p, x)
    m1, _, v1 = N.apply(w, x)
    # |tanh(s h)/s - h| <= s^2/3 per unit; the head sums 256 of them
    hw = np.abs(np.asarray(p["mu"][0])).sum(0)
    assert (np.abs(np.asarray(m1 - m0)) <= hw * 0.2 ** 2 / 3 + 1e-4).all()
    rel = np.abs(np.asarray(m1 - m0)).mean() / np.abs(np.asarray(m0)).mean()
    assert rel < 0.05, rel


def test_cannot_shrink():
    import jax
    from goi_jax.training import nets as N
    p = N.init_params(jax.random.PRNGKey(0), OBS, 2, (256, 256))
    with pytest.raises(ValueError):
        N.widen_params(jax.random.PRNGKey(1), p, OBS, (128, 256))


def test_section_input_zero_init_and_required():
    import jax
    import jax.numpy as jnp
    from goi_jax import env as E
    from goi_jax.training import nets as N
    from goi_jax.training import starts as St
    p = _trained_like(jax.random.PRNGKey(0))
    cfg = _cfg(section_input=True)
    q, opt, notes = N.warm_start(jax.random.PRNGKey(2), p, N.adam_init(p), cfg, OBS, sections=St.NS)
    assert q["pi_sec"].shape == (St.NS, 256) and any("section" in n for n in notes)
    raw = _obs()
    raw = raw.at[:, E.F["progress"]].set(jnp.linspace(0, 870, raw.shape[0]))
    sec = N.section_of(q, raw)
    assert sec.dtype == jnp.int32 and int(sec.min()) == 0 and int(sec.max()) == St.NS - 1
    assert N.section_of(p, raw) is None
    m0, _, v0 = N.apply(p, raw)
    m1, _, v1 = N.apply(q, raw, sec=sec)
    np.testing.assert_allclose(np.asarray(m1), np.asarray(m0), atol=1e-6)
    with pytest.raises(ValueError):
        N.apply(q, raw)
    # the section weights learn
    g = jax.grad(lambda z: jnp_sum(N.apply(z, raw, sec=sec)))(q)
    assert np.abs(np.asarray(g["pi_sec"])).sum() > 0 and np.abs(np.asarray(g["vf_sec"])).sum() > 0
    # opt tree matches the params tree
    assert jax.tree.structure(opt.m) == jax.tree.structure(q)


def _cfg(**kw):
    from goi_jax.training.config import TrainConfig
    return dataclasses.replace(TrainConfig(), **kw)


def test_warm_start_log_std_reset_and_floor():
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    p = N.init_params(jax.random.PRNGKey(0), OBS, 2, (256, 256), log_std_init=math.log(0.06))
    opt = N.adam_init(p)._replace(lr_scale=jnp.float32(0.3))
    cfg = _cfg(log_std_reset=math.log(0.3))
    q, o2, notes = N.warm_start(jax.random.PRNGKey(1), p, opt, cfg, OBS, same_run=False)
    np.testing.assert_allclose(np.exp(np.asarray(q["log_std"])), 0.3, rtol=1e-6)
    assert o2 is opt   # no architecture change: Adam state kept
    q, _, _ = N.warm_start(jax.random.PRNGKey(1), p, opt, cfg, OBS, same_run=True)   # a later chunk: no reset
    np.testing.assert_allclose(np.exp(np.asarray(q["log_std"])), 0.06, rtol=1e-5)
    q, _, _ = N.warm_start(jax.random.PRNGKey(1), p, opt, _cfg(min_log_std=math.log(0.15)), OBS)
    np.testing.assert_allclose(np.exp(np.asarray(q["log_std"])), 0.15, rtol=1e-6)
    # widening keeps Adam's moments for the old weights and the KL-adaptive LR scale
    q, o3, _ = N.warm_start(jax.random.PRNGKey(1), p, opt, _cfg(hidden=(512, 512)), OBS)
    assert int(o3.count) == int(opt.count) and float(o3.lr_scale) == pytest.approx(0.3)
    assert jax.tree.structure(o3.m) == jax.tree.structure(q)
    # adding a layer restarts them (count 0), still keeping the LR scale
    q, o4, _ = N.warm_start(jax.random.PRNGKey(1), p, opt, _cfg(hidden=(512, 512, 512)), OBS)
    assert int(o4.count) == 0 and float(o4.lr_scale) == pytest.approx(0.3)
    assert jax.tree.structure(o4.m) == jax.tree.structure(q)


def test_warm_start_keeps_adam_moments_when_exact():
    """Widening / section input: old weights keep their Adam moments and count (a restart makes the first steps
    ~10x larger than a trained run's, |m|/sqrt(v) ~0.08 in route-fast), new weights start at zero moments."""
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    from goi_jax.training import starts as St
    p = _trained_like(jax.random.PRNGKey(0))
    rnd = lambda k: jax.tree.map(lambda a: jnp.abs(jax.random.normal(jax.random.PRNGKey(k), a.shape)) + 0.1, p)  # noqa
    opt = N.AdamState(rnd(1), rnd(2), jnp.int32(5000), jnp.float32(0.4))
    q, o, notes = N.warm_start(jax.random.PRNGKey(3), p, opt, _cfg(hidden=(512, 512), section_input=True), OBS,
                               sections=St.NS)
    assert int(o.count) == 5000 and float(o.lr_scale) == pytest.approx(0.4)
    assert jax.tree.structure(o.m) == jax.tree.structure(q) and jax.tree.structure(o.v) == jax.tree.structure(q)
    np.testing.assert_array_equal(np.asarray(o.m["pi"][0][0])[:, :256], np.asarray(opt.m["pi"][0][0]))
    np.testing.assert_array_equal(np.asarray(o.v["pi"][1][0])[:256, :256], np.asarray(opt.v["pi"][1][0]))
    assert not np.asarray(o.v["pi"][1][0])[256:].any() and not np.asarray(o.m["mu"][0])[256:].any()
    np.testing.assert_array_equal(np.asarray(o.m["mu"][0])[:256], np.asarray(opt.m["mu"][0]))
    assert not np.asarray(o.m["pi_sec"]).any() and o.v["vf_sec"].shape == (St.NS, 512)
    # one Adam step on the warm-started net: old weights move exactly as in the old net (same grads there: the new
    # units' outgoing weights are zero, so the old block's gradients are unchanged)
    x = _obs()
    sec = N.section_of(q, x.at[:, 0].set(0.0))
    g0 = jax.grad(lambda z: jnp_sum(N.apply(z, x)))(p)
    g1 = jax.grad(lambda z: jnp_sum(N.apply(z, x, sec=sec)))(q)
    p0, _, _ = N.adam_step(p, g0, opt, 1e-3, max_grad_norm=1e9)
    p1, _, _ = N.adam_step(q, g1, o, 1e-3, max_grad_norm=1e9)
    np.testing.assert_allclose(np.asarray(p1["pi"][0][0])[:, :256], np.asarray(p0["pi"][0][0]), atol=1e-6)


def test_deepening_default_scale_keeps_the_policy():
    """The default deepen_scale keeps a low-std policy's KL to the original below target_kl (0.02) even for these
    heavy random weights (route-fast update 704's own net: KL 0.0008; at the old 0.2: 0.19)."""
    import jax
    from goi_jax.training import nets as N
    from goi_jax.training.config import TrainConfig
    p = _trained_like(jax.random.PRNGKey(0))
    p = dict(p, log_std=np.log(np.array([0.02, 0.05], np.float32)))
    w = N.widen_params(jax.random.PRNGKey(1), p, OBS, (512, 512, 512), TrainConfig().deepen_scale)
    x = jax.random.normal(jax.random.PRNGKey(4), (512, OBS))
    m0, ls, _ = N.apply(p, x)
    m1, _, _ = N.apply(w, x)
    kl = 0.5 * ((np.asarray(m1 - m0) / np.exp(np.asarray(ls))) ** 2).sum(-1)
    assert kl.mean() < 0.02, kl.mean()


def test_floor_holds_against_gradient():
    """A loss that pushes log_std down: with the floor applied after each Adam step it never goes below."""
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    p = N.init_params(jax.random.PRNGKey(0), 4, 2, (8,), log_std_init=-1.0)
    st = N.adam_init(p)
    floor = math.log(0.2)
    for _ in range(300):
        g = jax.grad(lambda q: jnp.sum(q["log_std"]))(p)
        p, st, _ = N.adam_step(p, g, st, 0.05)
        p = N.floor_log_std(p, floor)
    np.testing.assert_allclose(np.asarray(p["log_std"]), floor, atol=1e-6)


def test_entropy_bonus_raises_log_std():
    """ent_coef > 0: d(-ent_coef * entropy)/d log_std = -ent_coef per action dim, so Adam raises log_std."""
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    p = N.init_params(jax.random.PRNGKey(0), 4, 2, (8,), log_std_init=-2.0)
    g = jax.grad(lambda q: -0.005 * N.entropy(q["log_std"]))(p)
    np.testing.assert_allclose(np.asarray(g["log_std"]), -0.005)
    st = N.adam_init(p)
    p2, _, _ = N.adam_step(p, g, st, 1e-3)
    assert (np.asarray(p2["log_std"]) > -2.0).all()


# ---------------------------------------------------------------- trainer smoke tests (CPU, tiny)


def _tiny(**kw):
    from goi_jax.training.config import TrainConfig
    return TrainConfig(name="pytest-opts", parity_dir=str(DATA), route=str(ROUTE), num_envs=2, num_steps=4,
                       minibatch_size=4, num_minibatches=0, epochs=2, updates_per_call=2, total_steps=16,
                       episode_steps=3, **kw)


@needs_data
def test_trainer_with_all_options(tmp_path):
    """One device call with every option on (section input, 2x64 -> 3x96 warm start, floor, entropy, gamma):
    finite, floor held, ret_rms discounted with cfg.gamma, checkpoint plays through play.py."""
    import jax
    import jax.numpy as jnp
    from goi_jax import play
    from goi_jax.training import checkpoint as C
    from goi_jax.training import nets as N
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    jax.config.update("jax_compilation_cache_dir", str(ROOT / "runs" / ".jax-cache"))
    base = _tiny(hidden=(64, 64))
    env = St.make_env(base)
    tr0 = ppo.Trainer(env, base, 1)
    r0 = tr0.init(jax.random.PRNGKey(0))
    first0 = tr0.first(r0)
    cfg = dataclasses.replace(base, hidden=(96, 96, 96), section_input=True, min_log_std=math.log(0.5),
                              log_std_reset=math.log(0.4), ent_coef=0.01, gamma=0.9)
    params, opt, notes = N.warm_start(jax.random.PRNGKey(5), first0.params, first0.opt, cfg, env.obs_size,
                                      same_run=False, sections=St.NS)
    assert len(notes) >= 3
    tr = ppo.Trainer(env, cfg, 1)
    r = tr.init(jax.random.PRNGKey(1), params, opt, first0.obs_rms, first0.ret_rms)
    ret_before = np.asarray(r.ret)[0]
    r, m = tr.chunk(r)
    m = jax.device_get(m)
    for k in ("pg_loss", "value_loss", "approx_kl", "entropy", "explained_var"):
        assert np.isfinite(m[k]).all(), k
    assert (m["std"] >= 0.5 - 1e-6).all()
    first = tr.first(r)
    assert first.params["pi_sec"].shape == (St.NS, 96) and len(first.params["pi"]) == 3
    assert np.isfinite(ret_before).all()
    path = tmp_path / "ck" / "update_0000002.pkl"
    C.save(path, C.pack(cfg, first, 2, 16, 1.0))
    out = tmp_path / "play.json"
    rc = play.main([str(path), "--backend", "sim", "--steps", "3", "--along", "80", "--out", str(out)])
    assert rc == 0


def test_ret_rms_uses_cfg_gamma():
    """ppo.py discounts the reward-normalisation return with cfg.gamma (VecNormalize's gamma = PPO's gamma)."""
    import inspect
    from goi_jax.training import ppo
    src = inspect.getsource(ppo.Trainer._env_step)
    assert "ret = r.ret * cfg.gamma + rew" in src

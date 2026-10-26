"""Policy distillation in the PPO trainer (TrainConfig.teachers / distill_coef; goi_jax/training/ppo.py): the KL term,
teacher selection by route progress, a tiny training run that pulls a student toward a teacher, and the no-teacher
program left as it was.

The KL / selection tests are instant; the trainer tests need the recordings and compile the env step on CPU:
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/training/test_distill.py -q
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
ROUTE = Path(os.environ.get("GOI_ROUTE", ROOT / "dumps" / "obs" / "route.json"))
needs_data = pytest.mark.skipif(not ((DATA / "index.json").exists() and ROUTE.exists()),
                                reason=f"no recordings in {DATA} or no route row at {ROUTE}")
OBS = 105


def _teacher(key, log_std, head_gain=3.0, hidden=(64, 64), lo=0.0, rms_shift=0.0):
    """A teacher dict as ppo.load_teachers makes it, with a non-trivial action head and its own obs statistics."""
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    from goi_jax.training import normalize as Nz
    p = N.init_params(key, OBS, 2, hidden, log_std)
    w, b = p["mu"]
    p["mu"] = (w * (head_gain / 0.01), b + jax.random.normal(jax.random.fold_in(key, 1), b.shape))
    rms = Nz.RMS(jnp.full(OBS, rms_shift, jnp.float32), jnp.full(OBS, 1.0 + abs(rms_shift), jnp.float32),
                 jnp.float32(100.0))
    return dict(path=f"mem{lo:g}", lo=lo, params=p, obs_rms=rms, bounded_mean=True, norm_obs=True, clip_obs=10.0)


def test_kl_zero_for_the_student_itself_and_matches_closed_form():
    import jax
    import jax.numpy as jnp
    from goi_jax.training import ppo
    m = jax.random.normal(jax.random.PRNGKey(0), (50, 2))
    ls = jnp.array([-2.3, -0.4])
    assert float(jnp.max(jnp.abs(ppo.gauss_kl(m, jnp.broadcast_to(ls, m.shape), m, ls)))) == 0.0
    # 1-D closed form against numpy, and KL >= 0
    mt, st, ms, ss = 0.3, 0.2, -0.1, 0.5
    want = np.log(ss / st) + (st ** 2 + (mt - ms) ** 2) / (2 * ss ** 2) - 0.5
    got = ppo.gauss_kl(jnp.array([mt]), jnp.log(jnp.array([st])), jnp.array([ms]), jnp.log(jnp.array([ss])))
    np.testing.assert_allclose(float(got), want, rtol=1e-5)
    k = jax.random.split(jax.random.PRNGKey(1), 4)
    kl = ppo.gauss_kl(jax.random.normal(k[0], (100, 2)), jax.random.normal(k[1], (100, 2)),
                      jax.random.normal(k[2], (100, 2)), jax.random.normal(k[3], (2,)))
    assert float(kl.min()) >= 0.0


def test_loss_distill_term_is_zero_against_itself():
    """The trainer's loss with the student's own outputs as the teacher: distill exactly 0, loss unchanged."""
    import jax
    import jax.numpy as jnp
    from goi_jax.training import nets as N
    from goi_jax.training import ppo
    from goi_jax.training.config import TrainConfig
    cfg = TrainConfig(hidden=(64, 64), bounded_mean=True)
    tr = SimpleNamespace(cfg=cfg, mb_global=32)
    p = _teacher(jax.random.PRNGKey(3), -1.0)["params"]
    k = jax.random.split(jax.random.PRNGKey(4), 4)
    obs = jax.random.normal(k[0], (32, OBS))
    mean, log_std, _ = N.apply(p, obs, True)
    mb = dict(obs=obs, action=mean + 0.3 * jax.random.normal(k[1], mean.shape), logp=jax.random.normal(k[2], (32,)),
              adv=jax.random.normal(k[3], (32,)), ret=jnp.zeros(32))
    loss_fn = lambda q, b, c=None: ppo.Trainer._loss(tr, q, b, c)  # noqa: E731
    run = lambda b, c=None: jax.vmap(lambda q: loss_fn(q, b, c), axis_name=ppo.DA)(  # noqa: E731
        jax.tree.map(lambda a: a[None], p))
    base, aux0 = run(mb)
    assert set(aux0) == {"pg", "vl", "ent", "kl", "clipfrac"}
    mb_t = dict(mb, t_mean=mean, t_log_std=jnp.broadcast_to(log_std, mean.shape))
    with_t, aux1 = run(mb_t, jnp.float32(5.0))
    assert float(aux1["distill"][0]) == 0.0
    assert float(with_t[0]) == float(base[0])


def test_teacher_selection_by_progress():
    """parse order does not matter; the teacher with the largest lo <= progress acts (the lowest below every lo),
    each evaluated on the raw observation normalised with its own statistics."""
    import jax
    import jax.numpy as jnp
    from goi_jax import env as E
    from goi_jax.training import nets as N
    from goi_jax.training import normalize as Nz
    from goi_jax.training import ppo
    assert ppo.parse_teachers("b.pkl:385, /x:y/a.pkl:0,c.pkl:100") == [("/x:y/a.pkl", 0.0), ("c.pkl", 100.0),
                                                                     ("b.pkl", 385.0)]
    with pytest.raises(ValueError):
        ppo.parse_teachers("a.pkl:0,b.pkl:0")
    ts = [_teacher(jax.random.PRNGKey(10 + i), ls, lo=lo, rms_shift=0.5 * i)
          for i, (lo, ls) in enumerate(((0.0, -2.3), (100.0, -1.0), (385.0, 0.2)))]
    tp = tuple(ppo.Teacher(t["params"], t["obs_rms"]) for t in ts)
    prog = jnp.array([-5.0, 0.0, 99.99, 100.0, 384.9, 385.0, 900.0])
    raw = 2.0 * jax.random.normal(jax.random.PRNGKey(5), (prog.shape[0], OBS))
    raw = raw.at[:, E.F["progress"]].set(prog)
    m, ls, idx = ppo.Trainer._teacher(SimpleNamespace(teachers=ts), tp, raw)
    want = np.array([0, 0, 0, 1, 1, 2, 2])
    assert np.array_equal(np.asarray(idx), want)
    for j, i in enumerate(want):
        t = ts[i]
        mi, lsi, _ = N.apply(t["params"], Nz.normalize_obs(t["obs_rms"], raw[j:j + 1], 10.0), True)
        np.testing.assert_allclose(np.asarray(m[j]), np.asarray(mi[0]), atol=1e-6)
        np.testing.assert_allclose(np.asarray(ls[j]), np.asarray(lsi))
    # the three teachers really differ, so a wrong pick would show
    assert len({tuple(np.round(np.asarray(ls[j]), 3)) for j in (0, 3, 5)}) == 3


def test_load_teachers_from_checkpoints(tmp_path):
    """load_teachers reads each checkpoint's params, obs_rms and network options (sorted by lo)."""
    import jax
    from goi_jax.training import checkpoint as C
    from goi_jax.training import ppo
    from goi_jax.training.config import TrainConfig
    paths = []
    for i, lo in enumerate((385.0, 0.0)):
        t = _teacher(jax.random.PRNGKey(i), -1.0, lo=lo, rms_shift=float(i))
        cfg = TrainConfig(name=f"t{i}", hidden=(64, 64), bounded_mean=bool(i), clip_obs=5.0 + i)
        ck = dict(config=cfg.to_dict(), params=C.to_numpy(t["params"]), opt=None, obs_rms=tuple(t["obs_rms"]),
                  ret_rms=(0.0, 1.0, 1e-4), update=0)
        paths.append(tmp_path / f"t{i}.pkl")
        C.save(paths[-1], ck, keep_latest=False)
    ts = ppo.load_teachers(f"{paths[0]}:385,{paths[1]}:0")
    assert [t["lo"] for t in ts] == [0.0, 385.0] and [t["path"] for t in ts] == [str(paths[1]), str(paths[0])]
    assert ts[0]["bounded_mean"] and not ts[1]["bounded_mean"] and ts[0]["clip_obs"] == 6.0
    np.testing.assert_allclose(ts[0]["obs_rms"].mean, 1.0)


# ---------------------------------------------------------------- trainer (CPU, tiny)


def _tiny(**kw):
    from goi_jax.training.config import TrainConfig
    return TrainConfig(name="pytest-distill", parity_dir=str(DATA), route=str(ROUTE), num_envs=4, num_steps=8,
                       minibatch_size=16, num_minibatches=0, epochs=4, updates_per_call=4, total_steps=10_000,
                       episode_steps=6, hidden=(64, 64), bounded_mean=True, lr=1e-2, target_kl=None, kl_adapt=False,
                       **kw)


@needs_data
def test_distill_training_pulls_student_to_teacher():
    """A random-init student and one teacher (different init, large head, log_std -0.7 vs the student's 0): with
    distill_coef 1 the update reports distill_kl / distill_loss / distill_coef, and both fall over 8 updates; the
    coefficient follows its linear schedule."""
    import jax
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    t = _teacher(jax.random.PRNGKey(42), -0.7, head_gain=1.0, lo=0.0)
    cfg = _tiny(teachers="mem:0", distill_coef=1.0, distill_coef_end=0.5, distill_updates=8, distill_start=0)
    env = St.make_env(cfg)
    tr = ppo.Trainer(env, cfg, 1, teachers=[t])
    assert len(tr.device_args()) == 2
    r = tr.init(jax.random.PRNGKey(0))
    ms = []
    for _ in range(2):
        r, m = tr.chunk(r)
        ms.append(jax.device_get(m))
    m = {k: np.concatenate([x[k][0] for x in ms]) for k in ms[0]}
    for k in ("distill_kl", "distill_loss", "distill_coef", "distill_kl_0", "distill_share_0", "pg_loss"):
        assert np.isfinite(m[k]).all(), k
    np.testing.assert_allclose(m["distill_share_0"], 1.0)
    np.testing.assert_allclose(m["distill_kl_0"], m["distill_kl"], rtol=1e-5)
    np.testing.assert_allclose(m["distill_coef"], 1.0 - 0.5 * np.arange(8) / 8, rtol=1e-6)
    kl, dl = m["distill_kl"], m["distill_loss"]
    assert kl[0] > 0.1, kl
    assert kl[-2:].mean() < 0.6 * kl[:2].mean() and dl[-2:].mean() < 0.6 * dl[:2].mean(), (kl, dl)


@needs_data
def test_no_teachers_program_unchanged():
    """teachers "": the rollout's Transition has no teacher leaves, the device call takes no teacher argument and the
    metrics have no distill keys (checked once by hand when distillation was added: the lowered program of a
    no-teacher trainer was byte-identical to the one before)."""
    import jax
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    cfg = _tiny()
    env = St.make_env(cfg)
    tr = ppo.Trainer(env, cfg, 1)
    assert tr.teachers == [] and len(tr.device_args()) == 1
    r = tr.init(jax.random.PRNGKey(0))
    step = jax.pmap(lambda r_, p_, k_: tr._env_step(r_, p_, k_), axis_name=ppo.DA)
    _, (trans, ep) = jax.eval_shape(step, r, *tr.device_args(), jax.random.split(jax.random.PRNGKey(1), 1))
    assert trans.t_mean is None and trans.t_log_std is None
    assert len(jax.tree.leaves(trans)) == 6
    assert not any(k.startswith("dkl") for k in ep)
    upd = jax.pmap(lambda r_, p_: tr._update(r_, p_), axis_name=ppo.DA)
    _, metrics = jax.eval_shape(upd, r, *tr.device_args())
    assert not any(k.startswith("distill") for k in metrics)

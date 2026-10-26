"""The multi-step simulator (goi_jax/game/sim.py), the gymnax-style env (goi_jax/env.py) and the horizon report
(goi_jax/parity/horizon.py) against the game's recordings.

Skipped when the recordings are absent (dumps/ is gitignored). Each test compiles one jitted function (about 1-2
minutes each on CPU):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/env -q
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
ROUTE = Path(os.environ.get("GOI_ROUTE", ROOT / "dumps" / "obs" / "route.json"))
pytestmark = pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")


@pytest.fixture(scope="module")
def sim():
    from goi_jax.game import sim as SM
    return SM.Sim.from_recordings(DATA)


def test_sim_k1_matches_single_step_harness(sim):
    """One physics step of the sim from recorded pre-step states (controller, materials, crane and cursor moves all
    computed by the sim) equals the single-step harness (recorded motor speeds, materials and MovePosition targets)
    bit for bit, and its script outputs equal the recording: motor speeds, slide flags, tracked colliders,
    HammerReturn (c01 step 12) and the snap target the next step consumes."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import level as L
    from goi_jax.parity import harness as H
    from goi_jax.parity import states as S
    chain = S.load_chain(DATA / "coffee" / "c01-random-62.npz")
    steps = np.arange(1, 25)
    lv = sim.level
    states = [sim.state_from_recording(chain, t) for t in steps]
    st = jax.tree.map(lambda *a: np.stack(a), *states)
    jm, _, _ = lv.native_states(chain, steps, 0)
    model = lv.model_for(jm)  # the re-created joints' anchors move in the last digits between steps
    f = jax.jit(jax.vmap(lambda md, s, a: sim.physics_step(md, sim.params, s, a, axis_name="w"),
                         in_axes=(L.model_axes(), 0, 0), axis_name="w"))
    out, info = f(model, st, jnp.asarray(chain.action[steps]))
    out, info = jax.tree.map(np.asarray, (out, info))

    settings = H.engine_settings()
    jm2, hs, tgt, mask, _ = H.prepare(lv, chain, steps, settings)
    ref = H.Runner(lv, settings)(jm2, hs, tgt, mask)
    for name in ("xf_p", "xf_q", "c", "a", "v", "w"):
        assert np.array_equal(getattr(out.world.bodies, name), getattr(ref.bodies, name)), name
    for name in ("normal_impulse", "tangent_impulse", "friction"):
        assert np.array_equal(getattr(out.world.contacts, name), getattr(ref.contacts, name), equal_nan=True), name
    for name in ("impulse", "motor_impulse", "lower_impulse", "upper_impulse"):
        assert np.array_equal(getattr(out.world.joints, name), getattr(ref.joints, name)), name

    s1, s2 = chain.scr[1], chain.scr[2]
    assert np.array_equal(info.hinge_speed, s1["hj.motorSpeed"][steps])
    assert np.array_equal(info.slider_speed, s1["sj.motorSpeed"][steps])
    assert np.array_equal(out.fric.hammer_slide, s2["hammerSlide"][steps] > 0)
    assert np.array_equal(out.fric.pot_slide, s2["potSlide"][steps] > 0)
    for i, t in enumerate(steps):
        assert set(np.flatnonzero(out.fric.hammer_tracked[i])) == set(chain.hammer_keys[2][t].tolist())
        assert set(np.flatnonzero(out.fric.pot_tracked[i])) == set(chain.pot_keys[2][t].tolist())
    ev = chain.events
    snaps = ev[(ev["kind"] == S.EV_HAMMER_RETURN) & (ev["script"] == 1) & ((ev["flags"] & 1) == 1)]["step"]
    assert np.array_equal(info.snapped, np.isin(steps, snaps))
    assert info.snapped.any()
    # the snap target is the cursor's MovePosition target at the next step (recorded phase-2 move): the game reads
    # the Tip Transform, which differs from the body origin by the hierarchy round trip (~1 ulp)
    from goi_jax.game import unity
    tgt_next, mask_next = unity.move_targets(chain, steps + 1, lv)
    for i in np.flatnonzero(info.snapped):
        assert mask_next[i, sim.cursor]
        want = tgt_next[i, sim.cursor]
        assert np.all(np.abs(out.snap_target[i] - want) <= 2 * np.spacing(np.abs(want).max()))


def test_env_jit_vmap_smoke(sim):
    """reset / step under jit + vmap: fixed shapes, finite observations, rewards, auto-reset after max_steps."""
    if not ROUTE.exists():
        pytest.skip(f"no route row at {ROUTE}")
    import jax
    from goi_jax.env import GoiEnv
    env = GoiEnv.from_recordings(DATA, ROUTE, k=2, max_steps=2, axis_name="w")
    p = env.default_params
    reset = jax.jit(jax.vmap(env.reset, (0, None), axis_name="w"))
    step = jax.jit(jax.vmap(env.step, (0, 0, 0, None), axis_name="w"))
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    obs, st = reset(keys, p)
    assert obs.shape == (3, env.obs_size) and obs.dtype == np.float32
    assert np.isfinite(np.asarray(obs)).all()
    dones = []
    for i in range(3):
        keys = jax.random.split(keys[0], 3)
        a = jax.random.uniform(keys[1], (3, 2), minval=-1, maxval=1)
        obs, st, r, d, info = step(keys, st, a, p)
        assert obs.shape == (3, env.obs_size)
        assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(r)).all()
        dones.append(np.asarray(d))
    assert not dones[0].any() and dones[1].all() and not dones[2].any()  # max_steps = 2
    assert np.asarray(info["truncated"]).dtype == bool
    assert (np.asarray(st.t) == 1).all()


def test_horizon_runs():
    """The horizon report runs; with the recorded pose round trip injected (oracle), c01's first 30 steps stay
    within 1e-5 of the game."""
    from goi_jax.parity import horizon as Hz
    res = Hz.run(DATA, ["c01"], progress=False, max_actions=30, modes=("sim", "oracle"))
    text = Hz.format_report(res)
    assert "oracle" in text and "1e-05" in text and res["k"] == 1
    oracle = res["mode"] == "oracle"
    assert oracle.sum() == 1 and (~oracle).sum() == 1
    assert res["first"][1e-5][oracle][0] >= res["n"][oracle][0]  # never above 1e-5
    assert res["first_flip"][oracle][0] >= res["n"][oracle][0]
    assert (np.concatenate(res["err"]) < 1e-3).all()

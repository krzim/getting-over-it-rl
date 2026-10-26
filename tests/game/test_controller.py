"""The JAX controller and friction switch against the game's own per-step recordings (goi_jax.parity.record).

Skipped when the recordings are absent (dumps/ is gitignored). Record them with
    .venv/bin/python -m goi_jax.parity.record --out dumps/parity        (unsandboxed: it launches the game)
Run with
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/game -q
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
pytestmark = pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")

f32 = np.float32


@pytest.fixture(scope="module")
def chains():
    from goi_jax.parity import states
    return [c for _, c in states.iter_chains(DATA)]


@pytest.fixture(scope="module")
def row():
    from goi_jax.parity import states
    return states.controller_row(DATA)


def _inputs(c):
    """The controller's inputs per step: the state before the step (SCR0), the action, and the slider reaction torque
    as the controller read it (SCR1: Unity re-syncs the re-created joints before the controller reads it; SCR0's
    earlier read is a stale cache)."""
    s0, s1 = c.scr[0], c.scr[1]
    v = lambda *n: np.stack([s0[k] for k in n], -1)  # noqa: E731
    return dict(action=c.action, cursor=v("cursor.x", "cursor.y"), pot=v("P.x", "P.y", "P.z"), tip=v("T.x", "T.y", "T.z"),
                hub=v("H.x", "H.y", "H.z"), joint_angle=s0["hj.jointAngle"], reference_angle=s0["hj.referenceAngle"],
                reaction_torque=s1["sj.reactionTorque"],
                mouse_input=v("mouseInput.x", "mouseInput.y"), old_mouse=v("oldMouse.x", "oldMouse.y"),
                mva=s0["mouseVelocityAverage"], mw=v("mw.x", "mw.y"), mouse_snap=s0["mouseSnap"] > 0,
                pause_timer=s0["pauseInputTimer"], inputs_to_skip=s0["inputsToSkip"].astype(np.int32),
                old_angle=s0["oldAngle"])


def _expected(c):
    s1 = c.scr[1]
    v = lambda *n: np.stack([s1[k] for k in n], -1)  # noqa: E731
    return dict(hinge=s1["hj.motorSpeed"], slider=s1["sj.motorSpeed"], mouse_input=v("mouseInput.x", "mouseInput.y"),
                mva=s1["mouseVelocityAverage"], mw=v("mw.x", "mw.y"), target=c.cursor_target,
                velocity=v("cursor.vx", "cursor.vy"), moved=c.cursor_moves > 0)


def _cat(dicts):
    return {k: np.concatenate([d[k] for d in dicts]) for k in dicts[0]}


def test_curve_matches_unity(row):
    import jax
    from goi_jax.game import player
    for name in ("mouseCurve", "trackpadCurve", "stickCurve"):
        c = player.curve_from_row(row[name])
        xs, ys = np.array(row[name]["sampleX"], f32), np.array(row[name]["sampleY"], f32)
        got = np.asarray(jax.jit(jax.vmap(player.curve_eval, (None, 0)))(c, xs))
        assert np.array_equal(got, ys), f"{name}: {(got != ys).sum()} of {len(xs)} samples differ"


def test_q7_host_reference_float_model(chains, row):
    """Q7: Mono evaluates the controller's float expressions with double intermediates. The host oracle in that model
    matches every recorded output bit for bit; per-operation float32 does not."""
    from goi_jax.game import reference as ref
    const = dict(deadzone=row["deadzone"], angleEpsilon=row["angleEpsilon"], sensitivity=row["mouseSensitivity"],
                 curve=ref.curve_arrays(row["mouseCurve"]))
    exact = {"double": [0, 0], "float32": [0, 0]}
    for c in chains[::4]:
        x, want = _inputs(c), _expected(c)
        for i in range(len(c)):
            s = dict(action=x["action"][i], mouseInput=x["mouse_input"][i], oldMouse=x["old_mouse"][i], mva=x["mva"][i],
                     mw=x["mw"][i], mouseSnap=x["mouse_snap"][i], pauseInputTimer=x["pause_timer"][i],
                     inputsToSkip=x["inputs_to_skip"][i], C=x["cursor"][i], P=x["pot"][i], T=x["tip"][i], H=x["hub"][i],
                     jointAngle=x["joint_angle"][i], referenceAngle=x["reference_angle"][i],
                     reactionTorque=x["reaction_torque"][i], oldAngle=x["old_angle"][i])
            for mode in exact:
                o = ref.controller_step(const, s, mode)
                ok = o["hingeSpeed"] == want["hinge"][i] and o["sliderSpeed"] == want["slider"][i] and \
                    np.array_equal(np.array(o["mouseInput"], f32), want["mouse_input"][i])
                exact[mode][0] += ok
                exact[mode][1] += 1
    assert exact["double"][0] == exact["double"][1], exact
    assert exact["float32"][0] < exact["float32"][1], exact


def test_controller_bit_exact(chains, row):
    import jax
    import jax.numpy as jnp
    from goi_jax.game import player
    x = _cat([_inputs(c) for c in chains])
    want = _cat([_expected(c) for c in chains])
    p = player.params_from_row(row)
    state = player.ControllerState(jnp.asarray(x["mouse_input"]), jnp.asarray(x["old_mouse"]), jnp.asarray(x["mva"]),
                                   jnp.asarray(x["mw"]), jnp.asarray(x["mouse_snap"]), jnp.asarray(x["pause_timer"]),
                                   jnp.asarray(x["inputs_to_skip"]), jnp.asarray(x["old_angle"]))
    inp = player.ControllerInputs(*(jnp.asarray(x[k]) for k in player.ControllerInputs._fields))
    out = jax.jit(jax.vmap(player.controller_step, (None, 0, 0)))(p, state, inp)
    moved = want["moved"]
    checks = {
        "hinge_speed": (np.asarray(out.hinge_speed), want["hinge"], slice(None)),
        "slider_speed": (np.asarray(out.slider_speed), want["slider"], slice(None)),
        "mouse_input": (np.asarray(out.state.mouse_input), want["mouse_input"], slice(None)),
        "mva": (np.asarray(out.state.mva), want["mva"], slice(None)),
        "mw": (np.asarray(out.state.mw), want["mw"], slice(None)),
        "cursor_target": (np.asarray(out.cursor_target), want["target"], moved),
        "cursor_velocity": (np.asarray(out.cursor_velocity), want["velocity"], moved),
    }
    assert np.array_equal(np.asarray(out.cursor_move), moved)
    report = {}
    for name, (got, exp, m) in checks.items():
        g, e = got[m], exp[m]
        same = (g == e).reshape(len(g), -1).all(1)
        report[name] = (int(same.sum()), len(same), float(np.abs(g.astype(np.float64) - e).max()))
    print("controller bit-exact (exact, steps, max abs error):", report)
    for name, (n_ok, n, err) in report.items():
        assert n_ok == n, f"{name}: {n - n_ok} of {n} steps differ (max {err:.3g})"


def test_friction_switch(chains, row):
    import jax
    import jax.numpy as jnp
    from goi_jax.game import friction as fr
    p = fr.params_from_row(row)
    n_col = len(chains[0].names["colliders"])
    m = 16
    rows, s0, want = [], [], []
    for c in chains:
        for i in range(len(c)):
            cb = fr.np_callbacks(c.events, c.points, i, m)
            assert cb["n_total"] <= m
            rows.append(cb)
            ht, pt = np.zeros(n_col, bool), np.zeros(n_col, bool)
            ht[c.hammer_keys[0][i]] = True
            pt[c.pot_keys[0][i]] = True
            s0.append((c.scr[1]["hammerSlide"][i] > 0, c.scr[1]["potSlide"][i] > 0, ht, pt,
                       (c.scr[2]["tip.vx"][i], c.scr[2]["tip.vy"][i]), (c.scr[2]["pot.vx"][i], c.scr[2]["pot.vy"][i])))
            ht2, pt2 = np.zeros(n_col, bool), np.zeros(n_col, bool)
            ht2[c.hammer_keys[2][i]] = True
            pt2[c.pot_keys[2][i]] = True
            ev = c.events[c.events["step"] == i]
            returned = bool(((ev["kind"] == 4) & (ev["script"] == 0)).any())
            want.append((c.scr[2]["hammerSlide"][i] > 0, c.scr[2]["potSlide"][i] > 0, ht2, pt2, returned))
    cat = {k: np.stack([r[k] for r in rows]) for k in ("valid", "script", "kind", "collider", "terrain", "count",
                                                      "rel_x", "rel_y", "normal", "tangent")}
    buffer = np.where(cat["script"] == 0, 5, 20)

    def run(cb, normal, tangent, buf, rel, state, tip_v, pot_v):
        ratio = jnp.maximum(jnp.where(buf == 5, fr.contact_ratio(normal, tangent, cb.count, 5),
                                      fr.contact_ratio(normal, tangent, cb.count, 20)), 0)
        cb = cb._replace(ratio=ratio, rel_speed=fr.speed(rel))
        return fr.apply_callbacks(p, state, cb, fr.speed(tip_v), fr.speed(pot_v))

    cb = fr.Callbacks(jnp.asarray(cat["valid"]), jnp.asarray(cat["script"]), jnp.asarray(cat["kind"]),
                      jnp.asarray(cat["collider"]), jnp.asarray(cat["terrain"]), jnp.asarray(cat["count"]),
                      jnp.zeros(cat["count"].shape, jnp.float32), jnp.zeros(cat["count"].shape, jnp.float32))
    state = fr.FrictionState(*(jnp.asarray(np.stack([s[k] for s in s0])) for k in range(4)))
    tip_v = jnp.asarray(np.array([s[4] for s in s0], f32))
    pot_v = jnp.asarray(np.array([s[5] for s in s0], f32))
    rel = jnp.asarray(np.stack([cat["rel_x"], cat["rel_y"]], -1))
    new, hard = jax.jit(jax.vmap(run))(cb, jnp.asarray(cat["normal"]), jnp.asarray(cat["tangent"]), jnp.asarray(buffer),
                                       rel, state, tip_v, pot_v)
    for k, name in enumerate(("hammer_slide", "pot_slide", "hammer_tracked", "pot_tracked")):
        exp = np.stack([w[k] for w in want])
        got = np.asarray(getattr(new, name))
        bad = (got != exp).reshape(len(exp), -1).any(1)
        assert not bad.any(), f"{name}: {bad.sum()} of {len(bad)} steps differ"
    exp_ret = np.array([w[4] for w in want])
    assert np.array_equal(np.asarray(hard), exp_ret), f"hard hits: {np.asarray(hard).sum()} vs {exp_ret.sum()} recorded"


def test_hammer_return_rule(chains):
    from goi_jax.game import reference as ref
    seen = 0
    for c in chains:
        ev = c.events
        before = ev[(ev["kind"] == 4) & (ev["script"] == 0)]
        after = ev[(ev["kind"] == 4) & (ev["script"] == 1)]
        for b, a in zip(before, after):
            assert ref.hammer_return(b["a"], b["b"]) == bool(a["flags"] & 1)
            seen += 1
    assert seen > 0


def test_hammer_return_jax(chains):
    """The JAX HammerReturn: same decision as every recorded call, the double-precision comparison on a rounding edge,
    and its target (the Tip after the hit step) is where the cursor ends the following snap step."""
    import jax.numpy as jnp
    from goi_jax.game import player
    s = player.initial_state()
    for c in chains:
        ev = c.events
        before = ev[(ev["kind"] == 4) & (ev["script"] == 0)]
        after = ev[(ev["kind"] == 4) & (ev["script"] == 1)]
        for b, a in zip(before, after):
            _, snapped, _ = player.hammer_return(s, True, jnp.array([0.0, b["a"], 0.0]), jnp.array([0.0, b["b"], 0.0]))
            assert bool(snapped) == bool(a["flags"] & 1)
        for i in np.flatnonzero(c.scr[0]["mouseSnap"] > 0):
            tip = np.array([c.scr[2]["T.x"][i - 1], c.scr[2]["T.y"][i - 1], 0], f32)
            _, _, target = player.hammer_return(s, True, tip, np.zeros(3, f32))
            got = np.array([c.scr[2]["cursor.x"][i], c.scr[2]["cursor.y"][i]], f32)
            assert np.all(np.abs(got - np.asarray(target)) <= 2 * np.spacing(np.abs(got).max()))  # the move rounds
    # tip.y - 0.4 rounds (float32) onto pot.y although the exact difference is below it: Mono does not snap
    ty = f32(1.0)
    py = f32(ty - f32(0.4))
    assert float(ty) - float(f32(0.4)) < float(py) or float(ty) - float(f32(0.4)) > float(py)
    exact_below = float(ty) - float(f32(0.4)) < float(py)
    _, snapped, _ = player.hammer_return(s, True, jnp.array([0.0, ty, 0.0]), jnp.array([0.0, py, 0.0]))
    assert bool(snapped) == (not exact_below)

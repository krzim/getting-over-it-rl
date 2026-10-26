"""The space mechanics (goi_jax/game/space.py) against the space recordings (goi_jax/parity/record_space.py).

Skipped when the recordings are absent (dumps/ is gitignored). Compiles a few jitted steps (a few minutes on CPU):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/game/test_space.py -q
Full table: JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.space_parity --root dumps/parity-space
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPACE = Path(os.environ.get("GOI_SPACE_DIR", ROOT / "dumps" / "parity-space"))
PARITY = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
pytestmark = pytest.mark.skipif(not (SPACE / "index.json").exists(), reason=f"no space recordings in {SPACE}")
TIP = "Player/Hub/Slider/Handle/PoleMiddle/Tip"


def _chain(name):
    from goi_jax.parity import states as S
    return S.load_chain(SPACE / "space" / f"{name}.npz")


@pytest.fixture(scope="module")
def sim():
    from goi_jax.game import sim as SM
    return SM.Sim.from_recordings(SPACE)


@pytest.fixture(scope="module")
def step(sim):
    import jax
    from goi_jax.game import level as L
    return jax.jit(jax.vmap(lambda md, s, a: sim.physics_step(md, sim.params, s, a, axis_name="w"),
                            in_axes=(L.model_axes(), 0, 0), axis_name="w"))


def _run(sim, step, chain, steps, n=32):
    """physics_step from the recorded states at `steps` (padded to n): (out SimState, NAT0 of steps + 1)."""
    import jax
    import jax.numpy as jnp
    from goi_jax.parity import space_parity as SP
    lv = sim.level
    st = SP._stack([sim.state_from_recording(chain, int(t)) for t in steps])
    jm, _, _ = lv.native_states(chain, steps, 0)
    out, _ = step(lv.model_for(SP._pad(jm, n)), SP._pad(st, n), jnp.asarray(SP._pad(chain.action[steps], n)))
    out = jax.tree.map(lambda a: np.asarray(a)[:len(steps)], out)
    _, ref, _ = lv.native_states(chain, steps + 1, 0)
    return out, ref


def test_well_force_formula_exact(sim):
    """GravityControl's per-collider AddForce sum (5 wells, Mono double sqrMagnitude / magnitude) reproduces the
    recorded pending forces bit for bit: the pot body 3x (its 3 colliders), the hammer head 1x, at the body origins
    after the step."""
    import jax.numpy as jnp
    c = _chain("drift-swing")
    bn = c.names["body_names"]
    sp = sim.space
    n = 0
    for name, k in (("Player", 3), (TIP, 1)):
        b0, b1 = c.body(0, bn.index(name)), c.body(1, bn.index(name))
        for t in range(1, len(c), 7):
            fx, fy = (np.asarray(x) for x in sp.well_force(jnp.asarray(b1["xf"][t - 1][:2])))
            F = np.zeros(2, np.float32)
            for _ in range(k):
                for i in range(len(fx)):
                    F = (F + np.array([fx[i], fy[i]], np.float32)).astype(np.float32)
            assert np.array_equal(F, b0["force"][t]), (name, t, F, b0["force"][t])
            n += 1
    assert n > 100


def test_gravity_switches_and_forces(sim, step):
    """Every recorded step around the zone's Enters and Exits (entries from below, a low exit, the win): gravity after
    the step (SCR2) and the pending forces (NAT0 of the next step) exact; bodies within tolerance."""
    from goi_jax.parity import space_parity as SP
    n_switch = 0
    for name in ("enter-still", "enter-swing", "exit-low"):
        c = _chain(name)
        g2, g0 = c.field(2, "gravity.y"), c.field(0, "gravity.y")
        sw = np.nonzero(g2 != g0)[0]
        n_switch += len(sw)
        steps = np.unique(np.clip(np.concatenate([sw + d for d in (-1, 0, 1, 2)]), 2, len(c) - 2))[:32]
        out, ref = _run(sim, step, c, steps)
        assert np.array_equal(out.world.gravity[:, 1], g2[steps]), name
        dyn = np.asarray(sim.level.body_type) == 2
        assert np.array_equal(out.world.bodies.force[:, dyn], ref.bodies.force[:, dyn]), name
        assert SP.within(out.world.bodies.xf_p, ref.bodies.xf_p)[:, dyn].all(), name
    assert n_switch >= 4


def test_teleport_and_win(sim, step):
    """The teleport step: the climber and the cursor land where the game put them (within tolerance: x + shift per
    body; the pose-nudge path is bit-exact, test below), velocities kept, the teleporter flags as recorded at the
    action's end; the win: Won set on the step the first collider leaves through the top."""
    from goi_jax.parity import space_parity as SP
    for name in ("teleport-right", "teleport-left"):
        c = _chain(name)
        t = int(np.nonzero(np.abs(np.diff(c.body(1, 0)["xf"][:, 0])) > 5)[0][0])
        out, ref = _run(sim, step, c, np.array([t]))
        bd, rb = out.world.bodies, ref.bodies
        dyn = np.asarray(sim.level.body_type) == 2
        assert SP.within(bd.xf_p, rb.xf_p)[:, dyn].all(), name
        assert np.abs(bd.xf_p[0, sim.cursor] - rb.xf_p[0, sim.cursor]).max() == 0, name
        assert SP.within(bd.v, rb.v)[:, dyn].all(), name
        assert bool(out.world.new_fixture[0])
        tele = np.asarray(out.space.teleport[0])
        # shouldTeleport consumed, both sides disabled, the source re-enables itself next step (shouldEnable)
        assert not tele[:, 0].any() and not tele[:, 2].any() and tele[:, 1].sum() == 1, tele
    c = _chain("win-top")
    won = [a["won"] for a in c.meta["script_after"]]
    a = won.index(True)
    aos = np.asarray(c.meta["action_of_step"])
    steps = np.nonzero(aos == a)[0]
    out, _ = _run(sim, step, c, steps)
    first = int(np.argmax(out.space.won))
    assert out.space.won[first] and not out.space.won[:first].any()
    assert (out.world.gravity[:, 1] == 0).all()  # a high exit leaves gravity alone (the plugin's Win patch)


def test_teleport_through_transforms_exact():
    """With the pose nudge, the teleport moves the Player root Transform and re-syncs every climber body from the
    Transform hierarchy: bit-exact positions on both recorded teleports."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import pose
    from goi_jax.game import sim as SM
    sim = SM.Sim.from_recordings(SPACE, pose_nudge=True)
    for name in ("teleport-right", "teleport-left"):
        p = SPACE / "space" / f"{name}.npz"
        c = _chain(name)
        tf = pose.load_transforms(p)
        t = int(np.nonzero(np.abs(np.diff(c.body(1, 0)["xf"][:, 0])) > 5)[0][0])
        s = sim.state_from_recording(c, t, transforms=tf)
        jm, _, _ = sim.level.native_states(c, [t], 0)
        model = sim.level.model_for(jax.tree.map(lambda a: a[0], jm))
        out, _ = jax.jit(lambda md, s, a: sim.physics_step(md, sim.params, s, a))(model, s, jnp.asarray(c.action[t]))
        _, ref, _ = sim.level.native_states(c, [t + 1], 0)
        cb = sim.chain_bodies
        assert np.array_equal(np.asarray(out.world.bodies.xf_p)[cb], ref.bodies.xf_p[0][cb]), name


@pytest.mark.skipif(not (PARITY / "index.json").exists(), reason="no level-start recordings")
def test_space_off_the_zone_changes_nothing():
    """Below the zone the space step is a no-op: Sim(space=True) and Sim(space=False) give bit-identical worlds."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import level as L
    from goi_jax.game import sim as SM
    from goi_jax.parity import states as S
    on, off = SM.Sim.from_recordings(PARITY), SM.Sim.from_recordings(PARITY, space=False)
    chain = S.load_chain(PARITY / "coffee" / "c01-random-62.npz")
    steps = np.arange(1, 9)
    lv = on.level
    jm, _, _ = lv.native_states(chain, steps, 0)
    model = lv.model_for(jm)
    outs = []
    for sm in (on, off):
        st = jax.tree.map(lambda *a: np.stack(a), *[sm.state_from_recording(chain, int(t)) for t in steps])
        f = jax.jit(jax.vmap(lambda md, s, a, sm=sm: sm.physics_step(md, sm.params, s, a, axis_name="w"),
                             in_axes=(L.model_axes(), 0, 0), axis_name="w"))
        outs.append(jax.tree.map(np.asarray, f(model, st, jnp.asarray(chain.action[steps]))[0].world))
    for a, b in zip(jax.tree.leaves(outs[0]), jax.tree.leaves(outs[1])):
        assert np.array_equal(a, b, equal_nan=a.dtype.kind == "f")

"""Unity 2020.3's Physics2D work around b2World::Step, as far as one recorded K=1 step needs it.

`simulate(model, state, settings, dt, moves, axis_name)` = PhysicsScene2D.Simulate(dt) for one world:

1. Kinematic MovePosition (Rigidbody2D.MovePosition on the cursor and the crane, recorded as EVTS MovePosition
   events). Unity turns the pending target into the body's velocity for this step only: the solver sees
   v = (target - position) / dt, and after the step the body sits exactly on the target with the velocity it had
   before (recordings: NAT1 position == target bit for bit and NAT1 v == NAT0 v on every step; the cursor's script
   velocity `c4 - C` is what NAT0 holds). The kinematic's centre is put on the target too (its local centre is 0).
2. b2World::Step (goi_jax.box2d.step).

Material swaps (T5): HammerCollisions/PotSounds assign `sharedMaterial` in FixedUpdate, before Simulate. Unity
refreshes the fixture friction AND every existing contact's m_friction (sqrt mix) at assignment time: NAT0 (read just
before Simulate) already holds the new mix on all 1376 hammer-head contacts of the recorded swap steps, so the swap
step's solve uses the new friction. `contact_friction(...)` rebuilds the mix for the "refreshed after the solve"
alternative (goi_jax/parity/harness.py --t5): over the 291 swap steps, bodies within tolerance on 96.2% with the
recorded (new) mix vs 73.9% with the old one; the new mix is closer on 114 steps, the old on 2.

Not modelled (no effect on a recorded K=1 step): Transform sync of dirty Transforms (already in NAT0), interpolation,
callbacks (goi_jax.game.friction consumes them), the per-action pose round trip between steps (player.md §9.1; NAT0
of the next step already contains it).
"""
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import math2d as m
from goi_jax.box2d import step as box2d_step  # the function (package export)
from goi_jax.box2d.world import KINEMATIC

F32 = np.float32


def move_velocity(state, target, mask, dt):
    """Pre-step: the velocity Unity gives a kinematic body with a pending MovePosition, (target - p) / dt."""
    bd = state.bodies
    dt = jnp.asarray(dt, jnp.float32)
    v = (target - bd.xf_p) / dt
    use = mask & (bd.type == KINEMATIC)
    return state._replace(bodies=bd._replace(v=jnp.where(use[:, None], v, bd.v)))


def move_finish(state, before, target, mask):
    """Post-step: the body ends on its target (origin and centre) with its pre-step velocity."""
    bd = state.bodies
    use = mask & (bd.type == KINEMATIC)
    u2 = use[:, None]
    qc, qs = bd.xf_q[:, 0], bd.xf_q[:, 1]
    rx, ry = m.rot(qc, qs, bd.local_center[:, 0], bd.local_center[:, 1])
    c = jnp.stack([target[:, 0] + rx, target[:, 1] + ry], -1)
    return state._replace(bodies=bd._replace(xf_p=jnp.where(u2, target, bd.xf_p), c=jnp.where(u2, c, bd.c),
                                             v=jnp.where(u2, before.bodies.v, bd.v)))


def simulate(model, state, settings, dt, target=None, mask=None, axis_name=None, kinematic_move=True):
    """PhysicsScene2D.Simulate(dt): MovePosition velocities, b2World::Step, MovePosition finish."""
    if target is None or not kinematic_move:
        return box2d_step(model, state, settings, dt, axis_name=axis_name)
    before = state
    state = move_velocity(state, target, mask, dt)
    state = box2d_step(model, state, settings, dt, axis_name=axis_name)
    return move_finish(state, before, target, mask)


def contact_friction(fixture_friction, fa, fb):
    """b2MixFriction of the current fixture frictions for contact slots (fa, fb): float32 sqrt(fA * fB)."""
    prod = (np.take_along_axis(fixture_friction, fa, -1) * np.take_along_axis(fixture_friction, fb, -1)).astype(F32)
    return np.sqrt(prod, dtype=F32)


def move_targets(chain, steps, level):
    """(N, B, 2) MovePosition target per engine body and (N, B) mask, from the chain's EVTS (last call wins).
    Pending at step s's Simulate: a call made inside or after the previous step's Simulate (phase 2 callbacks /
    phase 3 of step s-1: HammerReturn's cursor snap from a collision callback), then overridden by this step's phase 0
    scripts and phase 1 crane. All 11 recorded phase-2 cursor moves land exactly at the next step's NAT1 (without
    them the cursor misses by up to 4.4 on those steps). A chain's first step cannot see a call from before it."""
    from goi_jax.parity import states as S
    steps = np.asarray(steps)
    N, B = len(steps), level.n_bodies
    tgt = np.zeros((N, B, 2), F32)
    mask = np.zeros((N, B), bool)
    ev = chain.events
    mv = ev[ev["kind"] == S.EV_MOVE_POSITION]
    # application order: (step it applies to, early-before-late within that step)
    applies = np.where(mv["phase"] >= 2, mv["step"] + 1, mv["step"])
    late_first = np.where(mv["phase"] >= 2, 0, 1)
    mv, applies = mv[np.lexsort((late_first, applies))], np.sort(applies)  # stable: event order kept within a group
    row = {int(s): i for i, s in enumerate(steps)}
    for e, s in zip(mv, applies):
        i = row.get(int(s))
        if i is None:
            continue
        b = level.game_to_engine.get(int(e["script"]))
        if b is None:
            continue
        tgt[i, b] = (e["a"], e["b"])
        mask[i, b] = True
    return tgt, mask

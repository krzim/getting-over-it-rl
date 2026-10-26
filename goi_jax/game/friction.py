"""The friction switch of the hammer head and the pot bottom (HammerCollisions / PotSounds) as pure JAX functions.

Each script keeps a `slide` flag and a set of tracked Terrain colliders, updated by Unity's collision callbacks after
every physics step; its FixedUpdate then assigns the collider's material for the next step: sliding (hammer
SlidingFriction 0.7, pot PotBottomFriction 0.3) or static (StaticFriction 1.0, PotBottomStaticFriction 0.8). The
material change reaches existing contacts' friction at once (native-box2d.md T5); Box2D mixes sqrt(f1 f2).

Callback rules (HammerCollisions.cs:115-307, PotSounds.cs:92-298; all only for colliders on layer Terrain, 10):
- Enter, hammer: needs at least one contact point; a hit with |relativeVelocity| > 14 calls HammerReturn (even for an
  already tracked collider); the collider is tracked if it is not yet. Enter, pot: tracked only if the pot tracks
  nothing yet (it follows one collider at a time), whatever the point count. Enter never changes `slide`.
- Stay, tracked colliders only: slide = (no points) or (speed > v and ratio > r), ratio = the max over the first
  min(points, buffer) reported points of (1 + |tangentImpulse|) / (1 + |normalImpulse|); hammer v = 0.3 (tip body
  speed), r = 5, buffer 5; pot v = 2 (pot body speed), r = 6, buffer 20. (The scripts' "point moved less than
  moveThreshold" test always passes, because the stored points are the shared buffer GetContacts just refilled.)
  The last Stay callback of a step wins.
- Exit: a Terrain collider is untracked; if nothing is tracked any more (for any collider), slide = true.
Verified against every recorded callback: 2283 Stay decisions, 2863 callbacks in sequence and the flags and tracked
sets after 4200 steps, all exact (tests/game/test_controller.py).

Impulses: use the ContactPoint2D values Unity reports (impulses of the last solve that touched the contact). They equal
the stored manifold impulses except after a TOI sub-step, which leaves the manifold's impulses at 0.

Interface:
    state = initial_state(n_colliders)                        # FrictionState, both sliding, nothing tracked
    hammer_mu, pot_mu = materials(params, state)              # what the two FixedUpdates assign before the step
    cb = Callbacks(...)                                        # this step's callbacks, padded, in Unity's order
    state, hard_hit = apply_callbacks(params, state, cb, tip_speed, pot_speed)
    ratio = contact_ratio(normal_impulse, tangent_impulse, count, buffer)   # per callback, from its points
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from goi_jax.game import floatmath as fm

f32 = jnp.float32
HAMMER, POT = 0, 1
ENTER, STAY, EXIT = 1, 2, 3


class FrictionParams(NamedTuple):
    hammer_static: jax.Array    # 1.0 StaticFriction
    hammer_sliding: jax.Array   # 0.7 SlidingFriction
    pot_static: jax.Array       # 0.8 PotBottomStaticFriction
    pot_sliding: jax.Array      # 0.3 PotBottomFriction
    hammer_speed: jax.Array     # 0.3
    hammer_ratio: jax.Array     # 5
    pot_speed: jax.Array        # 2
    pot_ratio: jax.Array        # 6
    hard_hit: jax.Array         # 14: |relativeVelocity| that sends HammerReturn


class FrictionState(NamedTuple):
    hammer_slide: jax.Array     # () bool
    pot_slide: jax.Array        # () bool
    hammer_tracked: jax.Array   # (n_colliders,) bool: Terrain colliders in HammerCollisions.collisionPoints
    pot_tracked: jax.Array      # (n_colliders,) bool: PotSounds.collisionPoints (at most one)


class Callbacks(NamedTuple):
    """One step's collision callbacks to the two scripts, padded to a fixed length M, in Unity's dispatch order."""
    valid: jax.Array            # (M,) bool
    script: jax.Array           # (M,) int32: HAMMER (callbacks of the Tip's collider) or POT (PotSounds' collider)
    kind: jax.Array             # (M,) int32: ENTER, STAY, EXIT
    collider: jax.Array         # (M,) int32: the other collider's index (Game.colliders order)
    terrain: jax.Array          # (M,) bool: the other object is on layer Terrain
    count: jax.Array            # (M,) int32: Collision2D.contactCount
    ratio: jax.Array            # (M,) float32: contact_ratio of its points (Stay only)
    rel_speed: jax.Array        # (M,) float32: |relativeVelocity| (Enter only; Mono magnitude)


def params_from_row(row: dict) -> FrictionParams:
    cc = row.get("codeConstants", {})
    return FrictionParams(
        hammer_static=f32(row["hammerStatic"]["friction"]), hammer_sliding=f32(row["hammerSliding"]["friction"]),
        pot_static=f32(row["potStatic"]["friction"]), pot_sliding=f32(row["potSliding"]["friction"]),
        hammer_speed=f32(cc.get("hammerSpeed", 0.3)), hammer_ratio=f32(cc.get("hammerRatio", 5)),
        pot_speed=f32(cc.get("potSpeed", 2)), pot_ratio=f32(cc.get("potRatio", 6)),
        hard_hit=f32(row.get("hammerHardHitThreshold", 14)))


def initial_state(n_colliders: int) -> FrictionState:
    """Start() of both scripts: sliding, nothing tracked (also what a plugin restore of a fresh state sets)."""
    z = jnp.zeros(n_colliders, bool)
    return FrictionState(jnp.bool_(True), jnp.bool_(True), z, z)


def materials(p: FrictionParams, s: FrictionState):
    """The friction of the hammer head's and the pot bottom's material for the coming step (the FixedUpdates)."""
    return (jnp.where(s.hammer_slide, p.hammer_sliding, p.hammer_static),
            jnp.where(s.pot_slide, p.pot_sliding, p.pot_static))


def contact_ratio(normal_impulse, tangent_impulse, count, buffer: int):
    """max over the first min(count, buffer) points of (1 + |t|) / (1 + |n|), each quotient evaluated in double and
    rounded (Mono). normal/tangent_impulse: (..., P) reported points in GetContacts order; 0 for no points."""
    n = jnp.abs(jnp.asarray(normal_impulse, f32))
    t = jnp.abs(jnp.asarray(tangent_impulse, f32))
    q = fm.dd_round(fm.dd_div(fm.two_sum(f32(1), t), fm.two_sum(f32(1), n)))
    k = jnp.arange(n.shape[-1])
    use = k < jnp.minimum(jnp.asarray(count)[..., None], buffer)
    return jnp.max(jnp.where(use, q, f32(0)), axis=-1)


def apply_callbacks(p: FrictionParams, s: FrictionState, cb: Callbacks, tip_speed, pot_speed):
    """Run one step's callbacks in order. tip_speed / pot_speed: the Tip's and the pot's body speed after the step
    (Rigidbody2D.velocity.magnitude, Mono). Returns (state, hard_hit): hard_hit is whether a hammer Enter sent
    HammerReturn (then run player.hammer_return)."""
    tip_speed, pot_speed = jnp.asarray(tip_speed, f32), jnp.asarray(pot_speed, f32)

    def one(carry, c):
        slide, tracked, hard = carry  # slide (2,), tracked (2, n), hard ()
        valid, script, kind, col, terrain, count, ratio, rel = c
        mine = tracked[script]
        is_tracked = mine[col]
        live = valid & terrain
        # Enter
        enter = live & (kind == ENTER)
        hammer_enter = enter & (script == HAMMER) & (count > 0)
        pot_enter = enter & (script == POT) & ~jnp.any(mine)
        hard = hard | (hammer_enter & (rel > p.hard_hit))
        add = (hammer_enter | pot_enter) & ~is_tracked
        # Stay
        stay = live & (kind == STAY) & is_tracked
        fast = jnp.where(script == HAMMER, (tip_speed > p.hammer_speed) & (ratio > p.hammer_ratio),
                         (ratio > p.pot_ratio) & (pot_speed > p.pot_speed))
        flag = (count == 0) | fast
        # Exit (the emptiness test runs for any collider)
        exit_ = valid & (kind == EXIT)
        remove = exit_ & terrain & is_tracked
        mine = mine.at[col].set(jnp.where(add, True, jnp.where(remove, False, is_tracked)))
        new_slide = jnp.where(stay, flag, jnp.where(exit_ & ~jnp.any(mine), True, slide[script]))
        return (slide.at[script].set(new_slide), tracked.at[script].set(mine), hard), None

    tracked = jnp.stack([s.hammer_tracked, s.pot_tracked])
    slide = jnp.stack([s.hammer_slide, s.pot_slide])
    xs = (cb.valid, cb.script, cb.kind, cb.collider, cb.terrain, cb.count, cb.ratio, cb.rel_speed)
    (slide, tracked, hard), _ = lax.scan(one, (slide, tracked, jnp.bool_(False)), xs)
    return FrictionState(slide[0], slide[1], tracked[0], tracked[1]), hard


def speed(v):
    """Rigidbody2D.velocity.magnitude under Mono (for tip_speed, pot_speed and |relativeVelocity|)."""
    v = jnp.asarray(v, f32)
    return fm.magnitude(v[..., 0], v[..., 1])


def np_callbacks(events, points, step: int, m: int, buffers=(5, 20)) -> dict:
    """Host helper for tests: one recorded step's callbacks (goi_jax.parity.states Chain.events / .points) as padded
    Callbacks fields (numpy), with each callback's reported points (normal, tangent impulses) for contact_ratio."""
    idx = np.flatnonzero((events["step"] == step) & (events["kind"] >= ENTER) & (events["kind"] <= EXIT))
    out = {k: np.zeros(m, dt) for k, dt in (("valid", bool), ("script", np.int32), ("kind", np.int32),
                                             ("collider", np.int32), ("terrain", bool), ("count", np.int32),
                                             ("rel_x", np.float32), ("rel_y", np.float32))}
    pn, pt = np.zeros((m, 32), np.float32), np.zeros((m, 32), np.float32)
    for j, e in enumerate(idx[:m]):
        x = events[e]
        out["valid"][j], out["script"][j], out["kind"][j] = True, x["script"], x["kind"]
        out["collider"][j], out["terrain"][j], out["count"][j] = max(x["collider"], 0), int(x["c"]) == 10, x["count"]
        out["rel_x"][j], out["rel_y"][j] = x["a"], x["b"]
        pts = points[points["event"] == e][:32]
        pn[j, :len(pts)], pt[j, :len(pts)] = pts["normalImpulse"], pts["tangentImpulse"]
    out["normal"], out["tangent"] = pn, pt
    out["n_used"] = len(idx[:m])
    out["n_total"] = len(idx)
    return out

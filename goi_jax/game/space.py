"""The space section (route progress ~760 and beyond): the zero-gravity zone, its gravity wells, the two teleporters
and the win, as the plugin's lockstep game runs them (plugin/Game.cs Step; docs/understanding/world.md §3-4).

Decompiled sources (`_decompiled/Assembly-CSharp/`): GravityControl.cs:39-50 (OnTriggerStay2D: the wells),
:52-55 (OnTriggerEnter2D: gravity 0), :57-85 (OnTriggerExit2D: a high exit rolls the credits, a low one sets -30);
Teleporter.cs:24-38 (Enter by the Player root's collider sets shouldTeleport while enabled, Exit sets shouldEnable),
:40-58 (LateUpdate; the plugin runs it on sim time right after the step, plugin/Game.cs Teleport); the plugin's Win
patch (plugin/Plugin.cs Patches.Win) replaces a high exit by Patches.Won = true (gravity and creditsUp untouched).

Measured on the space recordings (goi_jax/parity/record_space.py, dumps/parity-space: entries from below, low exits,
both teleporters, the win), all bit for bit:
- Trigger overlap: per (trigger, collider), any of the collider's pieces overlapping the trigger's box (b2TestOverlap:
  GJK distance of the cores within the two skins) at the bodies' poses right AFTER the step's Simulate. A collider's
  Enter is dispatched after the step at which it first overlaps, Stay from the next step on (never on the Enter
  step), Exit after the first step it no longer overlaps. Every gravity switch of 9 chains (11 Enters, 9 Exits)
  falls on the predicted step.
- Wells: on Stay, for every well (the 5 asteroid Transforms of GravityControl.gravityWells, read from the scene), the
  collider's attached body gets AddForce(2500 / |w - p|^2 * normalize(w - p)) with p = rb.position (the body origin
  after the step), accumulated well by well and collider by collider into the body's force, which the next step
  integrates (v += h * invMass * F) and then clears. Mono evaluates sqrMagnitude and magnitude in double (floatmath).
  The pot body carries 3 colliders (Player, PotCollider, PotCollider/Sides: up to 3x the force), the hammer head 1.
  Unity sends no Stay for a sleeping body.
- Gravity: an Enter of any collider sets (0, 0); an Exit with rb.position.y <= bounds.max.y - 5 sets (0, -30); a
  higher Exit sets Won (the win). Exits are applied before Enters within one step (Unity appends new pairs to its
  callback list; not seen otherwise in the recordings).
- Teleporters (hierarchy order: Invisible Walls/Teleporter at x ~103.4, then Teleporter (1) at x ~22.5; each targets
  the other's x): only the Player root's own collider counts. After the step's callbacks, per teleporter in order:
  shouldEnable -> enableTeleport; shouldTeleport -> the target is disabled, the Player root Transform and the cursor
  move by shift = targetX - player.x (float32), Physics2D.SyncTransforms re-places every climber body (positions
  only: angles and velocities stay; centre of mass b2Mul(xf, localCenter), c0 = c, as b2Body::SetTransform), the
  teleporter disables itself and sets shouldEnable. Bodies: with the pose nudge (pose.py) every position is the
  shifted Transform hierarchy's world position (bit-exact on both recorded teleports); without it, x + shift per
  body (bit-exact for 13 of 14 recorded bodies, the other within 1 ulp). Fixtures are re-synchronised (fat AABBs,
  b2Fixture::Synchronize with no displacement) and the next step looks for new contacts first (SetTransform's
  FindNewContacts).

Not modelled: the credits branch (creditsUp, gravity +1.2: the plugin's Win patch never lets it run), Unity's exact
callback order between an Exit and an Enter in one step, and new contacts made by the per-body SetTransform calls in
between (one FindNewContacts after all bodies moved instead).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import math2d as m
from goi_jax.box2d.pairs import synchronize_fixtures
from goi_jax.box2d.settings import EPSILON
from goi_jax.box2d.world import DYNAMIC, STATIC
from goi_jax.game import floatmath as fm
from goi_jax.game import pose

F32, I32 = np.float32, np.int32
f32 = jnp.float32

ZONE_COLLIDER = "ZeroGrav#0"
TELEPORTERS = ("Invisible Walls/Teleporter#0", "Invisible Walls/Teleporter (1)#0")  # hierarchy order (InLevel)
PLAYER_COLLIDER = "Player#0"  # Teleporter: coll.gameObject == player (the root's own collider)
# GravityControl.gravityWells (scene level1, ZeroGrav's MonoBehaviour): asteroid5 (1), asteroid4 (1), asteroid5,
# asteroid4, asteroid5 (2), children of Mountain at (-7.8421783, -0.6696552); identity rotations, unit scales. World
# position = parent + local in float32 (bit-exact against the recorded well forces).
_MOUNTAIN = (F32(-7.8421783447265625), F32(-0.6696552038192749))
_WELL_LOCAL = ((71.4000015258789, 424.6000061035156), (90.69999694824219, 441.8999938964844),
               (81.44217681884766, 392.43963623046875), (49.70000076293945, 406.20001220703125),
               (49.900001525878906, 445.70001220703125))
WELLS = np.array([[F32(F32(x) + _MOUNTAIN[0]), F32(F32(y) + _MOUNTAIN[1])] for x, y in _WELL_LOCAL], F32)
WELL_STRENGTH = F32(2500.0)
# Teleporter.targetX x: the other teleporter's Transform, Invisible Walls (69.24463) + its local x
_WALLS_X = F32(69.24462890625)
TARGET_X = np.array([F32(_WALLS_X + F32(-46.70000076293945)), F32(_WALLS_X + F32(34.20000076293945))], F32)
GRAVITY_OUT = F32(-30.0)       # GravityControl.cs:84
SHOULD_TELEPORT, SHOULD_ENABLE, ENABLE = 0, 1, 2


class SpaceState(NamedTuple):
    touch: jax.Array     # (1 + NT, NS) bool: (zone, teleporters) x tracked collider, overlapping after the last step
    teleport: jax.Array  # (NT, 3) bool per teleporter: shouldTeleport, shouldEnable, enableTeleport
    won: jax.Array       # () bool Patches.Won


class Space:
    """Host tables for the space mechanics of one level (see the module docstring)."""

    def __init__(self, level, chain_bodies, cursor, pot):
        lv = level
        fx = lv.fixture
        cols = lv.names["colliders"]
        fcol = fx["collider"].astype(I32)
        self.trigger_fix = []
        for name in (ZONE_COLLIDER,) + TELEPORTERS:
            f = np.nonzero(fcol == cols.index(name))[0]
            if len(f) != 1 or not fx["sensor"][f[0]]:
                raise ValueError(f"trigger {name}: expected one sensor fixture, got {f}")
            self.trigger_fix.append(int(f[0]))
        zf = self.trigger_fix[0]
        zlayer = int(fx["layer"][zf])
        masks = lv.layer_masks
        collides = ((np.asarray(masks, np.int64)[zlayer] >> fx["layer"]) & 1) != 0 if masks is not None \
            else np.ones(len(fcol), bool)
        btype = np.asarray(lv.body_type)
        tracked = np.nonzero((btype[fx["body"]] == DYNAMIC) & ~fx["sensor"].astype(bool) & collides)[0]
        self.tracked_cols = np.unique(fcol[tracked])                  # collider ids (level order)
        col_slot = {int(c): i for i, c in enumerate(self.tracked_cols)}
        self.n_tracked = len(self.tracked_cols)
        self.player_slot = col_slot[cols.index(PLAYER_COLLIDER)]
        # overlap lanes: (trigger row, trigger fixture, fixture, tracked collider slot); teleporters: the Player only
        lanes = [(0, zf, int(f), col_slot[int(fcol[f])]) for f in tracked]
        for t, tf in enumerate(self.trigger_fix[1:]):
            lanes += [(1 + t, tf, int(f), self.player_slot) for f in tracked if col_slot[int(fcol[f])] == self.player_slot]
        lanes = np.array(lanes, I32)
        self.lane_row, self.lane_trig, self.lane_fix, self.lane_slot = (lanes[:, i] for i in range(4))
        self.slot_body = np.array([int(fx["body"][np.nonzero(fcol == c)[0][0]]) for c in self.tracked_cols], I32)
        self.bodies = np.unique(self.slot_body)                       # bodies with tracked colliders
        self.max_k = int(max(np.sum(self.slot_body == b) for b in self.bodies))
        zv = fx["verts"][zf][:int(fx["count"][zf])]
        # Collider2D.bounds.max.y of the box (the plugin's route row reads the same value), minus 5 in double
        self.win_y = np.float64(F32(zv[:, 1].max())) - 5.0
        self.chain_bodies = np.asarray(chain_bodies, I32)
        self.moved = np.zeros(lv.n_bodies, bool)
        self.moved[self.chain_bodies] = True
        self.moved[int(cursor)] = True
        self.cursor = int(cursor)
        self.pot = int(pot)
        self.NT = len(TELEPORTERS)

    # ------------------------------------------------------------------ overlap

    def overlap(self, model, bd):
        """(1 + NT, NS) bool: b2TestOverlap of every tracked collider's pieces with each trigger (any piece), at the
        bodies' current poses; colliders of inactive or non-dynamic bodies never overlap."""
        from goi_jax.box2d.toi import distance
        fa, fb = jnp.asarray(self.lane_trig), jnp.asarray(self.lane_fix)
        ba, bb = model.fixture_body[fa], model.fixture_body[fb]

        def proxy(f):
            v = model.verts[f]
            return v[..., 0], v[..., 1], model.count[f], model.radius[f]
        pA, pB = proxy(fa), proxy(fb)
        xf = lambda b: (bd.xf_p[b, 0], bd.xf_p[b, 1], bd.xf_q[b, 0], bd.xf_q[b, 1])  # noqa: E731
        n = fa.shape[0]
        cache = (jnp.zeros(n, I32), jnp.zeros((n, 3), I32), jnp.zeros((n, 3), I32), jnp.zeros(n, f32))
        d, _ = distance(cache, pA, xf(ba), pB, xf(bb), jnp.ones(n, bool))
        r = pA[3] + pB[3]
        # b2Distance useRadii: distance -= rA + rB when it exceeds both rA + rB and epsilon, else 0; overlap < 10 eps
        sep = (d > r) & (d > F32(EPSILON))
        hit = ~sep | ((d - r) < F32(10 * EPSILON))
        hit = hit & (bd.type[bb] == DYNAMIC) & bd.active[bb]
        out = jnp.zeros((1 + self.NT) * self.n_tracked + 1, bool)
        flat = jnp.asarray(self.lane_row * self.n_tracked + self.lane_slot)
        out = out.at[jnp.where(hit, flat, out.shape[0] - 1)].set(True)[:-1]
        return out.reshape(1 + self.NT, self.n_tracked)

    def initial(self, model, world) -> SpaceState:
        """The space state of a recorded or placed world (no history): the colliders overlapping now count as
        touching (no Enter follows; a placement's restore dispatches it inside Simulate(0) and then restores the
        saved gravity), Won clear, each teleporter enabled unless the Player overlaps it (it can only be in there
        after arriving by teleport, which disables the target)."""
        touch = self.overlap(model, world.bodies)
        inside = touch[1:, self.player_slot]
        tele = jnp.stack([jnp.zeros(self.NT, bool), jnp.zeros(self.NT, bool), ~inside], -1)
        return SpaceState(touch=touch, teleport=tele, won=jnp.asarray(False))

    # ------------------------------------------------------------------ after one step

    def well_force(self, p):
        """GravityControl's per-collider AddForce sum over the wells for a body at p (2,) in Mono float semantics,
        as the 5 separate (f32) force vectors."""
        w = jnp.asarray(WELLS)
        gx, gy = w[:, 0] - p[0], w[:, 1] - p[1]
        sq = fm.dd_round(fm.dd_add(fm.two_prod(gx, gx), fm.two_prod(gy, gy)))   # sqrMagnitude
        nx, ny = fm.normalize(gx, gy)
        s = jnp.asarray(WELL_STRENGTH) / sq
        return m.mul(s, nx), m.mul(s, ny)

    def after_step(self, model, w, sp: SpaceState, ps=None):
        """The space callbacks and teleports after one Simulate: returns (world, space state, pose state).
        w: the world right after the step (callbacks for the hammer and pot already applied), ps: the climber's
        Transforms (pose_nudge) or None."""
        bd = w.bodies
        touch = self.overlap(model, bd)
        enter, leave = touch & ~sp.touch, sp.touch & ~touch
        slot_body = jnp.asarray(self.slot_body)
        stay = touch[0] & sp.touch[0] & bd.awake[slot_body]
        # zone: wells (Stay), gravity (Exit low -> -30, then Enter -> 0), the win (Exit high)
        y = bd.xf_p[slot_body, 1]
        high = _gt_double(y, self.win_y)   # rb.position.y > bounds.max.y - 5f (double)
        low_exit = jnp.any(leave[0] & ~high)
        won = sp.won | jnp.any(leave[0] & high)
        g = jnp.where(low_exit, jnp.stack([f32(0), jnp.asarray(GRAVITY_OUT)]), w.gravity)
        g = jnp.where(jnp.any(enter[0]), jnp.zeros(2, f32), g)
        force = bd.force
        for b in self.bodies:
            k = jnp.sum(stay & (slot_body == b))
            fx_, fy_ = self.well_force(bd.xf_p[b])
            F = force[b]
            for j in range(self.max_k):
                for i in range(len(WELLS)):
                    nf = jnp.stack([F[0] + fx_[i], F[1] + fy_[i]])
                    F = jnp.where(j < k, nf, F)
            force = force.at[b].set(F)
        bd = bd._replace(force=force)
        # teleporters: callbacks, then Teleport() per teleporter in order
        tele = sp.teleport
        pe = enter[1:, self.player_slot]
        pl = leave[1:, self.player_slot]
        tele = tele.at[:, SHOULD_TELEPORT].set(tele[:, SHOULD_TELEPORT] | (pe & tele[:, ENABLE]))
        tele = tele.at[:, SHOULD_ENABLE].set(tele[:, SHOULD_ENABLE] | pl)
        any_tp = jnp.zeros((), bool)
        moved = jnp.asarray(self.moved)
        cb = jnp.asarray(self.chain_bodies)
        for t in range(self.NT):
            en = tele[t, SHOULD_ENABLE]
            tele = tele.at[t, ENABLE].set(tele[t, ENABLE] | en).at[t, SHOULD_ENABLE].set(tele[t, SHOULD_ENABLE] & ~en)
            go = tele[t, SHOULD_TELEPORT]
            target = 1 - t
            tele = tele.at[target, ENABLE].set(tele[target, ENABLE] & ~go)
            px = ps.lp[0, 0] if ps is not None else bd.xf_p[self.pot, 0]
            shift = jnp.asarray(TARGET_X[t]) - px
            xp = bd.xf_p
            if ps is not None:  # the Transform hierarchy (pose.py): root moves, children follow, bodies re-synced
                ps_new = ps._replace(lp=ps.lp.at[0, 0].set(ps.lp[0, 0] + shift))
                xy = pose.body_positions(ps_new)
                xp_new = xp.at[cb].set(xy).at[self.cursor, 0].add(shift)
                ps = jax.tree.map(lambda a, b: jnp.where(go, a, b), ps_new, ps)
            else:
                xp_new = xp.at[:, 0].set(jnp.where(moved, xp[:, 0] + shift, xp[:, 0]))
            xp = jnp.where(go, xp_new, xp)
            cx, cy = m.xf_mul(xp[:, 0], xp[:, 1], bd.xf_q[:, 0], bd.xf_q[:, 1], bd.local_center[:, 0],
                              bd.local_center[:, 1])
            c = jnp.stack([cx, cy], -1)
            mv = (go & moved)[:, None]
            bd = bd._replace(xf_p=xp, c=jnp.where(mv, c, bd.c), c0=jnp.where(mv, c, bd.c0),
                             a0=jnp.where(go & moved, bd.a, bd.a0))
            tele = tele.at[t, SHOULD_TELEPORT].set(False)
            tele = tele.at[t, ENABLE].set(tele[t, ENABLE] & ~go).at[t, SHOULD_ENABLE].set(tele[t, SHOULD_ENABLE] | go)
            any_tp = any_tp | go
        w = w._replace(bodies=bd, gravity=g)
        # SyncTransforms -> b2Body::SetTransform: fixtures re-synchronised in place, then FindNewContacts
        synced = synchronize_fixtures(model, w, moved & any_tp & (bd.type != STATIC))
        w = jax.tree.map(lambda a, b: jnp.where(any_tp, a, b), synced, w)
        w = w._replace(new_fixture=w.new_fixture | any_tp)
        return w, SpaceState(touch=touch, teleport=tele, won=won), ps


def _gt_double(y, threshold: float):
    """y > threshold for float32 y and a float64 threshold (exact: compare against the threshold's float32 neighbours)."""
    t32 = F32(threshold)
    if np.float64(t32) == threshold:
        return y > t32
    hi = t32 if np.float64(t32) > threshold else np.nextafter(t32, F32(np.inf))
    return y >= hi

"""The real level in the JAX engine: goi_jax.box2d Model + State built from the game's own native Box2D world as the
recorder saw it (goi_jax/parity/states.py), and the inverse export for comparison.

    lv = load_level("dumps/parity")                   # static: geometry ('g' record), layers, proxy order, pairs
    jm, st = lv.native_state(chain, t)                # one recorded pre-Simulate world (NAT0 of step t)
    model = lv.model_for(jm)                          # Model with that step's joint frames
    out = lv.export(state_after)                      # recorded layout: bodies by native slot, contacts by ids

Index conventions (what makes the engine's orderings Box2D's):
- Bodies: the engine seeds islands from the highest body index down. Default (body_order="unity"): engine index =
  native list position + 1 for the moving bodies (the static ground body, Game index -1, last in the list, is engine
  0 at the identity transform), so seeds walk the recorded list from its tail (Tip first, Orange last). Stock Box2D
  seeds from m_bodyList's head, so this is an empirical match to the game, not something the C++ source explains. This is the order
  the game solves in: with the stock reading (body_order="box2d", seeds from the list head) the climber's joints are
  solved pot hinge first and the position iterations miss NAT1 by a median 2.8e-4; with this one they reproduce it
  bit for bit on 99% of steps (goi_jax/parity/harness.py).
- Fixtures: grouped by engine body; within a body reverse native fixture-list order (b2Body::CreateFixture prepends).
- Proxy order: the broad-phase proxy ids from a native dump (dumps/native/goi-rl-native-start.jsonl `proxyIds`,
  matched by collider path, component and piece) order the candidate pairs as b2BroadPhase::UpdatePairs does; without
  that dump, fixture order. Candidate orientation follows b2Contact::Create (polygon before circle, else lower proxy).
- Joints: native joint-list order is newest first, so engine joint j = reverse list position; the order differs
  between steps (the climber's joints are re-created each action), so joint frames and types are per step
  (`JointModel`, batched with `model_axes`). Joint bodies come from the joint name ("path#type": bodyA is the
  joint's own GameObject, bodyB its connected body, from the engine dump), checked against the native anchors.
- Contacts: slot i = native list position i, key = n - 1 - i (list newest first = descending key), with the stored
  manifold, feature keys, impulses and m_friction exactly as recorded (warm start as the game).

Filtering for the candidate pairs: Unity's layer matrix (dumps/engine-dump-start.jsonl `layerCollisionMasks` and
collider layers by path), joint-connected bodies (every Joint2D has enableCollision false), triggers (never solved),
same body, and Box2D's "one body dynamic". Every recorded contact is checked to be a candidate (`check_contacts`).

What is estimated rather than recorded: the broad-phase fat AABBs (not in the records). Static proxies get the shape
AABB + b2_aabbExtension (inserted once, never moved); moving proxies the same at the recorded pose. They decide only
which contacts Collide destroys (never a touching one) and which pairs the end-of-step FindNewContacts creates, so
the solve itself does not depend on them; the harness reports the contact-set agreement separately.
Not loaded: polygon centroids (only mass data uses them; Unity's mass is set verbatim) and Unity's contact flag bits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import shapes
from goi_jax.box2d.settings import MAX_POLYGON_VERTICES
from goi_jax.box2d.world import (AT_LOWER, AT_UPPER, DYNAMIC, EQUAL_LIMITS, INACTIVE_LIMIT, PRISMATIC, REVOLUTE, STATIC,
                                 WELD, Bodies, Contacts, Fixtures, Joints, Model, State)
from goi_jax.parity import states as S

F32, I32 = np.float32, np.int32
FLAG_AWAKE, FLAG_AUTOSLEEP, FLAG_BULLET, FLAG_FIXED_ROT, FLAG_ACTIVE = 0x2, 0x4, 0x8, 0x10, 0x20
JOINT_MODEL_FIELDS = ("joint_type", "joint_a", "joint_b", "local_anchor_a", "local_anchor_b", "local_axis_a",
                      "local_yaxis_a", "reference_angle")


class JointModel(NamedTuple):
    """The Model fields that change between steps (the joint list is re-created each action)."""
    joint_type: np.ndarray
    joint_a: np.ndarray
    joint_b: np.ndarray
    local_anchor_a: np.ndarray
    local_anchor_b: np.ndarray
    local_axis_a: np.ndarray
    local_yaxis_a: np.ndarray
    reference_angle: np.ndarray


def model_axes():
    """vmap in_axes for a Model whose joint fields are batched (JointModel stacked) and the rest shared."""
    return Model(**{f: (0 if f in JOINT_MODEL_FIELDS else None) for f in Model._fields})


def _engine_dump(root: Path):
    for p in (root / "engine-dump-start.jsonl", root.parent / "engine-dump-start.jsonl",
              Path("dumps/engine-dump-start.jsonl")):
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return None


def _shape_key(collider: str, piece: int, shape_raw: bytes) -> tuple:
    """Identify a piece across sessions: collider "path#component", piece, and the shape's meaningful bytes (type,
    radius, then circle centre or polygon centroid/vertices/normals/count; no vtable pointer or trailing bytes).
    Paths alone repeat (several Mountain/Rock)."""
    raw = bytes(shape_raw)
    typ = int.from_bytes(raw[8:12], "little")
    if typ == 0:
        return collider, int(piece), raw[8:0x18]
    n = min(max(int.from_bytes(raw[0x98:0x9C], "little"), 0), 8)
    return collider, int(piece), raw[8:0x18 + 8 * n] + raw[0x58:0x58 + 8 * n]


def _proxy_ids(root: Path):
    """Broad-phase proxy id per piece from a native dump (keyed by _shape_key), or None."""
    for p in (root.parent / "native" / "goi-rl-native-start.jsonl", Path("dumps/native/goi-rl-native-start.jsonl")):
        if p.exists():
            out = {}
            for line in p.read_text().splitlines():
                if '"kind":"fixture"' not in line:
                    continue
                r = json.loads(line)
                out[_shape_key(f"{r['collider']}#{r['component']}", r["piece"], bytes.fromhex(r["shapeRaw"]))] = int(r["proxyIds"][0])
            return out
    return None


@dataclass
class Level:
    """Static description of the level (one per recording session)."""
    names: dict
    n_bodies: int                 # engine bodies, ground included
    body_game: np.ndarray         # (B,) Game body index per engine body (-1 ground)
    body_type: np.ndarray         # (B,)
    native_slot: np.ndarray       # (B,) native list position of each engine body (ground = B - 1)
    fixture: dict                 # arrays per engine fixture: body, shape_type, radius, count, verts, normals,
    #                               friction, restitution, sensor, layer, collider, piece, proxy
    fixture_of: dict              # (collider index, piece) -> engine fixture
    pair_a: np.ndarray            # (P,) candidate fixture A (b2Contact::Create orientation), UpdatePairs order
    pair_b: np.ndarray
    pair_lookup: dict             # frozenset-free: (min f, max f) -> candidate index
    joint_bodies: dict            # joint name index -> (engine body A, engine body B)
    layer_masks: np.ndarray | None
    proxy_source: str
    model: Model                  # shared fields (joint fields are placeholders; see model_for)
    tip_fixtures: np.ndarray      # hammer head pieces (HammerCollisions' collider)
    pot_fixtures: np.ndarray      # pot bottom pieces (PotSounds' bottom collider)
    hammer_body: int
    pot_body: int
    game_to_engine: dict

    # ------------------------------------------------------------------ models

    def model_for(self, jm: JointModel) -> Model:
        """The shared Model with one step's (or a stacked batch of steps') joint fields."""
        return self.model._replace(**{f: jnp.asarray(getattr(jm, f)) for f in JOINT_MODEL_FIELDS})

    # ------------------------------------------------------------------ states

    def native_state(self, chain: S.Chain, t: int, which: int = 0, capacity: int = 96):
        """(JointModel, State) of recorded world `which` (0 = before Simulate, 1 = after) at step t, as numpy."""
        jm, st, _ = self.native_states(chain, [t], which, capacity)
        return jax_tree_index(jm, 0), jax_tree_index(st, 0)

    def native_states(self, chain: S.Chain, steps, which: int = 0, capacity: int = 96, fat: str = "track"):
        """Stacked (JointModel, State, info) for the given steps of a chain (numpy arrays with a leading step axis).
        info: per step dict of loader diagnostics (unmatched contacts, orientation mismatches, ...)."""
        steps = np.asarray(steps, int)
        N, B, F = len(steps), self.n_bodies, len(self.fixture["body"])
        # ---- bodies (native slot k -> engine body)
        nb = chain.bodies[which]
        order = chain.body_index[which][steps]
        expect = self.body_game[np.argsort(self.native_slot)][:-1]
        if not (order == expect[None, :]).all():
            raise ValueError("native body order differs from the geometry record's")
        eng = np.array([self.game_to_engine[int(g)] for g in expect])  # engine body per native slot

        def per_body(name, shape=(), dtype=F32, ground=0):
            out = np.full((N, B) + shape, ground, dtype)
            out[:, eng] = nb[name][steps].reshape((N, len(eng)) + shape)
            return out
        flags = per_body("flags", dtype=np.int64)
        xf = per_body("xf", (4,))
        q = np.stack([xf[..., 3], xf[..., 2]], -1)
        q[:, 0] = (1, 0)
        btype = np.broadcast_to(self.body_type, (N, B)).copy()
        bodies = _build(Bodies, dict(
            type=btype.astype(I32),
            active=np.where(btype == STATIC, True, (flags & FLAG_ACTIVE) != 0),
            awake=(flags & FLAG_AWAKE) != 0, allow_sleep=np.where(btype == STATIC, True, (flags & FLAG_AUTOSLEEP) != 0),
            bullet=(flags & FLAG_BULLET) != 0, continuous=(flags & FLAG_BULLET) != 0,
            sleep_time=per_body("sleepTime"), xf_p=xf[..., :2].copy(), xf_q=q,
            local_center=per_body("localCenter", (2,)), c0=per_body("c0", (2,)), c=per_body("c", (2,)),
            a0=per_body("a0"), a=per_body("a"), v=per_body("v", (2,)), w=per_body("w"),
            force=per_body("force", (2,)), torque=per_body("torque"), mass=per_body("mass"),
            inv_mass=per_body("invMass"), inertia=per_body("I"), inv_inertia=per_body("invI"),
            linear_damping=per_body("linearDamping"), angular_damping=per_body("angularDamping"),
            gravity_scale=per_body("gravityScale", ground=1)), N)
        # ---- fixtures: geometry friction, the scripts' current hammer / pot materials (SCR1 = after the swaps)
        fric = np.broadcast_to(self.fixture["friction"], (N, F)).copy()
        hf = chain.field(1, "hammerFriction")[steps]
        pf = chain.field(1, "potBottomFriction")[steps]
        fric[:, self.tip_fixtures] = hf[:, None]
        fric[:, self.pot_fixtures] = pf[:, None]
        if fat == "track" and which == 0:
            fat = self.track_fat(chain)[steps]
        else:
            fat = self.estimate_fat(bodies, swept=fat != "tight")
        fixtures = _build(Fixtures, dict(friction=fric.astype(F32),
                            restitution=np.broadcast_to(self.fixture["restitution"], (N, F)).astype(F32).copy(),
                            fat=fat), N)
        # ---- joints
        jm, joints = self._joints(chain, steps, which)
        # ---- contacts
        contacts, info = self._contacts(chain, steps, which, capacity)
        wd = chain.world[which]
        n = chain.n_contacts[which][steps]
        kw = dict(bodies=bodies, fixtures=fixtures, joints=joints, contacts=contacts,
                  gravity=wd["gravity"][steps].astype(F32), inv_dt0=wd["inv_dt0"][steps].astype(F32),
                  next_key=n.astype(I32), new_fixture=np.zeros(N, bool), overflow=np.zeros(N, I32))
        if "toi" in State._fields:  # TOI bookkeeping (toi.py): the recorded m_toiCount / m_toi / flags
            kw["toi"] = self._toi(chain, steps, which, capacity, nb, eng)
        st = State(**kw)
        return jm, st, info

    def _toi(self, chain, steps, which, C, nb, eng):
        from goi_jax.box2d import world as W
        N, B = len(steps), self.n_bodies
        cd, ncon = chain.contacts[which], chain.n_contacts[which]
        alpha0 = np.zeros((N, B), F32)
        alpha0[:, eng] = nb["alpha0"][steps]
        f = {k: np.zeros((N, C), d) for k, d in (("count", I32), ("toi", F32), ("flag", bool), ("enabled", bool))}
        f["toi"][:] = 1
        f["enabled"][:] = True
        for t, s in enumerate(steps):
            n = int(ncon[s])
            f["count"][t, :n] = cd["toiCount"][s, :n]
            f["toi"][t, :n] = cd["toi"][s, :n]
            fl = cd["flags"][s, :n].astype(np.int64)
            f["flag"][t, :n] = (fl & 0x20) != 0
            f["enabled"][t, :n] = (fl & S.CONTACT_ENABLED) != 0
        kw = dict(alpha0=alpha0, solved=np.zeros((N, C), bool), normal_impulse=np.zeros((N, C, 2), F32),
                  tangent_impulse=np.zeros((N, C, 2), F32), events=np.zeros(N, I32), **f)
        return _build(W.ContactTOI, kw, N)

    def estimate_fat(self, bodies: Bodies, swept=True) -> np.ndarray:
        """(N, F, 4) estimated fat AABBs. Static: shape AABB + b2_aabbExtension (inserted once, never moved). Moving
        (swept=True): what b2DynamicTree::MoveProxy would have built at the end of the previous step, from the
        sweep (c0, a0) -> current pose (b2Fixture::Synchronize) fattened with 2x the displacement; Box2D's real fat
        box is that one, or an older one that still contained it (then larger). swept=False: the current shape
        AABB + extension."""
        import jax
        from goi_jax.box2d import math2d as m2
        fb = self.fixture["body"]
        m = self.model

        def fat(bd):
            px, py = bd.xf_p[fb, 0], bd.xf_p[fb, 1]
            c, s = bd.xf_q[fb, 0], bd.xf_q[fb, 1]
            a2 = shapes.fixture_aabbs(m, px, py, c, s)
            if not swept:
                return jnp.stack(shapes.fatten(*a2), -1)
            c1, s1 = m2.rot_of(bd.a0)
            rx, ry = m2.rot(c1, s1, bd.local_center[:, 0], bd.local_center[:, 1])
            p1x, p1y = bd.c0[:, 0] - rx, bd.c0[:, 1] - ry
            a1 = shapes.fixture_aabbs(m, p1x[fb], p1y[fb], c1[fb], s1[fb])
            lo = (m2.bmin(a1[0], a2[0]), m2.bmin(a1[1], a2[1]), m2.bmax(a1[2], a2[2]), m2.bmax(a1[3], a2[3]))
            moved = jnp.stack(shapes.fatten(*lo, px - p1x[fb], py - p1y[fb]), -1)
            still = jnp.stack(shapes.fatten(*a2), -1)
            return jnp.where((bd.type[fb] == STATIC)[:, None], still, moved)
        f = jax.jit(jax.vmap(fat))
        return np.asarray(f(jax.tree.map(jnp.asarray, bodies)))

    def track_fat(self, chain: S.Chain) -> np.ndarray:
        """(N, F, 4) fat AABBs at every NAT0 of a chain, by replaying b2DynamicTree::MoveProxy through the chain:
        the first step's are estimate_fat's; after each recorded step every moving fixture is synchronized over
        its recorded sweep (NAT1 c0/a0 -> NAT1 pose: b2Body::SynchronizeFixtures) and then at the next NAT0 pose
        (the per-action Transform sync, b2Body::SetTransform: zero displacement); each re-fattens only when its
        swept AABB leaves the fat box. Exact for every proxy that re-fattened at least once in the chain. Cached."""
        cache = getattr(self, "_fat_cache", None)
        if cache is not None and cache[0] is chain:
            return cache[1]
        import jax
        from goi_jax.box2d import math2d as m2
        N = len(chain)
        _, st0, _ = self.native_states(chain, np.arange(N), 0, capacity=max(1, int(chain.n_contacts[0].max())),
                                       fat="swept")
        _, st1, _ = self.native_states(chain, np.arange(N), 1, capacity=max(1, int(chain.n_contacts[1].max())),
                                       fat="tight")
        fb = self.fixture["body"]
        m = self.model

        def sweep_box(bd):
            px, py = bd.xf_p[fb, 0], bd.xf_p[fb, 1]
            a2 = shapes.fixture_aabbs(m, px, py, bd.xf_q[fb, 0], bd.xf_q[fb, 1])
            c1, s1 = m2.rot_of(bd.a0)
            rx, ry = m2.rot(c1, s1, bd.local_center[:, 0], bd.local_center[:, 1])
            p1x, p1y = bd.c0[:, 0] - rx, bd.c0[:, 1] - ry
            a1 = shapes.fixture_aabbs(m, p1x[fb], p1y[fb], c1[fb], s1[fb])
            lo = jnp.stack((m2.bmin(a1[0], a2[0]), m2.bmin(a1[1], a2[1]), m2.bmax(a1[2], a2[2]),
                            m2.bmax(a1[3], a2[3])), -1)
            fat = jnp.stack(shapes.fatten(*(lo[:, k] for k in range(4)), px - p1x[fb], py - p1y[fb]), -1)
            return lo, fat

        def pose_box(bd):
            a = shapes.fixture_aabbs(m, bd.xf_p[fb, 0], bd.xf_p[fb, 1], bd.xf_q[fb, 0], bd.xf_q[fb, 1])
            return jnp.stack(a, -1), jnp.stack(shapes.fatten(*a), -1)
        sw_lo, sw_fat = (np.asarray(x) for x in jax.jit(jax.vmap(sweep_box))(jax.tree.map(jnp.asarray, st1.bodies)))
        po_lo, po_fat = (np.asarray(x) for x in jax.jit(jax.vmap(pose_box))(jax.tree.map(jnp.asarray, st0.bodies)))
        moving = (self.body_type[fb] != STATIC)
        out = np.empty_like(st0.fixtures.fat)
        cur = st0.fixtures.fat[0].copy()

        def contains(f, a):
            return (f[:, 0] <= a[:, 0]) & (f[:, 1] <= a[:, 1]) & (a[:, 2] <= f[:, 2]) & (a[:, 3] <= f[:, 3])
        for t in range(N):
            if t > 0:  # Transform sync at this NAT0 pose (zero displacement)
                upd = moving & ~contains(cur, po_lo[t])
                cur = np.where(upd[:, None], po_fat[t], cur)
            out[t] = cur
            # the step itself: island bodies (awake, non-static, after the step) synchronize over their sweep
            aw = st1.bodies.awake[t][fb] & moving
            upd = aw & ~contains(cur, sw_lo[t])
            cur = np.where(upd[:, None], sw_fat[t], cur)
        self._fat_cache = (chain, out)
        return out

    def _joints(self, chain, steps, which):
        N = len(steps)
        jidx = chain.joint_index[which][steps]  # (N, NJ) name index per native joint slot
        NJ = jidx.shape[1]
        raw = chain.joints_raw[which][steps]
        jd = chain.joints[which]
        jtype = jd[1]["type"][steps]
        z = lambda *s, dt=F32: np.zeros((N, NJ) + s, dt)
        out = dict(joint_type=z(dt=I32), joint_a=z(dt=I32), joint_b=z(dt=I32), local_anchor_a=z(2), local_anchor_b=z(2),
                   local_axis_a=z(2), local_yaxis_a=z(2), reference_angle=z())
        js = dict(enable_motor=z(dt=bool), motor_speed=z(), max_motor=z(), enable_limit=z(dt=bool), lower=z(), upper=z(),
                  frequency=z(), damping=z(), impulse=z(3), motor_impulse=z(), limit_state=z(dt=I32),
                  lower_impulse=z(), upper_impulse=z())
        for t in range(N):
            for k in range(NJ):
                if jidx[t, k] < 0:
                    raise ValueError("missing joint slot")
                e = NJ - 1 - k  # newest first -> engine index
                typ = int(jtype[t, k])
                d = {name: v[steps[t], k] for name, v in jd[typ].items()}
                a, b = self.joint_bodies[int(jidx[t, k])]
                out["joint_type"][t, e], out["joint_a"][t, e], out["joint_b"][t, e] = typ, a, b
                out["local_anchor_a"][t, e], out["local_anchor_b"][t, e] = d["localAnchorA"], d["localAnchorB"]
                out["reference_angle"][t, e] = d["referenceAngle"]
                if typ == REVOLUTE:
                    js["enable_motor"][t, e], js["enable_limit"][t, e] = bool(d["enableMotor"]), bool(d["enableLimit"])
                    js["motor_speed"][t, e], js["max_motor"][t, e] = d["motorSpeed"], d["maxMotorTorque"]
                    js["lower"][t, e], js["upper"][t, e] = d["lowerAngle"], d["upperAngle"]
                    lo, up = d["lowerImpulse"], d["upperImpulse"]
                    js["impulse"][t, e] = (d["impulse"][0], d["impulse"][1], F32(lo - up))  # v2.3 slot: z = lower - upper
                    js["motor_impulse"][t, e] = d["motorImpulse"]
                    js["lower_impulse"][t, e], js["upper_impulse"][t, e] = lo, up
                    js["limit_state"][t, e] = INACTIVE_LIMIT  # v2.4 keeps none; set_limit_states guesses v2.3's
                elif typ == PRISMATIC:
                    out["local_axis_a"][t, e], out["local_yaxis_a"][t, e] = d["localXAxisA"], d["localYAxisA"]
                    js["enable_motor"][t, e], js["enable_limit"][t, e] = bool(d["enableMotor"]), bool(d["enableLimit"])
                    js["motor_speed"][t, e], js["max_motor"][t, e] = d["motorSpeed"], d["maxMotorForce"]
                    js["lower"][t, e], js["upper"][t, e] = d["lowerTranslation"], d["upperTranslation"]
                    js["impulse"][t, e] = d["impulse"]
                    js["motor_impulse"][t, e] = d["motorImpulse"]
                    js["limit_state"][t, e] = d["limitState"]
                elif typ == WELD:
                    js["frequency"][t, e], js["damping"][t, e] = d["frequencyHz"], d["dampingRatio"]
                    js["impulse"][t, e] = d["impulse"]
                else:
                    raise ValueError(f"joint type {typ}")
        jm = JointModel(**out)
        missing = [f for f in Joints._fields if f not in js]
        if missing:  # a joint port added fields this loader does not know: zeros (cold) and say so once
            _warn_once(f"level: Joints fields {missing} not loaded (zeros)")
        joints = Joints(**{f: js[f] if f in js else np.zeros((N, NJ), F32) for f in Joints._fields})
        return jm, joints

    def set_limit_states(self, jm: JointModel, st: State, settings) -> State:
        """Box2D 2.3.1's revolute limit state for each recorded step (the game's v2.4 joint keeps none): what
        InitVelocityConstraints computes from the start pose, so the v2.3 port keeps the warm limit impulse only
        when the state is unchanged. Only used by the v2.3 revolute path."""
        bd, jt = st.bodies, st.joints
        ia, ib = jm.joint_a, jm.joint_b
        ang = np.take_along_axis(bd.a, ib, 1) - np.take_along_axis(bd.a, ia, 1) - jm.reference_angle
        ang = ang.astype(F32)
        slop2 = F32(2) * F32(settings.angular_slop)
        eq = np.abs(jt.upper - jt.lower) < slop2
        ls = np.where(eq, EQUAL_LIMITS, np.where(ang <= jt.lower, AT_LOWER,
                                                np.where(ang >= jt.upper, AT_UPPER, INACTIVE_LIMIT)))
        rev = (jm.joint_type == REVOLUTE) & jt.enable_limit
        ls = np.where(rev, ls, np.where(jm.joint_type == REVOLUTE, INACTIVE_LIMIT, jt.limit_state)).astype(I32)
        return st._replace(joints=jt._replace(limit_state=ls))

    def _contacts(self, chain, steps, which, C):
        N = len(steps)
        cd, ids, ncon = chain.contacts[which], chain.contact_ids[which], chain.n_contacts[which]
        z = lambda *s, dt=F32: np.zeros((N, C) + s, dt)
        pair = np.full((N, C), -1, I32)
        out = dict(fa=z(dt=I32), fb=z(dt=I32), key=z(dt=I32), touching=z(dt=bool), mtype=z(dt=I32), count=z(dt=I32),
                   local_normal=z(2), local_point=z(2), point_local=z(2, 2), point_id=z(2, dt=I32),
                   normal_impulse=z(2), tangent_impulse=z(2), friction=z(), restitution=z(), tangent_speed=z())
        info = []
        for t, s in enumerate(steps):
            n = int(ncon[s])
            if n > C:
                raise ValueError(f"{n} contacts > capacity {C}")
            bad = flipped = 0
            for i in range(n):
                ca, pa, cb, pb = (int(x) for x in ids[s, i])
                fa, fb = self.fixture_of[(ca, pa)], self.fixture_of[(cb, pb)]
                p = self.pair_lookup.get((min(fa, fb), max(fa, fb)), -1)
                if p < 0:
                    bad += 1
                elif (self.pair_a[p], self.pair_b[p]) != (fa, fb):
                    flipped += 1
                # A recorded contact that is not a candidate (a sensor overlap, e.g. the Splashes trigger: Box2D
                # keeps those in the contact list but never solves them) is loaded as an empty slot. Loading it as
                # candidate 0 made it a live, touching contact on the wrong fixture pair (metre-scale errors).
                pair[t, i] = p
                out["fa"][t, i], out["fb"][t, i], out["key"][t, i] = fa, fb, n - 1 - i
                fl = int(cd["flags"][s, i])
                out["touching"][t, i] = bool(fl & S.CONTACT_TOUCHING)
                out["mtype"][t, i], out["count"][t, i] = cd["manifoldType"][s, i], cd["pointCount"][s, i]
                out["local_normal"][t, i], out["local_point"][t, i] = cd["localNormal"][s, i], cd["localPoint"][s, i]
                out["point_local"][t, i] = cd["pointLocal"][s, i]
                out["point_id"][t, i] = cd["key"][s, i].astype(np.uint32).view(I32)
                out["normal_impulse"][t, i], out["tangent_impulse"][t, i] = cd["normalImpulse"][s, i], cd["tangentImpulse"][s, i]
                out["friction"][t, i], out["restitution"][t, i] = cd["friction"][s, i], cd["restitution"][s, i]
                out["tangent_speed"][t, i] = cd["tangentSpeed"][s, i]
            info.append(dict(not_candidate=bad, flipped=flipped,
                             disabled=int(sum(1 for i in range(n) if (int(cd["flags"][s, i]) & 6) == 2))))
        return _build(Contacts, dict(pair=pair, **out), N), info

    # ------------------------------------------------------------------ export

    def export(self, st) -> dict:
        """A (batched, leading axis N) engine State in the recorded layout:
        bodies: dict of (N, NB, ...) in native slot order with BODY_FIELDS names (xf = p.x, p.y, q.s, q.c);
        contacts: per step list of dicts keyed by (colliderA, pieceA, colliderB, pieceB) in list order (newest first);
        joints: (N, J, 3) impulse and (N, J) motor impulse by engine index."""
        bd = jax_np(st.bodies)
        order = np.argsort(self.native_slot)[:-1]  # engine body per native slot (ground excluded)
        g = lambda a: a[:, order]
        xf = np.concatenate([g(bd.xf_p), g(bd.xf_q)[..., 1:2], g(bd.xf_q)[..., 0:1]], -1)
        bodies = dict(xf=xf, c0=g(bd.c0), c=g(bd.c), a0=g(bd.a0), a=g(bd.a), v=g(bd.v), w=g(bd.w),
                      sleepTime=g(bd.sleep_time), awake=g(bd.awake), localCenter=g(bd.local_center))
        ct = jax_np(st.contacts)
        col, piece = self.fixture["collider"], self.fixture["piece"]
        contacts = []
        for t in range(ct.pair.shape[0]):
            live = np.nonzero(ct.pair[t] >= 0)[0]
            live = live[np.argsort(-ct.key[t, live], kind="stable")]
            rows = {}
            for pos, i in enumerate(live):
                fa, fb = int(ct.fa[t, i]), int(ct.fb[t, i])
                k = (int(col[fa]), int(piece[fa]), int(col[fb]), int(piece[fb]))
                rows[k] = dict(pos=pos, touching=bool(ct.touching[t, i]), count=int(ct.count[t, i]),
                               mtype=int(ct.mtype[t, i]), point_id=ct.point_id[t, i].copy(),
                               normal_impulse=ct.normal_impulse[t, i].copy(),
                               tangent_impulse=ct.tangent_impulse[t, i].copy(),
                               local_normal=ct.local_normal[t, i].copy(), point_local=ct.point_local[t, i].copy(),
                               friction=F32(ct.friction[t, i]))
            contacts.append(rows)
        jt = jax_np(st.joints)
        return dict(bodies=bodies, contacts=contacts, joint_impulse=jt.impulse, motor_impulse=jt.motor_impulse,
                    joints=jt)


_warned = set()


def _build(cls, kw, N):
    """cls(**kw), with fields another engine change added filled with zeros (and a one-time note)."""
    missing = [f for f in cls._fields if f not in kw and f not in getattr(cls, "_field_defaults", {})]
    if missing:
        _warn_once(f"level: {cls.__name__} fields {missing} not loaded (zeros)")
        for f in missing:
            kw[f] = np.zeros(N, F32)
    return cls(**{k: v for k, v in kw.items() if k in cls._fields})


def _warn_once(msg):
    if msg not in _warned:
        _warned.add(msg)
        print(msg, flush=True)


def jax_np(tree):
    return type(tree)(*(np.asarray(x) for x in tree))


def jax_tree_index(tree, i):
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):
        return type(tree)(*(jax_tree_index(x, i) for x in tree))
    return np.asarray(tree)[i]


# ---------------------------------------------------------------------------------------------------------------------

def load_level(root="dumps/parity", use_proxy_ids=True, body_order="unity") -> Level:
    """The level of a recording session: geometry.bin, the chains' names table, layer data and proxy ids."""
    root = Path(root)
    geo = S.parse_geometry((root / "geometry.bin").read_bytes())
    names = {k: geo[k] for k in ("colliders", "body_names", "joints")}
    nat = geo["bodies"]  # native list order, ground last
    B = len(nat)
    if nat[-1]["index"] != -1 or nat[-1]["type"] != STATIC:
        raise ValueError("expected the static ground body last in the native list")
    # The engine seeds islands from the highest body index down (stock Box2D: m_bodyList, newest first).
    # body_order="box2d": engine e <- native position B-1-e, i.e. seeds in recorded list order (Orange first).
    # body_order="unity" (default): engine e <- native position e for the moving bodies (ground stays engine 0), i.e.
    # seeds walk the recorded list from its tail (Tip, PoleMiddle, ..., Orange). Measured: with the game's own
    # velocities, the position iterations then reproduce NAT1 bit for bit on 99% of contact-free-TOI steps (stock
    # order: 0.2%, median 2.8e-4 off), because the climber's joints are solved Tip weld first, pot hinge last.
    if body_order == "box2d":
        native_slot = np.array([B - 1 - e for e in range(B)])
    elif body_order == "unity":
        native_slot = np.array([B - 1] + list(range(B - 1)))
    else:
        raise ValueError(body_order)
    body_game = np.array([nat[native_slot[e]]["index"] for e in range(B)])
    body_type = np.array([nat[native_slot[e]]["type"] for e in range(B)], I32)
    game_to_engine = {int(g): e for e, g in enumerate(body_game)}

    rows = _engine_dump(root)
    layer_of, masks, joint_conn = {}, None, {}
    if rows:
        for r in rows:
            if r["kind"] == "collider":
                layer_of[r["path"]] = int(r["layer"])
            elif r["kind"] == "settings":
                masks = np.array(r["layerCollisionMasks"], np.int64) & 0xFFFFFFFF
            elif r["kind"] == "joint":
                joint_conn[f"{r['path']}#{r['type']}"] = r.get("connected")
    proxies = _proxy_ids(root) if use_proxy_ids else None

    fx = {k: [] for k in ("body", "shape_type", "radius", "count", "verts", "normals", "friction", "restitution",
                          "sensor", "layer", "collider", "piece", "proxy")}
    for e in range(B):
        b = nat[native_slot[e]]
        for k in reversed(range(len(b["collider"]))):
            sh = S.shape_of(b["shape"][k])
            raw = b["fixture"][k]
            col, piece = int(b["collider"][k]), int(b["piece"][k])
            cname = names["colliders"][col]
            if sh["type"] == 0:
                v, n, cnt, st = np.array([sh["p"]], F32), np.zeros((1, 2), F32), 1, shapes.CIRCLE
            else:
                v, n, cnt, st = sh["vertices"].astype(F32), sh["normals"].astype(F32), len(sh["vertices"]), shapes.POLYGON
            vv, nn = shapes.pad(v, n)
            fx["body"].append(e)
            fx["shape_type"].append(st)
            fx["radius"].append(F32(sh["radius"]))
            fx["count"].append(cnt)
            fx["verts"].append(vv)
            fx["normals"].append(nn)
            fx["friction"].append(raw[0x30:0x34].view("<f4")[0])
            fx["restitution"].append(raw[0x34:0x38].view("<f4")[0])
            fx["sensor"].append(bool(raw[0x4A]))
            fx["layer"].append(layer_of.get(cname.rsplit("#", 1)[0], 0))
            fx["collider"].append(col)
            fx["piece"].append(piece)
            fx["proxy"].append(proxies.get(_shape_key(cname, piece, b["shape"][k].tobytes()), -1) if proxies else -1)
    fixture = {k: np.array(v) for k, v in fx.items()}
    fixture["verts"] = fixture["verts"].astype(F32).reshape(-1, MAX_POLYGON_VERTICES, 2)
    fixture["normals"] = fixture["normals"].astype(F32).reshape(-1, MAX_POLYGON_VERTICES, 2)
    for k in ("radius", "friction", "restitution"):
        fixture[k] = fixture[k].astype(F32)
    F = len(fixture["body"])
    if proxies and (fixture["proxy"] >= 0).all() and len(set(fixture["proxy"].tolist())) == F:
        proxy_order, proxy_source = fixture["proxy"], "native dump proxy ids"
    else:
        proxy_order, proxy_source = np.arange(F), "fixture order (no proxy ids)"
    fixture_of = {(int(c), int(p)): f for f, (c, p) in enumerate(zip(fixture["collider"], fixture["piece"]))}

    # joints: bodyA = own GameObject's body, bodyB = connected body
    body_names = names["body_names"]
    joint_bodies = {}
    for j, jname in enumerate(names["joints"]):
        path = jname.rsplit("#", 1)[0]
        conn = joint_conn.get(jname)
        if path in body_names and conn in body_names:
            joint_bodies[j] = (game_to_engine[body_names.index(path)], game_to_engine[body_names.index(conn)])
    pair_a, pair_b = candidate_pairs(fixture, body_type, joint_bodies.values(), masks, proxy_order)
    lookup = {(int(min(a, b)), int(max(a, b))): p for p, (a, b) in enumerate(zip(pair_a, pair_b))}

    cols = names["colliders"]
    hammer = [i for i, c in enumerate(cols) if c == "Player/Hub/Slider/Handle/PoleMiddle/Tip#0"]
    pot = [i for i, c in enumerate(cols) if c == "Player/PotCollider#0"]
    tip_fx = np.nonzero(np.isin(fixture["collider"], hammer))[0]
    pot_fx = np.nonzero(np.isin(fixture["collider"], pot))[0]
    J = 1
    zj = lambda *s, dt=F32: jnp.zeros((J,) + s, dt)
    model = Model(fixture_body=jnp.asarray(fixture["body"].astype(I32)),
                  shape_type=jnp.asarray(fixture["shape_type"].astype(I32)), radius=jnp.asarray(fixture["radius"]),
                  count=jnp.asarray(fixture["count"].astype(I32)), verts=jnp.asarray(fixture["verts"]),
                  normals=jnp.asarray(fixture["normals"]),
                  joint_type=zj(dt=jnp.int32), joint_a=zj(dt=jnp.int32), joint_b=zj(dt=jnp.int32),
                  local_anchor_a=zj(2), local_anchor_b=zj(2), local_axis_a=zj(2), local_yaxis_a=zj(2),
                  reference_angle=zj(), pair_a=jnp.asarray(pair_a), pair_b=jnp.asarray(pair_b))
    return Level(names=names, n_bodies=B, body_game=body_game, body_type=body_type, native_slot=native_slot,
                 fixture=fixture, fixture_of=fixture_of, pair_a=pair_a, pair_b=pair_b, pair_lookup=lookup,
                 joint_bodies=joint_bodies, layer_masks=masks, proxy_source=proxy_source, model=model,
                 tip_fixtures=tip_fx, pot_fixtures=pot_fx,
                 hammer_body=game_to_engine[body_names.index("Player/Hub/Slider/Handle/PoleMiddle/Tip")],
                 pot_body=game_to_engine[body_names.index("Player")], game_to_engine=game_to_engine)


def tip_points(level: Level) -> dict:
    """{tip fixture: (n, 2) authored PolygonCollider2D points of its vertices}, recovered from the session geometry
    (baked at the level start as a pure 1 + 2 ulp scale; goi_jax/game/placement.py)."""
    from goi_jax.game import placement
    fx = level.fixture
    return {int(f): placement.invert_bake(fx["verts"][f, :int(fx["count"][f])]) for f in level.tip_fixtures}


def with_tip_bake(level: Level, A, points: dict | None = None) -> Level:
    """A copy of `level` whose hammer-head pieces are the authored points baked through the 2x2 matrix A (one
    rebuild's collider bake, goi_jax/game/placement.py); everything else shared."""
    import dataclasses
    from goi_jax.game import placement
    points = tip_points(level) if points is None else points
    fixture = dict(level.fixture)
    verts, normals = np.array(fixture["verts"], F32), np.array(fixture["normals"], F32)
    for f, p in points.items():
        verts[f], normals[f] = placement.polygon(placement.bake_vertices(p, A))
    fixture["verts"], fixture["normals"] = verts, normals
    model = level.model._replace(verts=jnp.asarray(verts), normals=jnp.asarray(normals))
    return dataclasses.replace(level, fixture=fixture, model=model)


def candidate_pairs(fixture, body_type, joint_pairs, masks, proxy_order):
    """Vectorized candidate list: every fixture pair that can form a solver contact, sorted as UpdatePairs sorts
    (ascending (lower proxy id, higher proxy id)), oriented as b2Contact::Create (polygon first in polygon-circle;
    else the lower proxy id is A). Excluded: same body, a trigger, no dynamic body, joint-connected bodies, and
    Unity's layer matrix."""
    fb, sensor, layer = fixture["body"], fixture["sensor"], fixture["layer"]
    F = len(fb)
    dyn = body_type[fb] == DYNAMIC
    # Only pairs with a dynamic fixture: index the dynamic ones against all.
    d_idx = np.nonzero(dyn & ~sensor)[0]
    o_idx = np.nonzero(~sensor)[0]
    I, K = np.meshgrid(d_idx, o_idx, indexing="ij")
    I, K = I.ravel(), K.ravel()
    keep = fb[I] != fb[K]
    # each unordered pair once: if both dynamic keep only I < K
    keep &= ~(dyn[K] & (K <= I))
    I, K = I[keep], K[keep]
    jp = {(min(a, b), max(a, b)) for a, b in joint_pairs}
    if jp:
        ba, bb = np.minimum(fb[I], fb[K]), np.maximum(fb[I], fb[K])
        code = ba.astype(np.int64) * 100000 + bb
        jcode = np.array([a * 100000 + b for a, b in jp], np.int64)
        keep = ~np.isin(code, jcode)
        I, K = I[keep], K[keep]
    if masks is not None:
        keep = ((masks[layer[I]] >> layer[K]) & 1) != 0
        I, K = I[keep], K[keep]
    lo = np.where(proxy_order[I] < proxy_order[K], I, K)
    hi = np.where(proxy_order[I] < proxy_order[K], K, I)
    srt = np.lexsort((proxy_order[hi], proxy_order[lo]))
    lo, hi = lo[srt], hi[srt]
    st = fixture["shape_type"]
    swap = (st[lo] == shapes.CIRCLE) & (st[hi] == shapes.POLYGON)
    a = np.where(swap, hi, lo).astype(I32)
    b = np.where(swap, lo, hi).astype(I32)
    return a, b

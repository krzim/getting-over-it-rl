"""The plugin's lockstep game in JAX: one agent action = plugin/Game.cs Step (docs/understanding/overview.md §1.2), as
pure jit/vmap-able functions over a fixed-shape SimState.

    sim = Sim.from_recordings("dumps/parity")             # level, constants, canonical joint order (host, once)
    model, params = sim.model, sim.params                # pass both as jit ARGUMENTS (see player.py on closures)
    s = sim.state_from_recording(chain, t)               # a recorded pre-step world + script state (numpy pytree)
    s = jax.jit(sim.act, static_argnames="k")(model, params, s, action, k=4)    # one action: K physics steps

Per physics step (`physics_step`), in the plugin's order (plugin/Game.cs Step):
1. HammerCollisions / PotSounds FixedUpdate: the hammer-head and pot-bottom materials from the slide flags the last
   step's callbacks left (friction.materials); Unity refreshes the fixtures and every existing contact's mixed
   friction at assignment, before the solve (unity.py, T5).
2. PlayerControl.FixedUpdate (player.controller_step): inputs are the Transform positions of the pot, Tip and Hub
   (body origin, Transform z constants: pot/hub from params, Tip z in the state), the main hinge's jointAngle =
   ((bB.a - bA.a) - ref) * 57.29578 and referenceAngle * 57.29578 in degrees (both float32; bit-exact on every
   recorded step), and the slider's reaction torque (0 on the first step after the joints were re-created, else the
   previous step's prismatic impulse.y * (1/dt)). Writes the hinge motor speed (deg/s * 0.0174532924 -> rad/s, exact
   on all recorded steps), the slider motor speed, and the cursor MovePosition + velocity (or, on a HammerReturn snap
   step, the pending snap target instead).
3. simTime += dt; crane MovePosition (x0 + Mathf.Sin(simTime) * 0.01, y0), Mono double sum (exact on all recorded
   crane targets).
4. PhysicsScene2D.Simulate: kinematic MovePosition velocities, b2World::Step (Settings.game()), MovePosition finish
   (unity.py). The Box2D step is composed here from goi_jax.box2d's pieces (same calls as box2d.step) only to keep
   the velocities between the solve and SolveTOI (relativeVelocity of contacts that begin in a TOI sub-step).
5. Collision callbacks (`callbacks`): per (script, other collider) the touching state is the OR over the piece
   contacts of the Tip collider / Player/PotCollider#0; Enter / Stay / Exit from the previous step's state. Order:
   Unity keeps a list of the active pairs: new pairs are appended (ascending newest-contact key), the list is
   dispatched in order, and an Exit swap-removes its entry in place (the last entry moves into the slot and is
   dispatched next). Measured from the recorded native contacts of all 4200 steps: the callback SET matches on 4199
   (the miss: a contact that began and ended inside one step) and the order on 404 of the 426 steps with several
   callbacks (misses: the list order at a chain's first step, unknowable, and 3 pairs of simultaneous new pairs).
   Points: ContactPoint2D order = the pair's touching contacts oldest first, manifold points in order (2256 of
   2529 recorded non-TOI callbacks; count exact on all); impulses = the solve's, or the TOI sub-step's for a contact
   solved there (what Unity reports). relativeVelocity (Enter, hammer hard hit > 14) = the other body's minus the own
   body's point velocity at the first reported point, with the velocities at the start of the step (exact on every
   recorded non-TOI Enter) or, for a contact that went through TOI, after the solve (approximation).
   friction.apply_callbacks, then player.hammer_return (snap target for the next step).
After K steps (`end_action`): the plugin's pose update re-creates the climber's 5 joints. Measured: the native joint
order after a re-creation is always the same (canonical, `Sim.joint_order`), so only the climber joints' accumulated
impulses go cold (impulse, motor, limit impulses = 0; prismatic limit state inactive) and b2World::DestroyJoint's
SetAwake(true) wakes their bodies; the next step's reaction torque reads 0.

Pose nudge (Sim(pose_nudge=True), off by default; goi_jax/game/pose.py): Unity writes every moved body into its 3D
Transform after each Simulate and, after the once-per-action pose update, re-syncs the climber's bodies from those
Transforms (positions move by up to 3.8e-6, angles stay). With the setting on, SimState.pose carries the climber's
Transform locals: physics_step writes them after the step, the controller and HammerReturn read Transform positions
(pot, Hub, Tip, Tip z) from them, and end_action applies the Hub restore and the re-sync (positions and centres of
mass). The Transform write (rotations and positions) is bit-exact (pose.py); one K=4 action from a recorded state
ends bit-exact at the next action's start on 92% of actions with the setting on vs 6% off (c00, c17). Not yet
modelled with it: the re-created joints' anchors (Unity anchor * Transform.lossyScale, pose.joint_anchor), so it
stays off by default. goi_jax/parity/horizon.py compares sim (off), model (on) and oracle (the recorded nudge injected).

Not modelled:
- With pose_nudge off: the pose round trip above, and the Transform positions the controller reads within an action
  at K > 1 (they differ from the body origins in the last bits on 81% of the recorded steps 2..4, which changes the
  slider motor speed on 75% of them, by up to 1.5e-4); the Tip Transform z is kept constant.
- (Space, route progress ~760 and beyond: the zero-gravity zone, its wells, the teleporters and the win are modelled
  by goi_jax/game/space.py after each step when Sim(space=True), the default, and the state carries SimState.space.)
  The gift (Present, route ~525-630) never spawns. The lake trigger and Snake
  contact are observation-side (env.py / observe.py), not physics.
- Transform reads go through the 3D hierarchy in the game (Transform world position of a child body), which can
  differ from the body origin by an ulp: the HammerReturn snap target (the Tip Transform) is the Tip body origin here
  (1 ulp off on the recorded snap of c01 step 12).
- A contact that begins and ends inside one step (Unity reports Enter + Exit; seen once in 4200 steps).
- More than M active pairs (M = 16 by default; at most 4 recorded): new pairs beyond the list are not reported.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import box2d as jb
from goi_jax.box2d import math2d as m
from goi_jax.box2d.pairs import collide, find_new_contacts
from goi_jax.box2d.step import solve as box2d_solve, start_new_contacts
from goi_jax.box2d.toi import solve_toi
from goi_jax.box2d.gpu import graph
from goi_jax.box2d.world import CIRCLES, FACE_A, compact_model, compact_state, fixture_friction, is_compact
from goi_jax.game import floatmath as fm
from goi_jax.game import friction as fr
from goi_jax.game import player
from goi_jax.game import pose
from goi_jax.game import unity

F32, I32 = np.float32, np.int32
f32 = jnp.float32
TERRAIN = 10
DEG2RAD = F32(0.0174532924)  # Mathf.Deg2Rad
RAD2DEG = F32(57.29578)      # Mathf.Rad2Deg
TIP_COLLIDER = "Player/Hub/Slider/Handle/PoleMiddle/Tip#0"
POT_COLLIDER = "Player/PotCollider#0"
CLIMBER_JOINTS = ("Player#HingeJoint2D", "Player/Hub#HingeJoint2D", "Player/Hub/Slider#SliderJoint2D",
                  "Player/Hub/Slider/Handle#FixedJoint2D", "Player/Hub/Slider/Handle/PoleMiddle#FixedJoint2D")
BIG = np.int32(2 ** 30)


class SimParams(NamedTuple):
    """Constants of the lockstep game (a pytree: pass it as a jit argument)."""
    controller: player.ControllerParams
    friction: fr.FrictionParams
    crane_start: jax.Array   # (2,) craneStart (Rigidbody2D position at load)
    pot_z: jax.Array         # Player Transform z (0)
    hub_z: jax.Array         # Player/Hub Transform z (the plugin restores it after the pose update)
    dt: jax.Array            # Time.fixedDeltaTime


class SimState(NamedTuple):
    world: jb.State                    # Box2D world (bodies, fixtures, joints, contacts)
    ctrl: player.ControllerState       # PlayerControl fields
    fric: fr.FrictionState             # slide flags + tracked Terrain colliders
    touch: jax.Array                   # (2, NCOL) bool: (hammer, pot) x other collider touching after the last step
    pairs: jax.Array                   # (L,) int32: Unity's list of active (script, collider) pairs, flat id
    #                                    script * NCOL + collider, in callback order (-1 empty)
    n_pairs: jax.Array                 # int32 entries in `pairs`
    sim_time: jax.Array                # f32 plugin simTime (crane phase)
    snap_target: jax.Array             # (2,) cursor MovePosition pending from HammerReturn (used while mouse_snap)
    reaction_torque: jax.Array         # f32 slider reaction torque the controller reads next step
    tip_z: jax.Array                   # f32 Tip Transform z (PoseControl's; constant unless pose_nudge)
    pose: object = None                # pose.PoseState (the climber's Transform locals) when pose_nudge, else None
    space: object = None               # space.SpaceState (zone / teleporter triggers, Won) when Sim(space=True)


class StepInfo(NamedTuple):
    """Per physics step diagnostics (for tests and the horizon report)."""
    hinge_speed: jax.Array
    slider_speed: jax.Array
    hard_hit: jax.Array
    snapped: jax.Array
    n_callbacks: jax.Array
    toi_events: jax.Array


def _np(tree):
    return jax.tree.map(np.asarray, tree)


class Sim:
    """Static description of the lockstep game: the level, body/joint/fixture indices, callback tables. Host object;
    its methods are pure functions of (model, params, state, ...) and can be jitted/vmapped (closing over `self` only
    bakes small index tables)."""

    def __init__(self, level, model, params: SimParams, settings=None, joint_order=None, callbacks_max=16,
                 pose_nudge=False, pose_tilt=None, space=True):
        self.level = level
        self.pose_nudge = bool(pose_nudge)
        self.model = model
        self.params = params
        self.settings = settings or jb.Settings.game()
        self.M = int(callbacks_max)
        lv = level
        names = lv.names
        bn = names["body_names"]
        g2e = lv.game_to_engine
        body = lambda n: int(g2e[bn.index(n)])  # noqa: E731
        self.pot = int(lv.pot_body)
        self.tip = int(lv.hammer_body)
        self.hub = body("Player/Hub")
        self.arm = body("Player/Hub/Slider")
        self.cursor = body("Cursor")
        self.crane = body("Props/tower+crane")
        self.n_bodies = int(lv.n_bodies)
        self.chain_bodies = np.array([body(n) for n in pose.CHAIN], I32)  # pot .. Tip (pose.py), engine indices
        self.pose_tilt = np.asarray(pose.DEFAULT_TILT if pose_tilt is None else pose_tilt, F32)
        from goi_jax.game import space as sp_
        self.space = sp_.Space(lv, self.chain_bodies, self.cursor, self.pot) if space else None
        self._space_init = None
        # joints in the canonical (post re-creation) order: engine index e = NJ - 1 - native slot
        self.joint_order = np.asarray(joint_order)  # native slot -> joint name index
        NJ = len(self.joint_order)
        e_of = {int(n): NJ - 1 - k for k, n in enumerate(self.joint_order)}
        jn = names["joints"]
        self.hinge_j = e_of[jn.index("Player/Hub#HingeJoint2D")]
        self.slider_j = e_of[jn.index("Player/Hub/Slider#SliderJoint2D")]
        self.climber_j = np.array(sorted(e_of[jn.index(n)] for n in CLIMBER_JOINTS), I32)
        climber_bodies = set()
        for n in CLIMBER_JOINTS:
            a, b = lv.joint_bodies[jn.index(n)]
            climber_bodies |= {int(a), int(b)}
        self.climber_bodies = np.array(sorted(climber_bodies), I32)
        # fixtures -> colliders
        cols = names["colliders"]
        self.n_colliders = len(cols)
        fcol = lv.fixture["collider"].astype(I32)
        self.fixture_collider = fcol
        col_layer = np.zeros(self.n_colliders, I32)
        col_layer[fcol] = lv.fixture["layer"]
        self.collider_terrain = col_layer == TERRAIN
        self.collider_solid = (col_layer == TERRAIN) | (col_layer == 0)  # upstream's IsTouching filter
        self.tip_collider = cols.index(TIP_COLLIDER)
        self.pot_collider = cols.index(POT_COLLIDER)
        F = len(fcol)
        self.own = np.full(F, -1, I32)  # fixture -> script (0 hammer, 1 pot) or -1
        self.own[fcol == self.tip_collider] = fr.HAMMER
        self.own[fcol == self.pot_collider] = fr.POT
        self.tip_fixtures = np.asarray(lv.tip_fixtures, I32)
        self.pot_fixtures = np.asarray(lv.pot_fixtures, I32)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_recordings(cls, root="dumps/parity", chain=None, settings=None, compact_fixtures=False, **kw):
        """Level, constants and the canonical Model from a recording session (goi_jax.parity.record).
        compact_fixtures: the compact fixture layout (goi_jax/box2d/world.py compact_model): the ~1500 static
        fixtures' fat AABBs / friction / restitution are stored once in the Model and every world keeps only the
        rows of the ~30 fixtures on moving bodies (~36 KB less per world, and the per-step fixture kernels walk 30
        rows instead of 1500). Same floats, same results; state_from_recording then returns compact worlds."""
        from goi_jax.game import level as L
        from goi_jax.parity import states as S
        lv = L.load_level(root)
        row = S.controller_row(root)
        if chain is None:
            entry = S.load_index(root)["chains"][0]
            chain = S.load_chain(f"{root}/{entry['file']}")
        t = canonical_step(chain)
        jm, st, _ = lv.native_states(chain, [t], 0)
        jm = jax.tree.map(lambda a: a[0], jm)
        model = lv.model_for(jm)
        if compact_fixtures:
            model = compact_model(model, st)
        params = SimParams(controller=player.params_from_row(row), friction=fr.params_from_row(row),
                           crane_start=jnp.asarray(crane_start(chain, lv), f32),
                           pot_z=f32(chain.scr[0]["P.z"][t]), hub_z=f32(chain.scr[0]["H.z"][t]),
                           dt=f32(jb.GAME_DT))
        return cls(lv, model, params, settings, chain.joint_index[0][t], **kw)

    def state_from_recording(self, chain, t, capacity=96, transforms=None) -> SimState:
        """The SimState right before recorded step t (NAT0 world, SCR0 scripts), as numpy. Its joint order must be
        the canonical one (true after the first action of a chain; see canonical_step). With pose_nudge, the Transform
        locals come from `transforms` (pose.load_transforms of the chain file: TFM0 of step t) when given, else they
        are rebuilt from the bodies (pose.state_from_bodies)."""
        lv = self.level
        if not np.array_equal(chain.joint_index[0][t], self.joint_order):
            raise ValueError(f"step {t}: joint order {chain.joint_index[0][t]} is not the canonical "
                             f"{self.joint_order}")
        jm, st, _ = lv.native_states(chain, [t], 0, capacity)
        st = lv.set_limit_states(jm, st, self.settings)
        world = jax.tree.map(lambda a: np.asarray(a)[0], st)
        s0 = chain.scr[0]
        # gravity: Physics2D.gravity (SCR0), what the step integrates. The recorded native b2World gravity is garbage in
        # 12 of ~900 recorded chains (0, 0.9, 272, ... for a whole chain while the bodies fall at -30).
        world = world._replace(gravity=np.array([s0["gravity.x"][t], s0["gravity.y"][t]], F32))
        v2 = lambda a, b: np.array([s0[a][t], s0[b][t]], F32)  # noqa: E731
        ctrl = player.ControllerState(v2("mouseInput.x", "mouseInput.y"), v2("oldMouse.x", "oldMouse.y"),
                                      F32(s0["mouseVelocityAverage"][t]), v2("mw.x", "mw.y"),
                                      np.bool_(s0["mouseSnap"][t] > 0), F32(s0["pauseInputTimer"][t]),
                                      np.int32(s0["inputsToSkip"][t]), F32(s0["oldAngle"][t]))
        ht, pt = np.zeros(self.n_colliders, bool), np.zeros(self.n_colliders, bool)
        ht[chain.hammer_keys[0][t]] = True
        pt[chain.pot_keys[0][t]] = True
        fric = fr.FrictionState(np.bool_(s0["hammerSlide"][t] > 0), np.bool_(s0["potSlide"][t] > 0), ht, pt)
        touch, key = _np(self._touching(world.contacts))
        # Unity's pair list at a recorded state: unknowable; newest touching contact first fits best (see callbacks)
        order = np.argsort(-np.where(touch, key, -BIG).ravel(), kind="stable")[:min(int(touch.sum()), self.M)]
        pairs = np.full(self.M, -1, I32)
        pairs[:len(order)] = order
        tgt, mask = unity.move_targets(chain, [t], lv)
        snap = tgt[0, self.cursor] if mask[0, self.cursor] else world.bodies.xf_p[self.tip]
        ps = None
        if self.pose_nudge:
            if transforms is not None and not np.isnan(transforms[t, 0, 0, 0]):
                tf = transforms[t, 0]
                ps = pose.PoseState(tf[:, pose.LP].astype(F32), tf[:, pose.LQ].astype(F32))
            else:
                cb = self.chain_bodies
                z = np.array([s0["P.z"][t]] + [s0["H.z"][t]] * 4 + [s0["T.z"][t]], F32)
                ps = _np(pose.state_from_bodies(world.bodies.xf_p[cb], world.bodies.a[cb], z, self.pose_tilt))
        world = compact_state(self.model, world)  # no-op for a full-layout Model
        space = self.space_state(world) if self.space is not None else None
        return SimState(world=world, ctrl=_np(ctrl), fric=_np(fric), touch=touch, pairs=pairs, space=space,
                        n_pairs=np.int32(len(order)), sim_time=F32(s0["simTime"][t]), snap_target=snap.astype(F32),
                        reaction_torque=F32(chain.scr[1]["sj.reactionTorque"][t]), tip_z=F32(s0["T.z"][t]), pose=ps)

    def compact_state(self, s):
        """A SimState (any leading batch axes, e.g. a start bank or a checkpoint's archive) in this Sim's fixture
        layout: compacted if the Model is compact and s is not (world.compact_state); otherwise unchanged. A state
        without a space state (built before space.py) gets one (space_state) when this Sim models space."""
        s = s._replace(world=compact_state(self.model, s.world))
        if self.space is not None and s.space is None:
            s = s._replace(space=self.space_state(s.world))
        return s

    def space_state(self, world):
        """space.Space.initial for a world (any leading batch axes), as numpy: the trigger state of a recorded or placed
        state without history."""
        if self._space_init is None:
            self._space_init = jax.jit(lambda mdl, w: self.space.initial(mdl, w))
        lead = np.asarray(world.gravity).shape[:-1]
        f = self._space_init
        for _ in lead:
            f = jax.vmap(f, (None, 0))
        return _np(f(self.model, jax.tree.map(jnp.asarray, world)))

    def start_state(self, s, zero_gravity_inside=True):
        """A start-bank SimState (any leading batch axes) made ready for this Sim: compact_state, plus the space state
        when it has none (a bank built before space.py: space_state), and, with zero_gravity_inside, gravity (0, 0)
        where the climber already overlaps the zero-gravity zone with gravity (0, -30): the route placements' restore
        left -30 there (their Enter ran inside the restore's Simulate(0) before the saved gravity was put back), which a
        climber that got there in play never has (an Enter sets 0; only a collider leaving through the bottom sets
        -30 again)."""
        s = self.compact_state(s)
        if self.space is None:
            return s
        if zero_gravity_inside:
            inside = np.asarray(s.space.touch)[..., 0, :].any(-1)
            g = np.asarray(s.world.gravity)
            fix = inside & (g[..., 1] == F32(-30)) & (g[..., 0] == 0)
            g = np.where(fix[..., None], F32(0), g).astype(F32)
            s = s._replace(world=s.world._replace(gravity=g if isinstance(s.world.gravity, np.ndarray) else jnp.asarray(g)))
        return s

    # ------------------------------------------------------------------ one physics step

    def physics_step(self, model, params: SimParams, s: SimState, action, axis_name=None):
        """One plugin physics step (Game.Step's loop body) for one world. action: (2,) mouse input (already clamped)."""
        dt = params.dt
        w = s.world
        bd = w.bodies
        # 1. materials (HammerCollisions / PotSounds FixedUpdate), refreshed into existing contacts
        hm, pm = fr.materials(params.friction, s.fric)
        tip_f, pot_f = jnp.asarray(self.tip_fixtures), jnp.asarray(self.pot_fixtures)
        if is_compact(model, w.fixtures):  # their rows (the pieces are on moving bodies)
            tip_f, pot_f = model.fixture_slot[tip_f], model.fixture_slot[pot_f]
        fx = w.fixtures._replace(friction=w.fixtures.friction.at[tip_f].set(hm).at[pot_f].set(pm))
        ct = w.contacts
        mix = jnp.sqrt(m.mul(fixture_friction(model, fx, ct.fa), fixture_friction(model, fx, ct.fb)))
        swapped = (jnp.asarray(self.own)[ct.fa] >= 0) | (jnp.asarray(self.own)[ct.fb] >= 0)
        ct = ct._replace(friction=jnp.where((ct.pair >= 0) & swapped, mix, ct.friction))
        w = w._replace(fixtures=fx, contacts=ct)

        # 2. PlayerControl.FixedUpdate
        ja, jb_ = model.joint_a[self.hinge_j], model.joint_b[self.hinge_j]
        ref = model.reference_angle[self.hinge_j]
        joint_angle = m.mul((bd.a[jb_] - bd.a[ja]) - ref, RAD2DEG)
        if self.pose_nudge:  # Transform positions (pose.py)
            tf_pot, tf_hub = pose.world_position(s.pose.lp, s.pose.lq, 0), pose.world_position(s.pose.lp, s.pose.lq, 1)
            tf_tip = pose.world_position(s.pose.lp, s.pose.lq, len(pose.CHAIN) - 1)
        else:
            tf_pot = jnp.append(bd.xf_p[self.pot], params.pot_z)
            tf_tip = jnp.append(bd.xf_p[self.tip], s.tip_z)
            tf_hub = jnp.append(bd.xf_p[self.hub], params.hub_z)
        inputs = player.ControllerInputs(
            action=jnp.asarray(action, f32), cursor=bd.xf_p[self.cursor], pot=tf_pot, tip=tf_tip, hub=tf_hub,
            joint_angle=joint_angle,
            reference_angle=m.mul(ref, RAD2DEG), reaction_torque=s.reaction_torque)
        snap_pending = s.ctrl.mouse_snap
        out = player.controller_step(params.controller, s.ctrl, inputs)
        jt = w.joints
        ms = graph.set_rows(jt.motor_speed, self.hinge_j, m.mul(out.hinge_speed, DEG2RAD))  # .at[].set
        ms = graph.set_rows(ms, self.slider_j, out.slider_speed)
        w = w._replace(joints=jt._replace(motor_speed=ms))
        cursor_v = jnp.where(out.cursor_move, out.cursor_velocity, bd.v[self.cursor])
        w = w._replace(bodies=w.bodies._replace(v=w.bodies.v.at[self.cursor].set(cursor_v)))
        B = self.n_bodies
        tgt = jnp.zeros((B, 2), f32)
        mask = jnp.zeros(B, bool)
        tgt = tgt.at[self.cursor].set(jnp.where(out.cursor_move, out.cursor_target, s.snap_target))
        mask = mask.at[self.cursor].set(out.cursor_move | snap_pending)

        # 3. simTime and the crane
        t = s.sim_time + dt
        sn, _ = m.sin_cos(t)
        cx = fm.dd_round(fm.dd_add(fm.two_prod(sn, f32(0.01)), fm.pair(params.crane_start[0])))
        tgt = tgt.at[self.crane].set(jnp.stack([cx, params.crane_start[1]]))
        mask = mask.at[self.crane].set(True)

        # 4. Simulate
        before = w
        w = unity.move_velocity(w, tgt, mask, dt)
        w, pre_toi = self._box2d_step(model, w, dt, axis_name)
        w = unity.move_finish(w, before, tgt, mask)

        # 5. callbacks -> friction switch, HammerReturn
        cb, touch, pairs, n_pairs = self.callbacks(model, before.bodies, pre_toi, w, s.touch, s.pairs, s.n_pairs)
        bd1 = w.bodies
        fric, hard = fr.apply_callbacks(params.friction, s.fric, cb, fr.speed(bd1.v[self.tip]),
                                        fr.speed(bd1.v[self.pot]))
        ps, tip_z = s.pose, s.tip_z
        if self.pose_nudge:  # Unity writes the moved bodies into their Transforms after the step
            cb_ = jnp.asarray(self.chain_bodies)
            ps = pose.write(s.pose, bd1.xf_p[cb_], bd1.a[cb_], jnp.asarray(self.pose_tilt))
            tip_tf = pose.world_position(ps.lp, ps.lq, len(pose.CHAIN) - 1)
            pot_tf = pose.world_position(ps.lp, ps.lq, 0)
            tip_z = tip_tf[2]
        else:
            tip_tf = jnp.append(bd1.xf_p[self.tip], s.tip_z)
            pot_tf = jnp.append(bd1.xf_p[self.pot], params.pot_z)
        ctrl, snapped, snap_to = player.hammer_return(out.state, hard, tip_tf, pot_tf)
        inv_dt = f32(1) / dt
        rt = m.mul(w.joints.impulse[self.slider_j, 1], inv_dt)  # b2PrismaticJoint::GetReactionTorque
        space = s.space
        if self.space is not None and space is not None:  # zone, wells, teleports, win (space.py)
            w, space, ps_t = self.space.after_step(model, w, space, ps if self.pose_nudge else None)
            ps = ps_t if self.pose_nudge else ps
        s = SimState(world=w, ctrl=ctrl, fric=fric, touch=touch, pairs=pairs, n_pairs=n_pairs, sim_time=t,
                     snap_target=jnp.where(snapped, snap_to, s.snap_target), reaction_torque=rt, tip_z=tip_z, pose=ps,
                     space=space)
        toi_ev = w.toi.events if w.toi is not None else jnp.int32(0)
        return s, StepInfo(out.hinge_speed, out.slider_speed, hard, snapped, jnp.sum(cb.valid), toi_ev)

    def _box2d_step(self, model, state, dt, axis_name=None):
        """goi_jax.box2d.step.step, returning also the bodies between the solve and SolveTOI. Kept call for call
        identical to box2d.step (tests/env checks that the K=1 sim reproduces the single-step harness bit for bit)."""
        settings = self.settings
        dt = jnp.asarray(dt, jnp.float32)
        pallas = getattr(settings, "solver", "xla") == "pallas"
        # as box2d.step: FindNewContacts where new_fixture (normally nowhere; with Pallas the candidate scan is a
        # kernel run only there, and the "idle" GPU speed option skips the call when no world has one)
        state = start_new_contacts(model, state, settings, axis_name)
        state = state._replace(new_fixture=jnp.asarray(False))
        inv_dt = jnp.where(dt > 0, F32(1) / jnp.where(dt > 0, dt, F32(1)), F32(0))
        dt_ratio = m.mul(state.inv_dt0, dt)
        state = collide(model, state, settings)
        if pallas:  # as box2d.step (training/backend.select_solver's patch of box2d_solve is then not needed)
            from goi_jax.box2d.gpu.solve import solve as solve_pallas
            state = solve_pallas(model, state, settings, dt, inv_dt, dt_ratio, axis_name)
        else:
            state = box2d_solve(model, state, settings, dt, inv_dt, dt_ratio, axis_name)
        pre_toi = state.bodies
        if settings.continuous_physics:
            state = solve_toi(model, state, settings, dt, axis_name)
        state = state._replace(inv_dt0=jnp.where(dt > 0, inv_dt, state.inv_dt0))
        if settings.auto_clear_forces:
            b = state.bodies
            state = state._replace(bodies=b._replace(force=jnp.zeros_like(b.force), torque=jnp.zeros_like(b.torque)))
        return state, pre_toi

    # ------------------------------------------------------------------ callbacks

    def _touching(self, ct):
        """(2, NCOL) touching per (script, other collider) and the newest touching contact key per pair."""
        own = jnp.asarray(self.own)
        fcol = jnp.asarray(self.fixture_collider)
        live = (ct.pair >= 0) & ct.touching
        own_a, own_b = own[ct.fa], own[ct.fb]
        script = jnp.where(own_a >= 0, own_a, own_b)
        other = fcol[jnp.where(own_a >= 0, ct.fb, ct.fa)]
        hit = live & (script >= 0)
        flat = jnp.where(hit, script * self.n_colliders + other, 2 * self.n_colliders)
        touch = jnp.zeros(2 * self.n_colliders + 1, bool).at[flat].set(True)[:-1]
        key = jnp.full(2 * self.n_colliders + 1, -1, jnp.int32).at[flat].max(jnp.where(hit, ct.key, -1))[:-1]
        return touch.reshape(2, -1), key.reshape(2, -1)

    def callbacks(self, model, bd0, bd_pre_toi, w, touch0, pairs0, n_pairs0):
        """This step's Enter/Stay/Exit callbacks to HammerCollisions and PotSounds (friction.Callbacks, M slots in
        Unity's order), and the new touching / pair-list state. bd0: bodies at the start of the step (MovePosition
        velocities not applied), bd_pre_toi: after the solve, before SolveTOI, w: the world after the step."""
        NC = self.n_colliders
        ct, bd = w.contacts, w.bodies
        own = jnp.asarray(self.own)
        fcol = jnp.asarray(self.fixture_collider)
        live = (ct.pair >= 0) & ct.touching
        own_a, own_b = own[ct.fa], own[ct.fb]
        script = jnp.where(own_a >= 0, own_a, own_b)
        own_f = jnp.where(own_a >= 0, ct.fa, ct.fb)
        oth_f = jnp.where(own_a >= 0, ct.fb, ct.fa)
        other = fcol[oth_f]
        hit = live & (script >= 0)
        pid = jnp.where(hit, script * NC + other, 2 * NC)  # flat (script, collider) id per contact
        C = ct.pair.shape[0]
        # reported impulses: a contact solved in a TOI sub-step reports that solve's
        if w.toi is not None:
            solved = w.toi.solved
            ni = jnp.where(solved[:, None], w.toi.normal_impulse, ct.normal_impulse)
            ti = jnp.where(solved[:, None], w.toi.tangent_impulse, ct.tangent_impulse)
            via_toi = w.toi.count > 0
        else:
            ni, ti, via_toi = ct.normal_impulse, ct.tangent_impulse, jnp.zeros(C, bool)
        # point order within a pair: contacts oldest first (ascending key), points in manifold order
        same = (pid[:, None] == pid[None, :]) & hit[:, None] & hit[None, :]
        older = same & (ct.key[None, :] < ct.key[:, None])
        # points of the pair ahead of contact i (high bits) and how many contacts are ahead (low 10 bits), in one
        # (C, C) reduction: XLA otherwise writes `older` out (C x C per world) for a second reduction (any)
        assert C < 1024, "packed counts older contacts in 10 bits"
        packed = jnp.sum(jnp.where(older, ct.count[None, :] * 1024 + 1, 0), axis=1)
        before = packed >> 10
        q = fr.contact_ratio(ni[..., None], ti[..., None], jnp.ones((C, 2), I32), 1)  # (C, 2) per point
        pos = before[:, None] + jnp.arange(2)[None, :]
        inpt = hit[:, None] & (jnp.arange(2)[None, :] < ct.count[:, None])
        buf = jnp.where(script == fr.HAMMER, 5, 20)[:, None]
        q_use = jnp.where(inpt & (pos < buf), q, f32(0))
        n_flat = 2 * NC + 1
        ratio = jnp.zeros(n_flat, f32).at[pid].max(jnp.max(q_use, axis=1))[:-1]
        count = jnp.zeros(n_flat, I32).at[pid].add(jnp.where(hit, ct.count, 0))[:-1]
        key_new = jnp.full(n_flat, -1, I32).at[pid].max(jnp.where(hit, ct.key, -1))[:-1]
        touch = jnp.zeros(n_flat, bool).at[pid].set(True)[:-1]
        # relativeVelocity at the first reported point (oldest contact of the pair, point 0)
        first = hit & ((packed & 1023) == 0)
        px, py = world_points(model, ct, bd)
        p0x, p0y = px[:, 0], py[:, 0]

        def point_vel(b, bi):
            rx, ry = p0x - bd0.c[bi, 0], p0y - bd0.c[bi, 1]
            wx, wy = m.cross_sv(b.w[bi], rx, ry)
            return b.v[bi, 0] + wx, b.v[bi, 1] + wy
        ob, xb = model.fixture_body[own_f], model.fixture_body[oth_f]
        vo0, vx0 = point_vel(bd0, ob), point_vel(bd0, xb)
        vo1, vx1 = point_vel(bd_pre_toi, ob), point_vel(bd_pre_toi, xb)
        rel_x = jnp.where(via_toi, vx1[0] - vo1[0], vx0[0] - vo0[0])
        rel_y = jnp.where(via_toi, vx1[1] - vo1[1], vx0[1] - vo0[1])
        rel = fr.speed(jnp.stack([rel_x, rel_y], -1))
        rel_pair = jnp.zeros(n_flat, f32).at[jnp.where(first, pid, 2 * NC)].set(rel)[:-1]

        # Unity's order: a list of active pairs; this step's new pairs are appended (oldest newest-contact first),
        # then dispatched in list order, an Exit swap-removing its entry in place (the last entry moves into its
        # slot and is dispatched next). Matches 404 of the 426 recorded steps with several callbacks (most misses:
        # the unknowable list order at a chain's first step).
        L = self.M
        t0 = touch0.reshape(-1)
        new = touch & ~t0
        _, cand = jax.lax.top_k(-jnp.where(new, key_new, BIG), L)  # new pairs, ascending newest-contact key
        cand_ok = new[cand]
        slot = n_pairs0 + jnp.cumsum(cand_ok) - 1
        put = cand_ok & (slot < L)
        lst = pairs0.at[jnp.where(put, slot, L)].set(cand.astype(I32), mode="drop")
        n = jnp.minimum(n_pairs0 + jnp.sum(cand_ok), L).astype(I32)

        def dispatch(k, c):
            lst, n, i, out = c
            on = i < n
            e = lst[jnp.clip(i, 0, L - 1)]
            gone = on & ~touch[jnp.clip(e, 0, 2 * NC - 1)]
            out = graph.set_at(out, k, jnp.where(on, e, -1))   # out.at[k].set (k < L: fori_loop counter)
            last = lst[jnp.clip(n - 1, 0, L - 1)]
            lst2 = lst.at[jnp.clip(i, 0, L - 1)].set(last).at[jnp.clip(n - 1, 0, L - 1)].set(-1)
            lst = jnp.where(gone, lst2, lst)
            return lst, jnp.where(gone, n - 1, n), jnp.where(on & ~gone, i + 1, i), out
        lst, n, _, out = jax.lax.fori_loop(0, L, dispatch, (lst, n, jnp.int32(0), jnp.full(L, -1, I32)))
        valid = out >= 0
        idx = jnp.clip(out, 0, 2 * NC - 1)
        kind = jnp.where(~t0[idx], fr.ENTER, jnp.where(touch[idx], fr.STAY, fr.EXIT))
        col = idx % NC
        cb = fr.Callbacks(valid=valid, script=(idx // NC).astype(I32), kind=kind.astype(I32), collider=col.astype(I32),
                          terrain=jnp.asarray(self.collider_terrain)[col], count=jnp.where(touch[idx], count[idx], 0),
                          ratio=ratio[idx], rel_speed=rel_pair[idx])
        # pairs dropped from a full list are not tracked as touching (they enter once there is room)
        in_list = jnp.zeros(2 * NC + 1, bool).at[jnp.where(lst >= 0, lst, 2 * NC)].set(True)[:-1]
        return cb, (touch & in_list).reshape(2, NC), lst, n

    # ------------------------------------------------------------------ one action

    def end_action(self, s: SimState) -> SimState:
        """The plugin's once-per-action pose update re-creates the climber's joints: cold impulses, inactive
        prismatic limit, bodies woken (b2World::DestroyJoint), reaction torque 0. The joint order stays canonical."""
        jt = s.world.joints
        cj = [int(j) for j in np.asarray(self.climber_j).reshape(-1)]   # .at[cj].set(0) (graph.set_rows)
        jt = jt._replace(impulse=graph.set_rows(jt.impulse, cj, 0), motor_impulse=graph.set_rows(jt.motor_impulse, cj, 0),
                         lower_impulse=graph.set_rows(jt.lower_impulse, cj, 0),
                         upper_impulse=graph.set_rows(jt.upper_impulse, cj, 0))
        jt = jt._replace(limit_state=graph.set_rows(jt.limit_state, self.slider_j, 0))  # the prismatic one: inactive
        bd = s.world.bodies
        cb = jnp.zeros(self.n_bodies, bool).at[jnp.asarray(self.climber_bodies)].set(True) & (bd.type != jb.STATIC)
        bd = bd._replace(sleep_time=jnp.where(cb & ~bd.awake, F32(0), bd.sleep_time), awake=bd.awake | cb)
        ps = s.pose
        if self.pose_nudge:
            bd, ps = self.pose_resync(bd, s.pose)
        return s._replace(world=s.world._replace(joints=jt, bodies=bd), reaction_torque=f32(0), pose=ps)

    def pose_resync(self, bd, ps):
        """The pose update's Hub restore, then the climber's bodies re-synced from their Transforms (pose.py): new
        origins (x, y), centre of mass b2Mul(xf, localCenter), c0 = c; angles and velocities unchanged."""
        ps = pose.pose_update(ps)
        xy = pose.body_positions(ps)[1:]  # the pot (root) never moves
        idx = jnp.asarray(self.chain_bodies[1:])
        q = bd.xf_q[idx]
        cx, cy = m.xf_mul(xy[:, 0], xy[:, 1], q[:, 0], q[:, 1], bd.local_center[idx, 0], bd.local_center[idx, 1])
        c = jnp.stack([cx, cy], -1)
        bd = bd._replace(xf_p=bd.xf_p.at[idx].set(xy), c=bd.c.at[idx].set(c), c0=bd.c0.at[idx].set(c))
        return bd, ps

    def act(self, model, params, s: SimState, action, k: int = 4, axis_name=None):
        """One agent action (Game.Step): ClampMagnitude(action, 100), K physics steps, joint re-creation.
        Returns (state, StepInfo stacked over the K steps)."""
        a = clamp_magnitude(jnp.asarray(action, f32), f32(100))

        def one(st, _):
            return self.physics_step(model, params, st, a, axis_name)
        s, info = jax.lax.scan(one, s, None, length=k)
        return self.end_action(s), info


# ---------------------------------------------------------------------------------------------------------------------
# helpers

def clamp_magnitude(v, max_len):
    """Vector2.ClampMagnitude under Mono: if sqrMagnitude > max^2, normalized * max."""
    sq = fm.dd_round(fm.dd_add(fm.two_prod(v[0], v[0]), fm.two_prod(v[1], v[1])))
    big = sq > m.mul(max_len, max_len)
    n = jnp.stack(fm.normalize(v[0], v[1]))
    return jnp.where(big, fm.rounded(n * max_len), v)


def world_points(model, ct, bd):
    """b2WorldManifold::Initialize points (C, 2) x, y for every contact slot at the bodies' current transforms."""
    ia, ib = model.fixture_body[ct.fa], model.fixture_body[ct.fb]
    xfA = (bd.xf_p[ia, 0], bd.xf_p[ia, 1], bd.xf_q[ia, 0], bd.xf_q[ia, 1])
    xfB = (bd.xf_p[ib, 0], bd.xf_p[ib, 1], bd.xf_q[ib, 0], bd.xf_q[ib, 1])
    rA, rB = model.radius[ct.fa], model.radius[ct.fb]
    ln, lp, pl = ct.local_normal, ct.local_point, ct.point_local
    pAx, pAy = m.xf_mul(*xfA, lp[:, 0], lp[:, 1])
    pBx, pBy = m.xf_mul(*xfB, pl[:, 0, 0], pl[:, 0, 1])
    dx, dy = pBx - pAx, pBy - pAy
    far = m.dot(dx, dy, dx, dy) > m.mul(m.EPS, m.EPS)
    nnx, nny, _ = m.normalize(dx, dy)
    cnx, cny = jnp.where(far, nnx, F32(1)), jnp.where(far, nny, F32(0))
    circ = (m.mul(F32(0.5), (pAx + m.mul(rA, cnx)) + (pBx - m.mul(rB, cnx))),
            m.mul(F32(0.5), (pAy + m.mul(rA, cny)) + (pBy - m.mul(rB, cny))))
    fAnx, fAny = m.rot(xfA[2], xfA[3], ln[:, 0], ln[:, 1])
    fBnx, fBny = m.rot(xfB[2], xfB[3], ln[:, 0], ln[:, 1])
    fAp = m.xf_mul(*xfA, lp[:, 0], lp[:, 1])
    fBp = m.xf_mul(*xfB, lp[:, 0], lp[:, 1])
    is_c, is_a = ct.mtype == CIRCLES, ct.mtype == FACE_A
    xs, ys = [], []
    for j in range(2):
        qx, qy = m.xf_mul(*xfB, pl[:, j, 0], pl[:, j, 1])
        s_ = rA - m.dot(qx - fAp[0], qy - fAp[1], fAnx, fAny)
        fa = (m.mul(F32(0.5), (qx + m.mul(s_, fAnx)) + (qx - m.mul(rB, fAnx))),
              m.mul(F32(0.5), (qy + m.mul(s_, fAny)) + (qy - m.mul(rB, fAny))))
        qx, qy = m.xf_mul(*xfA, pl[:, j, 0], pl[:, j, 1])
        s_ = rB - m.dot(qx - fBp[0], qy - fBp[1], fBnx, fBny)
        fb = (m.mul(F32(0.5), (qx - m.mul(rA, fBnx)) + (qx + m.mul(s_, fBnx))),
              m.mul(F32(0.5), (qy - m.mul(rA, fBny)) + (qy + m.mul(s_, fBny))))
        xs.append(jnp.where(is_c, circ[0], jnp.where(is_a, fa[0], fb[0])))
        ys.append(jnp.where(is_c, circ[1], jnp.where(is_a, fa[1], fb[1])))
    return jnp.stack(xs, 1), jnp.stack(ys, 1)


def canonical_step(chain) -> int:
    """First step of a chain whose joint list has the order every re-creation leaves (climber joints at the head;
    a chain's first step can still hold the placement's order)."""
    names = chain.names.get("joints") or []
    ji = chain.joint_index[0]
    for t in range(len(ji)):
        if names and names[int(ji[t, 0])].startswith("Player"):
            return t
    raise ValueError("no step with the climber's joints at the head of the joint list")


def crane_start(chain, level) -> np.ndarray:
    """craneStart solved from the chain's recorded crane MovePosition targets (target = x0 + sin(simTime) * 0.01)."""
    from goi_jax.parity import states as S
    bn = level.names["body_names"]
    gi = bn.index("Props/tower+crane")
    ev = chain.events
    cr = ev[(ev["kind"] == S.EV_MOVE_POSITION) & (ev["phase"] == 1) & (ev["script"] == gi)]
    if not len(cr):
        raise ValueError("no crane MovePosition events in the chain")
    t = (chain.scr[1]["simTime"][cr["step"]] + F32(jb.GAME_DT)).astype(F32)
    s = np.sin(t.astype(np.float64)).astype(F32).astype(np.float64)
    x0 = (cr["a"].astype(np.float64) - s * np.float64(F32(0.01))).astype(F32)
    vals, counts = np.unique(x0, return_counts=True)
    return np.array([vals[np.argmax(counts)], cr["b"][0]], F32)

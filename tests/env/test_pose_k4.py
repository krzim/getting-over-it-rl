"""K=4 recordings (goi_jax.parity.record --k 4) and the Transform round trip model (goi_jax/game/pose.py).

Skipped when the K=4 recordings are absent (dumps/ is gitignored). Record them with
    .venv/bin/python -m goi_jax.parity.record --k 4 --plan k4 --out dumps/parity-k4     (unsandboxed: launches the game)
Run with
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/env/test_pose_k4.py -q
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_K4_DIR", ROOT / "dumps" / "parity-k4"))
pytestmark = pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no K=4 recordings in {DATA}")
CHAINS = ("c00", "c05", "c13", "c17")  # Coffee Rock hits / hits / smooth, level start swing

f32 = np.float32


@pytest.fixture(scope="module")
def chains():
    from goi_jax.game import pose
    from goi_jax.parity import states
    out = []
    for e, c in states.iter_chains(DATA):
        if any(Path(e["file"]).stem.startswith(p) for p in CHAINS):
            out.append((c, pose.load_transforms(DATA / e["file"])))
    return out


def _bodies(c, which, field):
    from goi_jax.game import pose
    bn = c.names["body_names"]
    return np.stack([c.body(which, bn.index(n))[field] for n in pose.CHAIN], 1)


def test_k4_layout(chains):
    """K records per action; within an action the next step starts where the last ended (no pose round trip, no joint
    re-creation); the joint list keeps the canonical order after the first action."""
    for c, _ in chains:
        aos = np.asarray(c.meta["action_of_step"])
        assert c.meta["k"] == 4 and (np.bincount(aos) == 4).all()
        mid = np.flatnonzero(aos[1:] == aos[:-1])
        xf0, xf1 = _bodies(c, 0, "xf"), _bodies(c, 1, "xf")
        assert (xf0[mid + 1] == xf1[mid]).all()
        ji = c.joint_index[0]
        assert (ji[4:] == ji[4]).all()


def test_k4_reaction_torque(chains):
    """The slider reaction torque the controller reads (SCR1): 0 on an action's first step (joints re-created), else
    the previous step's prismatic impulse.y * (1/dt) (sim.py physics_step), bit for bit."""
    n_nonzero = 0
    for c, _ in chains:
        aos = np.asarray(c.meta["action_of_step"])
        sj = c.names["joints"].index("Player/Hub/Slider#SliderJoint2D")
        imp = c.joint(1, sj)["impulse"][:, 1]
        first = np.r_[True, aos[1:] != aos[:-1]]
        want = np.where(first, f32(0), np.r_[f32(0), (imp[:-1] * (f32(1) / f32(1 / 120))).astype(f32)])
        rt = c.scr[1]["sj.reactionTorque"]
        assert np.array_equal(rt, want)
        n_nonzero += int((rt != 0).sum())
    assert n_nonzero > 1000


def test_transform_round_trip_exact(chains):
    """pose.py's Transform math against the recorded Transforms: world positions from the locals, the post-Simulate
    position write (given the recorded rotations), the pose update's Hub restore, and the nudged body positions at
    every action end: all bit-exact."""
    import jax
    from goi_jax.game import pose

    @jax.jit
    @jax.vmap
    def model(lp_old, lq_new, xy):
        lp = pose.write_positions(lp_old, lq_new, xy)
        ps = pose.pose_update(pose.PoseState(lp, lq_new))
        world = jax.numpy.stack([pose.world_position(lp, lq_new, i) for i in range(len(pose.CHAIN))])
        return lp, world, ps.lp, pose.body_positions(ps)

    n_ends = 0
    for c, tf in chains:
        assert tf is not None
        aos = np.asarray(c.meta["action_of_step"])
        T1, T2, T3 = tf[:, 1], tf[:, 2], tf[:, 3]
        xy = _bodies(c, 1, "xf")[..., :2]
        lp, world, lp3, rb = map(np.asarray, model(T1[..., pose.LP], T2[..., pose.LQ], xy))
        assert np.array_equal(lp, T2[..., pose.LP])
        assert np.array_equal(world, T2[..., pose.WP])
        end = np.flatnonzero(aos[1:] != aos[:-1])  # action ends with a next action
        assert np.array_equal(lp3[end], T3[end][..., pose.LP])
        assert np.array_equal(rb[end], _bodies(c, 0, "xf")[end + 1][..., :2])
        n_ends += len(end)
    assert n_ends > 300


def test_rotation_write_exact(chains):
    """Unity's post-Simulate rotation write (pose.rotation_from_body: XRotUpdateFromRBAngle with the C runtime's
    sinf/cosf/atan2f, the SIMD hierarchy products, SetLocalTR's normalisation) from the Transforms before the step and
    the bodies' new angles: every local rotation, local position and world rotation bit-exact on every step."""
    import jax
    from goi_jax.game import pose

    @jax.jit
    @jax.vmap
    def model(lp, lq, xy, a):
        ps = pose.write(pose.PoseState(lp, lq), xy, a)
        return ps.lp, ps.lq, jax.numpy.stack([pose.world_rotation(ps.lq, i) for i in range(len(pose.CHAIN))])

    n = 0
    for c, tf in chains:
        T1, T2 = tf[:, 1], tf[:, 2]
        lp, lq, wq = map(np.asarray, model(T1[..., pose.LP], T1[..., pose.LQ], _bodies(c, 1, "xf")[..., :2],
                                           _bodies(c, 1, "a")))
        assert np.array_equal(lq, T2[..., pose.LQ])
        assert np.array_equal(lp, T2[..., pose.LP])
        assert np.array_equal(wq, T2[..., pose.WQ])
        n += len(lq)
    assert n > 1000


def test_action_nudge_exact(chains):
    """From the Transforms at an action's start (TFM0) and the recorded bodies after each of its K steps, the modelled
    writes, the pose update's Hub restore and the re-sync give the next action's start positions bit for bit."""
    import jax
    from goi_jax.game import pose

    @jax.jit
    @jax.vmap
    def model(lp, lq, xys, angs):
        ps = pose.PoseState(lp, lq)
        for k in range(xys.shape[0]):
            ps = pose.write(ps, xys[k], angs[k])
        return pose.body_positions(pose.pose_update(ps))

    n_ends = n_plain = 0
    for c, tf in chains:
        aos = np.asarray(c.meta["action_of_step"])
        first = np.flatnonzero(np.r_[True, aos[1:] != aos[:-1]])
        f = first[first + 4 < len(aos)]
        idx = f[:, None] + np.arange(4)[None]
        xy, a = _bodies(c, 1, "xf")[..., :2], _bodies(c, 1, "a")
        rb = np.asarray(model(tf[f, 0][..., pose.LP], tf[f, 0][..., pose.LQ], xy[idx], a[idx]))
        want = _bodies(c, 0, "xf")[f + 4][..., :2]
        assert np.array_equal(rb, want)
        n_ends += len(f)
        n_plain += int((xy[f + 3] == want).all((-1, -2)).sum())
    assert n_ends > 300 and n_plain < n_ends // 2  # the nudge is real: most action ends move without it


def test_lossy_scale_and_joint_anchors(chains):
    """pose.lossy_scale is bit-exact on every recorded Transform, and every re-created climber joint's b2 local anchor
    (recorded at an action's first step) is the Unity anchor times the lossy scale (x, y) of the joint's own Transform
    (side A) or the connected body's (side B: the next chain member) after the previous pose update, with one constant
    Unity anchor per joint side recovered from the chain's first action (pose.unscale_anchor). Lossy scales stay
    within ~2e-6 of 1, so most joint sides keep a constant b2 anchor; the scaling must show on some of them."""
    import jax
    from goi_jax.game import pose

    lossy = jax.jit(jax.vmap(lambda lq: jax.numpy.stack([pose.lossy_scale(lq, i) for i in range(len(pose.CHAIN))])))
    n_checked = n_varying = 0
    for c, tf in chains:
        for which in range(3):
            assert np.array_equal(np.asarray(lossy(tf[:, which][..., pose.LQ])), tf[:, which][..., pose.WS])
        aos = np.asarray(c.meta["action_of_step"])
        f = np.flatnonzero(np.r_[True, aos[1:] != aos[:-1]])[1:]  # actions after a pose update
        ls = np.asarray(lossy(tf[f, 0][..., pose.LQ]))[..., :2]
        bn = c.names["body_names"]
        for ji, name in enumerate(c.names["joints"]):
            if not name.startswith("Player"):
                continue
            j = c.joint(0, ji)
            own = pose.CHAIN.index(name.split("#")[0])
            for key, i in (("localAnchorA", own), ("localAnchorB", own + 1)):
                rec = j[key][f]
                u = pose.unscale_anchor(rec[0], ls[0, i])
                assert np.array_equal(rec, (u[None] * ls[:, i]).astype(np.float32)), (name, key)
                n_checked += len(f)
                n_varying += int((rec != rec[0]).any())
    assert n_checked > 1000 and n_varying >= 5


def test_sim_with_pose_nudge_runs(chains):
    """Sim(pose_nudge=True) runs a K=4 action from a recorded state with its Transforms; the climber's bodies end
    within a millimetre of the game's after the action (the nudge is bit-exact, pose.py, but the re-created joints'
    anchors are not yet modelled in sim.py, and the physics is not bit-exact on every step)."""
    import jax
    from goi_jax.game import sim as SM
    from goi_jax.game import pose
    c, tf = chains[0]
    sim = SM.Sim.from_recordings(DATA, pose_nudge=True)
    aos = np.asarray(c.meta["action_of_step"])
    t = int(np.flatnonzero(np.r_[True, aos[1:] != aos[:-1]])[1])  # the second action: canonical joint order
    s = sim.state_from_recording(c, t, transforms=tf)
    assert s.pose is not None
    s1, _ = jax.jit(sim.act, static_argnames="k")(sim.model, sim.params, s, c.action[t], k=4)
    got = np.asarray(s1.world.bodies.xf_p)[sim.chain_bodies]
    want = _bodies(c, 0, "xf")[t + 4][..., :2]
    assert np.isfinite(got).all()
    assert np.abs(got - want).max() < 1e-3
    assert np.isfinite(np.asarray(pose.tip_z(s1.pose)))

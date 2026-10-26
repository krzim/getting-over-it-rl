"""PlayerControl.FixedUpdate as a pure JAX function: the per-physics-step controller of Getting Over It.

Everything is float32 and jit/vmap-able (no Python control flow on values). It reproduces the game's own float model,
Mono's double intermediates rounded where C# stores a value (Q7, settled by the recordings: bit-exact on every
recorded step; see tests/game/test_controller.py and goi_jax.game.reference for the host oracle). Logic follows the
decompile, PlayerControl.cs:225-327 (line numbers below); maths in docs/understanding/player.md §3.

Interface (all arrays; batch with vmap):

    params = params_from_row(controller_row)                 # constants (pass as a jit ARGUMENT, not a closure:
                                                             # XLA would rewrite `x / deadzone` into a reciprocal)
    state  = initial_state(tip_xy)                           # ControllerState
    out    = controller_step(params, state, ControllerInputs(...))
    out.state                 new ControllerState (mouseInput, oldMouse, mouseVelocityAverage, mw, mouseSnap, ...)
    out.hinge_speed           HingeJoint2D motor speed written (deg/s, clamped to +-800): the hub hinge motor
    out.slider_speed          SliderJoint2D motor speed written (clamped to +-50)
    out.cursor_move           whether the controller called fakeCursorRB.MovePosition this step (False on a snap step)
    out.cursor_target         its (last) MovePosition target: the kinematic cursor's position after the physics step
                              (on a snap step: the unchanged cursor position; use hammer_return's target instead)
    out.cursor_velocity       the velocity it then wrote (target - position: a displacement, not divided by dt)

After the physics step, a hammer hit (friction.apply_callbacks' `hard_hit`) runs `hammer_return`, which may set
mouseSnap and move the cursor onto the head for the next step: on that snap step out.cursor_move is False and
out.cursor_target is only the cursor's position before the step; the cursor actually ends the step at hammer_return's
target.

Inputs a simulator must supply each step (ControllerInputs): the mouse action (Rewired GetAxis "mouseX"/"mouseY"; in RL
mode the agent's action, sticks 0), the cursor body position, the Transform positions of the pot (Player), the hammer
head (Tip) and the hub (the hinge's GameObject) including z (only the 3D normalize at :307 uses z), the main hinge's
jointAngle and referenceAngle (degrees, Unity's convention: jointAngle = (bodyB angle - bodyA angle) - referenceAngle),
and the slider's reactionTorque as the controller reads it. In the plugin's lockstep game the climber's joints are
re-created at the end of every action, so the first physics step of each action reads reactionTorque 0 (every step at
K = 1); later steps read Unity's GetReactionTorque of the previous step (prismatic impulse.y * 1/dt).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import floatmath as fm

f32 = jnp.float32
DT = np.float32(0.008333334)  # Time.fixedDeltaTime


class Curve(NamedTuple):
    """An AnimationCurve's keys (non-weighted, ClampForever), padded by repeating the last key."""
    time: jax.Array
    value: jax.Array
    in_tangent: jax.Array
    out_tangent: jax.Array
    count: jax.Array  # int32: real keys


class ControllerParams(NamedTuple):
    deadzone: jax.Array        # PlayerControl.deadzone (0.02): hinge gain fades in over this cursor-hub distance
    angle_epsilon: jax.Array   # 1 degree: soft dead band on the motor error
    sensitivity: jax.Array     # mouseSensitivity (the plugin forces 1)
    curve: Curve               # mouseCurve (the plugin forces trackpad off) or trackpadCurve
    reach: jax.Array           # 3.5: the cursor's max distance from the pot (:302-305)


class ControllerState(NamedTuple):
    mouse_input: jax.Array     # (2,) smoothed mouse input
    old_mouse: jax.Array       # (2,) previous mouseInput (written, read only through mouse_input)
    mva: jax.Array             # () mouseVelocityAverage: the cursor's gain, gates the idle drift
    mw: jax.Array              # (2,) the point the motors steer to (cursor position at the start of the step)
    mouse_snap: jax.Array      # () bool, set by HammerReturn: skip input for one step
    pause_timer: jax.Array     # () pauseInputTimer (0 in RL play; Saviour.Load sets 1)
    inputs_to_skip: jax.Array  # () int32 (menu un-pause; 0 in RL play)
    old_angle: jax.Array       # () previous motor error (multiplied by 0 in the game)


class ControllerInputs(NamedTuple):
    action: jax.Array          # (2,) mouse axes
    cursor: jax.Array          # (2,) fakeCursorRB.position
    pot: jax.Array             # (3,) Player Transform position (P)
    tip: jax.Array             # (3,) Tip Transform position (T)
    hub: jax.Array             # (3,) hinge Transform position (H, Player/Hub)
    joint_angle: jax.Array     # () hj.jointAngle (deg)
    reference_angle: jax.Array  # () hj.referenceAngle (deg)
    reaction_torque: jax.Array  # () sj.reactionTorque as read at :322


class ControllerOutputs(NamedTuple):
    state: ControllerState
    hinge_speed: jax.Array
    slider_speed: jax.Array
    cursor_move: jax.Array
    cursor_target: jax.Array
    cursor_velocity: jax.Array
    num: jax.Array             # slider error term (|v5| * num2), for tests
    num2: jax.Array            # alignment of the head direction with the head-to-target direction, for tests


def curve_from_row(curve: dict, pad: int = 8) -> Curve:
    keys = curve["keys"]
    if len(keys) > pad:
        raise ValueError(f"{len(keys)} keys > pad {pad}")
    for k in keys:
        if k.get("weightedMode", "None") != "None":
            raise NotImplementedError("weighted AnimationCurve keys")
    if curve.get("preWrapMode", "ClampForever") not in ("ClampForever", "Clamp", "Once") or \
            curve.get("postWrapMode", "ClampForever") not in ("ClampForever", "Clamp", "Once"):
        raise NotImplementedError("looping AnimationCurve wrap modes")

    def col(name):
        v = [np.float32(k[name]) for k in keys]
        return jnp.asarray(np.array(v + [v[-1]] * (pad - len(v)), np.float32))
    return Curve(col("time"), col("value"), col("inTangent"), col("outTangent"), jnp.int32(len(keys)))


def params_from_row(row: dict) -> ControllerParams:
    """Constants from the plugin's controller row (GoiClient.controller(), dumps/parity/controller.json)."""
    curve = row["trackpadCurve"] if row.get("trackpad") else row["mouseCurve"]
    return ControllerParams(deadzone=f32(row["deadzone"]), angle_epsilon=f32(row["angleEpsilon"]),
                            sensitivity=f32(row["mouseSensitivity"]), curve=curve_from_row(curve),
                            reach=f32(row.get("codeConstants", {}).get("reach", 3.5)))


def initial_state(tip_xy=(0.0, 0.0)) -> ControllerState:
    """The state the plugin's restore leaves (plugin/Game.cs Restore): no input history, mw on the head, no pause."""
    z2 = jnp.zeros(2, f32)
    return ControllerState(z2, z2, f32(0), jnp.asarray(tip_xy, f32), jnp.bool_(False), f32(0), jnp.int32(0), f32(0))


def curve_eval(c: Curve, x):
    """Unity's AnimationCurve.Evaluate (native float32, from the segment's cached cubic; bit-exact against Unity's own
    Evaluate on 13,117 samples): t = x - key.time, ((c0 t + c1) t + c2) t + c3, clamped outside the keys."""
    x = jnp.asarray(x, f32)
    n = c.count
    last = n - 1
    i = jnp.clip(jnp.sum((c.time[1:] <= x) & (jnp.arange(1, c.time.shape[0]) < n)), 0, jnp.maximum(last - 1, 0))
    t0, t1, v0, v1 = c.time[i], c.time[i + 1], c.value[i], c.value[i + 1]
    dx = jnp.maximum(t1 - t0, f32(1e-4))
    dy = v1 - v0
    length = f32(1) / fm.rounded(dx * dx)
    d1, d2 = fm.rounded(c.out_tangent[i] * dx), fm.rounded(c.in_tangent[i + 1] * dx)
    c0 = fm.rounded((((d1 + d2) - dy) - dy) * length) / dx
    c1 = fm.rounded((((((dy + dy) + dy) - d1) - d1) - d2) * length)
    t = x - t0
    y = fm.rounded(t * (fm.rounded(t * (fm.rounded(t * c0) + c1)) + c.out_tangent[i])) + v0
    y = jnp.where(x > c.time[0], y, c.value[0])
    return jnp.where(x >= c.time[last], c.value[last], y)


def _repeat(t, length):
    """Mathf.Repeat: Clamp(t - Floor(t / length) * length, 0, length) (the product is exact here)."""
    k = jnp.floor(t / length)
    return jnp.clip(t - k * length, f32(0), length)


def _delta_angle(current, target):
    n = _repeat(target - current, f32(360))
    return jnp.where(n > f32(180), n - f32(360), n)


def controller_step(p: ControllerParams, s: ControllerState, x: ControllerInputs) -> ControllerOutputs:
    """One PlayerControl.FixedUpdate (the RL plugin's configuration: no stick input, loadFinished)."""
    a = jnp.asarray(x.action, f32)
    C, P, T, H = (jnp.asarray(v, f32) for v in (x.cursor, x.pot, x.tip, x.hub))
    snap = jnp.asarray(s.mouse_snap, bool)

    # :241-251 mouse input through the curve, then halfway from the last value (Lerp t = 0.5 rounds (a + b) / 2 once,
    # which equals the float32 sum halved)
    m = fm.rounded(a * p.sensitivity)
    k = curve_eval(p.curve, fm.magnitude(m[0], m[1]))
    m = fm.rounded(m * k)
    mi = fm.rounded((s.mouse_input + m)) * f32(0.5)
    old = s.mouse_input

    # :265-299 the pause / skip branches (not reached in RL play) and the normal one
    timer = s.pause_timer
    paused = timer > 0
    new_timer = jnp.where(paused, jnp.maximum(fm.dd_round(fm.dd_add(fm.pair(timer), fm.pair(-(DT * f32(2))))), f32(0)),
                          timer)
    one_minus = fm.two_sum(f32(1), -new_timer)  # 2 (1 - timer) - 1 in double: exact pair, doubled, minus 1
    k3 = jnp.clip(fm.dd_round(fm.dd_add((f32(2) * one_minus[0], f32(2) * one_minus[1]), fm.pair(f32(-1)))),
                  f32(0), f32(1))
    disabled = (~paused) & (timer < 0)
    skipping = (~paused) & (~disabled) & (s.inputs_to_skip > 0) & (fm.rounded(mi[0] * mi[0]) + fm.rounded(mi[1] * mi[1]) > 0)
    to_head = paused | disabled | skipping
    old = jnp.where(paused, fm.rounded(old * k3), jnp.where(skipping, jnp.zeros(2, f32), old))
    mi = jnp.where(paused, fm.rounded(mi * k3), jnp.where(disabled | skipping, fm.rounded(mi * f32(0)), mi))
    skip_left = jnp.where(skipping, s.inputs_to_skip - 1, s.inputs_to_skip)
    mva = jnp.where(to_head, f32(0), s.mva)
    mw = jnp.where(to_head, T[:2], C)

    # :300 the cursor's gain: Lerp(mva, max(0.1 |mi|, 0.001), 0.05) + 0.005
    target = jnp.maximum(fm.rounded(f32(0.1) * fm.magnitude(mi[0], mi[1])), f32(0.001))
    mva = fm.lerp(mva, target, f32(0.05)) + f32(0.005)
    # :301-305 cursor step, kept within reach of the pot
    v4 = C + fm.rounded(mi * mva)
    d = P[:2] - v4
    far = fm.magnitude(d[0], d[1]) > p.reach
    n = jnp.stack(fm.normalize(v4[0] - P[0], v4[1] - P[1]))
    v4 = jnp.where(far, P[:2] + fm.rounded(n * p.reach), v4)
    # :306-310 the head relative to the steering point, and how well the hammer points at it (3D normalize, z dropped)
    v5 = T[:2] - mw
    lhs = fm.normalize(T[0] - H[0], T[1] - H[1], T[2] - H[2])
    n5 = fm.normalize(v5[0], v5[1])
    num2 = fm.dot2(lhs[0], lhs[1], n5[0], n5[1])
    num = fm.rounded(fm.magnitude(v5[0], v5[1]) * num2)
    # :312-314 idle drift toward the head, after the reach clamp; MovePosition; velocity = displacement
    g = jnp.clip(f32(0.2) - jnp.minimum(mva, f32(0.2)), f32(0), f32(0.2))
    v4 = v4 + fm.rounded(fm.rounded(fm.rounded(f32(0.05) * v5) * f32(10)) * g)
    vel = v4 - C

    # a snap step (:233-236) keeps everything but the motors as it was
    keep = lambda new, cur: jnp.where(snap, cur, new)  # noqa: E731
    mi, old, mva, mw = keep(mi, s.mouse_input), keep(old, s.old_mouse), keep(mva, s.mva), keep(mw, s.mw)
    num, num2 = keep(num, f32(0)), keep(num2, f32(0))
    new_timer, skip_left = keep(new_timer, s.pause_timer), keep(skip_left, s.inputs_to_skip)

    # :316-321 hinge motor: angle error to the steering point, shaped, faded in over `deadzone`, clamped +-800 deg/s
    v6 = H[:2] - mw
    at = fm.atan2(v6[1], v6[0])
    arg = fm.dd_round(fm.dd_add(fm.dd_add(fm.two_prod(f32(57.29578), at), fm.pair(f32(-180))),
                                fm.pair(-jnp.asarray(x.reference_angle, f32))))
    e = -_delta_angle(arg, jnp.asarray(x.joint_angle, f32))
    e = jnp.where(e >= 0, f32(1), f32(-1)) * jnp.maximum(jnp.abs(e / f32(2)), jnp.abs(fm.rounded(e * e)))
    e = fm.rounded(e * jnp.clip(jnp.abs(e) / p.angle_epsilon, f32(0), f32(1)))
    scale = jnp.clip(fm.magnitude(v6[0], v6[1]) / p.deadzone, f32(0), f32(1))
    pw = fm.rounded(scale * scale)
    hinge = jnp.clip(fm.dd_round(fm.dd_mul_f(fm.two_prod(e, f32(3)), pw)), f32(-800), f32(800))
    # :322-325 slider motor: gain falls with the slider's reaction torque; clamped +-50
    g5 = f32(16) - jnp.maximum(fm.rounded(jnp.asarray(x.reaction_torque, f32) * f32(0.001)), f32(5))
    n6 = fm.dd_round(fm.dd_mul(*(2 * (fm.two_prod(num2, num2),))))  # Mathf.Pow(num2, 4)
    sl = fm.dd_round(fm.dd_mul_f(fm.dd_mul_f(fm.two_prod(-num, jnp.abs(num)), g5), n6))
    sl = jnp.clip(sl, f32(-50), f32(50))

    state = ControllerState(mi, old, mva, mw, jnp.bool_(False) & snap, new_timer, skip_left, e)
    return ControllerOutputs(state, hinge, sl, ~snap, jnp.where(snap, C, v4), jnp.where(snap, jnp.zeros(2, f32), vel),
                             num, num2)


def hammer_return(s: ControllerState, hard_hit, tip, pot):
    """PlayerControl.HammerReturn (PlayerControl.cs:192-199), run when a hammer Enter callback saw a hard hit
    (friction.apply_callbacks' hard_hit): unless the head is more than 0.4 below the pot, set mouseSnap and move the
    cursor onto the head. `tip`, `pot`: the Tip and Player Transform positions after the physics step (the callbacks
    run after Unity wrote the bodies back to their Transforms).

    Returns (state, snapped, target). target = tip[:2] is the cursor's MovePosition target: the kinematic cursor
    arrives there during the next physics step (the snap step, on which controller_step itself does not move it);
    recordings: the cursor after the snap step equals this target to within 1 ulp of its largest coordinate (the
    kinematic move rounds) on all 11 recorded snaps.

    The test `tip.y - 0.4 < pot.y` is one C# expression, so Mono compares the unrounded double difference; it is done
    here exactly with a two-sum (a float32 subtraction can round onto pot.y and flip the result)."""
    tip, pot = jnp.asarray(tip, f32), jnp.asarray(pot, f32)
    hi, lo = fm.two_sum(tip[1], f32(-0.4))
    below = (hi < pot[1]) | ((hi == pot[1]) & (lo < 0))
    snapped = jnp.asarray(hard_hit, bool) & ~below
    return s._replace(mouse_snap=s.mouse_snap | snapped), snapped, tip[:2]

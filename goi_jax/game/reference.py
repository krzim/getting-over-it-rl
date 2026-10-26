"""Host (numpy, scalar) reference of PlayerControl.FixedUpdate and AnimationCurve.Evaluate, in two float models.

This is the oracle the JAX port (goi_jax.game.player) is tested against, and the tool that settled Q7 (how Unity
2020.3's Mono evaluates float expressions). Logic follows the decompile; citations are PlayerControl.cs lines
(_decompiled/Assembly-CSharp/), Unity library semantics from UnityEngine.CoreModule (Mathf, Vector2, Vector3).

Float models:
- "double": every C# float expression is evaluated in float64 and rounded to float32 once, where it is stored (a local,
  a field, a method argument or a method's return value). Vector2/Vector3 operators are methods, so each component of
  their results is rounded; scalar expressions inside one statement keep double intermediates.
- "float32": every operation rounded to float32 (Mono's float32 optimisation).
Library calls are the same in both: Mathf.Sqrt/Atan2/Pow/Floor = (float)Math.X((double)f), a correctly rounded
float32 result in practice.

AnimationCurve.Evaluate (native, float32): Unity evaluates the segment's cubic from cached coefficients,
t' = x - key.time; ((c0 t' + c1) t' + c2) t' + c3 with c0 = (d1 + d2 - dy - dy) / dx^2 / dx,
c1 = (3 dy - 2 d1 - d2) / dx^2, c2 = outTangent, c3 = value, d = tangent * dx; clamped outside the keys
(ClampForever). Bit-exact against Unity's own Evaluate on 13,117 samples of the three curves (controller row samples).
"""
from __future__ import annotations

import numpy as np

f32, f64 = np.float32, np.float64


def curve_arrays(curve: dict):
    """(times, values, inTangents, outTangents) float32 arrays of a controller-row curve."""
    k = curve["keys"]
    return tuple(np.array([x[n] for x in k], f32) for n in ("time", "value", "inTangent", "outTangent"))


def curve_eval(curve, x) -> np.float32:
    """Unity's AnimationCurve.Evaluate for non-weighted keys with ClampForever wrap (all three game curves)."""
    t_, v, it, ot = curve if isinstance(curve, tuple) else curve_arrays(curve)
    x = f32(x)
    if not x > t_[0]:
        return v[0]
    if x >= t_[-1]:
        return v[-1]
    i = int(np.searchsorted(t_, x, side="right")) - 1
    dx = max(f32(t_[i + 1] - t_[i]), f32(1e-4))
    dy = f32(v[i + 1] - v[i])
    length = f32(f32(1) / f32(dx * dx))
    d1, d2 = f32(ot[i] * dx), f32(it[i + 1] * dx)
    c0 = f32(f32(f32(f32(f32(d1 + d2) - dy) - dy) * length) / dx)
    c1 = f32(f32(f32(f32(f32(f32(dy + dy) + dy) - d1) - d1) - d2) * length)
    t = f32(x - t_[i])
    return f32(f32(t * f32(f32(t * f32(f32(t * c0) + c1)) + ot[i])) + v[i])


class _Ops:
    """Scalar float ops under one float model. Every public method returns float32 (a stored value)."""

    def __init__(self, mode: str):
        assert mode in ("double", "float32")
        self.d = mode == "double"

    def w(self, x):  # widen for an intermediate
        return f64(x) if self.d else f32(x)

    def mag2(self, x, y):
        """Vector2.magnitude: (float)Math.Sqrt(x * x + y * y)."""
        x, y = self.w(x), self.w(y)
        return f32(np.sqrt(f64(x * x + y * y)))

    def mag3(self, x, y, z):
        x, y, z = self.w(x), self.w(y), self.w(z)
        return f32(np.sqrt(f64(x * x + y * y + z * z)))

    def dot2(self, ax, ay, bx, by):
        return f32(self.w(ax) * self.w(bx) + self.w(ay) * self.w(by))

    def lerp(self, a, b, t):
        """Vector2.Lerp component / Mathf.Lerp: a + (b - a) * Clamp01(t), rounded once (double) or per op."""
        t = f32(min(max(t, f32(0)), f32(1)))
        return f32(self.w(a) + (self.w(b) - self.w(a)) * self.w(t))


def normalize2(o: _Ops, x, y):
    m = o.mag2(x, y)
    return (f32(x / m), f32(y / m)) if m > f32(1e-5) else (f32(0), f32(0))


def normalize3(o: _Ops, x, y, z):
    m = o.mag3(x, y, z)
    return (f32(x / m), f32(y / m), f32(z / m)) if m > f32(1e-5) else (f32(0), f32(0), f32(0))


def repeat(t, length):
    """Mathf.Repeat: Clamp(t - Floor(t / length) * length, 0, length); the inner expression rounds once (its product is
    exact for the small integers Floor gives here, so both float models agree)."""
    k = f32(np.floor(f32(t / length)))
    return f32(min(max(f32(f64(t) - f64(k) * f64(length)), f32(0)), length))


def delta_angle(current, target):
    n = repeat(f32(target - current), f32(360))
    return f32(n - f32(360)) if n > f32(180) else n


def sign(x):
    return f32(1) if x >= 0 else f32(-1)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp01(x):
    return clamp(x, f32(0), f32(1))


def controller_step(c: dict, s: dict, mode: str = "double", lerp_rounds: bool = True) -> dict:
    """One PlayerControl.FixedUpdate (PlayerControl.cs:225-327) for the RL plugin's configuration (no stick input).

    c: constants: deadzone, angleEpsilon, sensitivity, trackpad, curve (curve_arrays tuple of the active curve).
    s: inputs (float32 scalars): action (2), mouseInput (2), mva, mw (2), mouseSnap, pauseInputTimer, inputsToSkip,
       C (cursor body position, 2), P/T/H (pot, tip, hinge Transform positions, 3 each), jointAngle, referenceAngle,
       reactionTorque (slider), oldAngle.
    lerp_rounds: Mathf.Lerp's return is rounded before the caller adds 0.005 (False: one rounding for the whole line,
       as if the JIT inlined it without narrowing).
    Returns the new state and what the controller wrote: hingeSpeed, sliderSpeed, cursorTarget (None if no
    MovePosition), cursorVelocity (None if not set), and the intermediates num, num2, e (motor error after shaping).
    """
    o = _Ops(mode)
    D = o.d
    out = dict(mouseInput=tuple(s["mouseInput"]), oldMouse=tuple(s.get("oldMouse", (0, 0))), mva=f32(s["mva"]),
               mw=tuple(s["mw"]), mouseSnap=bool(s["mouseSnap"]), pauseInputTimer=f32(s["pauseInputTimer"]),
               inputsToSkip=int(s["inputsToSkip"]), cursorTarget=None, cursorVelocity=None, moves=[])
    num = num2 = f32(0)
    C, P, T, H = (tuple(f32(v) for v in s[k]) for k in ("C", "P", "T", "H"))
    if out["mouseSnap"]:  # :233-236 a HammerReturn last step: no input this step, mw and the cursor as they are
        out["mouseSnap"] = False
    else:
        old = out["mouseInput"]                                         # :241/248
        sens = f32(c.get("sensitivity", 1))
        mx, my = f32(f32(s["action"][0]) * sens), f32(f32(s["action"][1]) * sens)  # :242/249
        k = curve_eval(c["curve"], o.mag2(mx, my))                         # :243/250
        mx, my = f32(mx * k), f32(my * k)
        mi = (o.lerp(old[0], mx, f32(0.5)), o.lerp(old[1], my, f32(0.5)))  # :244/251
        out["oldMouse"] = old
        # :253-264 sticks read 0 in RL mode (plugin/Plugin.cs VirtualMouse), so the stick branch never replaces it.
        mva = out["mva"]
        if out["pauseInputTimer"] > 0:                                    # :265-279
            p = f32(out["pauseInputTimer"] - f32(f32(0.008333334) * f32(2))) if not D else \
                f32(f64(out["pauseInputTimer"]) - f64(f32(0.008333334)) * 2)
            p = max(p, f32(0))
            out["pauseInputTimer"] = p
            k3 = clamp01(f32(o.w(2) * (o.w(1) - o.w(p)) - 1))
            out["oldMouse"] = (f32(old[0] * k3), f32(old[1] * k3))
            mi = (f32(mi[0] * k3), f32(mi[1] * k3))
            out["moves"].append(T[:2])
            mva, mw = f32(0), T[:2]
        elif out["pauseInputTimer"] < 0:                                  # :280-286
            mi = (f32(mi[0] * 0), f32(mi[1] * 0))
            out["moves"].append(T[:2])
            mva, mw = f32(0), T[:2]
        elif out["inputsToSkip"] > 0 and f32(f32(mi[0] * mi[0]) + f32(mi[1] * mi[1])) > 0:  # :287-295
            mi = (f32(mi[0] * 0), f32(mi[1] * 0))
            out["oldMouse"] = (f32(0), f32(0))
            out["moves"].append(T[:2])
            mva, mw = f32(0), T[:2]
            out["inputsToSkip"] -= 1
        else:
            mw = C                                                        # :298
        target = f32(max(f32(f32(0.1) * o.mag2(*mi)) if not D else f32(f64(f32(0.1)) * f64(o.mag2(*mi))), f32(0.001)))
        if lerp_rounds or not D:
            mva = f32(o.w(o.lerp(mva, target, f32(0.05))) + o.w(f32(0.005)))  # :300
        else:
            mva = f32(f64(mva) + (f64(target) - f64(mva)) * f64(f32(0.05)) + f64(f32(0.005)))
        v4 = (f32(C[0] + f32(mi[0] * mva)), f32(C[1] + f32(mi[1] * mva)))   # :301
        dpx, dpy = f32(P[0] - v4[0]), f32(P[1] - v4[1])
        if o.mag2(dpx, dpy) > f32(3.5):                                      # :302-305 reach from the pot
            nx, ny = normalize2(o, f32(v4[0] - P[0]), f32(v4[1] - P[1]))
            v4 = (f32(P[0] + f32(nx * f32(3.5))), f32(P[1] + f32(ny * f32(3.5))))
        v5 = (f32(T[0] - mw[0]), f32(T[1] - mw[1]))                          # :306
        lhs = normalize3(o, f32(T[0] - H[0]), f32(T[1] - H[1]), f32(T[2] - H[2]))[:2]  # :307 (3D, then drop z)
        n5 = normalize2(o, *v5)
        num2 = o.dot2(lhs[0], lhs[1], n5[0], n5[1])                          # :309
        num = f32(o.mag2(*v5) * num2)                                        # :310
        g = clamp(f32(f32(0.2) - min(mva, f32(0.2))), f32(0), f32(0.2))      # :312 idle drift toward the head
        dx, dy = f32(f32(f32(f32(0.05) * v5[0]) * f32(10)) * g), f32(f32(f32(f32(0.05) * v5[1]) * f32(10)) * g)
        v4 = (f32(v4[0] + dx), f32(v4[1] + dy))
        out["moves"].append(v4)                                              # :313
        out["cursorVelocity"] = (f32(v4[0] - C[0]), f32(v4[1] - C[1]))       # :314
        out.update(mouseInput=mi, mva=mva, mw=mw)
    if out["moves"]:
        out["cursorTarget"] = out["moves"][-1]
    mw = out["mw"]
    v6 = (f32(H[0] - mw[0]), f32(H[1] - mw[1]))                                # :316
    at = f32(np.arctan2(f64(v6[1]), f64(v6[0])))
    ref, ja = f32(s["referenceAngle"]), f32(s["jointAngle"])
    if D:
        arg = f32(f64(f32(57.29578)) * f64(at) - 180.0 - f64(ref))
    else:
        arg = f32(f32(f32(f32(57.29578) * at) - f32(180)) - ref)
    e = f32(-delta_angle(arg, ja))                                            # :317
    e = f32(sign(e) * max(abs(f32(e / f32(2))), abs(f32(e * e))))             # :318
    e = f32(e * clamp01(f32(abs(e) / f32(c.get("angleEpsilon", 1)))))         # :319
    scale = clamp01(f32(o.mag2(*v6) / f32(c["deadzone"])))
    pw = f32(f64(scale) ** 2)
    zero_term = f32(f32(e - f32(s.get("oldAngle", 0))) * f32(0))
    if D:
        hinge = f32((f64(e) * 3.0 + f64(zero_term)) * f64(pw))
    else:
        hinge = f32(f32(f32(e * f32(3)) + zero_term) * pw)
    hinge = clamp(hinge, f32(-800), f32(800))                                  # :320
    g5 = f32(f32(16) - max(f32(f32(s["reactionTorque"]) * f32(0.001)), f32(5)))  # :322
    n6 = f32(f64(num2) ** 4)                                                   # :323 Mathf.Pow
    if D:
        sl = f32(-f64(num) * abs(f64(num)) * f64(g5) * f64(n6))
    else:
        sl = f32(f32(f32(f32(-num) * abs(num)) * g5) * n6)
    sl = clamp(sl, f32(-50), f32(50))                                          # :324
    out.update(hingeSpeed=hinge, sliderSpeed=sl, num=num, num2=num2, e=e, oldAngle=e)
    return out


def hammer_return(tip_y, pot_y) -> bool:
    """PlayerControl.HammerReturn (PlayerControl.cs:192-199): snaps (mouseSnap, cursor MovePosition(tip)) unless the
    head is more than 0.4 below the pot."""
    # one C# expression: Mono compares the double difference (exact for float32 operands), not a float32-rounded one
    return not (float(f32(tip_y)) - float(f32(0.4)) < float(f32(pot_y)))

"""The between-action pose nudge: the climber's bodies round-trip through Unity's 3D Transforms.

What happens in the game (plugin/Game.cs Step; measured on the K=4 recordings, record version 3, which carry the
climber's Transforms: TFM0..2 with SCR0..2 and TFM3 after the once-per-action pose update):

1. After every PhysicsScene2D.Simulate Unity writes each moved body's pose into its Transform, parents first
   (Player, Hub, Slider, Handle, PoleMiddle, Tip). The position write is a world -> local conversion against the
   parents' new local TRS: target (body x, body y, the Transform's current world z), then for every ancestor from the
   root down: p = conj(q_anc) (p - t_anc) (`write_positions`; local scales are all 1). Bit-exact on all 10,560
   recorded steps for all six Transforms, given the new local rotations.
2. The Transform world position is computed child-up from the locals: p = t_par + quatMulVec(q_par, p) per ancestor,
   with Unity's quatMulVec (`quat_mul_vec`: u + u.x m0 + u.y m1 + u.z m2, summed as (u + x m0) + (y m1 + z m2)).
   Bit-exact on every recorded Transform position (`world_position`).
3. Once per action the pose update (PoseControl.LateUpdate, with the plugin saving and restoring
   `hub.transform.position`) moves Transforms of joint-holding bodies; the only lasting change to the chain is the
   Hub's restore, a SetPosition of its own world position: one more world -> local round trip of the Hub's local
   position (`pose_update`; exact on all 2,618 recorded action ends). The Transform change makes Unity re-sync the
   climber's bodies from their Transforms before the next step: every body's position becomes its Transform world
   position (x, y); angles and velocities stay (and the joints are re-created, sim.py end_action).
   Steps 1-3 reproduce the recorded nudge bit for bit at all 2,618 K=4 action ends, for all five child bodies
   (`body_positions`); the pot (the root) never moves.

4. The rotation write (UnityPlayer.dll 2020.3.25f1, disassembled with its public symbols): PhysicsManager2D::Simulate
   writes each body by Transform::SetPositionAndRotation(position, XRotUpdateFromRBAngle(transform.rotation,
   body angle)) (`rotation_from_body`): the Transform's world rotation q (CalculateGlobalRotation, SIMD products
   child-up: `world_rotation`) has its z angle theta = atan2f(q.z, q.w) replaced by the body angle, as
   (Rz(angle) * conj(Rz(theta))) * q, so the stored rotation depends on its history (and keeps the out-of-plane
   tilt of Handle and PoleMiddle). The world rotation is taken to a local one against the parents root first
   (InverseTransformPositionAndRotation: conj(q_parent) * q, SIMD) and SetLocalTR normalises it per lane
   (`normalize_rotation`). sinf, cosf and atan2f are the C runtime's (`sincosf`, `atan2f`): short double formulas
   rounded to float32, not the correctly rounded functions (those differ on ~0.5% of the writes).
   Bit-exact on all 10,560 recorded K=4 steps for all six Transforms (local and world rotations, local positions),
   and, from an action's first Transforms (TFM0) alone, the nudged bodies at all 2,618 action ends.

Functions are pure JAX on float32 with one rounding per C operation (products through math2d.mul, so XLA cannot fuse
them into FMAs). `load_transforms` decodes the TFM sections of version-3 records (numpy).
"""
from __future__ import annotations

import struct
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import lax

from goi_jax.box2d import math2d as m
from goi_jax.game import floatmath as fm

F32 = np.float32
f32 = jnp.float32
CHAIN = ("Player", "Player/Hub", "Player/Hub/Slider", "Player/Hub/Slider/Handle",
         "Player/Hub/Slider/Handle/PoleMiddle", "Player/Hub/Slider/Handle/PoleMiddle/Tip")
TAG_TRANSFORMS = 0x304D4654  # "TFM0" + which << 24 (plugin/GameRecord.cs RecTransforms)
# per Transform: localPosition(3) localRotation(4) localScale(3) position(3) rotation(4) lossyScale(3)
LP, LQ, LS, WP, WQ, WS = slice(0, 3), slice(3, 7), slice(7, 10), slice(10, 13), slice(13, 17), slice(17, 20)
_S2 = F32(0.70710677)
# Each body's out-of-plane rotation (world rotation = Rz(angle) * tilt): Handle and PoleMiddle are turned -90 deg
# about y (every K=4 recording); the others are flat.
DEFAULT_TILT = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, -_S2, 0, _S2], [0, -_S2, 0, _S2], [0, 0, 0, 1]], F32)


class PoseState(NamedTuple):
    """The climber's Transform locals, Player (the root) down to Tip: what Unity stores."""
    lp: jnp.ndarray  # (6, 3) localPosition
    lq: jnp.ndarray  # (6, 4) localRotation (x, y, z, w)


# ---------------------------------------------------------------------------------------------------------------------
# The C runtime's float trig in UnityPlayer.dll (UCRT / AMD libm sinf, cosf (FMA3 path) and atan2f, read from the
# disassembly). Each evaluates a short double-precision formula (truncated Taylor series, a 256-step atan table) and
# rounds it to float32, so the results are NOT the correctly rounded sin/cos/atan2 (that differs in ~0.1-1% of the
# calls). The double formulas are evaluated here in float-float (floatmath, ~48 bits) and rounded once: the same float
# unless the double result lies within ~1e-14 (relative) of a float32 rounding midpoint.

_S = (-0.16666666666666666, 0.008333333333333333, -0.0001984126984126984, 2.7557319223985893e-06)  # x^3 .. x^9
_C = (0.041666666666666664, -0.0013888888888888887, 2.4801587301587298e-05, -2.755731922398589e-07)  # x^4 .. x^10
_TWO_BY_PI = fm._host_pair(0.6366197723675814)
_PIBY2_1 = np.float64(1.5707963267341256)  # 33 significant bits: split exactly into two float32
_PIBY2_1H = F32(_PIBY2_1)
_PIBY2_1L = F32(_PIBY2_1 - np.float64(_PIBY2_1H))
_PIBY2_1T = fm._host_pair(6.077100506506192e-11)
_PIO4_BELOW = np.nextafter(F32(np.pi / 4), F32(0))  # largest float32 <= pi/4 (the double threshold)
_ATAN_T = np.arctan((np.arange(241) + 16) / 256.0)  # the binary's table atan(k / 256), k = 16..256 (to 1 ulp double)
_ATAN_TH = F32(_ATAN_T)
_ATAN_TL = F32(_ATAN_T - _ATAN_TH.astype(np.float64))


def _const(c, like):
    h, l = fm._host_pair(c) if not isinstance(c, tuple) else c
    return jnp.full_like(like, h), jnp.full_like(like, l)


def _dd_neg(a):
    return -a[0], -a[1]


def _dd_where(c, a, b):
    return jnp.where(c, a[0], b[0]), jnp.where(c, a[1], b[1])


def _dd_poly(x2, coefficients):
    """sum_k c_k x2^k (Horner) in float-float, coefficients lowest order first (double constants)."""
    s = _const(coefficients[-1], x2[0])
    for c in reversed(coefficients[:-1]):
        s = fm.dd_add(fm.dd_mul(s, x2), _const(c, x2[0]))
    return s


def _dd_gt(a, c):
    """float-float a > double constant c."""
    ch, cl = fm._host_pair(c)
    return (a[0] > ch) | ((a[0] == ch) & (a[1] > cl))


def sincosf(x):
    """(sinf(x), cosf(x)) as UnityPlayer.dll's C runtime computes them (float32 in, float32 out)."""
    x = jnp.asarray(x, f32)
    ax = jnp.abs(x)
    big = ax > _PIO4_BELOW
    mid = ax >= f32(0.0078125)
    small = ax >= f32(0.0001220703125)
    # |x| > pi/4: n = (int)(|x| * 2/pi + 0.5), r = |x| - n * piby2_1 - n * piby2_1tail
    q = fm.dd_add(fm.dd_mul_f(_const(_TWO_BY_PI, ax), ax), _const(0.5, ax))
    n = jnp.floor(q[0])
    n = jnp.where((q[0] == n) & (q[1] < 0), n - 1, n)
    n = jnp.where(big, n, f32(0))
    r = fm.dd_add(fm.pair(ax), _dd_neg(fm.two_prod(n, jnp.full_like(n, _PIBY2_1H))))
    r = fm.dd_add(r, _dd_neg(fm.two_prod(n, jnp.full_like(n, _PIBY2_1L))))
    r = fm.dd_add(r, _dd_neg(fm.dd_mul_f(_const(_PIBY2_1T, ax), n)))
    x2 = fm.dd_mul(r, r)
    sp = _dd_where(mid, _dd_poly(x2, _S), _const(_S[0], ax))           # 2^-13 <= |x| < 2^-7: x - x^3/6 only
    s = fm.dd_add(r, fm.dd_mul(sp, fm.dd_mul(r, x2)))
    cp = _dd_where(mid, fm.dd_mul(_dd_poly(x2, _C), fm.dd_mul(x2, x2)), (jnp.zeros_like(ax), jnp.zeros_like(ax)))
    c = fm.dd_add(fm.dd_add(_const(1.0, ax), (x2[0] * f32(-0.5), x2[1] * f32(-0.5))), cp)
    sv, cv = fm.dd_round(s), fm.dd_round(c)
    ni = n.astype(jnp.int32)
    odd = (ni & 1) == 1
    sin_r = jnp.where(odd, cv, sv)
    sin_r = jnp.where((ni & 3) >= 2, -sin_r, sin_r)
    cos_r = jnp.where(odd, sv, cv)
    cos_r = jnp.where((((ni + 1) >> 1) & 1) == 1, -cos_r, cos_r)
    sin_x = jnp.where(jnp.signbit(x), -sin_r, sin_r)
    sin_x = jnp.where(small, sin_x, x)
    cos_x = jnp.where(small, cos_r, f32(1))
    return sin_x, cos_x


def atan2f(y, x):
    """atan2f(y, x) as UnityPlayer.dll's C runtime computes it (float32 in, float32 out; finite inputs)."""
    y, x = jnp.asarray(y, f32), jnp.asarray(x, f32)
    ay, ax = jnp.abs(y), jnp.abs(x)
    ey = (lax.bitcast_convert_type(ay, jnp.int32) >> 23)
    ex = (lax.bitcast_convert_type(ax, jnp.int32) >> 23)
    diff = ey - ex
    xneg, yneg = jnp.signbit(x), jnp.signbit(y)
    sw = ay > ax
    num, den = jnp.where(sw, ax, ay), jnp.where(sw, ay, ax)
    den1 = jnp.where(den == 0, f32(1), den)
    r = fm.dd_div(fm.pair(num), fm.pair(den1))
    # table: k = (int)(r * 256 + 0.5), t = (256 num - k den) / (k num + 256 den), atan = (t + atan(k/256)) - t^3 c
    kq = fm.dd_add((r[0] * f32(256), r[1] * f32(256)), _const(0.5, r[0]))
    k = jnp.floor(kq[0])
    k = jnp.where((kq[0] == k) & (kq[1] < 0), k - 1, k)
    k = jnp.clip(k, f32(16), f32(256))
    tn = fm.dd_add(fm.pair(num * f32(256)), _dd_neg(fm.two_prod(k, den)))
    td = fm.dd_add(fm.two_prod(k, num), fm.pair(den * f32(256)))
    t = fm.dd_div(tn, (td[0], td[1]))
    ki = (k - f32(16)).astype(jnp.int32)
    tab = fm.dd_add(t, (jnp.asarray(_ATAN_TH)[ki], jnp.asarray(_ATAN_TL)[ki]))
    tab = fm.dd_add(tab, _dd_neg(fm.dd_mul(fm.dd_mul(fm.dd_mul(t, t), t), _const(0.33333333333224097, t[0]))))
    # 1e-4 <= r <= 1/16: r - (c7 - (c5 - r^2 c6) r^2) r^3
    r2 = fm.dd_mul(r, r)
    inner = fm.dd_add(_const(0.19999999999393223, r[0]), _dd_neg(fm.dd_mul(r2, _const(0.1428571356180717, r[0]))))
    inner = fm.dd_add(_const(0.3333333333333317, r[0]), _dd_neg(fm.dd_mul(inner, r2)))
    poly = fm.dd_add(r, _dd_neg(fm.dd_mul(inner, fm.dd_mul(r2, r))))
    res = _dd_where(_dd_gt(r, 0.0625), tab, _dd_where(_dd_gt(r, 0.0001) | (r[0] == f32(0.0001)), poly, r))
    res = _dd_where(sw, fm.dd_add(_const(1.5707963267948966, r[0]), _dd_neg(res)), res)
    res = _dd_where(xneg, fm.dd_add(_const(3.141592653589793, r[0]), _dd_neg(res)), res)
    out = fm.dd_round(res)
    out = jnp.where(yneg, -out, out)
    # the exponent-difference shortcuts the binary takes before the general formula
    pio2 = jnp.where(yneg, f32(-1.5707964), f32(1.5707964))
    pi = jnp.where(yneg, f32(-3.1415927), f32(3.1415927))
    out = jnp.where(~xneg & (diff < -13), y / jnp.where(ax == 0, f32(1), x), out)
    out = jnp.where(xneg & (diff < -26), pi, out)
    out = jnp.where(diff > 26, pio2, out)
    out = jnp.where(ax == 0, pio2, out)
    out = jnp.where(ay == 0, jnp.where(xneg, pi, y), out)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Unity's float32 Transform math

def quat_mul_vec(q, u):
    """Unity's quatMulVec (native, SIMD lanes): rotate u (..., 3) by q (..., 4). Bit-exact on every recorded Transform."""
    x, y, z, w = (q[..., i:i + 1] for i in range(4))
    c0 = jnp.array([-2, 2, -2], f32)
    c1 = jnp.array([-2, -2, 2], f32)
    c2 = jnp.array([2, -2, -2], f32)
    qyxw = jnp.concatenate([y, x, w], -1)
    qzwx = jnp.concatenate([z, w, x], -1)
    qwzy = jnp.concatenate([w, z, y], -1)
    m0 = m.mul(m.mul(c0, y), qyxw) - m.mul(m.mul(c2, z), qzwx)
    m1 = m.mul(m.mul(c1, z), qwzy) - m.mul(m.mul(c0, x), qyxw)
    m2 = m.mul(m.mul(c2, x), qzwx) - m.mul(m.mul(c1, y), qwzy)
    return (u + m.mul(u[..., 0:1], m0)) + (m.mul(u[..., 1:2], m1) + m.mul(u[..., 2:3], m2))


def quat_mul(a, b):
    """Quaternionf operator* (scalar code): a * b with every component ((w1 v2 + v1 w2) + ...) in C order."""
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    mm = m.mul
    return jnp.stack([((mm(aw, bx) + mm(ax, bw)) + mm(ay, bz)) - mm(az, by),
                      ((mm(aw, by) + mm(ay, bw)) + mm(az, bx)) - mm(ax, bz),
                      ((mm(aw, bz) + mm(az, bw)) + mm(ax, by)) - mm(ay, bx),
                      ((mm(aw, bw) - mm(ax, bx)) - mm(ay, by)) - mm(az, bz)], -1)


def quat_mul_simd(p, q):
    """p * q as the Transform hierarchy code computes it (SIMD lanes: CalculateGlobalRotation's parent * child and
    InverseTransformPositionAndRotation's conj(parent) * rotation): each component a chain of differences in its own
    order, e.g. x = -(((p_z q_y - p_y q_z) - p_w q_x) - p_x q_w)."""
    p0, p1, p2, p3 = (p[..., i] for i in range(4))
    q0, q1, q2, q3 = (q[..., i] for i in range(4))
    mm = m.mul
    return jnp.stack([-(((mm(p2, q1) - mm(p1, q2)) - mm(p3, q0)) - mm(p0, q3)),
                      -(((mm(p0, q2) - mm(p2, q0)) - mm(p3, q1)) - mm(p1, q3)),
                      -(((mm(p1, q0) - mm(p3, q2)) - mm(p2, q3)) - mm(p0, q1)),
                      ((mm(p3, q3) - mm(p0, q0)) - mm(p2, q2)) - mm(p1, q1)], -1)


def normalize_rotation(q):
    """SetLocalTR's normalisation of a stored rotation, per lane: q_i / sqrt(n_i) with the squared norm summed in a
    lane-dependent order (x: (zz + ww) + (xx + yy), y: (ww + xx) + (yy + zz), z: (xx + yy) + (zz + ww),
    w: (yy + zz) + (ww + xx)); identity lanes where n_i <= 1e-30."""
    s = m.mul(q, q)
    xx, yy, zz, ww = (s[..., i] for i in range(4))
    n = jnp.stack([(zz + ww) + (xx + yy), (ww + xx) + (yy + zz), (xx + yy) + (zz + ww), (yy + zz) + (ww + xx)], -1)
    ok = n > f32(1e-30)
    root = lax.optimization_barrier(jnp.sqrt(jnp.where(ok, n, f32(1))))
    return jnp.where(ok, q / root, jnp.array([0, 0, 0, 1], f32))


def conj(q):
    return q * jnp.array([-1, -1, -1, 1], f32)


def world_position(lp, lq, i: int):
    """Transform.position of chain member i from the locals (child-up; local scales are 1)."""
    p = lp[i]
    for j in range(i - 1, -1, -1):
        p = lp[j] + quat_mul_vec(lq[j], p)
    return p


def world_rotation(lq, i: int):
    """Transform.rotation of chain member i (CalculateGlobalRotation: child-up SIMD products). Bit-exact on every
    recorded Transform."""
    q = lq[i]
    for j in range(i - 1, -1, -1):
        q = quat_mul_simd(lq[j], q)
    return q


def _rotation_matrix(q):
    """The hierarchy code's quaternion -> 3x3 matrix (SIMD lanes), as three columns (..., 3), e.g. column 0 =
    (((-2y) y + (-2z) z) + 1, (2y) x + (2z) w, (-2y) w + (2z) x)."""
    x, y, z, w = (q[..., i] for i in range(4))
    mm, two, one = m.mul, f32(2), f32(1)
    c0 = jnp.stack([(mm(mm(-two, y), y) + mm(mm(-two, z), z)) + one, mm(mm(two, y), x) + mm(mm(two, z), w),
                    mm(mm(-two, y), w) + mm(mm(two, z), x)], -1)
    c1 = jnp.stack([mm(mm(-two, z), w) + mm(mm(two, x), y), (mm(mm(-two, z), z) + mm(mm(-two, x), x)) + one,
                    mm(mm(two, z), y) + mm(mm(two, x), w)], -1)
    c2 = jnp.stack([mm(mm(two, x), z) + mm(mm(two, y), w), mm(mm(-two, x), w) + mm(mm(two, y), z),
                    (mm(mm(-two, x), x) + mm(mm(-two, y), y)) + one], -1)
    return c0, c1, c2


def _mat_mul(p, a):
    """3x3 column product p a: column k = p0 a_k.x + (p1 a_k.y + p2 a_k.z)."""
    return tuple(m.mul(p[0], a[k][..., 0:1]) + (m.mul(p[1], a[k][..., 1:2]) + m.mul(p[2], a[k][..., 2:3]))
                 for k in range(3))


def lossy_scale(lq, i: int):
    """Transform.lossyScale of chain member i (GetWorldScaleLossy: the diagonal of R(conj(world rotation)) times the
    child-up product of the local rotation matrices; local scales are 1). Bit-exact on every recorded Transform.
    Unity's AnchoredJoint2D::ConfigureAnchors scales a joint's anchor by it (x, y) when the joint is (re)created:
    b2 localAnchorA = anchor * lossyScale(own Transform), localAnchorB = connectedAnchor * lossyScale(connected body's
    Transform); exact on every re-created climber joint of the K=4 recordings (see `joint_anchor`)."""
    rs = _rotation_matrix(lq[i])
    for j in range(i - 1, -1, -1):
        rs = _mat_mul(_rotation_matrix(lq[j]), rs)
    sm = _mat_mul(_rotation_matrix(conj(world_rotation(lq, i))), rs)
    return jnp.stack([sm[0][..., 0], sm[1][..., 1], sm[2][..., 2]], -1)


def joint_anchor(anchor, lq, i: int):
    """A re-created joint's b2 local anchor: the Unity anchor (2,) times the lossy scale (x, y) of Transform i."""
    return m.mul(jnp.asarray(anchor, f32), lossy_scale(lq, i)[:2])


def unscale_anchor(local_anchor, scale):
    """The Unity anchor a (float32) with a * scale == local_anchor, per component (numpy, host): the float nearest
    local_anchor / scale that reproduces it (searching +-8 ulps). Raises if none does."""
    v, s = np.asarray(local_anchor, F32), np.asarray(scale, F32)
    out = np.zeros_like(v)
    for k in np.ndindex(v.shape):
        base = np.array([v[k] / s[k] if s[k] != 0 else v[k]], F32)
        for d in sorted(range(-8, 9), key=abs):
            c = (base.view(np.int32) + d).view(F32)[0]
            if c * s[k] == v[k]:
                out[k] = c
                break
        else:
            raise ValueError(f"no float anchor reproduces {v[k]!r} at scale {s[k]!r}")
    return out


def _to_local(lp, lq, i: int, p):
    """World point p -> local coordinates of chain member i's parent (root first: conj(q) (p - t))."""
    for j in range(i):
        p = quat_mul_vec(conj(lq[j]), p - lp[j])
    return p


def _rotation_to_local(lq, i: int, q):
    """World rotation q -> local rotation of chain member i before normalisation (root first: conj(q_j) * q)."""
    for j in range(i):
        q = quat_mul_simd(conj(lq[j]), q)
    return q


# ---------------------------------------------------------------------------------------------------------------------
# The game's writes

def rotation_from_body(q, angle):
    """PhysicsUtility2D::XRotUpdateFromRBAngle: the world rotation Unity writes for a body at `angle` (rad) whose
    Transform has world rotation q: q' = q with w >= 0 (all four negated otherwise), its z angle theta =
    atan2f(q'.z, q'.w) (times 2, times 0.5), then (Rz(angle) * conj(Rz(theta))) * q', each Rz from sinf/cosf of the
    half angle. Keeps the Transform's out-of-plane rotation; the result depends on q (history), not only on angle."""
    q = jnp.asarray(q, f32)
    q = jnp.where(q[..., 3:4] < 0, -q, q)
    th = atan2f(q[..., 2], q[..., 3])
    h = m.mul(th + th, f32(0.5))
    sh, ch = sincosf(h)
    hb = m.mul(jnp.asarray(angle, f32), f32(0.5))
    sb, cb = sincosf(hb)
    z = jnp.zeros_like(sh)
    rinv = jnp.stack([-z, -z, -sh, ch], -1)
    rb = jnp.stack([z, z, sb, cb], -1)
    return quat_mul(quat_mul(rb, rinv), q)


def target_rotation(angle, tilt):
    """Approximate world rotation for a body at `angle` (rad) with no Transform history: Rz(angle) * tilt, tilt being
    the body's constant out-of-plane rotation (Ry(+-90 deg) for Handle and PoleMiddle, identity otherwise). Used only
    to build a PoseState without recorded Transforms (state_from_bodies)."""
    h = m.mul(jnp.asarray(angle, f32), f32(0.5))
    s, c = sincosf(h)
    z = jnp.zeros_like(h)
    rz = jnp.stack([z, z, s, c], -1)
    return quat_mul(rz, tilt)


def write_positions(lp_old, lq_new, xy):
    """Step 1: the new local positions after a Simulate. lp_old (6, 3) the locals before, lq_new (6, 4) the locals
    rotations after the write, xy (6, 2) the bodies' new positions. Parents first; each target z is the Transform's
    world z at that moment (new parents, own old local). Bit-exact given lq_new."""
    lp = lp_old
    for i in range(lp.shape[0]):
        zc = world_position(lp, lq_new, i)[2]
        p = _to_local(lp, lq_new, i, jnp.stack([xy[i, 0], xy[i, 1], zc]))
        lp = lp.at[i].set(p)
    return lp


def write_rotations(lq_old, angle, tilt=None):
    """Step 1, rotations: parents first, each body's new world rotation rotation_from_body(its current world
    rotation, body angle) is converted to a local rotation against the (new) parents and normalised (SetLocalTR).
    Bit-exact on all 10,560 recorded K=4 steps for all six Transforms. `tilt` is unused (kept for callers)."""
    lq = lq_old
    for i in range(lq.shape[0]):
        wq = rotation_from_body(world_rotation(lq, i), angle[i])
        lq = lq.at[i].set(normalize_rotation(_rotation_to_local(lq, i, wq)))
    return lq


def write(ps: PoseState, xy, angle, tilt=None) -> PoseState:
    """Unity's Transform write after one Simulate (Transform::SetPositionAndRotation per body, parents first):
    rotations, then positions. Bit-exact."""
    lq = write_rotations(ps.lq, angle)
    return PoseState(write_positions(ps.lp, lq, xy), lq)


def pose_update(ps: PoseState) -> PoseState:
    """Step 3, the Transform side of the once-per-action pose update: the Hub's position restore (a world -> local
    round trip of its local position). Bit-exact."""
    hw = world_position(ps.lp, ps.lq, 1)
    return ps._replace(lp=ps.lp.at[1].set(_to_local(ps.lp, ps.lq, 1, hw)))


def body_positions(ps: PoseState):
    """(6, 2) Transform world positions (x, y): the body positions after Unity re-syncs them from the Transforms."""
    return jnp.stack([world_position(ps.lp, ps.lq, i)[:2] for i in range(ps.lp.shape[0])])


def tip_z(ps: PoseState):
    """The Tip Transform's world z (PlayerControl reads tip.position)."""
    return world_position(ps.lp, ps.lq, len(CHAIN) - 1)[2]


def tilts_from(lq_rec, angle):
    """Each body's constant out-of-plane rotation, from one recorded state: conj(Rz(angle)) * world rotation, snapped
    to the nearest of identity / Ry(+90) / Ry(-90) (numpy, host)."""
    lq_rec = np.asarray(lq_rec, F32)
    out = np.zeros((len(CHAIN), 4), F32)
    s2 = F32(0.70710677)
    for i in range(len(CHAIN)):
        w = np.asarray(world_rotation(jnp.asarray(lq_rec), i), np.float64)
        h = float(angle[i]) / 2
        rz = np.array([0, 0, -np.sin(h), np.cos(h)])
        t = np.asarray(quat_mul(jnp.asarray(rz, f32), jnp.asarray(w, f32)), np.float64)
        cands = [np.array([0, 0, 0, 1], F32), np.array([0, s2, 0, s2], F32), np.array([0, -s2, 0, s2], F32)]
        out[i] = max(cands, key=lambda c: abs(float(np.dot(c, t))))
    return out


def state_from_bodies(xy, angle, z, tilt, lp_template=None) -> PoseState:
    """A PoseState for bodies at (xy, angle) when no recorded Transforms exist (K=1 recordings before version 3):
    world rotations target_rotation (not Unity's history-dependent values), local positions from the world positions
    with z (6,) the Transforms' world z (constant per body: the pose update pins the Hub's, the rest follow the rig)."""
    angle = jnp.asarray(angle, f32)
    tilt = jnp.asarray(tilt, f32)
    lq = jnp.tile(jnp.array([0, 0, 0, 1], f32), (len(CHAIN), 1))
    for i in range(len(CHAIN)):
        wq = target_rotation(angle[i], tilt[i])
        lq = lq.at[i].set(normalize_rotation(_rotation_to_local(lq, i, wq)))
    lp = jnp.zeros((len(CHAIN), 3), f32) if lp_template is None else jnp.asarray(lp_template, f32)
    for i in range(len(CHAIN)):
        lp = lp.at[i].set(_to_local(lp, lq, i, jnp.stack([jnp.asarray(xy[i, 0], f32), jnp.asarray(xy[i, 1], f32),
                                                          jnp.asarray(z[i], f32)])))
    return PoseState(lp, lq)


# ---------------------------------------------------------------------------------------------------------------------
# Recordings

def parse_transforms(record: bytes) -> dict:
    """{which: (names, (n, 20) float32)} for the TFM sections of one version-3 record."""
    from goi_jax.parity import states as S
    out = {}
    for tag, buf in S._sections(record, 8):
        if tag & 0x00FFFFFF != TAG_TRANSFORMS & 0x00FFFFFF:
            continue
        which = (tag >> 24) - 0x30
        (n,) = struct.unpack_from("<i", buf, 0)
        o, names, vals = 4, [], []
        for _ in range(n):
            (k,) = struct.unpack_from("<i", buf, o)
            names.append(buf[o + 4:o + 4 + k].decode("utf-8"))
            o += 4 + k
            vals.append(np.frombuffer(buf, "<f4", 20, o).copy())
            o += 80
        out[which] = (names, np.stack(vals))
    return out


def load_transforms(path) -> np.ndarray | None:
    """(N, 4, 6, 20) float32: per record, TFM0..3 of the six chain Transforms (NaN where a section is absent: TFM3
    exists only on an action's last step); None for recordings without Transforms (version < 3)."""
    from goi_jax.parity import states as S
    z = np.load(path, allow_pickle=False)
    recs = S.unpack_records(z["blob"], z["offsets"])
    out = np.full((len(recs), 4, len(CHAIN), 20), np.nan, F32)
    seen = False
    for t, r in enumerate(recs):
        for which, (names, v) in parse_transforms(r).items():
            idx = [names.index(n) for n in CHAIN]
            out[t, which] = v[idx]
            seen = True
    return out if seen else None

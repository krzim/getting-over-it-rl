"""Float32 JAX helpers for Mono's double-intermediate float expressions and Unity's Mathf calls.

Unity 2020.3's Mono evaluates a C# float expression in double precision and rounds it to float32 where it is stored (a
local, field, argument or return value): settled for GOI's PlayerControl by the recordings (Q7; tests/game). JAX here
runs in float32 without x64, so a double expression is evaluated as an unevaluated float-float sum hi + lo (~48 bits)
and rounded once. That reproduces the double result except when its exact value lies within ~2^-48 (relative) of a
float32 rounding midpoint.

Two XLA hazards are handled explicitly:
- multiply-add contraction: on CPU, `x + a * b` becomes one fused multiply-add (one rounding where C# rounds twice).
  `rounded(a * b)` keeps the product a separate rounded value (a select XLA cannot see through).
- simplifier folds: `(a + b) - a` is folded to `b` and kills Knuth's TwoSum error term; `two_sum` puts the rounded sum
  behind an optimization barrier.

The float-float core (split, two_prod, two_sum, add/mul/div, atan2) is ported from the cuphead-rl project
(src/cuphead_rl/sim/common/trig.py), where it was checked bit for bit against numpy float64 under jit on CPU.
"""
from __future__ import annotations

from fractions import Fraction

import jax.numpy as jnp
import numpy as np
from jax import lax

f32, u32, i32 = jnp.float32, jnp.uint32, jnp.int32
_NAN = np.float32(np.nan)


def rounded(v):
    """`v` rounded to float32 before whatever uses it (keeps a product out of a fused multiply-add)."""
    v = jnp.asarray(v)
    return jnp.where(jnp.isnan(v), _NAN, v)


def _barrier(x):
    return lax.optimization_barrier(x)


def _split(a):
    """a = hi + lo exactly, each with at most 12 significant bits (mask the low 12 stored mantissa bits)."""
    bits = lax.bitcast_convert_type(jnp.asarray(a, f32), u32)
    hi = lax.bitcast_convert_type(bits & u32(0xFFFFF000), f32)
    return hi, a - hi


def two_prod(a, b):
    """(p, e): p = fl(a b), a b = p + e exactly (Dekker on 12-bit halves, so contractions among them change nothing)."""
    a, b = jnp.asarray(a, f32), jnp.asarray(b, f32)
    p = rounded(a * b)
    ah, al = _split(a)
    bh, bl = _split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def two_sum(a, b):
    """(s, e): s = fl(a + b), a + b = s + e exactly (Knuth)."""
    a, b = jnp.asarray(a, f32), jnp.asarray(b, f32)
    s = _barrier(a + b)
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _fast_two_sum(a, b):
    s = _barrier(a + b)
    return s, b - (s - a)


def dd_add(a, b):
    """(hi, lo) + (hi, lo)."""
    s, e1 = two_sum(a[0], b[0])
    t, e2 = two_sum(a[1], b[1])
    s, e1 = _fast_two_sum(s, e1 + t)
    return _fast_two_sum(s, e1 + e2)


def dd_mul(a, b):
    p, e = two_prod(a[0], b[0])
    return _fast_two_sum(p, e + (rounded(a[0] * b[1]) + rounded(a[1] * b[0])))


def dd_mul_f(a, b):
    """(hi, lo) * float32."""
    p, e = two_prod(a[0], b)
    return _fast_two_sum(p, e + rounded(a[1] * b))


def dd_div(a, b):
    """(hi, lo) / (hi, lo), b != 0: two correction steps on the float32 quotient."""
    q1 = a[0] / b[0]
    p = dd_mul_f(b, q1)
    r = dd_add(a, (-p[0], -p[1]))
    q2 = r[0] / b[0]
    p = dd_mul_f(b, q2)
    r = dd_add(r, (-p[0], -p[1]))
    q3 = r[0] / b[0]
    q = _fast_two_sum(q1, q2)
    return dd_add(q, (q3, jnp.zeros_like(q3)))


def pair(x):
    x = jnp.asarray(x, f32)
    return x, jnp.zeros_like(x)


def dd_round(p):
    """A (hi, lo) pair rounded to the nearest float32: Mono storing a double into a float."""
    return (p[0] + p[1]).astype(f32)


def dd_sqrt_round(p):
    """float32(sqrt(hi + lo)) for hi + lo >= 0: a float32 square root, then one Newton correction from the exact
    residual (hi + lo) - r^2, so the result is the correctly rounded root of the float-float value (Mathf.Sqrt of a
    double sum, Vector2.magnitude)."""
    hi, lo = p
    r0 = jnp.sqrt(jnp.maximum(hi, f32(0)))
    sq = two_prod(r0, r0)
    res = (hi - sq[0]) + (lo - sq[1])  # hi - sq[0] is exact (Sterbenz) when r0 is within an ulp
    safe = jnp.where(r0 > 0, r0, f32(1))
    r = r0 + res / (f32(2) * safe)
    return jnp.where(r0 > 0, r, r0)


def magnitude(*components):
    """Vector2/Vector3.magnitude under Mono: (float)Math.Sqrt(x * x + y * y [+ z * z]) with a double sum."""
    acc = two_prod(components[0], components[0])
    for c in components[1:]:
        acc = dd_add(acc, two_prod(c, c))
    return dd_sqrt_round(acc)


def dot2(ax, ay, bx, by):
    """Vector2.Dot under Mono: lhs.x * rhs.x + lhs.y * rhs.y in double, rounded once."""
    return dd_round(dd_add(two_prod(ax, bx), two_prod(ay, by)))


def lerp(a, b, t):
    """Mathf.Lerp(a, b, t) under Mono for t in [0, 1]: a + (b - a) * t in double, rounded once at the return."""
    d = two_sum(b, -jnp.asarray(a, f32))
    return dd_round(dd_add(pair(a), dd_mul_f(d, t)))


def normalize(*components, eps=np.float32(1e-5)):
    """Vector2/Vector3.normalized: components / magnitude when magnitude > 1e-5, else zero (rounded per component)."""
    m = magnitude(*components)
    ok = m > eps
    safe = jnp.where(ok, m, f32(1))
    return tuple(jnp.where(ok, jnp.asarray(c, f32) / safe, f32(0)) for c in components)


# --- atan2 -----------------------------------------------------------------------------------------------------------

def _parts(decimal: str):
    exact, parts = Fraction(decimal), []
    for _ in range(3):
        p = np.float32(float(exact))
        parts.append(p)
        exact -= Fraction(float(p))
    return parts


_PIO2 = _parts("1.570796326794896619231321691639751442098584699687552910487472296153908203143104499314017412671058534")
_PI = _parts("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117068")


def _host_pair(value):
    hi = np.float32(value)
    return hi, np.float32(np.float64(value) - np.float64(hi))


def _poly(x, coefficients):
    ch, cl = _host_pair(coefficients[-1])
    s = (jnp.full_like(x[0], ch), jnp.full_like(x[0], cl))
    for c in reversed(coefficients[:-1]):
        kh, kl = _host_pair(c)
        s = dd_add(dd_mul(s, x), (jnp.full_like(x[0], kh), jnp.full_like(x[0], kl)))
    return s


_ATAN_COEFFS = [(-1.0) ** k / (2 * k + 1) for k in range(9)]  # atan(w)/w in w^2, |w| <= 1/16
_ATAN_TABLE = [_host_pair(np.arctan(k / 8.0)) for k in range(9)]


def _atan_unit(z):
    """atan(z) in float-float for 0 <= z <= 1: atan(k/8) + atan((z - k/8) / (1 + z k/8)), k = round(8 z)."""
    zh = z[0]
    k = jnp.clip(jnp.round(zh * f32(8.0)), 0.0, 8.0)
    c = k * f32(0.125)
    n = dd_add(z, (-c, jnp.zeros_like(zh)))
    d = dd_add(pair(jnp.ones_like(zh)), dd_mul_f(z, c))
    w = dd_div(n, d)
    s = _poly(dd_mul(w, w), _ATAN_COEFFS)
    a = dd_mul(w, s)
    ki = k.astype(i32)
    th = jnp.asarray(np.array([t[0] for t in _ATAN_TABLE], np.float32))[ki]
    tl = jnp.asarray(np.array([t[1] for t in _ATAN_TABLE], np.float32))[ki]
    return dd_add((th, tl), a)


def atan2(y, x):
    """Mathf.Atan2(y, x) = (float)Math.Atan2(y, x): the correctly rounded float32 angle (radians), IEEE signed zeros."""
    y, x = jnp.asarray(y, f32), jnp.asarray(x, f32)
    ax, ay = jnp.abs(x), jnp.abs(y)
    swap = ay > ax
    num, den = jnp.where(swap, ax, ay), jnp.where(swap, ay, ax)
    zero = den == 0
    safe = jnp.where(zero, f32(1.0), den)
    z = dd_div(pair(jnp.where(zero, f32(0.0), num)), pair(safe))
    th, tl = _atan_unit(z)
    sh, sl = dd_add((jnp.full_like(th, _PIO2[0]), jnp.full_like(th, _PIO2[1] + _PIO2[2])), (-th, -tl))
    th, tl = jnp.where(swap, sh, th), jnp.where(swap, sl, tl)
    mh, ml = dd_add((jnp.full_like(th, _PI[0]), jnp.full_like(th, _PI[1] + _PI[2])), (-th, -tl))
    neg_x = jnp.signbit(x)
    th, tl = jnp.where(neg_x, mh, th), jnp.where(neg_x, ml, tl)
    neg_y = jnp.signbit(y)
    th, tl = jnp.where(neg_y, -th, th), jnp.where(neg_y, -tl, tl)
    return (th + tl).astype(f32)

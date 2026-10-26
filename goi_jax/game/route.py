"""The level route (LevelSpline) and the plugin's progress, as pure float32 JAX functions.

The game measures progress with a Curvy spline: ProgressMeter's `spline.TFToDistance(spline.GetNearestPointTF(
player.position))` (docs/understanding/progress.md section 1); the plugin observes that through its guard against
forward leaps (plugin/Game.cs Progress) and the route ahead as `route.Interpolate(route.DistanceToTF(along + d))`
(Game.cs Observe). The spline data comes from the plugin's 'u' row (GoiClient.route(), plugin/GameRoute.cs).

GOI's LevelSpline is Linear with one cache step per segment (CacheDensity 1, every segment CacheSize 1 and step 1,
ApproximationDistances = [0, Length]), so every Curvy query reduces to closed forms on its 119 control points. The
float model is Mono's (goi_jax.game.floatmath): a C# float expression is evaluated in double and rounded where it is
stored, so e.g. `Vector3.sqrMagnitude` is one rounding of a double sum of squares, and `SegmentToTF` is
`(double)i / Count + 1.0 / Count * F` rounded once. Decompile references (Assembly-CSharp, FluffyUnderware.*):

    GetNearestPointTF            CurvySpline.cs:1400-1433   per segment F, then the true curve point, `<=` -> last
    GetNearestPointF             CurvySplineSegment.cs:1761-1794 (cache size 1: F = LinePointDistanceSqr's frag)
    LinePointDistanceSqr         DTMath.cs:126-144
    TFToSegment / TFToDistance   CurvySpline.cs:1021-1049
    LocalFToDistance             CurvySplineSegment.cs:1823-1838, getApproximationIndexINTERNAL :1940-1951
    DistanceToTF / ToSegment     CurvySpline.cs:1090-1148
    DistanceToLocalF             CurvySplineSegment.cs:1796-1821
    SegmentToTF                  CurvySpline.cs:1066-1076
    interpolateLinear            CurvySplineSegment.cs:2028-2033 (Vector3.LerpUnclamped)

Interface (vmap over a batch; RouteModel is a pytree of arrays, pass it as an argument):

    rm = load_route("dumps/obs/route.json")         # or route_model(GoiClient.route())
    tf = nearest_tf(rm, p3)                          # p3 = (x, y, z): the Player Transform position (z = rm.player_z)
    d  = tf_to_distance(rm, tf)                      # ProgressMeter's progress = raw_progress(rm, p3)
    q  = interpolate(rm, distance_to_tf(rm, s))      # the route point at arc length s, (x, y, z)
    p  = guarded_progress(rm, prev, p3)              # the plugin's Progress (obs field "progress")
    a  = ahead(rm, along)                            # (len(AHEAD), 3) route points along + AHEAD
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import floatmath as fm

f32, i32 = jnp.float32, jnp.int32
F32 = np.float32


class RouteModel(NamedTuple):
    p0: jax.Array        # (S, 3) segment start control point (= Approximation[0])
    p1: jax.Array        # (S, 3) next control point (= Approximation[1])
    dist: jax.Array      # (S,) segment Distance (arc length at its start)
    seg_len: jax.Array   # (S,) segment Length
    ad0: jax.Array       # (S,) ApproximationDistances[0]
    ad1: jax.Array       # (S,) ApproximationDistances[1]
    end_hi: jax.Array    # (S,) Distance + Length as a double, split hi + lo (DistanceToSegment's walk compares it)
    end_lo: jax.Array
    tf0_hi: jax.Array    # (S,) (double) i / Count, split
    tf0_lo: jax.Array
    inv_hi: jax.Array    # () 1.0 / Count, split
    inv_lo: jax.Array
    count: jax.Array     # () float32 Count
    length: jax.Array    # () spline Length
    ahead: jax.Array     # (A,) Game.Ahead
    max_advance: jax.Array  # () Game.MaxAdvance
    on_route: jax.Array     # () Game.OnRoute
    player_z: jax.Array     # () the Player Transform's z


def _split(x64):
    hi = np.asarray(x64, np.float64).astype(F32)
    lo = (np.asarray(x64, np.float64) - hi.astype(np.float64)).astype(F32)
    return hi, lo


def route_model(row: dict) -> RouteModel:
    """RouteModel from the plugin's route row (GoiClient.route() / route.json)."""
    if row["interpolation"] != "Linear" or row["closed"]:
        raise NotImplementedError(f"route port covers open Linear splines, got {row['interpolation']} closed={row['closed']}")
    segs = row["segments"]
    for s in segs:
        if s["cacheSize"] != 1 or s["stepSize"] != 1 or len(s["approximationDistances"]) != 2:
            raise NotImplementedError("route port covers one cache step per segment")
    S = len(segs)
    p0 = np.array([s["approximation"][:3] for s in segs], F32)
    p1 = np.array([s["approximation"][3:6] for s in segs], F32)
    if not (np.array_equal(p0, np.array([s["p0"] for s in segs], F32)) and
            np.array_equal(p1, np.array([s["p1"] for s in segs], F32))):
        raise ValueError("the approximation cache differs from the control points")
    dist = np.array([s["distance"] for s in segs], F32)
    seg_len = np.array([s["length"] for s in segs], F32)
    ad = np.array([s["approximationDistances"] for s in segs], F32)
    end_hi, end_lo = _split(dist.astype(np.float64) + seg_len.astype(np.float64))
    tf0_hi, tf0_lo = _split(np.arange(S, dtype=np.float64) / S)
    inv_hi, inv_lo = _split(1.0 / S)
    return RouteModel(p0=jnp.asarray(p0), p1=jnp.asarray(p1), dist=jnp.asarray(dist), seg_len=jnp.asarray(seg_len),
                      ad0=jnp.asarray(ad[:, 0]), ad1=jnp.asarray(ad[:, 1]), end_hi=jnp.asarray(end_hi),
                      end_lo=jnp.asarray(end_lo), tf0_hi=jnp.asarray(tf0_hi), tf0_lo=jnp.asarray(tf0_lo),
                      inv_hi=jnp.asarray(inv_hi), inv_lo=jnp.asarray(inv_lo), count=jnp.asarray(F32(S)),
                      length=jnp.asarray(F32(row["length"])), ahead=jnp.asarray(np.array(row["ahead"], F32)),
                      max_advance=jnp.asarray(F32(row["maxAdvance"])), on_route=jnp.asarray(F32(row["onRoute"])),
                      player_z=jnp.asarray(F32(row.get("playerZ", 0.0))))


def load_route(path) -> RouteModel:
    return route_model(json.loads(Path(path).read_text()))


# ------------------------------------------------------------------------------------------------- Mono float helpers

def _sum3(a, b, c):
    """x*x-style double sums of three float products, rounded once: Vector3.Dot / sqrMagnitude under Mono."""
    return fm.dd_round(fm.dd_add(fm.dd_add(a, b), c))


def _dot3(ax, ay, az, bx, by, bz):
    return _sum3(fm.two_prod(ax, bx), fm.two_prod(ay, by), fm.two_prod(az, bz))


def _sub(a, b):
    return fm.rounded(jnp.asarray(a, f32) - jnp.asarray(b, f32))


def div(a, b):
    """a / b correctly rounded (float32, as C# float division). XLA rewrites a division by an operand it sees as
    shared (a constant, or an unbatched array under vmap: `A / broadcast(B)`) into a multiply by the reciprocal, which
    is off by an ulp in ~0.5% of cases; one residual correction restores the rounded quotient."""
    a, b = jnp.asarray(a, f32), jnp.asarray(b, f32)
    q = a / b
    p, e = fm.two_prod(q, b)
    r = (a - p) - e  # a - q b, exact while q is within an ulp
    return fm.rounded(q + r / b)


def _le_dd(x, hi, lo):
    """x <= hi + lo (exact compare of a float with a double held as hi + lo)."""
    return (hi - x) >= -lo  # hi - x is exact where it matters (x within a factor 2 of hi)


def _lt_dd(hi, lo, x):
    """hi + lo < x."""
    return (x - hi) > lo


def _segment_to_tf(rm, idx, local_f):
    """CurvySpline.SegmentToTF: (double)idx / Count + 1.0 / Count * localF, rounded once."""
    base = (rm.tf0_hi[idx], rm.tf0_lo[idx])
    term = fm.dd_mul((rm.inv_hi, rm.inv_lo), fm.pair(local_f))
    return fm.dd_round(fm.dd_add(base, term))


def _lerp3(rm, idx, t):
    """interpolateLinear: Vector3.LerpUnclamped(P0, P1, Clamp01(t)), each component a + (b - a) * t in double."""
    t = jnp.clip(t, f32(0), f32(1))
    a, b = rm.p0[idx], rm.p1[idx]
    return tuple(fm.lerp(a[..., k], b[..., k], t) for k in range(3))


# ------------------------------------------------------------------------------------------------- Curvy queries

def _segment_f(rm, px, py, pz):
    """(S,) GetNearestPointF of every segment: the frag of DTMath.LinePointDistanceSqr(A0, A1, p)."""
    a, b = rm.p0, rm.p1
    vx, vy, vz = _sub(b[:, 0], a[:, 0]), _sub(b[:, 1], a[:, 1]), _sub(b[:, 2], a[:, 2])
    wx, wy, wz = _sub(px, a[:, 0]), _sub(py, a[:, 1]), _sub(pz, a[:, 2])
    num = _dot3(wx, wy, wz, vx, vy, vz)
    num2 = _dot3(vx, vy, vz, vx, vy, vz)
    safe = jnp.where(num2 > 0, num2, f32(1))
    frag = jnp.where(num <= 0, f32(0), jnp.where(num2 <= num, f32(1), div(num, safe)))
    return frag


def nearest_tf(rm: RouteModel, p3) -> jax.Array:
    """CurvySpline.GetNearestPointTF(p): per segment its nearest F on the cache, then the segment whose interpolated
    point is nearest (squared 3D distance, ties to the later segment), as a TF."""
    px, py, pz = (jnp.asarray(p3[k], f32) for k in range(3))
    S = rm.p0.shape[0]
    frag = _segment_f(rm, px, py, pz)
    idx = jnp.arange(S)
    qx, qy, qz = _lerp3(rm, idx, frag)
    dx, dy, dz = _sub(qx, px), _sub(qy, py), _sub(qz, pz)
    d2 = _sum3(fm.two_prod(dx, dx), fm.two_prod(dy, dy), fm.two_prod(dz, dz))
    best = S - 1 - jnp.argmin(d2[::-1])  # `<=` keeps the last minimum
    return _segment_to_tf(rm, best, frag[best])


def _tf_to_segment(rm, tf):
    """CurvySpline.TFToSegment (Clamp): (segment, localF)."""
    tf = jnp.clip(jnp.asarray(tf, f32), f32(0), f32(1))
    num = fm.rounded(tf * rm.count)
    seg = num.astype(i32)
    local = num - seg.astype(f32)
    last = seg >= rm.p0.shape[0]
    return jnp.where(last, rm.p0.shape[0] - 1, seg), jnp.where(last, f32(1), local)


def tf_to_distance(rm: RouteModel, tf) -> jax.Array:
    """CurvySpline.TFToDistance: segment.Distance + segment.LocalFToDistance(localF)."""
    seg, local = _tf_to_segment(rm, tf)
    frag = jnp.clip(local, f32(0), f32(1))  # getApproximationIndexINTERNAL: index 0 (step 1); localF 1 -> frag 1
    a0, a1 = rm.ad0[seg], rm.ad1[seg]
    num = _sub(a1, a0)
    within = fm.dd_round(fm.dd_add(fm.pair(a0), fm.two_prod(num, frag)))
    return fm.rounded(rm.dist[seg] + within)


def raw_progress(rm: RouteModel, p3) -> jax.Array:
    """ProgressMeter's progress: TFToDistance(GetNearestPointTF(p))."""
    return tf_to_distance(rm, nearest_tf(rm, p3))


def _approximately(a, b):
    """Mathf.Approximately(a, b)."""
    diff = jnp.abs(_sub(b, a))
    thr = fm.rounded(f32(1e-6) * jnp.maximum(jnp.abs(a), jnp.abs(b)))
    return diff < jnp.maximum(thr, f32(1.1210388e-44))


def distance_to_tf(rm: RouteModel, distance) -> jax.Array:
    """CurvySpline.DistanceToTF (Clamp)."""
    S = rm.p0.shape[0]
    d0 = jnp.asarray(distance, f32)
    d = jnp.clip(d0, f32(0), rm.length)
    at_end = d == rm.length
    # DistanceToSegment: the first segment whose Distance + Length (a double) is not below d; none -> the last
    below = _lt_dd(rm.end_hi, rm.end_lo, d)
    first = jnp.argmin(below)  # first False
    seg = jnp.where(below.all() | at_end, S - 1, first)
    local_d = jnp.where(at_end, fm.rounded(rm.dist[S - 1] + rm.seg_len[S - 1]), _sub(d, rm.dist[seg]))
    # DistanceToLocalF
    L = rm.seg_len[seg]
    ld = jnp.clip(local_d, f32(0), L)
    a0, a1 = rm.ad0[seg], rm.ad1[seg]
    num = fm.dd_div(fm.two_sum(ld, -a0), jax.lax.optimization_barrier(fm.two_sum(a1, -a0)))
    frac = fm.dd_round(num)
    local_f = jnp.where(ld == 0, f32(0), jnp.where(_approximately(ld, L), f32(1), frac))
    tf = _segment_to_tf(rm, seg, local_f)
    return jnp.where(d0 == rm.length, f32(1), tf)


def interpolate(rm: RouteModel, tf) -> tuple:
    """CurvySpline.Interpolate(tf): (x, y, z) in the spline's space (LevelSpline sits at the origin, unrotated)."""
    seg, local = _tf_to_segment(rm, tf)
    return _lerp3(rm, seg, local)


def point_at(rm: RouteModel, s) -> tuple:
    """Interpolate(DistanceToTF(s))."""
    return interpolate(rm, distance_to_tf(rm, s))


def _gap(rm, s, px, py):
    """Vector2.Distance(route point at s, position)."""
    qx, qy, _ = point_at(rm, s)
    return fm.magnitude(_sub(qx, px), _sub(qy, py))


MAX_GUARD_STEPS = 2400  # 0.5 per step: covers a leap across the whole 1026-long route


def guarded_progress(rm: RouteModel, prev, p3) -> jax.Array:
    """The plugin's Progress(position) (plugin/Game.cs): the raw progress, except that a forward leap of more than
    MaxAdvance counts only as far as the distance to the route keeps shrinking along it (unless the pot is within
    OnRoute of the new point). `prev` is the last observation's progress."""
    px, py = jnp.asarray(p3[0], f32), jnp.asarray(p3[1], f32)
    prev = jnp.asarray(prev, f32)
    nearest = raw_progress(rm, p3)
    s_hi, s_lo = fm.two_sum(prev, rm.max_advance)
    take = _le_dd(nearest, s_hi, s_lo) | (_gap(rm, nearest, px, py) <= rm.on_route)

    tol = F32(0.05)

    def cond(c):
        s, best, best_gap, stop, n = c
        return (~stop) & (s <= nearest) & (n < MAX_GUARD_STEPS)

    def body(c):
        s, best, best_gap, stop, n = c
        gap = _gap(rm, s, px, py)
        t_hi, t_lo = fm.two_sum(best_gap, f32(tol))
        brk = (gap - t_hi) > t_lo  # gap > bestGap + 0.05f, compared in double
        better = (~brk) & (gap <= best_gap)
        return (fm.rounded(s + f32(0.5)), jnp.where(better, s, best), jnp.where(better, gap, best_gap), brk, n + 1)

    # lanes that take the raw value start stopped, so a batched loop only runs for the lanes that walk
    init = (fm.rounded(prev + f32(0.5)), prev, _gap(rm, prev, px, py), take, jnp.zeros_like(prev, i32))
    walked = jax.lax.while_loop(cond, body, init)[1]
    return jnp.where(take, nearest, walked)


def ahead(rm: RouteModel, along) -> jax.Array:
    """(A, 3) route points at along + Game.Ahead[i] (the argument rounded to float, as C# passes it)."""
    s = fm.rounded(jnp.asarray(along, f32) + rm.ahead)
    x, y, z = jax.vmap(lambda v: point_at(rm, v))(s)
    return jnp.stack([x, y, z], -1)

"""observe.raycast's ray-vs-fixture tests as one Pallas kernel, one world per lane.

In XLA every world tests every ray against the grid cell's full row list: K (= the largest cell, 264 rows on the
level) + the moving rows, whatever its own cell holds (median 1 row, p90 33), so a 32-ray observation costs
32 x 283 polygon ray casts per world (~6.5 ms per env step at 16k worlds on the 3090). Here each lane walks only its
cell's rows (the block's largest count; worlds are grouped by count when gpu/kernel.SORT_LANES is on) and the moving
rows, each against every ray, with observe._cast_core's arithmetic on (block,) vectors: the same float32 operations,
the per-ray minimum fraction (min is exact in any order) and the origin-inside flag (the same for every ray).

`cast(om, st, rows, n_static, ox, oy, ex, ey)` -> (best fraction per ray, inside per ray) as observe.raycast forms
them; a custom_vmap function (one pallas_call over the batch). Interpret mode on CPU (tests/game/test_rays_gpu.py).
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl

F32 = np.float32
NG = 36  # per ray-table row: vx[8], vy[8], nx[8], ny[8], count, radius, circle, body
BLOCK = 128
NUM_WARPS = 4


class _Static:
    def __init__(self, v):
        self.v = v


jax.tree_util.register_pytree_node(_Static, lambda s: ((), s.v), lambda v, _: _Static(v))


def table(om):
    """The shared ray-table rows, flat (M * NG,)."""
    f = lambda x: jnp.asarray(x).astype(jnp.float32)  # noqa: E731
    v, n = om.ray_verts, om.ray_normals
    g = jnp.concatenate([v[..., 0], v[..., 1], n[..., 0], n[..., 1], f(om.ray_count)[:, None],
                         f(om.ray_radius)[:, None], f(om.ray_circle)[:, None], f(om.ray_body)[:, None]], 1)
    return g.reshape(-1)


def _kernel(cfg, geo_ref, rows_ref, w_ref, out_ref):
    """rows: (K + Mm, blk) ray-table rows (-1: none); w: ox, oy, n_static, ex[R], ey[R], then per body px, py, qc,
    qs, active (B each); out: best fraction per ray (R rows), inside (1 row)."""
    from goi_jax.game.observe import _cast_core
    K, Mm, R, B = cfg[:4]
    blk = w_ref.shape[1]
    lane = jax.lax.broadcasted_iota(jnp.int32, (blk,), 0)
    ox, oy = w_ref[0, :], w_ref[1, :]
    n_static = w_ref[2, :].astype(jnp.int32)
    EX, EY, BX = 3, 3 + R, 3 + 2 * R
    inf = jnp.full((blk,), jnp.inf, jnp.float32)
    for j in range(R):
        out_ref[j, :] = inf
    out_ref[R, :] = jnp.zeros((blk,), jnp.float32)

    def body_row(q, b):
        return w_ref[BX + q * B + jnp.clip(b, 0, B - 1), lane]

    def row(k, _):
        r = rows_ref[k, :]
        valid = r >= 0
        base = jnp.maximum(r, 0) * NG
        g = lambda q: geo_ref[base + q]  # noqa: E731  (per-lane gather from the shared table)
        b = g(35).astype(jnp.int32)
        px, py, qc, qs = (body_row(q, b) for q in range(4))
        valid = valid & (body_row(4, b) != 0)
        count = g(32).astype(jnp.int32)
        rad, circle = g(33), g(34) != 0
        vert = lambda i: (g(i), g(8 + i))  # noqa: E731
        normal = lambda i: (g(16 + i), g(24 + i))  # noqa: E731

        def ray(j, ins):
            frac, hit, inside = _cast_core(vert, normal, count, rad, circle, valid, px, py, qc, qs, ox, oy,
                                           w_ref[EX + j, :], w_ref[EY + j, :])
            out_ref[j, :] = jnp.minimum(out_ref[j, :], jnp.where(hit, frac, jnp.inf))
            return ins | inside
        ins = jax.lax.fori_loop(0, R, ray, jnp.zeros((blk,), bool))
        out_ref[R, :] = jnp.where(ins, F32(1), out_ref[R, :])
        return 0

    jax.lax.fori_loop(0, jnp.max(n_static), row, 0)
    jax.lax.fori_loop(K, K + Mm, row, 0)


def _call(cfg, geo, rows, w):
    from goi_jax.box2d.gpu import kernel as GK
    K, Mm, R, B, blk, interpret = cfg
    N = w.shape[0]
    Np = -(-N // blk) * blk
    rows, w = jax.lax.optimization_barrier((rows, w))
    order = GK.lane_order(w[:, 2], N, blk)  # worlds grouped by their cell's row count (SORT_LANES)
    rows, w = GK.to_minor(rows, N, Np, order), GK.to_minor(w, N, Np, order)
    kw = {}
    if not interpret:
        from jax.experimental.pallas import triton as plt
        kw["compiler_params"] = plt.CompilerParams(num_warps=NUM_WARPS, num_stages=1)
    spec = lambda r: pl.BlockSpec((r, blk), lambda i: (0, i))  # noqa: E731
    with GK.triton_mode(not interpret):
        out = pl.pallas_call(
            partial(_kernel, cfg), grid=(Np // blk,),
            in_specs=[pl.BlockSpec(geo.shape, lambda i: (0,)), spec(rows.shape[0]), spec(w.shape[0])],
            out_specs=spec(R + 1), out_shape=jax.ShapeDtypeStruct((R + 1, Np), jnp.float32),
            interpret=interpret, name="goi_rays")(geo, rows, w)
    return GK.from_minor(out, N, order)


@jax.custom_batching.custom_vmap
def _cast_call(cfg_box, geo, rows, w):
    return _call(cfg_box.v, geo, rows[None], w[None])[0]


@_cast_call.def_vmap
def _cast_rule(axis_size, in_batched, cfg_box, geo, rows, w):
    if in_batched[1]:
        raise NotImplementedError("rays kernel: the ray table must be shared by all worlds")
    rows, w = (a if b else jnp.broadcast_to(a, (axis_size,) + a.shape) for a, b in zip((rows, w), in_batched[2:]))
    return _call(cfg_box.v, geo, rows, w), True


def cast(om, st, rows, n_static, ox, oy, ex, ey, block=None, interpret=None):
    """observe.raycast's (best fraction per ray (R,), inside per ray (R,)) for one world (vmap batches it):
    rows (K + Mm,) = the cell's K row slots then the Mm moving rows, n_static = the cell's row count."""
    bd = st.bodies
    B = bd.type.shape[0]
    R = ex.shape[0]
    Mm = om.moving.shape[0]
    K = rows.shape[0] - Mm
    if interpret is None:
        interpret = jax.default_backend() == "cpu"
    f = lambda x: jnp.asarray(x).astype(jnp.float32)  # noqa: E731
    w = jnp.concatenate([jnp.stack([f(ox), f(oy), f(n_static)]), f(ex), f(ey), bd.xf_p[:, 0], bd.xf_p[:, 1],
                         bd.xf_q[:, 0], bd.xf_q[:, 1], f(bd.active)])
    out = _cast_call(_Static((K, Mm, R, B, block or BLOCK, interpret)), table(om), rows.astype(jnp.int32), w)
    return out[:R], jnp.broadcast_to(out[R] != 0, (R,))

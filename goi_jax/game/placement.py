"""How Unity bakes the hammer head's (Tip's) polygon into Box2D vertices, per world build (level load, placement, load).

Measured (goi_jax/parity/diagnose_bake.py; dumps/native, dumps/parity*, the authored PolygonCollider2D points):

1. The Tip's b2PolygonShape vertices are the authored collider points p (16 points, 15 used by the 4 convex pieces)
   through one 2x2 float32 matrix A, each product rounded, then summed:

       v.x = fl(fl(A00 * p.x) + fl(A01 * p.y)),   v.y = fl(fl(A10 * p.x) + fl(A11 * p.y))

   No translation, the same A for all four pieces. At the level start A = diag(1 + 2^-22, 1 + 2^-22) exactly: bit-exact
   on all 42 vertex coordinates of the native dump (goi-rl-native-start) and of both sessions' geometry.bin.
2. Every rebuild bakes a slightly different A: scale 1 + 2..4 ulp(1) per axis and a rotation of a few 1e-8 rad
   (`bake_matrix`; the smallest |r| matching every row of a chain is <= 112 * 2^-30 ~ 1.04e-7 rad on all 44 chains,
   the grid edge 160 is never needed; many grid points tie, so A is not uniquely identified). It is the body-relative
   transform of the collider (the 2D body frame against the Tip Transform's
   float32 3D matrix, built through the tilted Handle / PoleMiddle; INFERRED). With one A per chain from the grid
   scale (2..4, 2..4) ulp x rotation (-160..160) * 2^-30 rad, every Tip manifold the game stored is reproduced on all
   44 chains (3239 of 3239 rows; the session geometry: 2826); rotation alone (scale 2 ulp) suffices on 36.
3. Not reproduced from Transform data: the A of a given rebuild. Its inputs (the Transforms right after Restore's
   SetPositionAndRotation, before Simulate(0) writes the bodies back) are not in the records (TFM0 of a record is after
   that write-back; native body angles are after `b.rotation = s.rot`), and no float32 body angle (radians or degrees)
   together with the TFM0 matrix explains the rotation at the cold-start chains. A follows the placement's hammer
   angle, the same in both sessions for 0 deg (r = -112), 60 (-96), 180 (0) and the level start (0); 30 / 90 / 150 deg
   vary between chains. The per-angle table of one session predicts the other's rows: 875 of 891 (K=1, from the K=4
   session; geometry 777) and 2207 of 2348 (K=4, from K=1; geometry 2049).

`lossy_scale` (Unity's Transform.lossyScale from the chain's local rotations, bit-exact on all 34,320 recorded
Transforms) is the verified part of the 3D side, for whoever records the bake-state Transforms next.

Numpy, host-side; the authored points are recovered from a session's geometry with `invert_bake` (no game data here).
"""
from __future__ import annotations

import numpy as np

from goi_jax.box2d import shapes

F32 = np.float32
ULP1 = F32(2.0 ** -23)       # ulp(1)
ROT_UNIT = 2.0 ** -30        # the rotation grid of the measured tables (rad)
START_SCALE_ULPS = 2         # A at the level start: diag(1 + 2 ulp, 1 + 2 ulp)


def bake_matrix(scale_x_ulps=START_SCALE_ULPS, scale_y_ulps=None, rot_units=0, rot_units_10=None) -> np.ndarray:
    """(2, 2) float32 A: diag(1 + kx ulp, 1 + ky ulp) with off-diagonals -r and +r10 (rad = units * 2^-30)."""
    ky = scale_x_ulps if scale_y_ulps is None else scale_y_ulps
    r10 = rot_units if rot_units_10 is None else rot_units_10
    return np.array([[F32(1) + F32(scale_x_ulps) * ULP1, F32(-rot_units * ROT_UNIT)],
                     [F32(r10 * ROT_UNIT), F32(1) + F32(ky) * ULP1]], F32)


def bake_vertices(points, A) -> np.ndarray:
    """(..., 2) float32 vertices from authored points (..., 2) through A (one rounding per product and per sum)."""
    p = np.asarray(points, F32)
    A = np.asarray(A, F32)
    x = (p[..., 0] * A[0, 0]).astype(F32) + (p[..., 1] * A[0, 1]).astype(F32)
    y = (p[..., 0] * A[1, 0]).astype(F32) + (p[..., 1] * A[1, 1]).astype(F32)
    return np.stack([x.astype(F32), y.astype(F32)], -1)


def invert_bake(verts, scale_ulps=START_SCALE_ULPS) -> np.ndarray:
    """The authored points p with fl(p * (1 + k ulp)) == verts per coordinate (unique: the map is strictly increasing
    for k > 0). Raises if some coordinate has no preimage (the geometry was not baked by a pure scale)."""
    v = np.asarray(verts, F32)
    s = F32(1) + F32(scale_ulps) * ULP1
    guess = (v / s).astype(F32)
    out = np.full_like(guess, np.nan)
    for step in range(-4, 5):
        c = guess
        for _ in range(abs(step)):
            c = np.nextafter(c, F32(np.inf) if step > 0 else F32(-np.inf)).astype(F32)
        hit = (c * s).astype(F32) == v
        out = np.where(np.isnan(out) & hit, c, out)
    if np.isnan(out).any():
        raise ValueError("vertices are not a pure-scale bake of any float32 points")
    return out.astype(F32)


def polygon(verts) -> tuple[np.ndarray, np.ndarray]:
    """(verts, normals) padded to 8 the way level.py stores fixtures; normals as b2PolygonShape::Set derives them."""
    v = np.asarray(verts, F32)
    return shapes.pad(v, shapes._edge_normals(v))


# ---------------------------------------------------------------------------------------------------------------------
# Unity's Transform scale (verified on the recordings' TFM sections)

def quat_to_matrix(q) -> np.ndarray:
    """Unity's QuaternionToMatrix (float32, x2 = x + x form), (..., 4) -> (..., 3, 3)."""
    q = np.asarray(q, F32)
    x, y, z, w = (q[..., i] for i in range(4))
    two, one = F32(2), F32(1)
    x2, y2, z2 = x * two, y * two, z * two
    xx, yy, zz, xy, xz, yz = x * x2, y * y2, z * z2, x * y2, x * z2, y * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    m = np.empty(q.shape[:-1] + (3, 3), F32)
    m[..., 0, 0], m[..., 1, 0], m[..., 2, 0] = one - (yy + zz), xy + wz, xz - wy
    m[..., 0, 1], m[..., 1, 1], m[..., 2, 1] = xy - wz, one - (xx + zz), yz + wx
    m[..., 0, 2], m[..., 1, 2], m[..., 2, 2] = xz + wy, yz - wx, one - (xx + yy)
    return m


def matmul3(a, b) -> np.ndarray:
    """float32 3x3 product, each entry ((a0 b0 + a1 b1) + a2 b2)."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), F32)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = (a[..., i, 0] * b[..., 0, j] + a[..., i, 1] * b[..., 1, j]) + a[..., i, 2] * b[..., 2, j]
    return out


def global_rs(local_rotations) -> np.ndarray:
    """The chain's world rotation-scale matrix (local scales 1), accumulated leaf-up: R(q0) (R(q1) (... R(q_last)))."""
    lq = np.asarray(local_rotations, F32)
    m = quat_to_matrix(lq[..., -1, :])
    for i in range(lq.shape[-2] - 2, -1, -1):
        m = matmul3(quat_to_matrix(lq[..., i, :]), m)
    return m


def lossy_scale(local_rotations, world_rotation) -> np.ndarray:
    """Transform.lossyScale of the chain's last member: diag(R(conj(world rotation)) * global_rs). Bit-exact on every
    recorded Transform (all six chain members' TFM0..3 in dumps/parity-k4)."""
    wq = np.asarray(world_rotation, F32) * np.array([-1, -1, -1, 1], F32)
    m = matmul3(quat_to_matrix(wq), global_rs(local_rotations))
    return np.stack([m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)

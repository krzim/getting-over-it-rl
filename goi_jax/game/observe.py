"""The plugin's observation (plugin/Game.cs Observe) from a goi_jax.box2d State, as a pure float32 JAX function.

Layout (goi_env/client.py OBS_FIELDS, then AHEAD, then rays): 23 fields, 2 x 20 route-ahead offsets, n_rays pot rays,
head_rays hammer-head rays; plus the status the bridge sends after it (the pot touches the Snake).

    om    = build_obs_model(level, route_row, joint_model)       # host, once: ObsModel (pass it as a jit ARGUMENT)
    cfg   = ObsConfig(n_rays=32, ray_length=15.0, head_rays=0, head_length=4.0)   # static (hashable)
    carry = init_carry(om, model, state, progress)               # ObsCarry: the plugin's progress, lake / won flags
    obs, carry, snake = observe(om, cfg, state, mouse_input, mouse_speed, carry)
    carry = step_flags(om, model, state_before, state_after, carry)   # after EVERY physics step (sticky flags)

`state` is the world as the plugin reads it after an action (Rigidbody2D getters read the native b2Body: position =
xf.p, velocity = the centre-of-mass velocity, rotation = angle in degrees). mouse_input / mouse_speed are the
controller's mouseInput and mouseVelocityAverage after the action (goi_jax.game.player.ControllerState.mouse_input /
.mva). Field by field (Game.cs Observe, Mono float model from goi_jax.game.floatmath):

    pot x, y, vx, vy           b2Body xf.p / v of the pot (Player)
    pot_angle, pot_angvel      (angle * 57.29578f) * Deg2Rad: Unity's degree getters, then the plugin's conversion
    hammer cos, sin            (Tip.position - Hub.position).normalized (Vector2: double magnitude, float divides)
    hammer_angvel              Slider (the hub hinge's connected body) angularVelocity in rad/s, as above
    hammer_ext                 SliderJoint2D.jointTranslation (b2PrismaticJoint::GetJointTranslation)
    hammer_extvel              Vector2.Dot(Tip.velocity - Hub.velocity, d)
    head_contact, pot_contact  Rigidbody2D.IsTouching(filter Default|Terrain, no triggers): a touching contact
    fell_in_lake, won          sticky flags (carry; step_flags)
    progress                   route.guarded_progress of the Player Transform (x, y, player_z)
    cursor dx, dy (pot, head)  Cursor position minus the pot / the head
    mouse x, y, speed          controller state
    ahead_dx, dy               route.ahead(progress) minus the pot
    rays                       PhysicsScene2D.Raycast(origin, (cos a, sin a), length, solid).distance or length,
                               a = 2 pi i / n counter-clockwise from +x, from the pot (n_rays) then the head (head_rays)

Rays (raycast): the direction normalized in float, p2 = origin + length * dir, Box2D 2.3.1's b2PolygonShape /
b2CircleShape::RayCast (float32, C order, no radius skin) over every Default/Terrain non-trigger fixture of an active
body, the closest fraction f, the hit point as b2World's callback forms it ((1 - f) p1 + f p2) and the distance
|point - origin| in float: bit-exact on every recorded ray. Unity's queriesStartInColliders (true in the game) is
modelled as distance 0 for an origin inside a fixture (never seen in the recordings). Static fixtures are looked up
in a uniform grid built for the longest ray (each cell lists the static fixtures within reach of it).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import math2d as m
from goi_jax.box2d.shapes import CIRCLE
from goi_jax.box2d.world import PRISMATIC, STATIC
from goi_jax.game import floatmath as fm
from goi_jax.game import route as R

f32, i32 = jnp.float32, jnp.int32
F32 = np.float32
RAD2DEG, DEG2RAD = F32(57.29578), F32(0.0174532924)  # Mathf.Rad2Deg / Deg2Rad (Unity's C++ getters use the same)
N_FIELDS = 23
OBS_FIELDS = (
    "pot_x", "pot_y", "pot_vx", "pot_vy", "pot_angle", "pot_angvel",
    "hammer_cos", "hammer_sin", "hammer_angvel", "hammer_ext", "hammer_extvel",
    "head_contact", "pot_contact", "fell_in_lake", "won",
    "progress",
    "cursor_dx", "cursor_dy", "cursor_head_dx", "cursor_head_dy",
    "mouse_x", "mouse_y", "mouse_speed",
)  # = goi_env.client.OBS_FIELDS[:23] (that module needs no game; tests check they agree)


# observe's rays through the Pallas kernel (rays_gpu.py; same results). On with GOI_GPU_SPEED=rays (or all) on a GPU
# backend, or with this set (tests: interpret mode on CPU).
RAYS_KERNEL = False


class ObsConfig(NamedTuple):
    """Static (hashable) observation settings: GoiClient.configure's n_rays, ray_length, head_rays, head_length."""
    n_rays: int = 16
    ray_length: float = 10.0
    head_rays: int = 0
    head_length: float = 4.0

    @property
    def size(self):
        return N_FIELDS + 2 * 20 + self.n_rays + self.head_rays


class Grid(NamedTuple):
    """Static ray-table rows by cell, as a compressed list: cell c holds flat[start[c] : start[c] + count[c]]."""
    lo: jax.Array       # (2,) grid origin
    cell: jax.Array     # () cell size
    shape: jax.Array    # (2,) int32 cells in x, y
    start: jax.Array    # (Cx * Cy,) int32
    count: jax.Array    # (Cx * Cy,) int32
    flat: jax.Array     # (E,) int32 ray-table rows
    slots: jax.Array    # (K,) arange of the largest count: a ray tests K static rows (its shape is what matters)

    def rows(self, cell_index):
        k = self.slots
        i = jnp.clip(self.start[cell_index] + k, 0, self.flat.shape[0] - 1)
        return jnp.where(k < self.count[cell_index], self.flat[i], -1)


class ObsModel(NamedTuple):
    route: R.RouteModel
    pot: jax.Array       # engine body indices (int32 scalars)
    tip: jax.Array
    hub: jax.Array
    arm: jax.Array
    cursor: jax.Array
    slider_a: jax.Array  # SliderJoint2D: body A (Slider), body B (Handle), local anchors, local x axis of A
    slider_b: jax.Array
    slider_anchor_a: jax.Array
    slider_anchor_b: jax.Array
    slider_axis: jax.Array
    # ray table: every solid (Default / Terrain, non-trigger) fixture
    ray_fixture: jax.Array   # (M,) engine fixture index
    ray_body: jax.Array      # (M,)
    ray_circle: jax.Array    # (M,) bool
    ray_count: jax.Array     # (M,) vertex count
    ray_verts: jax.Array     # (M, 8, 2)
    ray_normals: jax.Array   # (M, 8, 2)
    ray_radius: jax.Array    # (M,)
    moving: jax.Array        # (Mm,) ray-table rows on non-static bodies (always tested)
    grid: Grid               # static rows by cell
    # contact flags, per engine fixture
    solid: jax.Array     # (F,) bool: passes the plugin's ContactFilter2D (layer Default|Terrain, not a trigger)
    tip_fix: jax.Array   # (F,) bool: on the hammer head's body
    pot_fix: jax.Array   # (F,) bool: on the pot's body
    snake: jax.Array     # (F,) bool: the Snake collider
    lake: jax.Array      # (F,) bool: the lake's RestartOnContact collider (Ground)
    player: jax.Array    # (F,) bool: on layer Player (what the lake reacts to)
    fixture_body: jax.Array  # (F,)
    # the space zone (GravityControl's trigger): leaving it through the top wins
    zone: jax.Array      # (4,) its fixture's AABB (min x, min y, max x, max y) incl. its skin
    zone_fix: jax.Array  # (F,) bool: fixtures whose trigger exit counts (non-static, layer collides with the zone's)
    win_hi: jax.Array    # () bounds.max.y - 5 in double, hi + lo
    win_lo: jax.Array


class ObsCarry(NamedTuple):
    progress: jax.Array  # () the plugin's `progress` (the last observation's)
    lake: jax.Array      # () bool RestartOnContact.resetting
    won: jax.Array       # () bool Patches.Won
    in_zone: jax.Array   # (F,) bool fixture overlapping the space zone's trigger


# ------------------------------------------------------------------------------------------------------------- build

def _fixture_aabb(verts, count, radius, circle):
    """Host: local AABB of each fixture (vertices + skin; circle: centre +- radius)."""
    out = np.zeros((len(count), 4), F32)
    for i in range(len(count)):
        v = verts[i, :max(1, int(count[i]))]
        r = radius[i]
        out[i] = (v[:, 0].min() - r, v[:, 1].min() - r, v[:, 0].max() + r, v[:, 1].max() + r)
    return out


def _world_aabb(local, px, py, c, s):
    """Host: AABB of a local box under a rigid transform (conservative)."""
    corners = np.array([[local[0], local[1]], [local[2], local[1]], [local[0], local[3]], [local[2], local[3]]])
    wx = c * corners[:, 0] - s * corners[:, 1] + px
    wy = s * corners[:, 0] + c * corners[:, 1] + py
    return np.array([wx.min(), wy.min(), wx.max(), wy.max()])


def build_obs_model(level, route_row: dict, joint_model, static_xf=None, cell=4.0, reach=15.0) -> ObsModel:
    """ObsModel from a goi_jax.game.level.Level, the plugin's route row (GoiClient.route()) and one step's JointModel
    (unbatched; only the slider joint's frame is read). static_xf: (B, 4) body poses (px, py, c, s) for the static
    bodies (default: identity, as the level loader puts the ground body). The grid lists, per `cell`-sized cell, every
    static solid fixture within `reach` (the longest ray) of it: rays longer than `reach` are not supported."""
    fx = level.fixture
    F = len(fx["body"])
    names = level.names["colliders"]
    bnames = level.names["body_names"]

    def body(path):
        return level.game_to_engine[bnames.index(path)]
    layer_default, layer_terrain = int(route_row["layerDefault"]), int(route_row["layerTerrain"])
    solid = np.isin(fx["layer"], [layer_default, layer_terrain]) & ~fx["sensor"].astype(bool)
    rows = np.nonzero(solid)[0]
    circle = fx["shape_type"][rows] == CIRCLE
    btype = level.body_type[fx["body"][rows]]
    moving = np.nonzero(btype != STATIC)[0].astype(np.int32)
    static_rows = np.nonzero(btype == STATIC)[0]
    # grid over the static rows (static bodies sit at the identity in the level loader)
    aabb = _fixture_aabb(fx["verts"][rows], fx["count"][rows], fx["radius"][rows], circle)
    if static_xf is not None:
        aabb = np.array([_world_aabb(aabb[k], *static_xf[fx["body"][rows[k]]]) if btype[k] == STATIC else aabb[k]
                         for k in range(len(rows))])
    sa = aabb[static_rows]
    lo = sa[:, :2].min(0) - reach - cell
    hi = sa[:, 2:].max(0) + reach + cell
    shape = np.ceil((hi - lo) / cell).astype(int)
    cells = []
    pad = reach + 1e-3 * max(1.0, reach)  # slack for the float32 cell lookup at run time (more rows never hurt)
    for iy in range(shape[1]):
        for ix in range(shape[0]):
            x0, y0 = lo[0] + ix * cell - pad, lo[1] + iy * cell - pad
            x1, y1 = x0 + cell + 2 * pad, y0 + cell + 2 * pad
            hit = (sa[:, 0] <= x1) & (sa[:, 2] >= x0) & (sa[:, 1] <= y1) & (sa[:, 3] >= y0)
            cells.append(static_rows[hit])
    count = np.array([len(c) for c in cells], np.int32)
    start = np.concatenate([[0], np.cumsum(count)[:-1]]).astype(np.int32)
    flat = np.concatenate([np.asarray(c, np.int32) for c in cells] + [np.zeros(1, np.int32)])
    grid = Grid(lo=jnp.asarray(lo.astype(F32)), cell=jnp.asarray(F32(cell)), shape=jnp.asarray(shape.astype(np.int32)),
                start=jnp.asarray(start), count=jnp.asarray(count), flat=jnp.asarray(flat),
                slots=jnp.arange(max(1, int(count.max())), dtype=jnp.int32))

    def mask(names_wanted):
        cols = [i for i, n in enumerate(names) if n in set(names_wanted)]
        return np.isin(fx["collider"], cols)
    tip, pot = body("Player/Hub/Slider/Handle/PoleMiddle/Tip"), body("Player")
    zone_m = mask([route_row["zeroGravCollider"]])
    zf = np.nonzero(zone_m)[0]
    if len(zf):
        za = _fixture_aabb(fx["verts"][zf], fx["count"][zf], fx["radius"][zf], fx["shape_type"][zf] == CIRCLE)
        zone = np.array([za[:, 0].min(), za[:, 1].min(), za[:, 2].max(), za[:, 3].max()], F32)
        zlayer = int(fx["layer"][zf[0]])
    else:
        zone, zlayer = np.array([np.inf, np.inf, -np.inf, -np.inf], F32), 0
    masks = level.layer_masks
    collides = ((masks[zlayer] >> fx["layer"]) & 1) != 0 if masks is not None else np.ones(F, bool)
    zone_fix = (level.body_type[fx["body"]] != STATIC) & ~fx["sensor"].astype(bool) & collides
    win = np.float64(F32(route_row["zeroGravMax"][1])) - 5.0
    win_hi, win_lo = R._split(win)
    # slider joint frame
    jt = np.asarray(joint_model.joint_type)
    sj = [j for j in range(len(jt)) if jt[j] == PRISMATIC]
    if len(sj) != 1:
        raise ValueError(f"expected one prismatic joint (the slider), got {len(sj)}")
    j = sj[0]
    a = lambda x: jnp.asarray(np.asarray(x))
    return ObsModel(
        route=R.route_model(route_row),
        pot=a(np.int32(pot)), tip=a(np.int32(tip)), hub=a(np.int32(body("Player/Hub"))),
        arm=a(np.int32(body("Player/Hub/Slider"))), cursor=a(np.int32(body("Cursor"))),
        slider_a=a(np.int32(joint_model.joint_a[j])), slider_b=a(np.int32(joint_model.joint_b[j])),
        slider_anchor_a=a(np.asarray(joint_model.local_anchor_a[j], F32)),
        slider_anchor_b=a(np.asarray(joint_model.local_anchor_b[j], F32)),
        slider_axis=a(np.asarray(joint_model.local_axis_a[j], F32)),
        ray_fixture=a(rows.astype(np.int32)), ray_body=a(fx["body"][rows].astype(np.int32)), ray_circle=a(circle),
        ray_count=a(fx["count"][rows].astype(np.int32)), ray_verts=a(fx["verts"][rows].astype(F32)),
        ray_normals=a(fx["normals"][rows].astype(F32)), ray_radius=a(fx["radius"][rows].astype(F32)),
        moving=a(moving), grid=grid,
        solid=a(solid), tip_fix=a(fx["body"] == tip), pot_fix=a(fx["body"] == pot),
        snake=a(mask([route_row["snakeCollider"]])), lake=a(mask(route_row["lakeColliders"])),
        player=a(fx["layer"] == int(route_row["layerPlayer"])), fixture_body=a(fx["body"].astype(np.int32)),
        zone=a(zone), zone_fix=a(zone_fix), win_hi=a(win_hi), win_lo=a(win_lo))


# ------------------------------------------------------------------------------------------------------------- rays

def ray_directions(n: int) -> np.ndarray:
    """(n, 2) float32: a = 2 * Mathf.PI * i / n (double, stored as float), then ((float)Math.Cos(a), (float)Math.Sin(a))."""
    pi = np.float64(F32(np.pi))
    a = (2.0 * pi * np.arange(n, dtype=np.float64) / n).astype(F32).astype(np.float64)
    return np.stack([np.cos(a), np.sin(a)], -1).astype(F32)


def _cast(om: ObsModel, st, rows, ox, oy, ex, ey, max_fraction=F32(1)):
    """Fractions of the ray (ox, oy) -> (ex, ey) against ray-table rows (any shape): (fraction, hit, inside).
    b2PolygonShape::RayCast / b2CircleShape::RayCast with maxFraction 1, and b2Shape::TestPoint of the origin."""
    valid = rows >= 0
    r = jnp.where(valid, rows, 0)
    b = om.ray_body[r]
    bd = st.bodies
    px, py = bd.xf_p[b, 0], bd.xf_p[b, 1]
    qc, qs = bd.xf_q[b, 0], bd.xf_q[b, 1]
    valid = valid & bd.active[b]
    verts, normals, count = om.ray_verts[r], om.ray_normals[r], om.ray_count[r]
    return _cast_core(lambda i: (verts[..., i, 0], verts[..., i, 1]), lambda i: (normals[..., i, 0], normals[..., i, 1]),
                      count, om.ray_radius[r], om.ray_circle[r], valid, px, py, qc, qs, ox, oy, ex, ey, max_fraction)


def _cast_core(vert, normal, count, rad, circle, valid, px, py, qc, qs, ox, oy, ex, ey, max_fraction=F32(1)):
    """_cast's arithmetic for gathered rows: vert(i) / normal(i) give vertex / normal i of each row, count, rad,
    circle, valid per row, (px, py, qc, qs) its body's transform (also used per lane by the ray kernel, rays_gpu.py,
    so both evaluate the same float32 operations)."""
    # polygon: the ray in the body frame
    p1x, p1y = m.xf_mul_t(px, py, qc, qs, ox, oy)
    p2x, p2y = m.xf_mul_t(px, py, qc, qs, ex, ey)
    dx, dy = p2x - p1x, p2y - p1y
    lower = jnp.zeros_like(p1x)
    upper = jnp.full_like(p1x, max_fraction)
    hit_edge = jnp.zeros(p1x.shape, bool)
    miss = jnp.zeros(p1x.shape, bool)
    inside = jnp.ones(p1x.shape, bool)
    for i in range(8):
        live = (i < count) & ~miss
        vx, vy = vert(i)
        nx, ny = normal(i)
        num = m.dot(nx, ny, vx - p1x, vy - p1y)
        den = m.dot(nx, ny, dx, dy)
        zero = den == 0
        miss = miss | (live & zero & (num < 0))
        safe = jnp.where(zero, F32(1), den)
        q = num / safe
        go_low = live & ~zero & (den < 0) & (num < m.mul(lower, den))
        go_up = live & ~zero & ~go_low & (den > 0) & (num < m.mul(upper, den))
        lower = jnp.where(go_low, q, lower)
        hit_edge = hit_edge | go_low
        upper = jnp.where(go_up, q, upper)
        miss = miss | (live & (upper < lower))
        # TestPoint: dot(normal, pLocal - v) > 0 -> outside
        inside = inside & ~((i < count) & (m.dot(nx, ny, p1x - vx, p1y - vy) > 0))
    poly_hit = hit_edge & ~miss
    poly_frac = lower
    # circle
    v0x, v0y = vert(0)
    cx, cy = m.xf_mul(px, py, qc, qs, v0x, v0y)
    sx, sy = ox - cx, oy - cy
    bb = m.dot(sx, sy, sx, sy) - m.mul(rad, rad)
    rx, ry = ex - ox, ey - oy
    cc = m.dot(sx, sy, rx, ry)
    rr = m.dot(rx, ry, rx, ry)
    sigma = m.mul(cc, cc) - m.mul(rr, bb)
    aa = -(cc + jnp.sqrt(jnp.maximum(sigma, F32(0))))
    circ_hit = ~((sigma < 0) | (rr < m.EPS)) & (aa >= 0) & (aa <= m.mul(max_fraction, rr))
    circ_frac = R.div(aa, jnp.where(rr > 0, rr, F32(1)))  # rr is the same for every world: see route.div
    circ_inside = bb <= 0  # TestPoint: dot(d, d) <= r * r
    frac = jnp.where(circle, circ_frac, poly_frac)
    hit = valid & jnp.where(circle, circ_hit, poly_hit)
    inside = valid & jnp.where(circle, circ_inside, inside)
    return frac, hit, inside


def unit_directions(dirs) -> np.ndarray:
    """Host: b2Vec2::Normalize of each direction in float32 (Unity normalizes the query direction): len = sqrtf(x x +
    y y), x * (1 / len)."""
    d = np.asarray(dirs, F32)
    n = np.sqrt((d[:, 0] * d[:, 0]).astype(F32) + (d[:, 1] * d[:, 1]).astype(F32)).astype(F32)
    inv = (F32(1) / n).astype(F32)
    return np.stack([(d[:, 0] * inv).astype(F32), (d[:, 1] * inv).astype(F32)], -1)


def raycast(om: ObsModel, st, origin, dirs, length):
    """Distances along (R, 2) directions from origin (2,), up to `length` (returned when nothing is hit): Unity's
    PhysicsScene2D.Raycast(origin, dir, length, Default|Terrain non-triggers).distance. What reproduces the game bit
    for bit (dumps/obs, 100% of hits): the direction normalized (float), p2 = p1 + length * dir, the closest
    fixture's fraction f, the hit point as b2World's RayCast callback forms it, (1 - f) * p1 + f * p2, and the distance
    |point - p1| in float32 (sqrtf of a float sum)."""
    ox, oy = origin[0], origin[1]
    L = jnp.asarray(length, f32)
    u = unit_directions(dirs)
    dx, dy = jnp.asarray(u[:, 0]), jnp.asarray(u[:, 1])
    ex, ey = ox + m.mul(L, dx), oy + m.mul(L, dy)  # input.p2 = p1 + distance * direction
    g = om.grid
    cell = jnp.floor((jnp.stack([ox, oy]) - g.lo) / g.cell).astype(i32)
    cell = jnp.clip(cell, 0, g.shape - 1)
    ci = cell[1] * g.shape[0] + cell[0]
    rows = jnp.concatenate([g.rows(ci), om.moving])  # (K + moving,)
    from goi_jax.box2d.gpu import speed_option
    if speed_option("rays") and jax.default_backend() != "cpu" or RAYS_KERNEL:
        from goi_jax.game import rays_gpu
        best, inside_any = rays_gpu.cast(om, st, rows, g.count[ci], ox, oy, ex, ey)
    else:
        frac, hit, inside = _cast(om, st, rows[None, :], ox, oy, ex[:, None], ey[:, None])
        best = jnp.min(jnp.where(hit, frac, jnp.inf), axis=1)
        inside_any = inside.any(axis=1)
    found = jnp.isfinite(best)
    f = jnp.where(found, best, F32(0))
    g1 = F32(1) - f
    hx, hy = m.mul(g1, ox) + m.mul(f, ex), m.mul(g1, oy) + m.mul(f, ey)
    dist = jnp.where(found, m.length(hx - ox, hy - oy), L)
    return jnp.where(inside_any, F32(0), dist)


# ------------------------------------------------------------------------------------------------------------- flags

def _touching(om: ObsModel, st, mine, other):
    """Any touching contact between a fixture in `mine` and one in `other` (bool[F] masks)."""
    ct = st.contacts
    live = (ct.pair >= 0) & ct.touching
    fa, fb = jnp.maximum(ct.fa, 0), jnp.maximum(ct.fb, 0)
    hit = (mine[fa] & other[fb]) | (mine[fb] & other[fa])
    return (live & hit).any()


def _zone_rows(model):
    """The fixtures zone_fix can select: a compact Model's non-static fixtures (zone_fix is False on static bodies by
    construction, build_obs_model), else None (every fixture)."""
    return getattr(model, "slot_fixture", None)


def zone_overlap(om: ObsModel, model, st):
    """(F,) bool: fixture overlaps the space-zone trigger (its world AABB, skin included, against the zone's box;
    exact for leaving through the top, which is all that wins). With a compact Model only the non-static fixtures
    are tested (the level's ~1500 static ones cannot be in zone_fix); same result."""
    from goi_jax.box2d import shapes
    bd = st.bodies
    rows = _zone_rows(model)
    if rows is None:
        sub, fix = model, jnp.arange(model.fixture_body.shape[-1])
    else:
        fix = rows
        sub = _Shapes(model.verts[rows], model.count[rows], model.radius[rows], model.shape_type[rows])
    b = model.fixture_body[fix]
    lx, ly, ux, uy = shapes.fixture_aabbs(sub, bd.xf_p[b, 0], bd.xf_p[b, 1], bd.xf_q[b, 0], bd.xf_q[b, 1])
    z = om.zone
    over = (lx <= z[2]) & (ux >= z[0]) & (ly <= z[3]) & (uy >= z[1])
    hit = om.zone_fix[fix] & over & bd.active[b]
    if rows is None:
        return hit
    return jnp.zeros(om.zone_fix.shape, bool).at[rows].set(hit)


class _Shapes(NamedTuple):
    """The Model fields shapes.fixture_aabbs reads, for a subset of fixtures."""
    verts: jax.Array
    count: jax.Array
    radius: jax.Array
    shape_type: jax.Array


def init_carry(om: ObsModel, model, st, progress) -> ObsCarry:
    """The carry for a state just reset / loaded / placed: the plugin's progress at that state (Snapshot.progress),
    flags clear."""
    return ObsCarry(progress=jnp.asarray(progress, f32), lake=jnp.asarray(False), won=jnp.asarray(False),
                    in_zone=zone_overlap(om, model, st))


def step_flags(om: ObsModel, model, st_before, st_after, carry: ObsCarry) -> ObsCarry:
    """After one physics step. Lake (RestartOnContact.OnCollisionEnter2D on Ground with a Player-layer collider):
    sticky once a Player-layer fixture touches it. Won (the plugin's GravityControl.OnTriggerExit2D patch): a fixture
    that overlapped the zone at the previous step's start no longer does at this step's start (Box2D updates trigger
    contacts in Collide, from the poses before the solve) with its body's y above bounds.max.y - 5. Unity dispatches
    the callback after Simulate returns, so the patch's rb.position is the post-step pose (st_after)."""
    lake = carry.lake | _touching(om, st_after, om.player, om.lake)
    now = zone_overlap(om, model, st_before)
    rows = _zone_rows(model)  # in_zone / now are False outside these (zone_overlap)
    fix = om.fixture_body if rows is None else om.fixture_body[rows]
    y = st_after.bodies.xf_p[fix, 1]
    above = (y - om.win_hi) > om.win_lo  # y > max.y - 5 (double)
    sel = (lambda a: a) if rows is None else (lambda a: a[rows])  # noqa: E731
    won = carry.won | (sel(carry.in_zone) & ~sel(now) & above).any()
    return carry._replace(lake=lake, won=won, in_zone=now)


# ------------------------------------------------------------------------------------------------------------- observe

def _sub(a, b):
    return fm.rounded(jnp.asarray(a, f32) - jnp.asarray(b, f32))


def slider_translation(om: ObsModel, st):
    """b2PrismaticJoint::GetJointTranslation: dot(pB - pA, bodyA world axis)."""
    bd = st.bodies
    a, b = om.slider_a, om.slider_b
    pax, pay = m.xf_mul(bd.xf_p[a, 0], bd.xf_p[a, 1], bd.xf_q[a, 0], bd.xf_q[a, 1], om.slider_anchor_a[0], om.slider_anchor_a[1])
    pbx, pby = m.xf_mul(bd.xf_p[b, 0], bd.xf_p[b, 1], bd.xf_q[b, 0], bd.xf_q[b, 1], om.slider_anchor_b[0], om.slider_anchor_b[1])
    ax, ay = m.rot(bd.xf_q[a, 0], bd.xf_q[a, 1], om.slider_axis[0], om.slider_axis[1])
    return m.dot(pbx - pax, pby - pay, ax, ay)


def observe(om: ObsModel, cfg: ObsConfig, st, mouse_input, mouse_speed, carry: ObsCarry):
    """(obs f32[cfg.size], carry with the new progress, snake bool): the bridge's observation of `st`."""
    bd = st.bodies
    pot, tip, hub, arm, cur = om.pot, om.tip, om.hub, om.arm, om.cursor
    px, py = bd.xf_p[pot, 0], bd.xf_p[pot, 1]
    vx, vy = bd.v[pot, 0], bd.v[pot, 1]
    angle = m.mul(m.mul(bd.a[pot], RAD2DEG), DEG2RAD)
    angvel = m.mul(m.mul(bd.w[pot], RAD2DEG), DEG2RAD)
    tx, ty = bd.xf_p[tip, 0], bd.xf_p[tip, 1]
    hx, hy = bd.xf_p[hub, 0], bd.xf_p[hub, 1]
    dx, dy = fm.normalize(_sub(tx, hx), _sub(ty, hy))
    arm_w = m.mul(m.mul(bd.w[arm], RAD2DEG), DEG2RAD)
    ext = slider_translation(om, st)
    rvx, rvy = _sub(bd.v[tip, 0], bd.v[hub, 0]), _sub(bd.v[tip, 1], bd.v[hub, 1])
    extvel = fm.dot2(rvx, rvy, dx, dy)
    head = _touching(om, st, om.tip_fix, om.solid)
    potc = _touching(om, st, om.pot_fix, om.solid)
    snake = _touching(om, st, om.pot_fix, om.snake)
    prog = R.guarded_progress(om.route, carry.progress, (px, py, om.route.player_z))
    cx, cy = bd.xf_p[cur, 0], bd.xf_p[cur, 1]
    mouse_input = jnp.asarray(mouse_input, f32)
    one = lambda b: jnp.where(b, F32(1), F32(0))
    fields = [px, py, vx, vy, angle, angvel, dx, dy, arm_w, ext, extvel, one(head), one(potc), one(carry.lake),
              one(carry.won), prog, _sub(cx, px), _sub(cy, py), _sub(cx, tx), _sub(cy, ty), mouse_input[0],
              mouse_input[1], jnp.asarray(mouse_speed, f32)]
    q = R.ahead(om.route, prog)
    ahead = jnp.stack([_sub(q[:, 0], px), _sub(q[:, 1], py)], -1).reshape(-1)
    parts = [jnp.stack([jnp.asarray(f, f32) for f in fields]), ahead]
    if cfg.n_rays:
        parts.append(raycast(om, st, (px, py), ray_directions(cfg.n_rays), F32(cfg.ray_length)))
    if cfg.head_rays:
        parts.append(raycast(om, st, (tx, ty), ray_directions(cfg.head_rays), F32(cfg.head_length)))
    return jnp.concatenate(parts), carry._replace(progress=prog), snake

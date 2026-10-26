"""Decoder and loader for the plugin's per-step recordings (bridge command 'w', plugin/GameRecord.cs).

One record = one physics step at K = 1, run exactly as the lockstep game runs it:

    SCR0  script state before HammerCollisions/PotSounds/PlayerControl.FixedUpdate
    SCR1  script state right after PlayerControl.FixedUpdate (what the controller computed and wrote)
    NAT0  native Box2D world right before PhysicsScene2D.Simulate (after the crane's MovePosition)
    NAT1  native Box2D world right after Simulate (before the once-per-action pose update re-creates the climber's joints)
    SCR2  script state after the step (callbacks and teleports done, pose update not yet)
    EVTS  events in between, in the order they happened (MovePosition calls, collision callbacks, HammerReturn)
    CPTS  the ContactPoint2Ds each collision callback saw (version >= 2): Unity's reported impulses, which differ from
          the stored manifold impulses after a TOI sub-step
    TIME  ms for the whole recorded step, ms of it reading native memory, world generation, the action the controller read

A chain (record.py) stores the raw records back to back; `load_chain` decodes a chain file into padded arrays:

    c = load_chain("dumps/parity/coffee/c03-smooth.npz")
    c.scr[0]["mouseInput.x"]          # (N,) script field before the step; c.scr[1] after the controller, c.scr[2] after
    c.action                          # (N, 2) the mouse input the controller read
    c.bodies[1]["v"]                  # (N, NB, 2) native linear velocity after Simulate (bodies in native list order)
    c.body_index                      # (N, NB) Game body index of each native body (names: c.body_names)
    c.contacts[0]["normalImpulse"]    # (N, NC, 2) per manifold point, before Simulate; c.n_contacts[0] (N,) valid rows
    c.events                          # structured (E,) with a `step` column
    c.cursor_target                   # (N, 2) the controller's MovePosition target for the cursor (NaN if none)

Native layouts: docs/understanding/native-box2d.md §3 (Unity 2020.3.25f1 x64).
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC_RECORD, MAGIC_GEOMETRY = 0x31524F47, 0x31474F47
TAG_SCRIPT, TAG_NATIVE, TAG_EVENTS, TAG_TIME, TAG_POINTS = 0x30524353, 0x3054414E, 0x53545645, 0x454D4954, 0x53545043

# Game.ScriptFields (plugin/GameRecord.cs); checked against the controller row's "scriptFields" when present.
SCRIPT_FIELDS = (
    "mouseInput.x", "mouseInput.y", "oldMouse.x", "oldMouse.y", "mouseVelocityAverage", "mw.x", "mw.y", "mouseSnap",
    "pauseInputTimer", "inputsToSkip", "oldAngle",
    "hj.motorSpeed", "hj.maxMotorTorque", "sj.motorSpeed", "sj.maxMotorForce",
    "pc.motor.motorSpeed", "pc.motor.maxMotorTorque", "pc.slider.motorSpeed", "pc.slider.maxMotorTorque",
    "hj.jointAngle", "hj.referenceAngle", "sj.reactionTorque", "sj.jointTranslation",
    "P.x", "P.y", "P.z", "T.x", "T.y", "T.z", "H.x", "H.y", "H.z",
    "cursor.x", "cursor.y", "cursor.vx", "cursor.vy", "cursorTf.x", "cursorTf.y", "cursorTf.z",
    "tip.vx", "tip.vy", "pot.vx", "pot.vy",
    "hammerSlide", "potSlide", "hammerFriction", "potBottomFriction",
    "simTime", "gravity.x", "gravity.y", "sensitivity", "trackpad", "loadFinished",
    "pot.x", "pot.y", "tip.x", "tip.y", "hub.x", "hub.y", "pot.rotation", "tip.rotation", "hub.rotation", "arm.rotation",
    "sj.reactionForce.x", "sj.reactionForce.y", "hj.reactionForce.x", "hj.reactionForce.y",
)
EV_ENTER, EV_STAY, EV_EXIT, EV_HAMMER_RETURN, EV_MOVE_POSITION = 1, 2, 3, 4, 5
# Event phases: 0 scripts, 1 crane (before Simulate), 2 inside Simulate (callbacks), 3 after Simulate.
EVENT_DTYPE = np.dtype([("phase", "<i4"), ("kind", "<i4"), ("script", "<i4"), ("collider", "<i4"), ("other", "<i4"),
                        ("count", "<i4"), ("a", "<f4"), ("b", "<f4"), ("c", "<f4"), ("d", "<f4"), ("flags", "<i4")])
# Collision events: a, b = relativeVelocity, c = the other object's layer (10 = Terrain), flags = slide after the call
# (bit 0) | tracked-collider count after it << 8. HammerReturn: script 0 before / 1 after, a = tip y, b = pot y, flags =
# mouseSnap. MovePosition: script = body index, a, b = target, c, d = the body's position at the call.
TERRAIN_LAYER = 10
# CPTS (record version >= 2): the ContactPoint2Ds a collision callback saw, in GetContacts order.
POINT_DTYPE = np.dtype([("event", "<i4"), ("px", "<f4"), ("py", "<f4"), ("nx", "<f4"), ("ny", "<f4"),
                        ("separation", "<f4"), ("normalImpulse", "<f4"), ("tangentImpulse", "<f4"),
                        ("collider", "<i4"), ("other", "<i4")])

BODY_BYTES, JOINT_BYTES, CONTACT_BYTES, WORLD_BYTES = 0xC8, 0x150, 4 + 0x58 + 0x14, 0x88
FIXTURE_BYTES, SHAPE_BYTES = 0x58, 0xA0

# b2Body fields (offset, dtype, count).
BODY_FIELDS = {
    "type": (0x00, "<i4", 1), "flags": (0x04, "<u2", 1), "islandIndex": (0x08, "<i4", 1),
    "xf": (0x0C, "<f4", 4),  # p.x, p.y, q.s, q.c
    "localCenter": (0x1C, "<f4", 2), "c0": (0x24, "<f4", 2), "c": (0x2C, "<f4", 2), "a0": (0x34, "<f4", 1),
    "a": (0x38, "<f4", 1), "alpha0": (0x3C, "<f4", 1),
    "v": (0x40, "<f4", 2), "w": (0x48, "<f4", 1), "force": (0x4C, "<f4", 2), "torque": (0x54, "<f4", 1),
    "mass": (0x90, "<f4", 1), "invMass": (0x94, "<f4", 1), "unity98": (0x98, "<f4", 2), "I": (0xA0, "<f4", 1),
    "invI": (0xA4, "<f4", 1), "linearDamping": (0xA8, "<f4", 1), "angularDamping": (0xAC, "<f4", 1),
    "gravityScale": (0xB0, "<f4", 1), "sleepTime": (0xB4, "<f4", 1), "unityB8": (0xB8, "<i4", 1),
}
# Contact record: b2Contact flags, then bytes 0x70..0xc8 (child indices, Unity ints, manifold, radii), then
# 0x110..0x124 (toiCount, toi, friction, restitution, tangentSpeed). Offsets below are within that record.
_M = 4 + 0x10  # manifold start
CONTACT_FIELDS = {
    "flags": (0x00, "<u4", 1), "childA": (0x04, "<i4", 1), "childB": (0x08, "<i4", 1), "unity78": (0x0C, "<i4", 2),
    "localPointP": (_M + 0x00, "<f4", 2), "normalImpulse0": (_M + 0x08, "<f4", 1), "tangentImpulse0": (_M + 0x0C, "<f4", 1),
    "key0": (_M + 0x10, "<u4", 1),
    "localPointP1": (_M + 0x14, "<f4", 2), "normalImpulse1": (_M + 0x1C, "<f4", 1), "tangentImpulse1": (_M + 0x20, "<f4", 1),
    "key1": (_M + 0x24, "<u4", 1),
    "localNormal": (_M + 0x28, "<f4", 2), "localPoint": (_M + 0x30, "<f4", 2), "manifoldType": (_M + 0x38, "<i4", 1),
    "pointCount": (_M + 0x3C, "<i4", 1), "radii": (4 + 0x50, "<f4", 2),
    "toiCount": (4 + 0x58, "<i4", 1), "toi": (4 + 0x5C, "<f4", 1), "friction": (4 + 0x60, "<f4", 1),
    "restitution": (4 + 0x64, "<f4", 1), "tangentSpeed": (4 + 0x68, "<f4", 1),
}
CONTACT_TOUCHING, CONTACT_ENABLED = 0x2, 0x4
J = 0x98  # derived joint fields start
JOINT_COMMON = {"type": (0x08, "<i4", 1), "indexA": (0x70, "<i4", 1), "indexB": (0x74, "<i4", 1)}
JOINT_FIELDS = {
    1: {  # revolute (v2.4 solver state)
        "localAnchorA": (J, "<f4", 2), "localAnchorB": (J + 0x08, "<f4", 2), "impulse": (J + 0x10, "<f4", 3),
        "motorImpulse": (J + 0x1C, "<f4", 1), "lowerImpulse": (J + 0x20, "<f4", 1), "upperImpulse": (J + 0x24, "<f4", 1),
        "enableMotor": (J + 0x28, "u1", 1), "maxMotorTorque": (J + 0x2C, "<f4", 1), "motorSpeed": (J + 0x30, "<f4", 1),
        "enableLimit": (J + 0x34, "u1", 1), "referenceAngle": (J + 0x38, "<f4", 1), "lowerAngle": (J + 0x3C, "<f4", 1),
        "upperAngle": (J + 0x40, "<f4", 1), "angle": (J + 0x84, "<f4", 1), "axialMass": (J + 0x88, "<f4", 1)},
    2: {  # prismatic (v2.3.1)
        "localAnchorA": (J, "<f4", 2), "localAnchorB": (J + 0x08, "<f4", 2), "localXAxisA": (J + 0x10, "<f4", 2),
        "localYAxisA": (J + 0x18, "<f4", 2), "referenceAngle": (J + 0x20, "<f4", 1), "impulse": (J + 0x24, "<f4", 3),
        "motorImpulse": (J + 0x30, "<f4", 1), "lowerTranslation": (J + 0x34, "<f4", 1),
        "upperTranslation": (J + 0x38, "<f4", 1), "maxMotorForce": (J + 0x3C, "<f4", 1), "motorSpeed": (J + 0x40, "<f4", 1),
        "enableLimit": (J + 0x44, "u1", 1), "enableMotor": (J + 0x45, "u1", 1), "limitState": (J + 0x48, "<i4", 1)},
    8: {  # weld (v2.3.1)
        "frequencyHz": (J, "<f4", 1), "dampingRatio": (J + 0x04, "<f4", 1), "bias": (J + 0x08, "<f4", 1),
        "localAnchorA": (J + 0x0C, "<f4", 2), "localAnchorB": (J + 0x14, "<f4", 2), "referenceAngle": (J + 0x1C, "<f4", 1),
        "gamma": (J + 0x20, "<f4", 1), "impulse": (J + 0x24, "<f4", 3)},
}
WORLD_FIELDS = {"bodyCount": (0x10, "<i4", 1), "jointCount": (0x14, "<i4", 1), "gravity": (0x58, "<f4", 2),
                "allowSleep": (0x60, "u1", 1), "inv_dt0": (0x78, "<f4", 1), "warmStarting": (0x7C, "u1", 1),
                "continuousPhysics": (0x7D, "u1", 1), "subStepping": (0x7E, "u1", 1), "stepComplete": (0x7F, "u1", 1),
                "flags": (0x80, "<i4", 1)}


def decode(raw: np.ndarray, fields: dict) -> dict:
    """Fields of a (..., nbytes) uint8 array of structs, as arrays (..., count) or (...) for count 1."""
    raw = np.ascontiguousarray(raw)
    out = {}
    for name, (off, dt, n) in fields.items():
        size = np.dtype(dt).itemsize
        a = raw[..., off:off + size * n].copy().view(dt)
        out[name] = a[..., 0] if n == 1 else a
    return out


def _sections(data: bytes, offset: int):
    while offset < len(data):
        tag, n = struct.unpack_from("<ii", data, offset)
        yield tag & 0xFFFFFFFF, data[offset + 8:offset + 8 + n]
        offset += 8 + n


def _keys(buf, o):
    (n,) = struct.unpack_from("<i", buf, o)
    return np.frombuffer(buf, "<i4", n, o + 4).copy(), o + 4 + 4 * n


def _script(buf):
    (n,) = struct.unpack_from("<i", buf, 0)
    f = np.frombuffer(buf, "<f4", n, 4).copy()
    hk, o = _keys(buf, 4 + 4 * n)
    pk, o = _keys(buf, o)
    return f, hk, pk


def _native(buf):
    (nw,) = struct.unpack_from("<i", buf, 0)
    if nw < 0:
        return None
    o = 4
    world = np.frombuffer(buf, np.uint8, nw, o).copy()
    o += nw
    out = {"world": world}
    for name, size, ids in (("bodies", BODY_BYTES, 1), ("joints", JOINT_BYTES, 1), ("contacts", CONTACT_BYTES, 4)):
        (n,) = struct.unpack_from("<i", buf, o)
        o += 4
        rec = np.frombuffer(buf, np.uint8, n * (4 * ids + size), o).reshape(n, 4 * ids + size)
        o += n * (4 * ids + size)
        out[name + "_id"] = rec[:, :4 * ids].copy().view("<i4")
        out[name] = rec[:, 4 * ids:].copy()
    return out


def parse_record(data: bytes) -> dict:
    """One 'w' record as {"scr": [3 x (F,)], "hammer_keys": [3], "pot_keys": [3], "nat": [2 x dict], "events": (E,),
    "time_ms", "native_ms", "generation", "action"}."""
    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != MAGIC_RECORD:
        raise ValueError(f"not a step record (magic {magic:#x})")
    r = {"version": version, "scr": [None] * 3, "hammer_keys": [None] * 3, "pot_keys": [None] * 3, "nat": [None] * 2}
    for tag, buf in _sections(data, 8):
        base, which = tag & 0x00FFFFFF, (tag >> 24) - 0x30
        if base == TAG_SCRIPT & 0x00FFFFFF:
            r["scr"][which], r["hammer_keys"][which], r["pot_keys"][which] = _script(buf)
        elif base == TAG_NATIVE & 0x00FFFFFF:
            r["nat"][which] = _native(buf)
        elif tag == TAG_EVENTS:
            (n,) = struct.unpack_from("<i", buf, 0)
            r["events"] = np.frombuffer(buf, EVENT_DTYPE, n, 4).copy()
        elif tag == TAG_POINTS:
            (n,) = struct.unpack_from("<i", buf, 0)
            r["points"] = np.frombuffer(buf, POINT_DTYPE, n, 4).copy()
        elif tag == TAG_TIME:
            ms, nms, gen, ax, ay = struct.unpack_from("<ffiff", buf, 0)
            r.update(time_ms=ms, native_ms=nms, generation=gen, action=np.array([ax, ay], np.float32))
    return r


def parse_geometry(data: bytes) -> dict:
    """The 'g' reply: per body in native list order its Game body index (-1 the ground), b2BodyType and fixtures
    (collider index, piece, raw b2Fixture, raw shape), then the index tables (collider "path#component", body paths,
    joint "path#type")."""
    magic, version, nb = struct.unpack_from("<iii", data, 0)
    if magic != MAGIC_GEOMETRY:
        raise ValueError(f"not a geometry reply (magic {magic:#x})")
    o, bodies = 12, []
    for _ in range(max(nb, 0)):
        index, btype, nf = struct.unpack_from("<iii", data, o)
        o += 12
        width = 8 + FIXTURE_BYTES + SHAPE_BYTES
        rec = np.frombuffer(data, np.uint8, nf * width, o).reshape(nf, width)
        o += rec.size
        ids = rec[:, :8].copy().view("<i4")
        bodies.append({"index": index, "type": btype, "collider": ids[:, 0], "piece": ids[:, 1],
                       "fixture": rec[:, 8:8 + FIXTURE_BYTES].copy(), "shape": rec[:, 8 + FIXTURE_BYTES:].copy()})
    (n,) = struct.unpack_from("<i", data, o)
    colliders, body_names, joints = data[o + 4:o + 4 + n].decode("utf-8").split("\n\n")[:3]
    return {"version": version, "bodies": bodies, "colliders": colliders.split("\n"),
            "body_names": body_names.split("\n"), "joints": joints.rstrip("\n").split("\n")}


def shape_of(shape_raw: np.ndarray) -> dict:
    """A b2Shape's raw bytes: type (0 circle, 2 polygon), radius, and the circle centre or polygon vertices/normals."""
    s = decode(shape_raw[None], {"type": (0x08, "<i4", 1), "radius": (0x0C, "<f4", 1), "p": (0x10, "<f4", 2),
                                 "centroid": (0x10, "<f4", 2), "vertices": (0x18, "<f4", 16),
                                 "normals": (0x58, "<f4", 16), "count": (0x98, "<i4", 1)})
    s = {k: v[0] for k, v in s.items()}
    n = int(np.clip(s["count"], 0, 8)) if s["type"] == 2 else 0
    s["vertices"], s["normals"] = s["vertices"].reshape(8, 2)[:n], s["normals"].reshape(8, 2)[:n]
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Chains

def pack_records(records: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Raw records back to back (uint8) and their start offsets (int64, len n + 1)."""
    offsets = np.zeros(len(records) + 1, np.int64)
    offsets[1:] = np.cumsum([len(r) for r in records])
    return np.frombuffer(b"".join(records), np.uint8).copy(), offsets


def unpack_records(blob: np.ndarray, offsets: np.ndarray) -> list[bytes]:
    b = blob.tobytes()
    return [b[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _pad(rows: list[np.ndarray], width: int, fill=0) -> np.ndarray:
    shape = (len(rows), width) + (rows[0].shape[1:] if rows else ())
    out = np.full(shape, fill, rows[0].dtype if rows else np.uint8)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


@dataclass
class Chain:
    """A decoded recording chain; see the module docstring."""
    meta: dict
    action: np.ndarray
    scr: list                   # 3 x structured view: dict name -> (N,)
    scr_raw: np.ndarray          # (3, N, F)
    hammer_keys: list            # 3 x list of arrays
    pot_keys: list
    events: np.ndarray           # structured with "step"
    points: np.ndarray           # structured (POINT_DTYPE + "step"; "event" indexes `events`), empty before version 2
    body_index: list             # 2 x (N, NB) Game body index per native body slot
    bodies: list                 # 2 x dict of (N, NB, ...)
    joint_index: list
    joints_raw: list             # 2 x (N, NJ, JOINT_BYTES)
    joints: list                 # 2 x dict type -> dict of (N, NJ, ...) (rows of other types are garbage)
    contact_ids: list            # 2 x (N, NC, 4): collider A, piece A, collider B, piece B
    contacts: list               # 2 x dict of (N, NC, ...)
    n_contacts: list             # 2 x (N,)
    world: list                  # 2 x dict of (N,)
    time_ms: np.ndarray
    native_ms: np.ndarray
    record_bytes: np.ndarray
    cursor_target: np.ndarray    # (N, 2) the controller's (last) cursor MovePosition target in phase 0, NaN if none
    cursor_moves: np.ndarray     # (N,) how many cursor MovePosition calls the scripts made
    names: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.action)

    def field(self, which: int, name: str) -> np.ndarray:
        return self.scr_raw[which, :, SCRIPT_FIELDS.index(name)]

    def body(self, which: int, index: int) -> dict:
        """Native fields of one Game body (index into names["body_names"]) per step: the native list order can change
        within a chain, so rows are picked by the recorded body index. Missing steps hold zeros."""
        return _pick(self.bodies[which], self.body_index[which], index)

    def joint(self, which: int, index: int) -> dict:
        """Native fields of one joint (index into names["joints"]) per step, decoded for its own type."""
        n = self.names["joints"][index] if self.names.get("joints") else ""
        jtype = 2 if n.endswith("SliderJoint2D") else 8 if n.endswith("FixedJoint2D") else 1
        return _pick(self.joints[which][jtype], self.joint_index[which], index)


def _pick(fields: dict, slots: np.ndarray, index: int) -> dict:
    hit = slots == index
    row = np.where(hit.any(1), hit.argmax(1), -1)
    out = {}
    for k, v in fields.items():
        a = v[np.arange(len(row)), np.maximum(row, 0)]
        a[row < 0] = 0
        out[k] = a
    return out


def decode_chain(records: list[bytes], meta: dict | None = None, names: dict | None = None) -> Chain:
    recs = [parse_record(r) for r in records]
    meta, names = meta or {}, names or {}
    scr_raw = np.stack([np.stack([r["scr"][k] for r in recs]) for k in range(3)])
    scr = [{n: scr_raw[k, :, i] for i, n in enumerate(SCRIPT_FIELDS)} for k in range(3)]
    ev = []
    for i, r in enumerate(recs):
        e = r.get("events", np.zeros(0, EVENT_DTYPE))
        ee = np.zeros(len(e), EVENT_DTYPE.descr + [("step", "<i4")])
        for f in EVENT_DTYPE.names:
            ee[f] = e[f]
        ee["step"] = i
        ev.append(ee)
    events = np.concatenate(ev) if ev else np.zeros(0)
    pts, base = [], 0
    for i, r in enumerate(recs):
        p = r.get("points", np.zeros(0, POINT_DTYPE))
        pp = np.zeros(len(p), POINT_DTYPE.descr + [("step", "<i4")])
        for f in POINT_DTYPE.names:
            pp[f] = p[f]
        pp["event"] += base
        pp["step"] = i
        base += len(ev[i])
        pts.append(pp)
    points = np.concatenate(pts) if pts else np.zeros(0, POINT_DTYPE.descr + [("step", "<i4")])
    body_names = names.get("body_names", [])
    cursor = body_names.index("Cursor") if "Cursor" in body_names else meta.get("cursor_index", -1)
    target = np.full((len(recs), 2), np.nan, np.float32)
    moves = np.zeros(len(recs), np.int32)
    for i, e in enumerate(ev):
        m = e[(e["kind"] == EV_MOVE_POSITION) & (e["phase"] == 0) & (e["script"] == cursor)]
        moves[i] = len(m)
        if len(m):
            target[i] = (m["a"][-1], m["b"][-1])
    out = dict(body_index=[], bodies=[], joint_index=[], joints_raw=[], joints=[], contact_ids=[], contacts=[],
               n_contacts=[], world=[])
    for k in range(2):
        nat = [r["nat"][k] for r in recs]
        nb = max(len(n["bodies"]) for n in nat)
        nj = max(len(n["joints"]) for n in nat)
        nc = max(1, max(len(n["contacts"]) for n in nat))
        braw = _pad([n["bodies"] for n in nat], nb)
        out["body_index"].append(_pad([n["bodies_id"][:, 0] for n in nat], nb, -1))
        out["bodies"].append(decode(braw, BODY_FIELDS))
        jraw = _pad([n["joints"] for n in nat], nj)
        out["joint_index"].append(_pad([n["joints_id"][:, 0] for n in nat], nj, -1))
        out["joints_raw"].append(jraw)
        jd = {t: {**decode(jraw, JOINT_COMMON), **decode(jraw, f)} for t, f in JOINT_FIELDS.items()}
        out["joints"].append(jd)
        out["contact_ids"].append(_pad([n["contacts_id"] for n in nat], nc, -1))
        craw = _pad([n["contacts"] for n in nat], nc)
        cd = decode(craw, CONTACT_FIELDS)
        cd["normalImpulse"] = np.stack([cd.pop("normalImpulse0"), cd.pop("normalImpulse1")], -1)
        cd["tangentImpulse"] = np.stack([cd.pop("tangentImpulse0"), cd.pop("tangentImpulse1")], -1)
        cd["key"] = np.stack([cd.pop("key0"), cd.pop("key1")], -1)
        cd["pointLocal"] = np.stack([cd.pop("localPointP"), cd.pop("localPointP1")], -2)
        out["contacts"].append(cd)
        out["n_contacts"].append(np.array([len(n["contacts"]) for n in nat], np.int32))
        out["world"].append(decode(np.stack([n["world"] for n in nat]), WORLD_FIELDS))
    return Chain(meta=meta, action=np.stack([r["action"] for r in recs]), scr=scr, scr_raw=scr_raw,
                 hammer_keys=[[r["hammer_keys"][k] for r in recs] for k in range(3)],
                 pot_keys=[[r["pot_keys"][k] for r in recs] for k in range(3)], events=events, points=points,
                 time_ms=np.array([r["time_ms"] for r in recs], np.float32),
                 native_ms=np.array([r["native_ms"] for r in recs], np.float32),
                 record_bytes=np.array([len(r) for r in records], np.int64), cursor_target=target, cursor_moves=moves,
                 names=names, **out)


def load_chain(path) -> Chain:
    """A chain file written by goi_jax.parity.record (npz: blob, offsets, meta json, names json)."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    names = json.loads(str(z["names"])) if "names" in z else {}
    return decode_chain(unpack_records(z["blob"], z["offsets"]), meta, names)


def load_index(root="dumps/parity") -> dict:
    """The recordings index (record.py writes <root>/index.json): chains, controller row, geometry file."""
    return json.loads((Path(root) / "index.json").read_text())


def iter_chains(root="dumps/parity", group: str | None = None):
    """(entry, Chain) for every chain in the index (optionally one group, e.g. "coffee" or "start")."""
    root = Path(root)
    for entry in load_index(root)["chains"]:
        if group is None or entry["group"] == group:
            yield entry, load_chain(root / entry["file"])


def controller_row(root="dumps/parity") -> dict:
    return json.loads((Path(root) / "controller.json").read_text())

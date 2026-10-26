"""Cross-check a native Box2D dump (plugin/NativeBox2D.cs, bridge command 'n' or launch flag -goi-native) against the
Unity-side values it carries, and against an EngineDump taken at the same moment.

    python scripts/check_native.py dumps/native/goi-rl-native-1.jsonl [--engine dumps/native/goi-rl-engine-state-1.jsonl]
    python scripts/check_native.py --series dumps/native/goi-rl-native-{3..12}.jsonl   # consecutive steps (T5)

Prints, per check, how many items agree bit-exactly and the worst difference otherwise. Stdlib only.
"""
import argparse
import collections
import json
import math
import struct

RAD2DEG = struct.unpack("<f", struct.pack("<f", 57.29578))[0]  # Mathf.Rad2Deg as float32


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def load(path):
    rows = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["kind"]].append(r)
    return rows


def rot(c, s, x, y):
    return c * x - s * y, s * x + c * y


def area(pts):
    return 0.5 * sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts)))  # > 0 counter-clockwise


def check_bodies(d):
    n = exact_pos = exact_vel = exact_mass = exact_inertia = exact_com = exact_rot = exact_w = 0
    worst_rot = 0.0
    for b in d["body"]:
        if b["path"] is None:
            continue
        n += 1
        x, y, s, c = b["xf"]
        exact_pos += [bits(x), bits(y)] == [bits(v) for v in b["apiPosition"]]
        exact_vel += [bits(v) for v in b["linearVelocity"]] == [bits(v) for v in b["apiVelocity"]]
        exact_mass += bits(b["mass"]) == bits(b["apiMass"]) or b["type"] != 2  # Box2D zeroes non-dynamic masses
        exact_inertia += bits(b["I"]) == bits(b["apiInertia"])
        exact_com += [bits(v) for v in b["sweep"][:2]] == [bits(v) for v in b["apiCenterOfMass"]]
        a = f32(b["sweep"][7])  # sweep: localCenter(2) c0(2) c(2) a0 a alpha0; JSON floats need re-rounding to float32
        exact_rot += bits(f32(a * RAD2DEG)) == bits(b["apiRotation"])  # one float32 multiply
        exact_w += bits(f32(f32(b["angularVelocity"]) * RAD2DEG)) == bits(b["apiAngularVelocity"])
        worst_rot = max(worst_rot, abs(a * RAD2DEG - b["apiRotation"]))
    print(f"bodies with a Rigidbody2D: {n}; bit-exact vs API: position {exact_pos}, velocity {exact_vel}, "
          f"rotation (sweep.a * Rad2Deg) {exact_rot} (worst {worst_rot:.3g} deg), angular velocity {exact_w}, "
          f"mass (dynamic; others are 0 natively) {exact_mass}, inertia {exact_inertia}, centre of mass {exact_com}")
    print("native body-list order (newest first):",
          ", ".join(f"{b['index']}:{(b['path'] or '<static ground>').split('/')[-1]}" for b in d["body"]))
    odd = [(b["path"], b["unity98"], b["unityB8"]) for b in d["body"] if b["unity98"] != [1, 1]]
    print("bodies whose Unity floats at 0x98 are not (1, 1):", odd or "none",
          "| sleepTime:", {(b["path"] or "ground").split("/")[-1]: b["sleepTime"] for b in d["body"] if b["sleepTime"]})


def check_pieces(d, engine=None):
    # Keyed by the native collider object: paths repeat (several "Mountain/Rock"s).
    pieces = collections.Counter(f["colliderNative"] for f in d["fixture"])
    shapes = {u["native"]: u["shapeCount"] for u in d["ucollider"] if u["active"]}
    bad = [(k, shapes.get(k, 0), pieces.get(k, 0)) for k in set(shapes) | set(pieces) if shapes.get(k, 0) != pieces.get(k, 0)]
    unmapped = sum(f["collider"] is None for f in d["fixture"])
    radii = collections.Counter(f["radius"] for f in d["fixture"] if f.get("shapeType") == 2)
    print(f"fixtures: {len(d['fixture'])}, unmapped to a collider: {unmapped}; colliders whose piece count != "
          f"shapeCount: {len(bad)} of {len(shapes)} {bad[:5]}; polygon radii: {dict(radii)}")
    print("fixture fields vs collider: sensor == isTrigger", sum(bool(f["sensor"]) == f["colliderTrigger"] for f in d["fixture"]),
          "| friction == collider.friction", sum(bits(f["friction"]) == bits(f["colliderFriction"]) for f in d["fixture"]),
          "| filters", dict(collections.Counter((f["categoryBits"], f["maskBits"], f["groupIndex"]) for f in d["fixture"])))
    if engine is None:
        return
    body = {b["addr"]: b for b in d["body"]}
    names = collections.Counter((u["path"], u["component"]) for u in d["ucollider"])
    native = collections.defaultdict(float)
    ccw = 0
    for f in d["fixture"]:
        if f.get("shapeType") != 2:
            continue
        x, y, s, c = body[f["body"]]["xf"]
        v = f["vertices"]
        pts = [(x + px, y + py) for px, py in (rot(c, s, v[i], v[i + 1]) for i in range(0, len(v), 2))]
        native[(f["collider"], f["component"])] += area(pts)
        ccw += area(pts) > 0
    outline = {}
    for c in engine["collider"]:
        if not c.get("tris") or c["type"] == "CircleCollider2D":
            continue
        v, t = c["verts"], c["tris"]
        outline[(c["path"], c["component"])] = sum(
            abs(area([(v[2 * t[i + k]], v[2 * t[i + k] + 1]) for k in range(3)])) for i in range(0, len(t), 3))
    rel = sorted((abs(native[k] - a) / max(a, 1e-9), k) for k, a in outline.items() if k in native and names[k] == 1)
    print(f"area of the native pieces vs Unity's outline mesh, {len(rel)} polygon/box colliders with unique paths: worst relative "
          f"difference {rel[-1][0]:.2e} ({rel[-1][1][0]}), {sum(r < 1e-5 for r, _ in rel)} within 1e-5, "
          f"{sum(r < 1e-3 for r, _ in rel)} within 1e-3; polygon pieces counter-clockwise: {ccw}")


def check_contacts(d):
    api = collections.defaultdict(list)
    for u in d["ucollider"]:
        for p in u["contacts"]:
            api[frozenset([p["collider"], p["other"]])].append(p)
    native = collections.defaultdict(list)
    touching = 0
    for c in d["contact"]:
        if c["flags"] & 2 and c["flags"] & 4:  # touching and enabled
            touching += 1
            native[frozenset([c["colliderA"], c["colliderB"]])] += c["points"]
    # Each API point is reported once from each side: halve.
    n_api = sum(len(v) for v in api.values()) // 2
    n_native = sum(len(v) for v in native.values())
    worst, compared, same = 0.0, 0, 0
    for pair, pts in native.items():
        a = sorted(p["normalImpulse"] for p in api.get(pair, []))[::2] if pair in api else []
        b = sorted(p["normalImpulse"] for p in pts)
        if len(a) == len(b):
            compared += len(b)
            same += sum(bits(x) == bits(y) for x, y in zip(a, b))
            worst = max([worst] + [abs(x - y) for x, y in zip(a, b)])
    print(f"contacts in the world list: {len(d['contact'])}, touching+enabled: {touching}, their manifold points: "
          f"{n_native}; ContactPoint2D from GetContacts: {n_api}; normalImpulse bit-exact {same} of {compared} "
          f"compared (worst {worst:.3g}); pairs only native {len(set(native) - set(api))}, only API {len(set(api) - set(native))}")
    fixtures = {f["addr"]: f for f in d["fixture"]}
    mix = sum(bits(c["friction"]) == bits(f32(math.sqrt(fixtures[c["fixtureA"]]["friction"] * fixtures[c["fixtureB"]]["friction"])))
              for c in d["contact"])
    print(f"contact friction == float32(sqrt(fA * fB)) of its fixtures' current friction: {mix} of {len(d['contact'])}")


def check_joints(d):
    body = {b["addr"]: b for b in d["body"]}
    by_path = {b["path"]: b for b in d["body"] if b["path"]}
    for j in d["joint"]:
        if "localAnchorA" not in j or j["type"] == 2:  # a slider's anchors are apart by its translation
            continue
        A, B = by_path.get(j["bodyA"]), by_path.get(j["bodyB"])
        if A is None or B is None:
            continue
        pa = [A["xf"][0] + v for v in rot(A["xf"][3], A["xf"][2], *j["localAnchorA"])]
        pb = [B["xf"][0] + v for v in rot(B["xf"][3], B["xf"][2], *j["localAnchorB"])]
        pa[1] += A["xf"][1] - A["xf"][0]
        pb[1] += B["xf"][1] - B["xf"][0]
        gap = math.dist(pa, pb)
        # Unity's anchor, put through the joint object's full 3D Transform and back into the body's 2D frame.
        ua = j.get("unityAnchor")
        proj = None
        if ua is not None and "tfRotation" in j:
            qx, qy, qz, qw = j["tfRotation"]
            sx, sy, _ = j["tfLossyScale"]
            vx, vy = ua[0] * sx, ua[1] * sy
            wx = (1 - 2 * (qy * qy + qz * qz)) * vx + 2 * (qx * qy - qz * qw) * vy
            wy = 2 * (qx * qy + qz * qw) * vx + (1 - 2 * (qx * qx + qz * qz)) * vy
            proj = rot(A["xf"][3], -A["xf"][2], wx, wy)
        print(f"joint {j['index']} {j['unityType']} {j['joint']}: native anchors A {j['localAnchorA']} B {j['localAnchorB']}; "
              f"Unity anchor {ua} connectedAnchor {j.get('unityConnectedAnchor')}; world gap {gap * 1000:.3f} mm"
              + (f"; Unity anchor through the 3D Transform: ({proj[0]:.7g}, {proj[1]:.7g})" if proj else ""))


def check_series(paths):
    """T5: for contacts that persist across consecutive steps, does m_friction follow the Tip's material swap?"""
    prev = None
    for p in paths:
        d = load(p)
        fx = {f["addr"]: f for f in d["fixture"]}
        tip = [u for u in d["ucollider"] if u["path"].endswith("/Tip")][0]
        cur = {}
        for c in d["contact"]:
            if "Tip" in (c["colliderA"] or "") + (c["colliderB"] or ""):
                fa, fb = fx[c["fixtureA"]]["friction"], fx[c["fixtureB"]]["friction"]
                cur[c["addr"]] = (c["friction"], fa, fb, c["flags"] & 2)
        kept = set(cur) & set(prev or {})
        follows = sum(bits(cur[a][0]) == bits(f32(math.sqrt(cur[a][1] * cur[a][2]))) for a in kept)
        changed = sum(prev[a][1:3] != cur[a][1:3] for a in kept)
        print(f"{p.split('/')[-1]}: Tip material {tip['material']}; Tip contacts {len(cur)} ({sum(v[3] > 0 for v in cur.values())} "
              f"touching), kept from last dump {len(kept)}, fixture friction changed on {changed}, contact friction == "
              f"sqrt(current fA*fB) on {follows} of {len(kept)}")
        prev = cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("native", nargs="+")
    ap.add_argument("--engine")
    ap.add_argument("--series", action="store_true")
    a = ap.parse_args()
    if a.series:
        return check_series(a.native)
    d = load(a.native[0])
    w = d["world"][0]
    print({k: v for k, v in w.items() if not k.startswith("raw")})
    check_bodies(d)
    check_pieces(d, load(a.engine) if a.engine else None)
    check_contacts(d)
    check_joints(d)


if __name__ == "__main__":
    main()

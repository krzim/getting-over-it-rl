"""Record parity chains in the space section (route >= 760 m) that cross the zero-gravity zone's boundary, go through
a teleporter and leave through the top (the win), from the real game (bridge 'W', K = 4).

The route placements (record_route.py) put the climber inside the ZeroGrav trigger: the placement's restore runs the
trigger Enter inside Simulate(0) and then sets the saved gravity back (-30), so those chains never see an Enter, a
gravity switch or a teleport. Here every chain starts from a placement that is edited before it is loaded: the
exported state (plugin/Game.cs Export, format 1) gets the climber (Player and every body under it, the cursor and the
controller's mouse target mw) moved by (dx, dy), every climber body's velocity set, and the gravity set. A chain that
starts outside the zone then enters it on its own during the recording.

After every recorded action the state is saved and exported once more (no side effects: Capture only reads) to keep
the script state the records do not carry: Patches.Won, GravityControl.creditsUp and the 3 flags per teleporter
(shouldTeleport, shouldEnable, enableTeleport). meta["script_after"] is that per action.

    .venv/bin/python -m goi_jax.parity.record_space --out dumps/parity-space     # unsandboxed: launches the game

The last line printed is DONE or STOPPED (reason).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import traceback
from pathlib import Path

import numpy as np

from goi_jax.parity import record as RC
from goi_jax.parity import states

BODY_BYTES = 50          # Export: pos 2f, vel 2f, rot, spin, z, quat 4f, active, awake, type i32
HEADER = 20              # format, nBodies, nHinges, nTeleporters, nColliders

# name, placement (along, lift, angle), pot target (x, y), climber velocity, gravity y, actions kind, actions
CHAINS = [
    ("enter-still", (775.0, 0.5, 180.0), (58.6, 354.6), (-0.3, 15.0), -30.0, "still", 90),
    ("enter-gentle", (775.0, 0.5, 180.0), (58.2, 354.4), (-0.6, 15.5), -30.0, "gentle", 90),
    ("enter-swing", (775.0, 0.5, 90.0), (59.4, 351.0), (-0.2, 17.0), -30.0, "swing", 120),
    ("exit-low", (775.0, 0.5, 180.0), (58.8, 361.5), (0.0, -6.0), 0.0, "still", 40),
    ("teleport-right", (775.0, 0.5, 0.0), (99.0, 420.0), (9.0, 0.0), 0.0, "still", 90),
    ("teleport-left", (775.0, 0.5, 180.0), (27.0, 430.0), (-9.0, 0.5), 0.0, "gentle", 90),
    ("win-top", (775.0, 0.5, 0.0), (70.0, 465.5), (0.0, 10.0), 0.0, "still", 40),
    ("drift-swing", (775.0, 0.5, 90.0), (60.0, 400.0), (0.0, 0.0), 0.0, "swing", 120),
    ("drift-random", (775.0, 0.5, 90.0), (50.0, 425.0), (1.0, -1.0), 0.0, "random", 120),
]


def parse_export(data: bytes, nb: int) -> dict:
    fmt, n_bodies, n_hinges, n_tele, n_col = struct.unpack_from("<5i", data, 0)
    assert fmt == 1 and n_bodies == nb, (fmt, n_bodies, nb)
    o = HEADER + nb * BODY_BYTES
    gravity = struct.unpack_from("<2f", data, o)
    o += 32 + 24
    flags = struct.unpack_from("<8?", data, o)
    o += 8
    tele = struct.unpack_from(f"<{3 * n_tele}?", data, o)
    return {"gravity": list(gravity), "won": flags[0], "creditsUp": flags[4], "teleport": list(tele)}


def edit_export(data: bytes, body_names: list[str], pot_xy, vel, gravity_y) -> bytes:
    """The exported state with the climber moved so the pot body sits at pot_xy, its bodies at velocity vel (angular
    velocity 0), the cursor and mw moved with it, and gravity (0, gravity_y)."""
    nb = len(body_names)
    buf = bytearray(data)
    pot = body_names.index("Player")
    px, py = struct.unpack_from("<2f", buf, HEADER + pot * BODY_BYTES)
    dx, dy = np.float32(pot_xy[0]) - np.float32(px), np.float32(pot_xy[1]) - np.float32(py)
    for i, n in enumerate(body_names):
        if n == "Player" or n.startswith("Player/") or n == "Cursor":
            o = HEADER + i * BODY_BYTES
            x, y = struct.unpack_from("<2f", buf, o)
            struct.pack_into("<2f", buf, o, np.float32(x) + dx, np.float32(y) + dy)
            if n != "Cursor":
                struct.pack_into("<2f", buf, o + 8, *vel)
                struct.pack_into("<f", buf, o + 20, 0.0)
    o = HEADER + nb * BODY_BYTES
    struct.pack_into("<2f", buf, o, 0.0, gravity_y)
    mx, my = struct.unpack_from("<2f", buf, o + 8)
    struct.pack_into("<2f", buf, o + 8, np.float32(mx) + dx, np.float32(my) + dy)
    return bytes(buf)


def gentle(n, rng):
    a, out, scale = np.zeros(2), [], rng.uniform(1.0, 3.0)
    for _ in range(n):
        a += -0.05 * a + rng.normal(0, 0.12 * scale, 2)
        out.append(np.clip(a, -scale, scale))
    return np.array(out, np.float32)


def record_one(client, spec, names, k, seed):
    name, (along, lift, angle), pot_xy, vel, gy, kind, n = spec
    rng = np.random.default_rng(seed)
    obs = client.place(along, lift, angle, 0.0)
    if obs is None:
        raise RuntimeError(f"placement at {along} does not fit")
    sid = client.save_state()
    data = client.export_state(sid)
    client.drop_state(sid)
    bn = names["body_names"]
    new = edit_export(data, bn, pot_xy, vel, gy)
    sid = client.import_state(new)
    obs = client.load_state(sid)
    client.drop_state(sid)
    acts = gentle(n, rng) if kind == "gentle" else RC.actions(kind, n, rng)
    meta = dict(name=name, along=along, lift=lift, angle=angle, pot_target=list(pot_xy), velocity=list(vel),
                gravity_start=gy, kind=kind, seed=seed, k=k, obs0=[float(x) for x in obs])
    recs, aos, after, obs_after = [], [], [], []
    t = time.perf_counter()
    for i, a in enumerate(acts):
        r = RC.record_action(client, float(a[0]), float(a[1]))
        if len(r) != k:
            raise RuntimeError(f"record_action returned {len(r)} records, expected K = {k}")
        recs += r
        aos += [i] * k
        sid = client.save_state()
        after.append(parse_export(client.export_state(sid), len(bn)))
        client.drop_state(sid)
        o = client.observe()
        obs_after.append([float(o[j]) for j in range(23)])
    meta.update(action_of_step=aos, actions_requested=acts.tolist(), actions=len(acts), steps=len(acts),
                wall_s=time.perf_counter() - t, script_after=after, obs_after=obs_after)
    return recs, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dumps/parity-space")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="chain names to record (default: all not yet recorded)")
    ap.add_argument("--max-seconds", type=float, default=600.0)
    args = ap.parse_args(argv)
    from goi_jax import bridge

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"chains": []}
    done = {e["file"] for e in index["chains"]}
    specs = [(i, s) for i, s in enumerate(CHAINS) if f"space/{s[0]}.npz" not in done and (not args.only or s[0] in args.only)]
    print(f"{len(specs)} chains to record", flush=True)
    if not specs:
        print("DONE")
        return
    if bridge.running_games():
        bridge.stop_all()
    t0 = time.perf_counter()
    proc, client = bridge.start(headless=True, k=args.k, n_rays=0)
    print(f"game connected in {time.perf_counter() - t0:.1f} s", flush=True)
    status = "DONE"
    try:
        client.reset()
        row = client.controller()
        (out / "controller.json").write_text(json.dumps(row, indent=1))
        geo = client.geometry()
        (out / "geometry.bin").write_bytes(geo)
        names = states.parse_geometry(geo)
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        index["controller"], index["geometry"] = "controller.json", "geometry.bin"
        t_start = time.monotonic()
        for i, spec in specs:
            if time.monotonic() - t_start > args.max_seconds:
                status = f"STOPPED: time budget of {args.max_seconds:.0f} s used; run again to record the rest"
                break
            file = f"space/{spec[0]}.npz"
            try:
                recs, meta = record_one(client, spec, names, args.k, 7000 + i)
            except RuntimeError as e:
                print(f"skip {file}: {e}", flush=True)
                continue
            chain = states.decode_chain(recs, meta, names)
            blob, offsets = states.pack_records(recs)
            (out / "space").mkdir(exist_ok=True)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, meta=json.dumps(meta), names=json.dumps(names))
            won = [a["won"] for a in meta["script_after"]]
            grav = [a["gravity"][1] for a in meta["script_after"]]
            tele = [a["teleport"] for a in meta["script_after"]]
            pot = chain.field(2, "pot.x"), chain.field(2, "pot.y")
            jumps = int((np.abs(np.diff(pot[0])) > 5).sum())
            entry = dict(file=file, group="space", section="space", kind=spec[5], actions=meta["actions"],
                         steps=meta["actions"], k=args.k, seed=meta["seed"],
                         start_pot=[float(pot[0][0]), float(pot[1][0])], end_pot=[float(pot[0][-1]), float(pot[1][-1])],
                         won_at=(won.index(True) if True in won else None), gravity_changes=int(np.sum(np.diff(grav) != 0)),
                         gravity_values=sorted(set(grav)), teleports=jumps, wall_s=meta["wall_s"],
                         contacts_max=int(chain.n_contacts[0].max()))
            index["chains"].append(entry)
            index_path.write_text(json.dumps(index, indent=1))
            print(json.dumps(entry), flush=True)
        client.close()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        index_path.write_text(json.dumps(index, indent=1))
        time.sleep(2)
        bridge.stop_all()
    print(f"{len(index['chains'])} chains in {index_path}")
    print(status)


if __name__ == "__main__":
    main()

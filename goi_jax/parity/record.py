"""Record per-step parity chains from the real game (bridge command 'w', plugin/GameRecord.cs).

Each chain starts the climber somewhere (the level start, or placed on the route at Coffee Rock, route progress
60-120), optionally lets it settle with a still mouse (unrecorded actions), then records N actions at K physics steps
each with one kind of mouse input:

    still   no input (the cursor drifts onto the head, the hammer goes loose)
    smooth  a slowly wandering input (Ornstein-Uhlenbeck, |a| up to ~8)
    swing   circles of the cursor (amplitude 6-16, a few rad/s, direction flips)
    hits    the head lifted, then slammed down onto the rock (bursts up to ~20: hard hits, HammerReturn)
    random  independent uniform inputs in [-12, 12]^2 each step (the policy's action scale, goi_env/env.py MOUSE)
    mixed   segments of the above

K = 1 (--k 1, bridge 'w'): one record per action. K > 1 (--k 4, bridge 'W', plugin >= record version 3): K records
per action, one per physics step, exactly as Game.Step runs the action (the pose update and the climber-joint
re-creation once, after the last step; steps 2..K read the previous step's slider reaction torque). The chain's meta
holds k and "action_of_step" (the action index of every record); "steps" and "actions" count actions (records =
actions x k).
Version-3 records also carry the climber's Transforms (TFM0..3, goi_jax/game/pose.py).

Output (gitignored): <out>/<group>/<name>.npz per chain (raw records, decoded by goi_jax.parity.states.load_chain),
<out>/controller.json (the controller row), <out>/geometry.bin (every fixture of the world, once per session), and
<out>/index.json listing every chain with its start, kind, seed and timing.

    .venv/bin/python -m goi_jax.parity.record --out dumps/parity            # run unsandboxed: it launches the game
    .venv/bin/python -m goi_jax.parity.record --plan small                  # 4 short chains, to try it
    .venv/bin/python -m goi_jax.parity.record --k 4 --plan k4 --out dumps/parity-k4    # 22 chains x 120 actions at K=4

The last line printed is DONE (all chains recorded) or STOPPED (with the reason; rerun to resume).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from goi_jax.parity import states

KINDS = ("still", "smooth", "swing", "hits", "random", "mixed")


def actions(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 2) float32 mouse inputs of one kind."""
    if kind == "still":
        return np.zeros((n, 2), np.float32)
    if kind == "smooth":
        a, out, scale = np.zeros(2), [], rng.uniform(2.0, 8.0)
        for _ in range(n):
            a += -0.04 * a + rng.normal(0, 0.35 * scale / 3, 2)
            out.append(np.clip(a, -scale, scale))
        return np.array(out, np.float32)
    if kind == "swing":
        amp, w, phase, out = rng.uniform(6, 16), rng.uniform(0.04, 0.12), rng.uniform(0, 2 * np.pi), []
        sign = 1.0
        for t in range(n):
            if rng.random() < 0.01:
                sign = -sign
            phase += sign * w
            out.append((amp * np.cos(phase), amp * np.sin(phase)))
        return np.array(out, np.float32)
    if kind == "hits":
        out = []
        while len(out) < n:
            up, down = int(rng.integers(15, 40)), int(rng.integers(8, 25))
            ax = rng.uniform(-3, 3)
            out += [(ax, rng.uniform(4, 10))] * up + [(rng.uniform(-6, 6), -rng.uniform(10, 20))] * down
            out += [(0.0, 0.0)] * int(rng.integers(0, 15))
        return np.array(out[:n], np.float32)
    if kind == "random":
        return rng.uniform(-12, 12, (n, 2)).astype(np.float32)
    if kind == "mixed":
        parts, left = [], n
        while left > 0:
            m = min(left, int(rng.integers(20, 60)))
            parts.append(actions(str(rng.choice(KINDS[:5])), m, rng))
            left -= m
        return np.concatenate(parts)
    raise ValueError(kind)


def plan(which: str) -> list[dict]:
    """The chains to record: group, start (along on the route or None for the level start), kind, steps (actions),
    settle (still actions before recording)."""
    chains = []
    if which == "k4small":
        return [dict(group="coffee", along=80.0, kind="hits", steps=12, settle=5, seed=900),
                dict(group="start", along=None, kind="random", steps=12, settle=0, seed=901)]
    if which == "k4":
        # Coffee Rock: 8 places x 2 kinds, then the start: 6 kinds; 120 actions (480 physics steps) each. Every third
        # chain starts right after the placement (cold), the rest after 15 still actions.
        seed, i = 400, 0
        order = ["hits", "smooth", "random", "swing", "mixed", "hits", "random", "smooth"]
        for a, along in enumerate([62.0, 70.0, 78.0, 86.0, 94.0, 102.0, 110.0, 118.0]):
            for j in range(2):
                chains.append(dict(group="coffee", along=along, kind=order[(a + 3 * j) % len(order)], steps=120,
                                   settle=0 if i % 3 == 0 else 15, seed=seed))
                seed, i = seed + 1, i + 1
        for kind in ("smooth", "swing", "hits", "random", "mixed", "still"):
            chains.append(dict(group="start", along=None, kind=kind, steps=120, settle=0 if kind == "still" else 8,
                               seed=seed))
            seed += 1
        return chains
    if which == "small":
        return [dict(group="coffee", along=80.0, kind="smooth", steps=40, settle=30, seed=0),
                dict(group="coffee", along=90.0, kind="hits", steps=40, settle=30, seed=1),
                dict(group="start", along=None, kind="swing", steps=40, settle=0, seed=2),
                dict(group="coffee", along=100.0, kind="still", steps=20, settle=0, seed=3)]
    seed = 100
    # Coffee Rock: 8 places x 3 kinds, 150 steps each; every third chain is recorded from the first step after
    # placement (cold world, no settling), the rest after 60 still steps.
    alongs = [62.0, 70.0, 78.0, 86.0, 94.0, 102.0, 110.0, 118.0]
    order = ["smooth", "hits", "swing", "random", "mixed", "still", "hits", "smooth"]
    for i, along in enumerate(alongs):
        for j in range(3):
            kind = order[(i + 3 * j) % len(order)] if j < 2 else ("hits" if i % 2 else "swing")
            chains.append(dict(group="coffee", along=along, kind=kind, steps=150, settle=0 if (i + j) % 3 == 0 else 60,
                               seed=seed))
            seed += 1
    for kind in ("smooth", "swing", "hits", "random", "still"):
        chains.append(dict(group="start", along=None, kind=kind, steps=120, settle=0 if kind == "still" else 30,
                           seed=seed))
        seed += 1
    return chains


def place(client, along: float, rng: np.random.Generator):
    for lift in (0.5, 1.0, 0.0, 2.0):
        for angle in rng.permutation([90, 60, 120, 30, 150, 0, 180]):
            obs = client.place(along, lift, float(angle))
            if obs is not None:
                return obs, lift, float(angle)
    return None, None, None


def record_action(client, mouse_x, mouse_y) -> list[bytes]:
    """Bridge 'W': one action at the configured K as K per-step records (plugin record version >= 3)."""
    import struct
    data = bytes(client._call(b"W" + struct.pack("<ff", mouse_x, mouse_y)))
    (n,), o, out = struct.unpack_from("<i", data, 0), 4, []
    for _ in range(n):
        (m,) = struct.unpack_from("<i", data, o)
        out.append(data[o + 4:o + 4 + m])
        o += 4 + m
    return out


def record_chain(client, spec: dict, names: dict, k: int = 1) -> tuple[list[bytes], dict]:
    rng = np.random.default_rng(spec["seed"])
    meta = dict(spec)
    if spec["along"] is None:
        obs = client.reset()
        meta.update(lift=None, angle=None)
    else:
        obs, lift, angle = place(client, spec["along"], rng)
        if obs is None:
            raise RuntimeError(f"the climber fits nowhere at {spec['along']}")
        meta.update(lift=lift, angle=angle)
    for _ in range(spec["settle"]):
        obs = client.step(0.0, 0.0)
    meta["start_pot"] = [float(obs[0]), float(obs[1])]
    meta["start_progress"] = float(obs[15])
    acts = actions(spec["kind"], spec["steps"], rng)
    t = time.perf_counter()
    if k == 1:
        recs = [client.record_step(float(a[0]), float(a[1])) for a in acts]
        meta["action_of_step"] = list(range(len(recs)))
    else:
        recs, aos = [], []
        for i, a in enumerate(acts):
            r = record_action(client, float(a[0]), float(a[1]))
            if len(r) != k:
                raise RuntimeError(f"record_action returned {len(r)} records, expected K = {k}")
            recs += r
            aos += [i] * k
        meta["action_of_step"] = aos
    meta["wall_s"] = time.perf_counter() - t
    meta["actions_requested"] = acts.tolist()
    meta["k"] = k
    meta["actions"] = len(acts)
    return recs, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dumps/parity")
    ap.add_argument("--plan", default="full", choices=("full", "small", "k4", "k4small"))
    ap.add_argument("--k", type=int, default=1, help="physics steps per action (1: bridge 'w'; > 1: bridge 'W')")
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--max-seconds", type=float, default=240.0,
                    help="stop after this long (resumable: chains already in the index are skipped next time)")
    args = ap.parse_args(argv)
    from goi_jax import bridge  # launching needs WSL interop: run unsandboxed

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"chains": []}
    done = {e["file"] for e in index["chains"]}
    specs = plan(args.plan)
    stale = bridge.running_games()
    if stale:
        print(f"stale games running before launch (killed): {stale}", flush=True)
        bridge.stop_all()
    t_launch = time.perf_counter()
    proc, client = bridge.start(headless=bool(args.headless), k=args.k, n_rays=0)
    print(f"game connected in {time.perf_counter() - t_launch:.1f} s", flush=True)
    status = "DONE"
    try:
        client.reset()
        row = client.controller()
        (out / "controller.json").write_text(json.dumps(row, indent=1))
        if row.get("scriptFields") and tuple(row["scriptFields"]) != states.SCRIPT_FIELDS:
            raise RuntimeError("the plugin's script fields differ from goi_jax.parity.states.SCRIPT_FIELDS")
        geo = client.geometry()
        (out / "geometry.bin").write_bytes(geo)
        names = states.parse_geometry(geo)
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        index["controller"], index["geometry"] = "controller.json", "geometry.bin"
        t_start = time.monotonic()
        for i, spec in enumerate(specs):
            if time.monotonic() - t_start > args.max_seconds:
                status = f"STOPPED: time budget of {args.max_seconds:.0f} s used; run again to record the rest"
                break
            name = f"{spec['group']}/c{i:02d}-{spec['kind']}" + ("" if spec["along"] is None else f"-{spec['along']:.0f}")
            file = name + ".npz"
            if file in done:
                continue
            try:
                recs, meta = record_chain(client, spec, names, args.k)
            except RuntimeError as e:
                print(f"skip {name}: {e}", flush=True)
                continue
            chain = states.decode_chain(recs, meta, names)  # proves the records decode before they are kept
            blob, offsets = states.pack_records(recs)
            (out / spec["group"]).mkdir(exist_ok=True)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, meta=json.dumps(meta), names=json.dumps(names))
            entry = dict(file=file, **{k: meta[k] for k in ("group", "along", "kind", "steps", "settle", "seed", "lift",
                                                             "angle", "start_pot", "start_progress", "k", "actions")},
                         ms_median=float(np.median(chain.time_ms)), ms_max=float(chain.time_ms.max()),
                         native_ms_median=float(np.median(chain.native_ms)),
                         kb_median=float(np.median(chain.record_bytes) / 1024), kb_max=float(chain.record_bytes.max() / 1024),
                         contacts_max=int(chain.n_contacts[0].max()), wall_s=meta["wall_s"],
                         hammer_returns=int(((chain.events["kind"] == states.EV_HAMMER_RETURN)
                                             & (chain.events["script"] == 1) & (chain.events["flags"] == 1)).sum()))
            index["chains"].append(entry)
            index_path.write_text(json.dumps(index, indent=1))
            print(f"{name}: {meta['actions']} actions, {len(recs)} steps, {entry['ms_median']:.2f} ms median ({entry['ms_max']:.1f} max), "
                  f"{entry['kb_median']:.1f} KB median, {entry['contacts_max']} contacts max, "
                  f"{entry['hammer_returns']} hammer returns, wall {meta['wall_s']:.1f} s", flush=True)
        client.close()
    except Exception as e:
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        time.sleep(2)
        bridge.stop_all()
    total = sum(e["steps"] * e.get("k", 1) for e in index["chains"])
    print(f"{len(index['chains'])} chains, {total} physics steps in {index_path}")
    print(status)


if __name__ == "__main__":
    main()

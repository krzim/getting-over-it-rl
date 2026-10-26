"""Record a route-wide start bank and per-section parity chains from the real game (bridge 'W', K = 4).

Group "route" (training starts): the climber placed every --every m along the whole route (0..864) the way upstream
places it (goi_env/env.py GoiEnv._place: lifts 0.5, 1, 0, -0.5, 2, 3, 4 above the route point, hammer angles 90, 60,
120, ... until it fits), in up to four variants per spot:

    a  upstream's order, dx 0                       (what upstream's --place seeds)
    b  hammer leaning first (angles 30, 150, 60, ...), dx 0
    c  upstream's order, pot 1 m ahead (dx +1)      (upstream --place-at ALONG:DX)
    d  upstream's order, pot 1 m behind (dx -1)

each left to settle 30 still actions (upstream's settle) and then recorded for --actions actions (still for a,
gentle wandering input for b-d). Spots where nothing fits are skipped; a variant that lands on the same placement as
an earlier one of the spot, or that ends in the lake / at the win while settling, is skipped. Also the level start
(reset; cold and settled).

Group "section" (parity spot checks): per section of scripts/sections.py SECTIONS (and the level start), one chain
placed a little into the section, settled 15 actions, then --section-actions actions of mixed input.

Every index entry carries along, the placement (lift, angle, dx), start_progress (the plugin's progress after
settling), the section name of that progress, and obs0 (the plugin's full observation right before the first recorded
step, 32 pot rays of 15 m: the JAX env's ObsConfig) for the obs-at-load check.

    .venv/bin/python -m goi_jax.parity.record_route --out dumps/parity-route      # unsandboxed: launches the game

Resumable (chains already in the index are skipped). The last line is DONE or STOPPED (reason).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

from goi_jax.parity import record as RC
from goi_jax.parity import states

# scripts/sections.py SECTIONS (route progress where each section starts; the last is the win)
SECTIONS = [(9, "lakeside"), (60, "rocks"), (120, "tree and slope"), (176, "chimney"), (204, "Construction Wall"),
            (228, "girder"), (245, "hut"), (256, "garden"), (290, "playground"), (310, "boxes"), (335, "furniture"),
            (385, "orange stretch"), (415, "chapel"), (465, "roofs"), (525, "anvil"), (560, "snow"), (630, "bucket"),
            (660, "Ice Mountain"), (735, "radio mast"), (760, "space"), (864, "win")]
LIFTS = (0.5, 1.0, 0.0, -0.5, 2.0, 3.0, 4.0)                          # goi_env/env.py GoiEnv._place
ANGLES = (90, 60, 120, 30, 150, 0, 180, 210, 330, 240, 300, 270)
LEAN = (30, 150, 60, 120, 0, 180, 90, 210, 330, 240, 300, 270)
VARIANTS = {"a": (ANGLES, 0.0, "still"), "b": (LEAN, 0.0, "gentle"), "c": (ANGLES, 1.0, "gentle"),
            "d": (ANGLES, -1.0, "gentle")}
LAKE, WON, PROGRESS = 13, 14, 15


def section_of(progress: float) -> str:
    """Name of the section containing route progress (below the first: 'start'; at/after the win: 'win')."""
    name = "start"
    for lo, n in SECTIONS:
        if progress >= lo:
            name = n
    return name


def gentle(n, rng):
    """A slow, small wandering mouse input (|a| <= ~3)."""
    a, out, scale = np.zeros(2), [], rng.uniform(1.0, 3.0)
    for _ in range(n):
        a += -0.05 * a + rng.normal(0, 0.12 * scale, 2)
        out.append(np.clip(a, -scale, scale))
    return np.array(out, np.float32)


def plan(every=5.0, n_actions=8, section_actions=60, variants="abcd"):
    specs, seed = [], 5000
    specs.append(dict(group="route", name="route/start-cold", along=None, kind="still", steps=n_actions, settle=0,
                      seed=seed))
    specs.append(dict(group="route", name="route/start-settled", along=None, kind="gentle", steps=n_actions,
                      settle=30, seed=seed + 1))
    seed += 2
    for along in np.arange(0.0, 864.0 + 1e-6, every):
        for v in variants:
            specs.append(dict(group="route", name=f"route/r{along:05.1f}-{v}", along=float(along), variant=v,
                              kind=VARIANTS[v][2], steps=n_actions, settle=30, seed=seed))
            seed += 1
    specs.append(dict(group="section", name="section/s00-start", along=None, kind="mixed", steps=section_actions,
                      settle=8, seed=seed, section="start"))
    seed += 1
    for i, ((lo, name), (hi, _)) in enumerate(zip(SECTIONS[:-1], SECTIONS[1:])):
        along = float(lo + min(8.0, (hi - lo) / 3))
        specs.append(dict(group="section", name=f"section/s{i + 1:02d}-{name.replace(' ', '_')}", along=along,
                          variant="a", kind="mixed", steps=section_actions, settle=15, seed=seed, section=name))
        seed += 1
    return specs


def place(client, along, variant):
    angles, dx, _ = VARIANTS[variant]
    for lift in LIFTS:
        for angle in angles:
            obs = client.place(along, lift, float(angle), dx)
            if obs is not None:
                return obs, lift, float(angle), dx
    return None, None, None, None


def record_one(client, spec, k, seen):
    rng = np.random.default_rng(spec["seed"])
    meta = dict(spec)
    if spec["along"] is None:
        obs = client.reset()
        meta.update(lift=None, angle=None, dx=None)
    else:
        obs, lift, angle, dx = place(client, spec["along"], spec.get("variant", "a"))
        if obs is None:
            raise RuntimeError(f"the climber fits nowhere at {spec['along']}")
        key = (spec["along"], lift, angle, dx)
        if spec["group"] == "route" and key in seen:
            raise RuntimeError(f"same placement as an earlier variant ({lift}, {angle}, {dx})")
        seen.add(key)
        meta.update(lift=lift, angle=angle, dx=dx)
    for _ in range(spec["settle"]):
        obs = client.step(0.0, 0.0)
        if obs[LAKE] > 0.5 or obs[WON] > 0.5:
            raise RuntimeError("fell in the lake / won while settling")
    meta["obs0"] = [float(x) for x in obs]
    meta["start_pot"] = [float(obs[0]), float(obs[1])]
    meta["start_progress"] = float(obs[PROGRESS])
    meta["section"] = spec.get("section") or section_of(float(obs[PROGRESS]))
    acts = gentle(spec["steps"], rng) if spec["kind"] == "gentle" else RC.actions(spec["kind"], spec["steps"], rng)
    t = time.perf_counter()
    recs, aos = [], []
    for i, a in enumerate(acts):
        r = RC.record_action(client, float(a[0]), float(a[1]))
        if len(r) != k:
            raise RuntimeError(f"record_action returned {len(r)} records, expected K = {k}")
        recs += r
        aos += [i] * k
    meta.update(action_of_step=aos, wall_s=time.perf_counter() - t, actions_requested=acts.tolist(), k=k,
                actions=len(acts))
    return recs, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="dumps/parity-route")
    ap.add_argument("--every", type=float, default=5.0)
    ap.add_argument("--actions", type=int, default=8)
    ap.add_argument("--section-actions", type=int, default=60)
    ap.add_argument("--variants", default="abcd")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=480.0)
    args = ap.parse_args(argv)
    from goi_jax import bridge

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"chains": []}
    tried = set(index.get("skipped", []))
    done = {e["file"] for e in index["chains"]} | tried
    specs = [s for s in plan(args.every, args.actions, args.section_actions, args.variants)
             if s["name"] + ".npz" not in done]
    print(f"{len(specs)} chains to record", flush=True)
    if not specs:
        print("DONE")
        return
    if bridge.running_games():
        bridge.stop_all()
    t0 = time.perf_counter()
    proc, client = bridge.start(headless=True, k=args.k, n_rays=32, ray_length=15.0)
    print(f"game connected in {time.perf_counter() - t0:.1f} s", flush=True)
    status, n_new, n_skip = "DONE", 0, 0
    try:
        client.reset()
        row = client.controller()
        (out / "controller.json").write_text(json.dumps(row, indent=1))
        geo = client.geometry()
        (out / "geometry.bin").write_bytes(geo)
        names = states.parse_geometry(geo)
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        index["controller"], index["geometry"] = "controller.json", "geometry.bin"
        t_start, seen = time.monotonic(), set()
        for spec in specs:
            if time.monotonic() - t_start > args.max_seconds:
                status = f"STOPPED: time budget of {args.max_seconds:.0f} s used; run again to record the rest"
                break
            file = spec["name"] + ".npz"
            try:
                recs, meta = record_one(client, spec, args.k, seen)
            except RuntimeError as e:
                n_skip += 1
                index.setdefault("skipped", []).append(file)
                index.setdefault("skip_reason", {})[file] = str(e)
                continue
            chain = states.decode_chain(recs, meta, names)
            blob, offsets = states.pack_records(recs)
            (out / spec["group"]).mkdir(exist_ok=True)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, meta=json.dumps(meta), names=json.dumps(names))
            entry = dict(file=file, **{k: meta.get(k) for k in (
                "group", "along", "variant", "kind", "steps", "settle", "seed", "lift", "angle", "dx", "start_pot",
                "start_progress", "section", "k", "actions", "obs0")},
                contacts_max=int(chain.n_contacts[0].max()), wall_s=meta["wall_s"])
            index["chains"].append(entry)
            n_new += 1
            if n_new % 20 == 0:
                index_path.write_text(json.dumps(index))
                print(f"{n_new} recorded, {n_skip} skipped (last {file}, progress {meta['start_progress']:.1f}, "
                      f"{meta['section']})", flush=True)
        client.close()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        index_path.write_text(json.dumps(index))
        time.sleep(2)
        bridge.stop_all()
    print(f"{n_new} recorded, {n_skip} skipped this run; {len(index['chains'])} chains in {index_path}")
    print(status)


if __name__ == "__main__":
    main()

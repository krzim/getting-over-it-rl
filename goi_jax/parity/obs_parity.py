"""Observation parity: the JAX observe/route/reward (goi_jax/game/observe.py, route.py, reward.py) against the
plugin's own Observe on the same states.

Record (launches the game; run unsandboxed):

    .venv/bin/python -m goi_jax.parity.obs_parity record --out dumps/obs

writes <out>/route.json (bridge 'u': LevelSpline and the flag colliders, with probes of the game's spline queries),
<out>/geometry.bin, <out>/controller.json, and chains <out>/chains/*.npz. A chain places the climber on the route (or
resets it), settles, then for every step t records the K=1 step ('w') and the plugin's observation right after it
('o', no step). Observation t is of the state after step t: the native world right before the NEXT step's Simulate
(NAT0 of record t+1: the pose round trip at the end of an action has moved the climber's child bodies by ~1e-6 since
NAT1, and nothing moves bodies between Observe and that point). So a chain of N observed steps holds N+1 records,
obs[0] being the observation before the first record (after placing / settling).

Compare (CPU is fine):

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.obs_parity compare --data dumps/obs

prints the per-field table: max abs error, bit-exact share, share within tolerance. The last line printed is DONE or
STOPPED.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

N_RAYS, RAY_LENGTH, HEAD_RAYS, HEAD_LENGTH = 32, 15.0, 8, 4.0


# --------------------------------------------------------------------------------------------------------- recording

def _actions(kind, n, rng):
    if kind == "still":
        return np.zeros((n, 2), np.float32)
    if kind == "random":
        return rng.uniform(-12, 12, (n, 2)).astype(np.float32)
    if kind == "swing":
        amp, w, ph = rng.uniform(6, 16), rng.uniform(0.04, 0.12), rng.uniform(0, 2 * np.pi)
        return np.array([(amp * np.cos(ph + w * t), amp * np.sin(ph + w * t)) for t in range(n)], np.float32)
    if kind == "flick":  # big moves: flings, falls, long jumps of progress
        out = []
        while len(out) < n:
            a = rng.uniform(-30, 30, 2)
            out += [a] * int(rng.integers(3, 10)) + [(0.0, 0.0)] * int(rng.integers(0, 20))
        return np.array(out[:n], np.float32)
    raise ValueError(kind)


def _place(client, along, rng):
    for lift in (0.5, 1.0, 0.0, 2.0, 3.0):
        for angle in rng.permutation([90, 60, 120, 30, 150, 0, 180]):
            obs = client.place(along, lift, float(angle))
            if obs is not None:
                return obs
    return None


def record(out: Path, max_seconds=420.0, steps=40, settle=20):
    from goi_jax import bridge
    from goi_jax.parity import states
    out.mkdir(parents=True, exist_ok=True)
    (out / "chains").mkdir(exist_ok=True)
    index_path = out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"chains": []}
    done = {e["file"] for e in index["chains"]}
    proc, client = bridge.start(headless=True, k=1, n_rays=N_RAYS, ray_length=RAY_LENGTH, head_rays=HEAD_RAYS,
                                head_length=HEAD_LENGTH)
    status = "DONE"
    try:
        client.reset()
        route = client.route()
        (out / "route.json").write_text(json.dumps(route))
        (out / "controller.json").write_text(json.dumps(client.controller(), indent=1))
        geo = client.geometry()
        (out / "geometry.bin").write_bytes(geo)
        names = states.parse_geometry(geo)
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        length = float(route["length"])
        index.update(config=dict(n_rays=N_RAYS, ray_length=RAY_LENGTH, head_rays=HEAD_RAYS, head_length=HEAD_LENGTH,
                                 k=1), route_length=length)
        kinds = ("random", "swing", "flick", "still")
        specs = [dict(along=None, kind=k, seed=i) for i, k in enumerate(("flick", "random", "flick"))]
        alongs = np.arange(4.0, length - 4.0, max(4.0, (length - 8.0) / 70))
        specs += [dict(along=float(a), kind=kinds[i % len(kinds)], seed=100 + i) for i, a in enumerate(alongs)]
        t0 = time.monotonic()
        for i, spec in enumerate(specs):
            if time.monotonic() - t0 > max_seconds:
                status = f"STOPPED: time budget of {max_seconds:.0f} s used; run again to record the rest"
                break
            file = f"chains/o{i:03d}-{spec['kind']}" + ("-start" if spec["along"] is None else f"-{spec['along']:.0f}") + ".npz"
            if file in done:
                continue
            rng = np.random.default_rng(spec["seed"])
            obs0 = client.reset() if spec["along"] is None else _place(client, spec["along"], rng)
            if obs0 is None:
                print(f"skip {file}: the climber fits nowhere there", flush=True)
                continue
            for _ in range(settle if spec["along"] is not None else 0):
                obs0 = client.step(0.0, 0.0)
            obs, snake, recs = [obs0], [client.on_snake], []
            acts = _actions(spec["kind"], steps + 1, rng)
            for t in range(steps + 1):
                recs.append(client.record_step(float(acts[t, 0]), float(acts[t, 1])))
                if t < steps:
                    obs.append(client.observe())
                    snake.append(client.on_snake)
            states.decode_chain(recs[:2], spec, names)  # decodes
            blob, offsets = states.pack_records(recs)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, obs=np.array(obs, np.float32),
                                snake=np.array(snake), actions=acts, meta=json.dumps(spec), names=json.dumps(names))
            index["chains"].append(dict(file=file, **spec, steps=steps))
            index_path.write_text(json.dumps(index, indent=1))
            o = np.array(obs)
            print(f"{file}: progress {o[0, 15]:.1f} -> {o[-1, 15]:.1f}, lake {o[:, 13].max():.0f}, "
                  f"snake {max(snake):d}, head {o[:, 11].mean():.2f}, pot {o[:, 12].mean():.2f}", flush=True)
        client.close()
    except Exception as e:
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        time.sleep(2)
        bridge.stop_all()
    print(f"{len(index['chains'])} chains in {index_path}")
    print(status)


def _record_chain(client, names, actions, stop=None):
    """record_step + observe per action, plus one more record (the state of the last observation is that record's
    NAT0); stops early once stop(obs) is true. Returns (records, obs, snake)."""
    from goi_jax.parity import states
    obs, snake, recs = [client.observe()], [client.on_snake], []
    for t in range(len(actions)):
        recs.append(client.record_step(float(actions[t, 0]), float(actions[t, 1])))
        if t == len(actions) - 1 or (stop is not None and stop(obs[-1]) and len(recs) > 1):
            break
        obs.append(client.observe())
        snake.append(client.on_snake)
    obs = obs[:len(recs)]
    snake = snake[:len(recs)]
    states.decode_chain(recs[:1], {}, names)
    return recs, obs, snake


def record_extra(out: Path, max_seconds=420.0, guard_tries=3000):
    """Second session: (1) guard probes: place(along, dx, lift, angle) at random, which runs the plugin's Progress from
    progress = along at the placed pot (obs pot x, y): <out>/guard.npz; (2) chains aimed at the flags: falls into the
    lake beside the start, drops along the Snake, flings at the top of the space zone (x-prefixed chains)."""
    from goi_jax import bridge
    from goi_jax.parity import states
    index_path = out / "index.json"
    index = json.loads(index_path.read_text())
    proc, client = bridge.start(headless=True, k=1, n_rays=N_RAYS, ray_length=RAY_LENGTH, head_rays=HEAD_RAYS,
                                head_length=HEAD_LENGTH)
    status = "DONE"
    t0 = time.monotonic()
    try:
        client.reset()
        names = states.parse_geometry(client.geometry())
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        length = float(index["route_length"])
        rng = np.random.default_rng(7)
        if not (out / "guard.npz").exists():
            rows = []
            for i in range(guard_tries):
                along = np.float32(rng.uniform(0, length))
                dx, lift = np.float32(rng.uniform(-12, 12)), np.float32(rng.uniform(-6, 6))
                obs = client.place(float(along), float(lift), float(rng.choice([0, 90, 180, 270])), float(dx))
                if obs is not None:
                    rows.append([along, obs[0], obs[1], obs[15]])
                if time.monotonic() - t0 > max_seconds / 3:
                    break
            np.savez_compressed(out / "guard.npz", rows=np.array(rows, np.float32))
            print(f"guard probes: {len(rows)} placements kept of {i + 1}", flush=True)
        specs = []
        for dx in (-6.0, -10.0, -14.0, -20.0, -26.0):
            specs.append(dict(kind="lake", along=2.0, dx=dx, lift=1.0, steps=160))
        for along in np.arange(20.0, 460.0, 20.0):
            for dx in (-5.0, 5.0):
                specs.append(dict(kind="snake", along=float(along), dx=dx, lift=2.0, steps=40))
        for along in (990.0, 1000.0, 1005.0, 1010.0, 1015.0):
            specs.append(dict(kind="top", along=along, dx=0.0, lift=0.5, steps=120))
        done = {e["file"] for e in index["chains"]}
        for i, sp in enumerate(specs):
            if time.monotonic() - t0 > max_seconds:
                status = f"STOPPED: time budget of {max_seconds:.0f} s used; run again to record the rest"
                break
            file = f"chains/x{i:03d}-{sp['kind']}-{sp['along']:.0f}{sp['dx']:+.0f}.npz"
            if file in done:
                continue
            obs0 = None
            for lift in (sp["lift"], sp["lift"] + 1, sp["lift"] + 3, 0.0):
                for angle in (90, 60, 120, 30, 150, 0, 180):
                    obs0 = client.place(sp["along"], lift, float(angle), sp["dx"])
                    if obs0 is not None:
                        break
                if obs0 is not None:
                    break
            if obs0 is None:
                print(f"skip {file}: no room", flush=True)
                continue
            if sp["kind"] == "top":  # fling: big up/down mouse bursts
                acts = np.array([(0.0, 30.0)] * 10 + [(0.0, -30.0)] * 10, np.float32)
                acts = np.tile(acts, (sp["steps"] // 20 + 1, 1))[: sp["steps"]]
            else:
                acts = np.zeros((sp["steps"], 2), np.float32)
            stop = (lambda o: o[13] > 0) if sp["kind"] == "lake" else None
            recs, obs, snake = _record_chain(client, names, acts, stop)
            blob, offsets = states.pack_records(recs)
            meta = dict(sp, seed=0)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, obs=np.array(obs, np.float32),
                                snake=np.array(snake), actions=acts[:len(recs)], meta=json.dumps(meta),
                                names=json.dumps(names))
            index["chains"].append(dict(file=file, **meta))
            index_path.write_text(json.dumps(index, indent=1))
            o = np.array(obs)
            print(f"{file}: {len(o)} obs, progress {o[0, 15]:.1f} -> {o[-1, 15]:.1f}, lake {o[:, 13].max():.0f}, "
                  f"won {o[:, 14].max():.0f}, snake {int(max(snake))}, pot y {o[:, 1].min():.1f}..{o[:, 1].max():.1f}",
                  flush=True)
        client.close()
    except Exception as e:
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        time.sleep(2)
        bridge.stop_all()
    print(status)


def target_specs(data: Path, won_only=False):
    """Chains aimed at the two flags no wandering chain sets: the pot dropped onto the Snake (a chute of 62 pieces from
    beside the bucket down to the start) at points spread over its height, and the hammer swung up out of the top of the
    space zone (GravityControl's trigger) from a pot placed just below it."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import level as L
    from goi_jax.game import route as R
    lv = L.load_level(data)
    route_row = json.loads((data / "route.json").read_text())
    rm = R.route_model(route_row)
    fx, cols = lv.fixture, lv.names["colliders"]
    sn = np.nonzero(fx["collider"] == cols.index(route_row["snakeCollider"]))[0]
    v = np.concatenate([fx["verts"][k][:fx["count"][k]] for k in sn])
    s = np.arange(0, float(route_row["length"]), 0.5, dtype=np.float32)
    pts = np.asarray(jax.jit(jax.vmap(lambda x: jnp.stack(R.point_at(rm, x)[:2])))(s))
    specs = []
    picks = v[np.argsort(v[:, 1])][:: max(1, len(v) // 14)]
    for q in ([] if won_only else picks):
        j = int(np.argmin(((pts - q[None]) ** 2).sum(-1)))
        specs.append(dict(kind="snake", along=float(s[j]), dx=float(q[0] - pts[j, 0]), lift=float(q[1] - pts[j, 1]) + 1.2,
                          steps=40, mouse=(0.0, 0.0)))
    top = float(route_row["zeroGravMax"][1])
    j = int(np.argmin(np.abs(pts[:, 1] - (top - 3.0)) + 1e3 * ((pts[:, 0] < -5) | (pts[:, 0] > 125))))
    for drop in ((1.2, 1.6, 2.0) if won_only else (2.5, 3.0, 3.5, 4.0)):
        for mouse in ((0.0, 30.0), (10.0, 30.0), (-10.0, 30.0)):
            specs.append(dict(kind="top", along=float(s[j]), dx=0.0, lift=float(top - drop - pts[j, 1]), steps=30,
                              mouse=mouse))
    return specs


def record_targets(out: Path, max_seconds=420.0, won_only=False):
    from goi_jax import bridge
    from goi_jax.parity import states
    index_path = out / "index.json"
    index = json.loads(index_path.read_text())
    specs = target_specs(out, won_only)
    proc, client = bridge.start(headless=True, k=1, n_rays=N_RAYS, ray_length=RAY_LENGTH, head_rays=HEAD_RAYS,
                                head_length=HEAD_LENGTH)
    status = "DONE"
    t0 = time.monotonic()
    try:
        client.reset()
        names = states.parse_geometry(client.geometry())
        names = {k: names[k] for k in ("colliders", "body_names", "joints")}
        done = {e["file"] for e in index["chains"]}
        for i, sp in enumerate(specs):
            if time.monotonic() - t0 > max_seconds:
                status = f"STOPPED: time budget of {max_seconds:.0f} s used; run again to record the rest"
                break
            file = f"chains/{'z' if won_only else 'y'}{i:03d}-{sp['kind']}-{sp['along']:.0f}.npz"
            if file in done:
                continue
            obs0 = None
            angles = (0.0, 180.0, 30.0, 150.0) if sp["kind"] == "top" else (90.0, 60.0, 120.0, 30.0, 150.0, 0.0, 180.0)
            for extra in (0.0, 0.5, 1.0, 2.0):
                for angle in angles:
                    obs0 = client.place(sp["along"], sp["lift"] + extra, angle, sp["dx"])
                    if obs0 is not None:
                        break
                if obs0 is not None:
                    break
            if obs0 is None:
                print(f"skip {file}: no room", flush=True)
                continue
            acts = np.tile(np.array(sp["mouse"], np.float32), (sp["steps"], 1))
            recs, obs, snake = _record_chain(client, names, acts)
            blob, offsets = states.pack_records(recs)
            meta = dict(sp, seed=0)
            np.savez_compressed(out / file, blob=blob, offsets=offsets, obs=np.array(obs, np.float32),
                                snake=np.array(snake), actions=acts[:len(recs)], meta=json.dumps(meta),
                                names=json.dumps(names))
            index["chains"].append(dict(file=file, **meta))
            index_path.write_text(json.dumps(index, indent=1))
            o = np.array(obs)
            print(f"{file}: {len(o)} obs, won {o[:, 14].max():.0f}, snake {int(max(snake))}, "
                  f"pot y {o[:, 1].min():.1f}..{o[:, 1].max():.1f}", flush=True)
        client.close()
    except Exception as e:
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        time.sleep(2)
        bridge.stop_all()
    print(status)


# --------------------------------------------------------------------------------------------------------- compare

def load_obs_chain(path):
    from goi_jax.parity import states
    z = np.load(path, allow_pickle=False)
    chain = states.decode_chain(states.unpack_records(z["blob"], z["offsets"]), json.loads(str(z["meta"])),
                                json.loads(str(z["names"])))
    return chain, z["obs"], z["snake"]


FIELD_TOL = {"progress": 0.0, "ahead": 1e-5, "ray": 1e-4, "head_ray": 1e-4}


def field_names(cfg):
    from goi_jax.game.observe import OBS_FIELDS
    from goi_env.client import AHEAD
    return (list(OBS_FIELDS) + [f"ahead{d}_{ax}" for d in AHEAD for ax in ("dx", "dy")]
            + [f"ray{i}" for i in range(cfg.n_rays)] + [f"head_ray{i}" for i in range(cfg.head_rays)])


def _group(name):
    for g in ("head_ray", "ray", "ahead"):
        if name.startswith(g):
            return g
    return name


def compare(data: Path, n_chains=0, progress=True, chains=None):
    """Per chain and step: the JAX observation of the recorded state vs the plugin's. Returns a dict of arrays:
    jax (S, n), game (S, n), snake_jax, snake_game, t0 (bool: the chain's first observation, whose previous progress
    is unknown), names."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import level as L
    from goi_jax.game import observe as O
    from goi_jax.parity import states
    index = json.loads((data / "index.json").read_text())
    c = index["config"]
    cfg = O.ObsConfig(c["n_rays"], c["ray_length"], c["head_rays"], c["head_length"])
    route_row = json.loads((data / "route.json").read_text())
    lv = L.load_level(data)
    entries = index["chains"][: n_chains or None] if chains is None else [e for e in index["chains"] if e["file"] in chains]
    om = None
    run = None
    out = dict(jax=[], game=[], snake_jax=[], snake_game=[], t0=[], chain=[], lake_jax=[], won_jax=[])
    flags = None
    for k, e in enumerate(entries):
        chain, obs, snake = load_obs_chain(data / e["file"])
        n = len(obs)
        steps = np.arange(n)
        jm, st, _ = lv.native_states(chain, steps, 0, capacity=max(1, int(chain.n_contacts[0][:n].max())), fat="tight")
        if om is None:
            om = O.build_obs_model(lv, route_row, L.jax_tree_index(jm, 0), reach=max(cfg.ray_length, cfg.head_length))

            def one(om, st, mi, ms, prev, lake, won):
                carry = O.ObsCarry(progress=prev, lake=lake, won=won, in_zone=jnp.zeros(om.solid.shape, bool))
                o, _, sn = O.observe(om, cfg, st, mi, ms, carry)
                return o, sn
            run = jax.jit(jax.vmap(one, in_axes=(None, 0, 0, 0, 0, 0, 0)))
        mi = np.stack([chain.field(0, "mouseInput.x"), chain.field(0, "mouseInput.y")], -1)[:n]
        ms = chain.field(0, "mouseVelocityAverage")[:n]
        prev = np.concatenate([obs[:1, 15], obs[:-1, 15]]).astype(np.float32)
        st = jax.tree.map(jnp.asarray, st)
        o, sn = run(om, st, jnp.asarray(mi), jnp.asarray(ms), jnp.asarray(prev), jnp.asarray(obs[:, 13] > 0),
                    jnp.asarray(obs[:, 14] > 0))
        out["jax"].append(np.asarray(o))
        out["game"].append(obs)
        out["snake_jax"].append(np.asarray(sn))
        out["snake_game"].append(snake)
        t0 = np.zeros(n, bool)
        t0[0] = True
        out["t0"].append(t0)
        out["chain"].append(np.full(n, k))
        # sticky flags: step_flags over the chain's physics steps (NAT0 -> NAT1 of record t gives observation t + 1)
        if flags is None:
            def flag_run(om, model, st0, st1, carry):
                def body(c, x):
                    c = O.step_flags(om, model, x[0], x[1], c)
                    return c, (c.lake, c.won)
                return jax.lax.scan(body, carry, (st0, st1))[1]
            flags = jax.jit(flag_run)
        _, st1, _ = lv.native_states(chain, steps, 1, capacity=max(1, int(chain.n_contacts[1][:n].max())), fat="tight")
        st1 = jax.tree.map(jnp.asarray, st1)
        st0 = st
        carry = O.init_carry(om, lv.model, jax.tree.map(lambda x: x[0], st0), obs[0, 15])
        carry = carry._replace(lake=jnp.asarray(obs[0, 13] > 0), won=jnp.asarray(obs[0, 14] > 0))
        lk, wn = flags(om, lv.model, st0, st1, carry)
        out["lake_jax"].append(np.concatenate([obs[:1, 13] > 0, np.asarray(lk)[:-1]]))
        out["won_jax"].append(np.concatenate([obs[:1, 14] > 0, np.asarray(wn)[:-1]]))
        if progress:
            print(f"{e['file']}: {n} observations", flush=True)
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["names"] = field_names(cfg)
    if (data / "guard.npz").exists():  # place() probes: Progress from progress = along at the placed pot
        from goi_jax.game import route as R
        rows = np.load(data / "guard.npz")["rows"]
        rm = om.route

        def guard(rm, along, x, y):
            p3 = (x, y, rm.player_z)
            raw = R.raw_progress(rm, p3)
            return R.guarded_progress(rm, along, p3), raw
        gj, raw = jax.jit(jax.vmap(guard, in_axes=(None, 0, 0, 0)))(rm, *(jnp.asarray(rows[:, k]) for k in range(3)))
        gj, raw = np.asarray(gj), np.asarray(raw)
        walked = raw > rows[:, 0] + np.float32(3)
        res["guard"] = dict(jax=gj, game=rows[:, 3], raw=raw, walked=walked)
    return res


def table(res) -> str:
    names = res["names"]
    J, G = res["jax"], res["game"]
    ok_prev = ~res["t0"]
    groups = {}
    for i, nm in enumerate(names):
        groups.setdefault(_group(nm), []).append(i)
    lines = [f"{'field':16s} {'n':>6s} {'max |err|':>11s} {'bit-exact':>10s} {'within tol':>11s}  tol"]
    for g, cols in groups.items():
        rows = ok_prev if g in ("progress", "ahead") else np.ones(len(J), bool)
        j, gm = J[rows][:, cols], G[rows][:, cols]
        err = np.abs(j.astype(np.float64) - gm)
        tol = FIELD_TOL.get(g, 1e-5)
        within = err <= max(tol, 0.0)
        lines.append(f"{g:16s} {err.size:6d} {err.max():11.3g} {np.mean(j == gm):10.2%} {np.mean(within):11.2%}  {tol:g}")
    sn = res["snake_jax"] == res["snake_game"]
    lines.append(f"{'snake (status)':16s} {sn.size:6d} {'':11s} {np.mean(sn):10.2%}   (set in {res['snake_game'].sum()})")
    for nm, col in (("lake (stepped)", 13), ("won (stepped)", 14)):
        ok = res[nm.split()[0] + "_jax"] == (G[:, col] > 0)
        lines.append(f"{nm:16s} {ok.size:6d} {'':11s} {np.mean(ok):10.2%}   (set in {(G[:, col] > 0).sum()})")
    if "guard" in res:
        g = res["guard"]
        lines.append(f"{'guard probes':16s} {len(g['game']):6d} {np.abs(g['jax'] - g['game']).max():11.3g} "
                     f"{np.mean(g['jax'] == g['game']):10.2%}   ({g['walked'].sum()} walked the guard, "
                     f"{(g['game'] != g['raw']).sum()} differ from the raw progress)")
    return "\n".join(lines)


def compare_main(data: Path, n_chains=0):
    try:
        res = compare(data, n_chains)
        print(table(res))
        np.savez_compressed(data / "compare.npz", **{k: v for k, v in res.items() if k not in ("names", "guard")},
                            names=np.array(res["names"]))
        print("DONE")
    except Exception as e:
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", default="dumps/obs")
    r.add_argument("--max-seconds", type=float, default=420.0)
    x = sub.add_parser("record-extra")
    x.add_argument("--out", default="dumps/obs")
    x.add_argument("--max-seconds", type=float, default=420.0)
    y = sub.add_parser("record-targets")
    y.add_argument("--out", default="dumps/obs")
    y.add_argument("--max-seconds", type=float, default=420.0)
    y.add_argument("--won-only", action="store_true", help="only the space-zone exits, from just below its top")
    c = sub.add_parser("compare")
    c.add_argument("--data", default="dumps/obs")
    c.add_argument("--chains", type=int, default=0, help="only the first N chains (0: all)")
    args = ap.parse_args(argv)
    if args.cmd == "record":
        record(Path(args.out), args.max_seconds)
    elif args.cmd == "record-targets":
        record_targets(Path(args.out), args.max_seconds, args.won_only)
    elif args.cmd == "record-extra":
        record_extra(Path(args.out), args.max_seconds)
    else:
        compare_main(Path(args.data), args.chains)


if __name__ == "__main__":
    sys.exit(main())

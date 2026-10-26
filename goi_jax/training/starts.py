"""Where training episodes start.

Bank ("bottom" starts): recorded game states, SimState pre-step states of the parity chains, every `start_every`
recorded physics steps from each chain's first canonical step. start_group:
    coffee   Coffee Rock (cfg.parity_dir: 24 chains placed on the route at progress 62-118, varied mouse input)
    start    the level start (cfg.parity_dir, 5 chains)
    route    the whole route (cfg.route_dir, goi_jax/parity/record_route.py: the climber placed every 5 m from 0 to
             864 the way upstream places it, up to 4 placement variants per spot, settled 30 actions, then 8 K=4
             actions recorded; plus the level start)
    all      coffee + start + route
    searched / <group>+searched   the states of a searched-start bank (Go-Explore search in the sim,
             goi_jax/training/search.py, scripts/search_starts.py; file: cfg.extra["searched_bank"],
             $GOI_SEARCHED_BANK or runs/search/bank.pkl), alone or added to a recorded group; drawn like any bank
             state (by section / "weak")
Built once on the host and cached under runs/.cache/; make_env then leaves out (bank_keep) the cfg.bank_skip sections
(default "": none; space is modelled, goi_jax/game/space.py) and route states in free flight (cfg.route_max_air_speed). Every bank state carries its route progress (the JAX
ProgressMeter at that state) and its section (scripts/sections.py SECTIONS; below 9 m: "start").

Which bank state an episode starts from (start_sampling):
    entry    uniform over bank states (the old behaviour; a section's share = its share of the bank)
    section  a section uniformly (among sections with states), then a state of it uniformly
    weak     a section with weight (1 - p) + 0.05, p = (crossed + 0.5) / (tries + 1) over the recent episodes started
             from it (crossed: the held best progress reached the next section's start, as scripts/sections.py
             counts it; counts decay by sec_decay at every step that ends episodes), then a state of it uniformly.
             Per device, on device (Archive.sec_tries / sec_wins).

Archive (upstream train.py Starts' saved starts, Go-Explore style): one SimState per bin_size of route progress, saved
the first time an episode of this device gets there with the pot or the hammer touching something and the episode
not ending; a `saved_starts` share of resets draw a filled bin (inside `window`) with weight 1/sqrt(1 + times drawn).
All on device: fixed (archive_bins, ...) arrays, one writer per bin per step.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import env as E
from goi_jax.game import route as R

CACHE = Path("runs/.cache")
# scripts/sections.py SECTIONS: route progress where each section starts (index 0: below the first, "start").
SECTIONS = [(0, "start"), (9, "lakeside"), (60, "rocks"), (120, "tree and slope"), (176, "chimney"),
            (204, "Construction Wall"), (228, "girder"), (245, "hut"), (256, "garden"), (290, "playground"),
            (310, "boxes"), (335, "furniture"), (385, "orange stretch"), (415, "chapel"), (465, "roofs"),
            (525, "anvil"), (560, "snow"), (630, "bucket"), (660, "Ice Mountain"), (735, "radio mast"), (760, "space"),
            (864, "win")]
SECTION_LO = np.array([lo for lo, _ in SECTIONS], np.float32)
SECTION_NEXT = np.array([lo for lo, _ in SECTIONS[1:]] + [np.inf], np.float32)   # crossing goal per section
NS = len(SECTIONS)
GROUPS = ("coffee", "start", "route", "all")


def section_index(progress):
    """Section index (into SECTIONS) of route progress; numpy or jax."""
    xp = jnp if isinstance(progress, jax.Array) else np
    return xp.clip(xp.searchsorted(xp.asarray(SECTION_LO), progress, side="right") - 1, 0, NS - 1)


def bank_entries(root, group, every=None, route_dir="dumps/parity-route"):
    """[(root, index entry)] of the chains a start group draws from."""
    from goi_jax.parity import states as S
    if group not in GROUPS:
        raise ValueError(f"start_group {group!r}: one of {GROUPS}")
    out = []
    if group in ("coffee", "start", "all"):
        out += [(Path(root), e) for e in S.load_index(root)["chains"]
                if (group == "all" and e["group"] in ("coffee", "start")) or e["group"] == group]
    if group in ("route", "all"):
        if not (Path(route_dir) / "index.json").exists():
            raise ValueError(f"start_group {group!r} needs the route recordings in {route_dir} "
                             "(goi_jax/parity/record_route.py)")
        out += [(Path(route_dir), e) for e in S.load_index(route_dir)["chains"] if e["group"] == "route"]
    return out


def _bank_init():
    """Worker setup: CPU only, and XLA's persistent cache for every compile (Level.track_fat jits a fresh closure
    per chain; with the cache a chain costs ~0.7 s instead of ~3 s)."""
    jax.config.update("jax_platforms", "cpu")
    jax.config.update("jax_compilation_cache_dir", str(Path("runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


def world_ok(chain, t) -> bool:
    """The recorded b2World fields at NAT0 of step t are sane. Some route recordings (12 of 684 chains) carry a
    garbage world block for the whole chain (bodyCount 0 or ~1e9, gravity (0, 0) / (0, 272) / NaN, inv_dt0 0): the
    loader would hand the sim that gravity and dt ratio, so such states are left out of the bank."""
    w = chain.world[0]
    g = w["gravity"][t]
    return bool(0 < int(w["bodyCount"][t]) < 1000 and abs(float(w["inv_dt0"][t]) - 120.0) < 0.01
                and np.isfinite(g).all() and g[0] == 0 and g[1] in (0.0, -30.0))


def _states_of(sim, files, every, capacity):
    from goi_jax.game import sim as SM
    from goi_jax.parity import states as S
    states, where = [], []
    for f in files:
        chain = S.load_chain(f)
        for t in range(SM.canonical_step(chain), len(chain.action), every):
            if not world_ok(chain, t):
                continue
            try:
                states.append(sim.state_from_recording(chain, t, capacity))
                where.append((str(f), t))
            except ValueError:   # a step whose joint order is not the canonical one
                continue
    return states, where


def _bank_part(args):
    """Bank states of some chains (a worker process): ([SimState], [(file, step)])."""
    sim_root, files, every, capacity = args
    from goi_jax.game import sim as SM
    return _states_of(SM.Sim.from_recordings(sim_root), files, every, capacity)


def build_bank(sim, root="dumps/parity", group="coffee", every=10, capacity=96, cache=True,
               route_dir="dumps/parity-route", workers=8):
    """(SimState with a leading bank axis as numpy, [(root/file, step)]). Many chains: built in `workers` CPU
    processes (each loads the level from `root`; the route recordings' level is the same, checked by byte compare of
    the loaded fixtures)."""
    entries = bank_entries(root, group, every, route_dir)
    if not entries:
        raise ValueError(f"no recorded chains of group {group!r} in {root}")
    roots = sorted({str(r.resolve()) for r, _ in entries})
    stamp = json.dumps([2, group, every, capacity, roots, [(Path(r) / "index.json").stat().st_mtime_ns for r in roots]])
    path = CACHE / f"bank-{group}-{every}-{hashlib.sha1(stamp.encode()).hexdigest()[:10]}.pkl"
    if cache and path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    files = [r / e["file"] for r, e in entries]
    if len(files) < 40 or workers <= 1:
        states, where = _states_of(sim, files, every, capacity)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        parts = [(str(root), files[i::workers], every, capacity) for i in range(workers)]
        import sys
        main = sys.modules["__main__"]
        saved = {k: main.__dict__[k] for k in ("__file__", "__spec__") if k in main.__dict__}
        main.__dict__.pop("__file__", None)
        main.__spec__ = None
        try:   # spawned workers must not re-run the caller's script (they only need this module)
            with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_bank_init) as ex:
                got = list(ex.map(_bank_part, parts))
        finally:
            main.__dict__.update(saved)
        order = {str(f): i for i, f in enumerate(files)}
        pairs = sorted(((w, s) for st, wh in got for s, w in zip(st, wh)), key=lambda x: (order[x[0][0]], x[0][1]))
        states, where = [s for _, s in pairs], [w for w, _ in pairs]
    bank = jax.tree.map(lambda *a: np.stack(a), *states)
    out = (bank, where)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def bank_progress(env, bank, chunk=256):
    """(N,) route progress (ProgressMeter, the observation's) of every bank state, on the default device."""
    rm = env.default_params.obs.route
    pot = env.sim.pot
    f = jax.jit(jax.vmap(lambda p: R.raw_progress(rm, (p[0], p[1], rm.player_z))))
    xy = np.asarray(bank.world.bodies.xf_p)[:, pot]
    return np.concatenate([np.asarray(f(jnp.asarray(xy[i:i + chunk]))) for i in range(0, len(xy), chunk)])


class BankIndex(NamedTuple):
    """Bank states grouped by section, for two-stage draws (host constants, closed over by the trainer)."""
    order: np.ndarray       # (N,) bank indices sorted by section
    start: np.ndarray       # (NS,) first position of each section in `order`
    count: np.ndarray       # (NS,) bank states per section
    section: np.ndarray     # (N,) section of each bank state
    progress: np.ndarray    # (N,) route progress of each bank state


def bank_index(progress) -> BankIndex:
    sec = section_index(np.asarray(progress, np.float32)).astype(np.int32)
    order = np.argsort(sec, kind="stable").astype(np.int32)
    count = np.bincount(sec, minlength=NS).astype(np.int32)
    start = (np.cumsum(count) - count).astype(np.int32)
    return BankIndex(order, start, count, sec, np.asarray(progress, np.float32))


def section_logits(bi: BankIndex, mode, tries=None, wins=None):
    """(NS,) log weights of the sections a bank draw picks from (-inf: no states)."""
    have = jnp.asarray(bi.count) > 0
    if mode == "weak" and tries is not None:
        p = (wins + 0.5) / (tries + 1.0)
        w = jnp.log((1.0 - p) + 0.05)
    else:
        w = jnp.zeros(NS, jnp.float32)
    return jnp.where(have, w, -jnp.inf)


def bank_draw(bi: BankIndex, key, n, mode="section", tries=None, wins=None):
    """(n,) bank indices and their sections, by start_sampling `mode` (entry | section | weak)."""
    k1, k2 = jax.random.split(key)
    N = len(bi.section)
    if mode == "entry":
        i = jax.random.randint(k1, (n,), 0, N)
        return i, jnp.asarray(bi.section)[i]
    s = jax.random.categorical(k1, section_logits(bi, mode, tries, wins), shape=(n,))
    cnt = jnp.asarray(bi.count)[s]
    j = jnp.minimum((jax.random.uniform(k2, (n,)) * cnt).astype(jnp.int32), jnp.maximum(cnt - 1, 0))
    return jnp.asarray(bi.order)[jnp.asarray(bi.start)[s] + j], s


def section_update(ar, ep_sec, done, held_best, decay):
    """Archive with the section stats updated for the episodes ending now (ep_sec >= 0: started from the bank)."""
    if ar.sec_tries is None:
        return ar
    m = done & (ep_sec >= 0)
    s = jnp.where(m, ep_sec, NS)
    won = m & (held_best >= jnp.asarray(SECTION_NEXT)[jnp.clip(ep_sec, 0, NS - 1)])
    d = jnp.where(jnp.any(m), jnp.float32(decay), jnp.float32(1.0))
    tries = (ar.sec_tries * d).at[s].add(1.0, mode="drop")
    wins = (ar.sec_wins * d).at[jnp.where(won, s, NS)].add(1.0, mode="drop")
    return ar._replace(sec_tries=tries, sec_wins=wins)


def make_env(cfg, axis_name="w"):
    """GoiEnv with cfg's reward / episode settings and the start bank as its `starts`."""
    one = None
    from goi_jax.parity import states as S
    for e in S.load_index(cfg.parity_dir)["chains"]:
        if e["group"] == "start" or one is None:
            one = [(e["file"], None)]
            if e["group"] == "start":
                break
    env = E.GoiEnv.from_recordings(cfg.parity_dir, cfg.route, starts=one, k=cfg.k, capacity=cfg.contact_capacity,
                                   held_records=cfg.held_records, stall_ends=True, snake_ends=cfg.snake_ends,
                                   axis_name=axis_name, time_penalty=cfg.time_penalty, action_cost=cfg.action_cost,
                                   mouse=cfg.mouse, recovery=cfg.recovery, regress=cfg.regress,
                                   air_bonus=cfg.air_bonus, speed_bonus=cfg.speed_bonus,
                                   stall_steps=cfg.stall_steps if cfg.stall_seconds > 0 else 0,
                                   max_steps=cfg.episode_steps,
                                   compact_fixtures=getattr(cfg, "compact_fixtures", False))
    bank, where = group_bank(env, cfg)
    progress = bank_progress(env, bank)
    keep = bank_keep(env, cfg, bank, where, progress)
    if not keep.any():
        raise ValueError(f"start_group {cfg.start_group!r}: every bank state filtered out (bank_skip {cfg.bank_skip!r})")
    if not keep.all():
        bank = jax.tree.map(lambda a: a[keep], bank)
        where = [w for w, k in zip(where, keep) if k]
        progress = progress[keep]
    env.default_params = env.default_params._replace(starts=jax.tree.map(jnp.asarray, bank))
    env.bank_where = where
    env.bank_index = bank_index(progress)
    env.bank_dropped = int((~keep).sum())
    return env


SEARCHED_BANK = "runs/search/bank.pkl"   # default searched-start bank (scripts/search_starts.py --out)


def split_group(group):
    """start_group -> (recorded group or None, searched?): "all" -> ("all", False), "all+searched" -> ("all", True),
    "searched" -> (None, True)."""
    parts = [g.strip() for g in str(group).split("+") if g.strip()]
    searched = "searched" in parts
    rest = [g for g in parts if g != "searched"]
    if len(rest) > 1 or (not rest and not searched):
        raise ValueError(f"start_group {group!r}: one of {GROUPS}, optionally '+searched', or 'searched'")
    return (rest[0] if rest else None), searched


def searched_bank_path(cfg):
    """The searched-start bank file: cfg.extra["searched_bank"], else $GOI_SEARCHED_BANK, else SEARCHED_BANK."""
    extra = getattr(cfg, "extra", None) or {}
    return Path(extra.get("searched_bank") or os.environ.get("GOI_SEARCHED_BANK") or SEARCHED_BANK)


def group_bank(env, cfg):
    """(bank SimState numpy with leading N, where) of cfg.start_group: build_bank's recorded states, plus (start_group
    "<group>+searched" or "searched") the states of a searched-start bank (goi_jax/training/search.py save_bank;
    where = ("searched:<file>", cell)). Both are put in the env's fixture layout before they are joined."""
    base, searched = split_group(cfg.start_group)
    parts, where = [], []
    if base is not None:
        b, w = build_bank(env.sim, cfg.parity_dir, base, cfg.start_every, cfg.contact_capacity, route_dir=cfg.route_dir)
        parts.append(b)
        where += list(w)
    if searched:
        path = searched_bank_path(cfg)
        if not path.exists():
            raise ValueError(f"start_group {cfg.start_group!r} needs a searched-start bank at {path} "
                             "(scripts/search_starts.py)")
        with open(path, "rb") as f:
            sb = pickle.load(f)
        parts.append(sb["states"])
        where += [(f"searched:{path}", int(c)) for c in sb["cell"]]
    if len(parts) == 1:
        return parts[0], where
    parts = [env.sim.compact_state(b) for b in parts]
    try:
        bank = jax.tree.map(lambda *a: np.concatenate([np.asarray(x) for x in a]), *parts)
    except ValueError as e:   # e.g. another contact_capacity, or a compact searched bank for a full-layout env
        raise ValueError(f"searched-start bank {searched_bank_path(cfg)} does not match the recorded bank's layout "
                         f"(compact_fixtures / contact_capacity): {e}") from e
    return bank, where


def bank_keep(env, cfg, bank, where, progress, chunk=256):
    """(N,) bool: bank states kept. Left out: states in cfg.bank_skip sections, and route-recording states in free
    flight (pot and hammer head touching nothing and the pot faster than cfg.route_max_air_speed): about a quarter of
    the route placements are mid-fall when recorded (placed over a drop or in space at gravity -30), not settled
    starts. Coffee / start chains (recorded with varied input) keep their airborne states."""
    sec = section_index(np.asarray(progress, np.float32))
    skip = {s.strip() for s in (cfg.bank_skip or "").split(",") if s.strip()}
    unknown = skip - {n for _, n in SECTIONS}
    if unknown:
        raise ValueError(f"bank_skip: unknown sections {sorted(unknown)}")
    keep = ~np.isin(sec, [i for i, (_, n) in enumerate(SECTIONS) if n in skip])
    route = np.array([Path(w[0]).resolve().is_relative_to(Path(cfg.route_dir).resolve()) for w in where])
    if cfg.route_max_air_speed > 0 and route.any():
        p = env.default_params
        f = jax.jit(jax.vmap(lambda s: reset_from(env, p, s)[0]))
        n = len(where)
        obs = []
        for i in range(0, n, chunk):
            idx = np.r_[i:min(i + chunk, n)]
            idx = np.r_[idx, np.full(chunk - len(idx), idx[-1])]   # one shape, one compile
            obs.append(np.asarray(f(jax.tree.map(lambda a: jnp.asarray(a[idx]), bank)))[:min(chunk, n - i)])
        obs = np.concatenate(obs)
        held = (obs[:, E.F["pot_contact"]] > 0) | (obs[:, E.F["head_contact"]] > 0)
        speed = np.linalg.norm(np.asarray(bank.world.bodies.v)[:, env.sim.pot], axis=-1)
        keep &= ~(route & ~held & (speed > cfg.route_max_air_speed))
    return keep


def reset_from(env: E.GoiEnv, p: E.EnvParams, s):
    """GoiEnv.reset for a given SimState s (the same code, without drawing from p.starts)."""
    rm = p.obs.route
    bd = s.world.bodies
    prog = R.raw_progress(rm, (bd.xf_p[env.sim.pot, 0], bd.xf_p[env.sim.pot, 1], rm.player_z))
    from goi_jax.game import observe as O
    carry = O.init_carry(p.obs, p.model, s.world, prog)
    raw, carry, _ = env._observe(p, s, carry)
    pr = raw[E.F["progress"]]
    i32, f32 = jnp.int32, jnp.float32
    st = E.EnvState(sim=s, carry=carry, best_progress=pr, held_best=pr, progress=pr, stall_mark=pr,
                    stalled=i32(0), air=i32(0), t=i32(0), actions=jnp.zeros((E.PREV_ACTIONS, 2), f32))
    return env._full_obs(p, raw, st), st


# ---------------------------------------------------------------- archive


class Archive(NamedTuple):
    states: object          # SimState with leading (nb,)
    filled: jax.Array       # (nb,) bool
    starts: jax.Array       # (nb,) f32 times drawn
    progress: jax.Array     # (nb,) f32 progress when saved
    sec_tries: jax.Array = None   # (NS,) f32 decayed bank episodes started per section (start_sampling "weak")
    sec_wins: jax.Array = None    # (NS,) f32 of those, decayed count that crossed into the next section


def archive_init(bank, nb):
    one = jax.tree.map(lambda a: a[0], bank)
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (nb,) + a.shape), one)
    z = jnp.zeros(NS, jnp.float32)
    return Archive(states, jnp.zeros(nb, bool), jnp.zeros(nb, jnp.float32), jnp.zeros(nb, jnp.float32), z, z)


def archive_complete(ar: Archive) -> Archive:
    """An archive from an older checkpoint (no section stats) with zeroed ones."""
    if ar.sec_tries is None:
        z = np.zeros(NS, np.float32)
        ar = ar._replace(sec_tries=z, sec_wins=z)
    return ar


def archive_bins_ok(cfg, nb):
    """(nb,) bool: bins inside cfg.window."""
    lo, hi = cfg.window or (-np.inf, np.inf)
    b = np.arange(nb) * cfg.bin_size
    return jnp.asarray((b >= lo) & (b < hi))


def archive_save(ar: Archive, cfg, nb, sims, progress, held, ok):
    """Save each env's post-action SimState into its progress bin where (held, ok, bin empty); one writer per bin."""
    n = progress.shape[0]
    b = jnp.clip(jnp.floor(progress / cfg.bin_size).astype(jnp.int32), 0, nb - 1)
    want = ok & held & ~ar.filled[b] & (progress >= 0)
    slot = jnp.where(want, b, nb)
    winner = jnp.full(nb + 1, -1, jnp.int32).at[slot].max(jnp.arange(n, dtype=jnp.int32))
    write = want & (winner[slot] == jnp.arange(n))
    idx = jnp.where(write, b, nb)                                   # nb: dropped
    states = jax.tree.map(lambda A, s: A.at[idx].set(s, mode="drop"), ar.states, sims)
    filled = ar.filled.at[idx].set(True, mode="drop")
    prog = ar.progress.at[idx].set(progress, mode="drop")
    return ar._replace(states=states, filled=filled, progress=prog)


def archive_draw(ar: Archive, key, bins_ok, n):
    """(n,) bin indices drawn with weight 1/sqrt(1 + starts) over filled bins in the window; (n,) valid."""
    usable = ar.filled & bins_ok
    logits = jnp.where(usable, -0.5 * jnp.log1p(ar.starts), -jnp.inf)
    b = jax.random.categorical(key, logits, shape=(n,))
    return b, jnp.broadcast_to(usable.any(), (n,))

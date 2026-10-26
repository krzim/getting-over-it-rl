"""Go-Explore style start-state search in the simulator: save/load is free (a SimState is a pytree of arrays), so
instead of learning to get somewhere, branch many envs from states already reached and keep whatever gets further.

Archive: one SimState per cell, cell = (route-progress bin of `bin_size` m, hammer-direction sector of
`angle_cells`). Every cell maps to the section (starts.SECTIONS) its bin starts in. A cell keeps its best state:
the furthest progress (by more than `tol` m), else, within `tol`, the one reached in the fewest actions from the
lineage's seed (`time`). Only physically valid, settled states are saved (`state_ok` below: finite bodies and
contact impulses, no contact-slot overflow, positions inside +-1e4, pot speed under `max_speed`, the pot or the
hammer head touching something for `min_held` consecutive actions, episode not terminated by lake / win / Snake).

One round (all on device, one jit; `rounds` rounds per device call in a lax.fori_loop):
  1. pick `branches` cells: weight 1/sqrt(1 + times chosen), x (1 + frontier_bonus) where the next bin has no cell
     yet (a frontier), optionally limited to a progress window; pick "section" gives every section with cells the
     same total weight, pick "cell" weights cells directly;
  2. reset every branch from its cell (starts.reset_from: the env's observation, ray casts included);
  3. `horizon` actions of GoiEnv.step_env, vmapped over the branches (axis "w" like the trainer): action =
     use_policy * policy mean + OU noise (per branch a sigma from the set and a correlation rho), clipped to [-1, 1];
     a branch that dies (lake, win, Snake, invalid state) restarts from its own cell with fresh noise;
  4. after every action, valid settled states are offered to the archive (one writer per cell per step).
The policy (if any) is a training checkpoint's network + observation normaliser, passed as arguments, so one
compiled program serves "random smooth mouse" (use_policy 0) and "policy plus noise" (use_policy 1).

Output (save_bank): a pickle {"states": SimState (N, ...) numpy, "progress", "section", "time", "root", "cell",
"chosen", "meta"} of the filled cells; starts.make_env loads it for start_group "searched" / "<group>+searched".
"""
from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import env as E
from goi_jax.training import nets as N
from goi_jax.training import normalize as Nz
from goi_jax.training import starts as St

F = E.F
WA = "w"


@dataclass(frozen=True)
class SearchConfig:
    bin_size: float = 2.0
    angle_cells: int = 4             # hammer-direction sectors per bin
    max_progress: float = 760.0      # no cells from here on ("space": no zero gravity / wells in the sim)
    branches: int = 8192             # envs per round
    horizon: int = 96                # actions per round (k=4: 3.2 s of game time)
    rounds: int = 1                  # rounds per device call
    min_held: int = 2                # consecutive actions touching something before a state may be saved
    max_speed: float = 6.0           # pot speed (m/s) above which a state is not saved
    tol: float = 0.5                 # progress within tol (m) counts as equal; then fewer actions wins
    frontier_bonus: float = 3.0
    pick: str = "section"            # section | cell
    rand_sigmas: tuple = (0.3, 0.6, 1.0)     # OU noise std per branch without a policy
    pol_sigmas: tuple = (0.05, 0.15, 0.35)   # with a policy (its own std collapsed to ~0.06)
    rho: tuple = (0.8, 0.97)         # OU correlation per action, uniform in this range per branch

    @property
    def n_bins(self):
        return int(np.ceil(self.max_progress / self.bin_size))

    @property
    def n_cells(self):
        return self.n_bins * self.angle_cells


class Archive(NamedTuple):
    states: object          # SimState (NC, ...)
    filled: jax.Array       # (NC,) bool
    progress: jax.Array     # (NC,) f32 route progress of the saved state
    time: jax.Array         # (NC,) i32 actions from the lineage's seed
    chosen: jax.Array       # (NC,) f32 times picked
    root: jax.Array         # (NC,) i32 section of the lineage's seed
    found: jax.Array        # (NC,) i32 round the cell was first filled (-1: seed)


class Policy(NamedTuple):
    params: dict
    rms: Nz.RMS
    use: jax.Array          # f32 0 (random smooth mouse) or 1 (policy mean + noise)


def cell_section(sc: SearchConfig):
    """(NC,) section index of every cell (the section its bin starts in)."""
    lo = (np.arange(sc.n_cells) // sc.angle_cells) * sc.bin_size
    return St.section_index(lo.astype(np.float32)).astype(np.int32)


def cell_of(sc: SearchConfig, progress, hcos, hsin):
    b = jnp.clip(jnp.floor(progress / sc.bin_size).astype(jnp.int32), 0, sc.n_bins - 1)
    ang = jnp.arctan2(hsin, hcos)
    a = jnp.clip(jnp.floor((ang + jnp.pi) / (2 * jnp.pi) * sc.angle_cells).astype(jnp.int32), 0, sc.angle_cells - 1)
    return b * sc.angle_cells + a


def archive_init(sc: SearchConfig, one_state) -> Archive:
    nc = sc.n_cells
    states = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (nc,) + np.shape(a)), one_state)
    return Archive(states, jnp.zeros(nc, bool), jnp.zeros(nc, jnp.float32), jnp.zeros(nc, jnp.int32),
                   jnp.zeros(nc, jnp.float32), jnp.full(nc, -1, jnp.int32), jnp.full(nc, -1, jnp.int32))


def archive_update(ar: Archive, sc: SearchConfig, sims, cell, progress, time, root, cand, round_i=-1):
    """Offer n states (sims with leading n) to their cells; returns (archive, number of newly filled cells).
    Replacement: empty cell, or progress > old + tol, or progress >= old - tol with fewer actions. Among the
    candidates of one cell the furthest wins (ties: the highest lane index)."""
    nc = sc.n_cells
    n = progress.shape[0]
    old_p, old_t, filled = ar.progress[cell], ar.time[cell], ar.filled[cell]
    better = cand & (~filled | (progress > old_p + sc.tol) | ((progress >= old_p - sc.tol) & (time < old_t)))
    slot = jnp.where(better, cell, nc)
    top = jnp.full(nc + 1, -jnp.inf, jnp.float32).at[slot].max(progress)
    is_top = better & (progress == top[slot])
    lane = jnp.arange(n, dtype=jnp.int32)
    winner = jnp.full(nc + 1, -1, jnp.int32).at[jnp.where(is_top, slot, nc)].max(lane)
    write = is_top & (winner[slot] == lane)
    idx = jnp.where(write, cell, nc)
    new = jnp.sum((write & ~filled).astype(jnp.int32))
    states = jax.tree.map(lambda A, s: A.at[idx].set(s, mode="drop"), ar.states, sims)
    found = ar.found.at[jnp.where(write & ~filled, cell, nc)].set(jnp.int32(round_i), mode="drop")
    return ar._replace(states=states, filled=ar.filled.at[idx].set(True, mode="drop"),
                       progress=ar.progress.at[idx].set(progress, mode="drop"),
                       time=ar.time.at[idx].set(time.astype(jnp.int32), mode="drop"),
                       root=ar.root.at[idx].set(root.astype(jnp.int32), mode="drop"), found=found), new


def pick_logits(ar: Archive, sc: SearchConfig, cell_sec, lo, hi):
    """(NC,) log weights of the cells a round branches from (-inf: not pickable)."""
    nb, A = sc.n_bins, sc.angle_cells
    bin_filled = ar.filled.reshape(nb, A).any(-1)
    nxt = jnp.concatenate([bin_filled[1:], jnp.ones(1, bool)])           # the last bin has no frontier
    frontier = jnp.repeat(~nxt, A)
    w = (1.0 + sc.frontier_bonus * frontier) / jnp.sqrt(1.0 + ar.chosen)
    ok = ar.filled & (ar.progress >= lo) & (ar.progress < hi)
    w = jnp.where(ok, w, 0.0)
    if sc.pick == "section":
        tot = jax.ops.segment_sum(w, cell_sec, num_segments=St.NS)
        w = w / jnp.maximum(tot[cell_sec], 1e-12)
    return jnp.where(w > 0, jnp.log(jnp.maximum(w, 1e-30)), -jnp.inf)


def state_ok(sim, pot, max_speed):
    """Physically valid: finite bodies and contact impulses, no overflowed contacts, sane positions, pot speed."""
    bd, ct = sim.world.bodies, sim.world.contacts
    fin = jnp.array(True)
    for x in (bd.xf_p, bd.v, bd.a, bd.w, ct.normal_impulse, ct.tangent_impulse):
        fin = fin & jnp.all(jnp.isfinite(x))
    pos = jnp.all(jnp.abs(bd.xf_p) < 1e4)
    speed = jnp.sqrt(jnp.sum(bd.v[pot] ** 2))
    return fin & pos & (sim.world.overflow == 0), speed < max_speed


class Searcher:
    """The device program for one (env, SearchConfig). env: starts.make_env's GoiEnv (axis_name "w")."""

    def __init__(self, env: E.GoiEnv, sc: SearchConfig, bounded_mean=True, clip_obs=10.0, norm_obs=True):
        if sc.pick not in ("section", "cell"):
            raise ValueError(f"pick {sc.pick!r}: section | cell")
        if len(sc.rand_sigmas) != len(sc.pol_sigmas):
            raise ValueError("rand_sigmas and pol_sigmas need the same length")
        self.env, self.sc = env, sc
        self.bounded_mean, self.clip_obs, self.norm_obs = bounded_mean, clip_obs, norm_obs
        self.cell_sec = jnp.asarray(cell_section(sc))
        # the env's params with a one-state reset bank: reset_from never reads it, and the program's shapes then do
        # not depend on the bank
        dp = env.default_params
        self.p = dp._replace(starts=jax.tree.map(lambda a: jnp.asarray(a[:1]), dp.starts))
        self._round = jax.jit(self._call, donate_argnums=(0,))
        self._seed = jax.jit(self._seed_chunk, donate_argnums=(0,))

    # ---------------------------------------------------------------- seeding

    def _seed_chunk(self, ar, sims, mask, p):
        sc, env = self.sc, self.env
        obs, est = jax.vmap(lambda s: St.reset_from(env, p, s))(sims)
        prog = est.progress
        ok, _ = jax.vmap(lambda s: state_ok(s, env.sim.pot, jnp.inf))(sims)
        cand = mask & ok & (prog >= 0) & (prog < sc.max_progress)
        cell = cell_of(sc, prog, obs[:, F["hammer_cos"]], obs[:, F["hammer_sin"]])
        root = St.section_index(prog).astype(jnp.int32)
        ar, new = archive_update(ar, sc, sims, cell, prog, jnp.zeros_like(cell), root, cand, -1)
        return ar

    def seed(self, ar: Archive, states, chunk=256) -> Archive:
        """Insert host states (SimState with leading N, numpy) as seeds (time 0, root = their own section)."""
        p = self.p
        n = jax.tree.leaves(states)[0].shape[0]
        for i in range(0, n, chunk):
            idx = np.r_[i:min(i + chunk, n)]
            m = np.zeros(chunk, bool)
            m[:len(idx)] = True
            idx = np.r_[idx, np.full(chunk - len(idx), idx[-1])]
            ar = self._seed(ar, jax.tree.map(lambda a: jnp.asarray(a[idx]), states), jnp.asarray(m), p)
        return ar

    # ---------------------------------------------------------------- one round

    def _act(self, pol: Policy, obs):
        x = Nz.normalize_obs(pol.rms, obs, self.clip_obs) if self.norm_obs else obs
        # section-aware checkpoints (TrainConfig.section_input) need the section ids of the raw observation
        mean, _, _ = N.apply(pol.params, x, self.bounded_mean, N.section_of(pol.params, obs))
        return mean * pol.use

    def _one_round(self, ar: Archive, key, round_i, pol: Policy, lo, hi, p):
        sc, env, B = self.sc, self.env, self.sc.branches
        k_pick, k_sig, k_rho, k_run = jax.random.split(key, 4)
        logits = pick_logits(ar, sc, self.cell_sec, lo, hi)
        cells = jax.random.categorical(k_pick, logits, shape=(B,))
        live = jnp.isfinite(logits[cells])                       # False only if nothing is pickable
        ar = ar._replace(chosen=ar.chosen.at[jnp.where(live, cells, sc.n_cells)].add(1.0, mode="drop"))
        s0 = jax.tree.map(lambda a: a[cells], ar.states)
        t0, root = ar.time[cells], ar.root[cells]
        obs0, est0 = jax.vmap(lambda s: St.reset_from(env, p, s))(s0)
        sig_set = jnp.where(pol.use > 0, jnp.asarray(sc.pol_sigmas, jnp.float32),
                            jnp.asarray(sc.rand_sigmas, jnp.float32))
        sig = sig_set[jax.random.randint(k_sig, (B,), 0, len(sc.rand_sigmas))][:, None]
        rho = jax.random.uniform(k_rho, (B, 1), minval=sc.rho[0], maxval=sc.rho[1])
        step = jax.vmap(env.step_env, (0, 0, 0, None), axis_name=WA)
        pot = env.sim.pot

        def body(c, k):
            ar, obs, est, u, t, held_run, st = c
            k_n, k_s = jax.random.split(k)
            u = rho * u + jnp.sqrt(1 - rho * rho) * sig * jax.random.normal(k_n, (B, 2))
            a = jnp.clip(self._act(pol, obs) + u, -1.0, 1.0)
            obs2, est2, _, term, _, _ = step(jax.random.split(k_s, B), est, a, p)
            ok, slow = jax.vmap(lambda s: state_ok(s, pot, sc.max_speed))(est2.sim)
            held = (obs2[:, F["pot_contact"]] > 0) | (obs2[:, F["head_contact"]] > 0)
            held_run = jnp.where(held, held_run + 1, 0)
            t = t + 1
            prog = est2.progress
            cand = (live & ok & ~term & slow & (held_run >= sc.min_held) & (prog >= 0) & (prog < sc.max_progress))
            cell = cell_of(sc, prog, obs2[:, F["hammer_cos"]], obs2[:, F["hammer_sin"]])
            ar, new = archive_update(ar, sc, est2.sim, cell, prog, t0 + t, root, cand, round_i)
            won = obs2[:, F["won"]] > 0
            dead = term | ~ok
            st = dict(new=st["new"] + new, deaths=st["deaths"] + jnp.sum(dead.astype(jnp.int32)),
                      invalid=st["invalid"] + jnp.sum((~ok).astype(jnp.int32)),
                      wins=st["wins"] + jnp.sum(won.astype(jnp.int32)),
                      best=jnp.maximum(st["best"], jnp.max(jnp.where(ok & live, prog, -jnp.inf))))
            est = _tree_sel(dead, est0, est2)
            obs = jnp.where(dead[:, None], obs0, obs2)
            u = jnp.where(dead[:, None], 0.0, u)
            t = jnp.where(dead, 0, t)
            held_run = jnp.where(dead, 0, held_run)
            return (ar, obs, est, u, t, held_run, st), None

        z = jnp.int32(0)
        st0 = dict(new=z, deaths=z, invalid=z, wins=z, best=jnp.float32(-jnp.inf))
        c0 = (ar, obs0, est0, jnp.zeros((B, 2), jnp.float32), jnp.zeros(B, jnp.int32), jnp.zeros(B, jnp.int32), st0)
        (ar, *_, st), _ = jax.lax.scan(body, c0, jax.random.split(k_run, sc.horizon))
        return ar, st

    def _call(self, ar, key, round0, pol, lo, hi, p):
        def one(i, c):
            ar, st = c
            ar, s = self._one_round(ar, jax.random.fold_in(key, i), round0 + i, pol, lo, hi, p)
            st = dict(new=st["new"] + s["new"], deaths=st["deaths"] + s["deaths"], invalid=st["invalid"] + s["invalid"],
                      wins=st["wins"] + s["wins"], best=jnp.maximum(st["best"], s["best"]))
            return ar, st
        z = jnp.int32(0)
        return jax.lax.fori_loop(0, self.sc.rounds, one,
                                 (ar, dict(new=z, deaths=z, invalid=z, wins=z, best=jnp.float32(-jnp.inf))))

    def args(self, ar, key, round0, pol, lo=-np.inf, hi=np.inf):
        return (ar, key, jnp.int32(round0), pol, jnp.float32(lo), jnp.float32(hi), self.p)

    def compile(self, ar, pol):
        """Ahead-of-time compile (no device work; call before taking the GPU lock)."""
        return self._round.lower(*self.args(ar, jax.random.PRNGKey(0), 0, pol)).compile()

    def run(self, ar, key, round0, pol, lo=-np.inf, hi=np.inf):
        """`rounds` rounds; returns (archive, stats dict of device scalars)."""
        return self._round(*self.args(ar, key, round0, pol, lo, hi))


def _tree_sel(mask, a, b):
    return jax.tree.map(lambda x, y: jnp.where(mask.reshape(mask.shape + (1,) * (x.ndim - 1)), x, y), a, b)


# ---------------------------------------------------------------- policy


def no_policy(obs_size, hidden=(256, 256)) -> Policy:
    params = N.init_params(jax.random.PRNGKey(0), obs_size, 2, tuple(hidden))
    return Policy(params, Nz.rms_init((obs_size,)), jnp.float32(0.0))


def load_policy(path) -> tuple[Policy, dict]:
    """(Policy with use = 1, the checkpoint's config dict) from a training checkpoint."""
    from goi_jax.training import checkpoint as C
    ck = C.load(path)
    params = jax.tree.map(jnp.asarray, ck["params"])
    rms = Nz.RMS(*(jnp.asarray(x) for x in ck["obs_rms"]))
    return Policy(params, rms, jnp.float32(1.0)), ck["config"]


def with_use(pol: Policy, use: float) -> Policy:
    return pol._replace(use=jnp.float32(use))


# ---------------------------------------------------------------- host: summaries, bank file


def coverage(ar_host: dict, sc: SearchConfig):
    """Per section: cells filled, bins with any cell / bins in the section (below max_progress), furthest progress.
    ar_host: numpy filled / progress / root arrays."""
    filled, prog, root = (np.asarray(ar_host[k]) for k in ("filled", "progress", "root"))
    sec = cell_section(sc)
    bins_f = filled.reshape(sc.n_bins, sc.angle_cells).any(-1)
    bin_sec = sec[::sc.angle_cells]
    rows = []
    for i, (lo, name) in enumerate(St.SECTIONS[:-1]):
        nb = int((bin_sec == i).sum())
        if nb == 0:
            continue
        m = filled & (sec == i)
        rows.append(dict(section=name, lo=lo, bins=nb, bins_filled=int((bins_f & (bin_sec == i)).sum()),
                         cells=int(m.sum()), top=float(prog[m].max()) if m.any() else None,
                         reached_from=float(prog[filled & (root == i)].max()) if (filled & (root == i)).any() else None))
    return rows


def save_bank(path, ar: Archive, sc: SearchConfig, meta: dict):
    """Write the filled cells as a start bank (atomic). Also the resume point of scripts/search_starts.py."""
    f = np.asarray(ar.filled)
    keep = np.nonzero(f)[0]
    states = jax.tree.map(lambda a: np.asarray(a)[keep], ar.states)
    prog = np.asarray(ar.progress)[keep]
    out = dict(states=states, progress=prog, section=St.section_index(prog).astype(np.int32),
               time=np.asarray(ar.time)[keep], root=np.asarray(ar.root)[keep], cell=keep.astype(np.int32),
               chosen=np.asarray(ar.chosen)[keep], found=np.asarray(ar.found)[keep],
               search=asdict(sc), meta=meta)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return out


def load_bank(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def archive_from_bank(bank: dict, sc: SearchConfig) -> Archive:
    """Rebuild a device archive from a saved bank (same SearchConfig cell layout)."""
    if dict(bank["search"]).get("angle_cells") != sc.angle_cells or bank["search"].get("bin_size") != sc.bin_size \
            or bank["search"].get("max_progress") != sc.max_progress:
        raise ValueError("bank was searched with another cell layout (bin_size / angle_cells / max_progress)")
    one = jax.tree.map(lambda a: a[0], bank["states"])
    ar = archive_init(sc, one)
    c = jnp.asarray(bank["cell"])
    return ar._replace(states=jax.tree.map(lambda A, s: A.at[c].set(jnp.asarray(s)), ar.states, bank["states"]),
                       filled=ar.filled.at[c].set(True), progress=ar.progress.at[c].set(jnp.asarray(bank["progress"])),
                       time=ar.time.at[c].set(jnp.asarray(bank["time"])), chosen=ar.chosen.at[c].set(jnp.asarray(bank["chosen"])),
                       root=ar.root.at[c].set(jnp.asarray(bank["root"])), found=ar.found.at[c].set(jnp.asarray(bank["found"])))

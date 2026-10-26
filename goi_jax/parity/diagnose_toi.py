"""TOI events, engine vs game, on every recorded step with a TOI event (NAT1 toiCount > 0 on some contact):
which contacts the engine counted (ContactTOI.count) against the game's m_toiCount, the TOI bodies' alpha0, and the
ContactPoint2D impulses Unity reported from the TOI sub-step solve against the engine's (ContactTOI.normal_impulse).

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_toi [--chains c15 ...] [--set name=value ...]

Per step: 'same' when the per-contact TOI counts agree, else the contacts that differ (list position in NAT1 = newest
first, and the creation order rank in NAT0 when it existed).

    ... -m goi_jax.parity.diagnose_toi --ties

lists the first min-TOI search of every TOI step on the engine's pre-TOI state: the candidates exactly tied at the
minimum alpha (contact-list positions, newest first) and which of them the game made its TOI contact. Last line DONE or
STOPPED.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import importlib

import jax
import jax.numpy as jnp

from goi_jax import box2d as jb
from goi_jax.box2d import math2d as m
from goi_jax.box2d import toi as T
from goi_jax.box2d.world import empty_toi
from goi_jax.game import level as L
from goi_jax.game import unity
from goi_jax.parity import harness as H
from goi_jax.parity import states as S
from goi_jax.parity.diagnose_step import run_steps


def toi_steps(chain):
    cd, n = chain.contacts[1], chain.n_contacts[1]
    return np.array([s for s in range(len(chain)) if (cd["toiCount"][s, :n[s]] > 0).any()], int)


def compare_counts(level, chain, steps, out):
    """Per step: list of (key, game count, engine count, NAT1 list pos, existed in NAT0) where they differ."""
    col, piece = level.fixture["collider"], level.fixture["piece"]
    rows = []
    for t, s in enumerate(steps):
        ct = out.contacts
        live = np.nonzero(np.asarray(ct.pair)[t] >= 0)[0]
        eng = {}
        for i in live:
            fa, fb = int(ct.fa[t, i]), int(ct.fb[t, i])
            eng[(int(col[fa]), int(piece[fa]), int(col[fb]), int(piece[fb]))] = int(out.toi.count[t, i])
        ids1, n1 = chain.contact_ids[1][s], chain.n_contacts[1][s]
        ids0 = {tuple(int(x) for x in chain.contact_ids[0][s, i]) for i in range(chain.n_contacts[0][s])}
        diff = []
        for i in range(n1):
            k = tuple(int(x) for x in ids1[i])
            g = int(chain.contacts[1]["toiCount"][s, i])
            e = eng.get(k, -1)
            if g != e:
                diff.append((k, g, e, i, k in ids0))
        rows.append(diff)
    return rows


def first_search(level, chain, steps, settings):
    """The engine's first min-TOI search of each step: (toi (N, C), candidate mask (N, C), list position (N, C))."""
    ST = importlib.import_module("goi_jax.box2d.step")
    jm, st, tgt, mask, _ = H.prepare(level, chain, steps, settings)

    def one(model, state, tg, mk):
        dt = jnp.float32(jb.GAME_DT)
        state = unity.move_velocity(state, tg, mk, dt)
        state = ST.collide(model, state, settings)
        state = ST.solve(model, state, settings, dt, jnp.float32(1) / dt, m.mul(state.inv_dt0, dt))
        B, C = state.bodies.type.shape[0], state.contacts.pair.shape[0]
        tst = empty_toi(B, C)
        _, tst, found, ms, ma = T._search(model, state, tst, settings, jnp.asarray(True), None)
        _, pos, _ = T._list_order(state.contacts)
        cand = tst.flag & (tst.toi < 1) & (tst.toi == ma) & found
        return tst.toi, cand, pos, state.contacts
    key = settings
    if key not in _FIRST:
        _FIRST[key] = jax.jit(jax.vmap(one, in_axes=(L.model_axes(), 0, 0, 0)))
    n = len(steps)
    jm, st, tgt, mask = H._pad((jm, st, tgt, mask), 64)  # one compile for every chain
    out = _FIRST[key](level.model_for(jm), jax.tree.map(jnp.asarray, st), jnp.asarray(tgt), jnp.asarray(mask))
    return H._cut(jax.tree.map(np.asarray, out), n)


_FIRST = {}


def ties(level, root, chains, settings):
    col, piece = level.fixture["collider"], level.fixture["piece"]
    tally = {}
    for e in S.load_index(root)["chains"]:
        stem = Path(e["file"]).stem
        if chains and not any(stem.startswith(c) for c in chains):
            continue
        chain = S.load_chain(Path(root) / e["file"])
        steps = toi_steps(chain)
        if not len(steps):
            continue
        toi, cand, pos, ct = first_search(level, chain, steps, settings)
        for t, s in enumerate(steps):
            idx = np.nonzero(cand[t])[0]
            if len(idx) < 2:
                continue
            g1 = {tuple(int(x) for x in chain.contact_ids[1][s, i]): int(chain.contacts[1]["toiCount"][s, i])
                  for i in range(chain.n_contacts[1][s])}
            items = []
            for i in sorted(idx, key=lambda i: pos[t, i]):
                fa, fb = int(ct.fa[t, i]), int(ct.fb[t, i])
                k = (int(col[fa]), int(piece[fa]), int(col[fb]), int(piece[fb]))
                items.append((int(pos[t, i]), g1.get(k, -1), k))
            picked = [p for p, g, _ in items if g > 0]
            kind = ("several" if len(picked) > 1 else  # more than one tied contact got an event: pick unknown
                    "newest" if picked and picked[0] == items[0][0] else
                    "oldest" if picked and picked[-1] == items[-1][0] else "other")
            a = float(toi[t, idx[0]])
            tally[(a == 0, kind)] = tally.get((a == 0, kind), 0) + 1
            print(f"{stem:<20} {s:4d} alpha {a:.9g} tied (pos, game toiCount): "
                  f"{[(p, g, k) for p, g, k in items]}  game took the {kind}")
    print("tally (alpha == 0, game's pick): " + ", ".join(f"{k}: {v}" for k, v in sorted(tally.items())))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    ap.add_argument("--ties", action="store_true", help="list exact ties of the first min-TOI search")
    args = ap.parse_args(argv)
    try:
        kw = dict(s.split("=", 1) for s in args.set)
        settings = H.engine_settings(**{k: H._parse_value(v) for k, v in kw.items()})
        level = L.load_level(args.root)
        if args.ties:
            ties(level, args.root, args.chains, settings)
            print("DONE")
            return
        runner = None
        n_same = n_diff = 0
        for e in S.load_index(args.root)["chains"]:
            stem = Path(e["file"]).stem
            if args.chains and not any(stem.startswith(c) for c in args.chains):
                continue
            chain = S.load_chain(Path(args.root) / e["file"])
            steps = toi_steps(chain)
            if not len(steps):
                continue
            out, runner = run_steps(level, chain, steps, settings, runner, batch=64)
            res = H.compare(level, chain, steps, out)
            for t, (s, diff) in enumerate(zip(steps, compare_counts(level, chain, steps, out))):
                ok = res["c"]["ok"][t] & res["v"]["ok"][t] & res["a"]["ok"][t] & res["w"]["ok"][t]
                tag = "same" if not diff else "DIFF " + "; ".join(
                    f"{level.names['colliders'][k[0]].split('/')[-1][:12]}#{k[1]}-"
                    f"{level.names['colliders'][k[2]].split('/')[-1][:12]}#{k[3]} game {g} eng {en} pos {p}"
                    f"{'' if old else ' new'}" for k, g, en, p, old in diff)
                n_same += not diff
                n_diff += bool(diff)
                print(f"{stem:<20} {s:4d} {'ok ' if ok else 'BAD'} |dv| {res['v']['abs'][t]:8.2g}  "
                      f"events {int(out.toi.events[t])}  {tag}")
        print(f"TOI steps with the same per-contact TOI counts: {n_same}, different: {n_diff}")
        print("DONE")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

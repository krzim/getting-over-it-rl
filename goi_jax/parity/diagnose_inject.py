"""Counterfactual: would the engine match the game if its Collide produced the game's manifolds for the hammer head?

For every recorded step, the hammer-head (Tip) contacts whose NAT1 manifold the game computed with Collide at the NAT0
pose (no TOI on the contact, the Tip not advanced by a TOI event) get that recorded manifold injected into the engine's
Collide (b2Contact::Evaluate's output; warm-start carry-over by feature key then runs as usual). Everything else is the
harness's run. If a step's failure comes from the Tip's per-rebuild baked shape (goi_jax/parity/diagnose_shapes.py),
injection moves it into tolerance; what stays out has another cause.

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_inject [--chains c02 c03] [--all-dynamic]

--all-dynamic injects for every contact with a dynamic body (pot included), not only the Tip's.
Prints the harness report for the injected run; last line DONE or STOPPED.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import box2d as jb
from goi_jax.box2d import pairs as P
from goi_jax.game import level as L
from goi_jax.game import unity
from goi_jax.parity import harness as H
from goi_jax.parity import states as S

F32 = np.float32
_CTX = {"inj": None, "calls": 0}
_orig_evaluate = P.nc.evaluate


def _patched_evaluate(model, fa, fb, xa, xb, linear_slop):
    man = _orig_evaluate(model, fa, fb, xa, xb, linear_slop)
    inj = _CTX["inj"]
    _CTX["calls"] += 1
    if inj is None or _CTX["calls"] != 1:  # only b2ContactManager::Collide (the step's first Evaluate)
        return man
    m = inj["mask"]
    w = lambda new, old: jnp.where(m, new, old)
    return dict(man, mtype=w(inj["mtype"], man["mtype"]), count=w(inj["count"], man["count"]),
                local_normal=(w(inj["ln"][:, 0], man["local_normal"][0]), w(inj["ln"][:, 1], man["local_normal"][1])),
                local_point=(w(inj["lp"][:, 0], man["local_point"][0]), w(inj["lp"][:, 1], man["local_point"][1])),
                point_local=tuple((w(inj["pl"][:, p, 0], man["point_local"][p][0]),
                                   w(inj["pl"][:, p, 1], man["point_local"][p][1])) for p in range(2)),
                point_id=(w(inj["keys"][:, 0], man["point_id"][0]), w(inj["keys"][:, 1], man["point_id"][1])))


def injection(level, chain, steps, C, all_dynamic=False):
    """(N, C) injection arrays: NAT1 manifolds of the qualifying contacts, in the loader's NAT0 slots."""
    steps = np.asarray(steps)
    N = len(steps)
    order = np.argsort(level.native_slot)[:-1]
    tip_slot = int(np.nonzero(order == level.hammer_body)[0][0])
    tipf = set(int(f) for f in level.tip_fixtures)
    fbody = level.fixture["body"]
    out = dict(mask=np.zeros((N, C), bool), mtype=np.zeros((N, C), np.int32), count=np.zeros((N, C), np.int32),
               ln=np.zeros((N, C, 2), F32), lp=np.zeros((N, C, 2), F32), pl=np.zeros((N, C, 2, 2), F32),
               keys=np.zeros((N, C, 2), np.int32))
    c1, i1, n1 = chain.contacts[1], chain.contact_ids[1], chain.n_contacts[1]
    i0, n0 = chain.contact_ids[0], chain.n_contacts[0]
    for t, s in enumerate(steps):
        slots = {tuple(int(x) for x in i0[s, i]): i for i in range(int(n0[s]))}
        for i in range(int(n1[s])):
            n = int(c1["pointCount"][s, i])
            if n == 0 or c1["toiCount"][s, i] > 0:
                continue
            k = tuple(int(x) for x in i1[s, i])
            fa, fb = level.fixture_of[k[:2]], level.fixture_of[k[2:]]
            if all_dynamic:
                ok = level.body_type[fbody[fa]] == jb.DYNAMIC or level.body_type[fbody[fb]] == jb.DYNAMIC
                bodies = [int(fbody[fa]), int(fbody[fb])]
                adv = any(chain.bodies[1]["alpha0"][s, int(np.nonzero(order == b)[0][0])] != 0
                          for b in bodies if b in order)
                ok = ok and not adv
            else:
                ok = (fa in tipf or fb in tipf) and chain.bodies[1]["alpha0"][s, tip_slot] == 0
            if not ok or k not in slots:
                continue
            j = slots[k]
            out["mask"][t, j] = True
            out["mtype"][t, j], out["count"][t, j] = c1["manifoldType"][s, i], n
            out["ln"][t, j], out["lp"][t, j] = c1["localNormal"][s, i], c1["localPoint"][s, i]
            out["pl"][t, j] = c1["pointLocal"][s, i]
            out["keys"][t, j] = c1["key"][s, i].astype(np.uint32).view(np.int32)
    return out


class InjectRunner:
    def __init__(self, level, settings):
        self.level, s = level, settings

        def one(model, st, tgt, mask, inj):
            _CTX["inj"], _CTX["calls"] = inj, 0
            try:
                return unity.simulate(model, st, s, jb.GAME_DT, tgt, mask, axis_name="w")
            finally:
                _CTX["inj"] = None
        self._run = jax.jit(jax.vmap(one, in_axes=(L.model_axes(), 0, 0, 0, 0), axis_name="w"))

    def __call__(self, jm, st, tgt, mask, inj):
        P.nc.evaluate = _patched_evaluate
        try:
            out = self._run(self.level.model_for(jm), jax.tree.map(jnp.asarray, st), jnp.asarray(tgt),
                            jnp.asarray(mask), jax.tree.map(jnp.asarray, inj))
        finally:
            P.nc.evaluate = _orig_evaluate
        return jax.tree.map(np.asarray, out)


def run(root="dumps/parity", chains=None, settings=None, batch=512, capacity=96, all_dynamic=False):
    root = Path(root)
    level = L.load_level(root)
    settings = settings or H.engine_settings()
    runner = InjectRunner(level, settings)
    entries = S.load_index(root)["chains"]
    if chains:
        entries = [e for e in entries if any(Path(e["file"]).stem.startswith(c) for c in chains)]
    parts, t0, n_inj = [], time.time(), 0
    for e in entries:
        chain = S.load_chain(root / e["file"])
        for steps in H._batches(len(chain), batch):
            jm, st, tgt, mask, info = H.prepare(level, chain, steps, settings, capacity)
            inj = injection(level, chain, steps, capacity, all_dynamic)
            n_inj += int(inj["mask"].sum())
            n = len(steps)
            jm_p, st_p, tgt_p, mask_p, inj_p = H._pad((jm, st, tgt, mask, inj), batch)
            out = H._cut(runner(jm_p, st_p, tgt_p, mask_p, inj_p), n)
            parts.append(dict(res=H.compare(level, chain, steps, out), cls=H.classify(level, chain, steps),
                              chain=np.array([e["file"]] * n), step=steps,
                              loader=dict(not_candidate=np.array([i["not_candidate"] for i in info]),
                                          flipped=np.array([i["flipped"] for i in info]))))
        print(f"{e['file']}: {len(chain)} steps ({time.time() - t0:.0f} s)", flush=True)
    print(f"injected manifolds: {n_inj}")
    return H._concat(parts), level, settings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*")
    ap.add_argument("--all-dynamic", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--save")
    args = ap.parse_args(argv)
    from goi_jax.parity import report
    try:
        kw = dict(s.split("=", 1) for s in args.set)
        settings = H.engine_settings(**{k: H._parse_value(v) for k, v in kw.items()})
        res, level, settings = run(args.root, args.chains, settings, all_dynamic=args.all_dynamic)
        print(report.format_report(res, level, settings, worst=args.worst))
        if args.save:
            report.save(res, args.save)
        print("DONE")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

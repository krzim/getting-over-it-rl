"""One recorded K=1 step under the microscope: run the engine on NAT0 (as the harness does, one world) and print,
side by side with the game's NAT1, the bodies' errors, every touching contact (manifold count / keys / local points,
impulses, TOI count and cached toi, the engine's TOI solve impulses) and the collision callbacks with the
ContactPoint2Ds Unity reported.

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_step c03 134 [--set name=value ...] [--fits F]

--fits uses the hammer head's per-chain shapes fitted by goi_jax.parity.diagnose_shapes (its --fits cache).

Last line DONE or STOPPED.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from goi_jax.game import level as L
from goi_jax.parity import harness as H
from goi_jax.parity import states as S

F32 = np.float32


def find_chain(root, prefix):
    for e in S.load_index(root)["chains"]:
        if Path(e["file"]).stem.startswith(prefix):
            return e, S.load_chain(Path(root) / e["file"])
    raise KeyError(prefix)


def run_steps(level, chain, steps, settings, runner=None, batch=None):
    """Engine State after each step (numpy, leading axis = steps) and the Runner."""
    runner = runner or H.Runner(level, settings)
    jm, st, tgt, mask, _ = H.prepare(level, chain, steps, settings)
    n = len(steps)
    b = batch or n
    jm, st, tgt, mask = H._pad((jm, st, tgt, mask), b)
    return H._cut(runner(jm, st, tgt, mask), n), runner


def describe(level, chain, s, out, t=0):
    names = level.names
    col = names["colliders"]
    bn = names["body_names"]
    exp = level.export(out)
    order = np.argsort(level.native_slot)[:-1]
    g1, g0 = chain.bodies[1], chain.bodies[0]
    print("bodies (dynamic): |dc| |da| |dv| |dw|  engine toi alpha0 vs game alpha0")
    for k, e in enumerate(order):
        if level.body_type[e] != 2:
            continue
        dc = np.abs(exp["bodies"]["c"][t, k] - g1["c"][s, k]).max()
        da = abs(exp["bodies"]["a"][t, k] - g1["a"][s, k])
        dv = np.abs(exp["bodies"]["v"][t, k] - g1["v"][s, k]).max()
        dw = abs(exp["bodies"]["w"][t, k] - g1["w"][s, k])
        ea0 = np.asarray(out.toi.alpha0)[t, e] if out.toi is not None else np.nan
        moved = np.abs(g1["c"][s, k] - g0["c"][s, k]).max()
        print(f"  {bn[level.body_game[e]].split('/')[-1][:14]:<14} {dc:9.2g} {da:9.2g} {dv:9.2g} {dw:9.2g}   "
              f"a0 eng {ea0:.4f} game {g1['alpha0'][s, k]:.4f}  game c0==NAT0 c {bool((g1['c0'][s, k] == g0['c'][s, k]).all())} "
              f"|step move| {moved:.3g}")
    ct = out.contacts
    toi = out.toi
    print(f"engine TOI events: {int(np.asarray(toi.events)[t]) if toi is not None else 'n/a'}")
    cd, ids, nc = chain.contacts[1], chain.contact_ids[1], chain.n_contacts[1][s]
    rows = exp["contacts"][t]
    # engine slot per key
    slot_of = {}
    fa, fb = np.asarray(ct.fa)[t], np.asarray(ct.fb)[t]
    for i in np.nonzero(np.asarray(ct.pair)[t] >= 0)[0]:
        k = (int(level.fixture["collider"][fa[i]]), int(level.fixture["piece"][fa[i]]),
             int(level.fixture["collider"][fb[i]]), int(level.fixture["piece"][fb[i]]))
        slot_of[k] = i
    print("contacts after the step (game NAT1 touching, or engine touching):")
    seen = set()
    for i in range(nc):
        k = tuple(int(x) for x in ids[s, i])
        seen.add(k)
        r = rows.get(k)
        gt = bool(cd["flags"][s, i] & S.CONTACT_TOUCHING)
        if not gt and not (r and r["touching"]):
            continue
        gn = int(cd["pointCount"][s, i])
        print(f"  {col[k[0]].split('/')[-1][:18]}#{k[1]} - {col[k[2]].split('/')[-1][:22]}#{k[3]}  game: fl={int(cd['flags'][s, i]):#x} "
              f"n={gn} toiC={cd['toiCount'][s, i]} toi={cd['toi'][s, i]:.5f} keys={cd['key'][s, i][:gn].tolist()} "
              f"ni={cd['normalImpulse'][s, i][:gn]} ti={cd['tangentImpulse'][s, i][:gn]} pl={cd['pointLocal'][s, i][:gn].tolist()}")
        if r is None:
            print("      engine: (no contact)")
            continue
        j = slot_of[k]
        en = r["count"]
        ex = ""
        if toi is not None:
            ex = (f" toiC={int(np.asarray(toi.count)[t, j])} toi={float(np.asarray(toi.toi)[t, j]):.5f} "
                  f"flag={bool(np.asarray(toi.flag)[t, j])} solved={bool(np.asarray(toi.solved)[t, j])} "
                  f"toi_ni={np.asarray(toi.normal_impulse)[t, j][:en]}")
        print(f"      engine: touch={int(r['touching'])} n={en} keys={r['point_id'][:en].astype(np.uint32).tolist()} "
              f"ni={r['normal_impulse'][:en]} ti={r['tangent_impulse'][:en]} pl={r['point_local'][:en].tolist()}{ex}")
    for k, r in rows.items():
        if k not in seen and r["touching"]:
            print(f"  ENGINE ONLY {col[k[0]]}#{k[1]} - {col[k[2]]}#{k[3]} n={r['count']}")
    ev = chain.events[chain.events["step"] == s]
    pts = chain.points[chain.points["step"] == s]
    print("events:")
    base = np.nonzero(chain.events["step"] == s)[0]
    for j, e in zip(base, ev):
        kind = {1: "Enter", 2: "Stay", 3: "Exit", 4: "HammerReturn", 5: "MovePosition"}.get(int(e["kind"]), e["kind"])
        extra = ""
        if e["kind"] in (1, 2, 3):
            extra = f" coll={col[e['collider']].split('/')[-1]} other={col[e['other']].split('/')[-1]} relv=({e['a']:.4g},{e['b']:.4g})"
            for p in pts[pts["event"] == j]:
                extra += (f"\n        point ({p['px']:.6f},{p['py']:.6f}) n=({p['nx']:.5f},{p['ny']:.5f}) sep={p['separation']:.3g} "
                          f"ni={p['normalImpulse']:.6g} ti={p['tangentImpulse']:.6g}")
        print(f"   phase {e['phase']} {kind} script={e['script']} count={e['count']}{extra}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chain")
    ap.add_argument("steps", type=int, nargs="+")
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--fits", help="Tip shapes fitted by goi_jax.parity.diagnose_shapes (--fits cache .npz)")
    args = ap.parse_args(argv)
    try:
        kw = dict(s.split("=", 1) for s in args.set)
        settings = H.engine_settings(**{k: H._parse_value(v) for k, v in kw.items()})
        level = L.load_level(args.root)
        e, chain = find_chain(args.root, args.chain)
        if args.fits:
            import dataclasses

            import jax.numpy as jnp

            from goi_jax.parity import diagnose_shapes as DSH
            v, n, _, _ = DSH.fit_all(args.root, level, None, args.fits)[e["file"]]
            level = dataclasses.replace(level, model=level.model._replace(verts=jnp.asarray(v), normals=jnp.asarray(n)))
        out, _ = run_steps(level, chain, np.array(args.steps), settings)
        for t, s in enumerate(args.steps):
            print(f"===== {e['file']} step {s}")
            describe(level, chain, s, out, t)
        print("DONE")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

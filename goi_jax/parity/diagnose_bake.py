"""The hammer head's per-rebuild collider bake (goi_jax/game/placement.py), measured on the recordings' manifolds.

Every placement rebuilds the physics world and Unity re-bakes the Tip's polygon pieces: vertices = authored points
through a 2x2 float32 matrix A (scale 1 + k ulp, rotation r * 2^-30 rad). The game's stored manifolds show the result
(diagnose_shapes collect_rows: NAT1 manifolds of Tip contacts computed by Collide at the NAT0 pose). This tool

    scan     scores every chain's rows on a grid of A (scale (kx, ky) ulps x rotation r), cached per session (.npz);
    table    per placement hammer angle (index "angle"; None = the level start), the A with the most rows over the chains
             of that angle in one session;
    harness  runs the single-step parity harness with each chain's Tip baked through A, A taken from the table of
             ANOTHER session (--table-from: held out; nothing fitted on the evaluated data), or from the chain's own
             scan (--own: in-sample, an upper bound), or the session geometry (--geometry).

    nice -n 19 env JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_bake --root dumps/parity-k4 \
        --scan runs/bake-k4.npz                                          # scan (cached; ~minutes)
    ... --root dumps/parity --scan runs/bake-k1.npz --table-from runs/bake-k4.npz --harness [--chains c03 ...]

Last line DONE or STOPPED.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import level as L
from goi_jax.game import placement as PL
from goi_jax.parity import diagnose_shapes as D
from goi_jax.parity import states as S

F32 = np.float32
SCALES = ((2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (3, 4), (4, 3))
ROTS = tuple(range(-160, 161, 4))


def grid():
    return [(kx, ky, r) for kx, ky in SCALES for r in ROTS]


def _score_batch(level, rows, cands, points):
    """(C,) rows matched with the Tip baked through each candidate matrix (C, 2, 2)."""
    fx = level.fixture
    C, R = len(cands), len(rows["step"])
    F = len(fx["count"])
    tipf = np.array(sorted(points))
    vt = np.zeros((C, len(tipf), 8, 2), F32)
    nt = np.zeros_like(vt)
    for c, A in enumerate(cands):
        for i, f in enumerate(tipf):
            vt[c, i], nt[c, i] = PL.polygon(PL.bake_vertices(points[int(f)], A))
    slot = np.full(F, -1)
    slot[tipf] = np.arange(len(tipf))
    ci = np.repeat(np.arange(C), R)

    def side(f):
        f = np.tile(f, C)
        s = slot[f]
        v = np.where((s >= 0)[:, None, None], vt[ci, np.maximum(s, 0)], fx["verts"][f])
        n = np.where((s >= 0)[:, None, None], nt[ci, np.maximum(s, 0)], fx["normals"][f])
        return f, v, n
    fa, va, na = side(rows["fa"])
    fb, vb, nb = side(rows["fb"])
    big = {k: (tuple(np.tile(x, C) for x in v) if isinstance(v, tuple) else np.tile(v, (C,) + (1,) * (v.ndim - 1)))
           for k, v in rows.items()}
    man = D._eval(jnp.asarray(np.concatenate([va, vb])), jnp.asarray(np.concatenate([na, nb])),
                  jnp.asarray(np.concatenate([fx["count"][fa], fx["count"][fb]]).astype(np.int32)),
                  jnp.asarray(np.concatenate([fx["radius"][fa], fx["radius"][fb]])),
                  jnp.asarray(np.concatenate([fx["shape_type"][fa], fx["shape_type"][fb]]).astype(np.int32)),
                  tuple(jnp.asarray(x) for x in big["xa"]), tuple(jnp.asarray(x) for x in big["xb"]))
    ok = D._compare(jax.tree.map(np.asarray, man), big)
    return ok.reshape(C, R).sum(1)


def scan(root, cache=None, chains=None, chunk=48):
    """{file: (R, scores over grid())} for every chain with Tip rows; cached in `cache` (.npz) when given."""
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        return {str(f): (int(r), s) for f, r, s in zip(z["files"], z["rows"], z["scores"])}
    level = L.load_level(root)
    points = L.tip_points(level)
    g = grid()
    mats = [PL.bake_matrix(kx, ky, r) for kx, ky, r in g]
    out, t0 = {}, time.time()
    for e in S.load_index(root)["chains"]:
        if chains and not any(Path(e["file"]).stem.startswith(c) for c in chains):
            continue
        rows = D.collect_rows(level, S.load_chain(Path(root) / e["file"]))
        R = len(rows["step"])
        if R == 0:
            out[e["file"]] = (0, np.zeros(len(g), int))
            continue
        sc = np.concatenate([_score_batch(level, rows, mats[i:i + chunk], points) for i in range(0, len(g), chunk)])
        out[e["file"]] = (R, sc)
        b = int(np.argmax(sc))
        print(f"{e['file']:<26} rows {R:4d} start-A {sc[g.index((2, 2, 0))]:4d} best {sc[b]:4d} at {g[b]} "
              f"({time.time() - t0:.0f} s)", flush=True)
    if cache:
        fs = list(out)
        np.savez_compressed(cache, files=np.array(fs), rows=np.array([out[f][0] for f in fs]),
                            scores=np.stack([out[f][1] for f in fs]))
    return out


def angle_key(entry):
    a = entry.get("angle")
    return "start" if a is None else f"{float(a):g}"


def table(root, scans):
    """{angle key: (kx, ky, r)}: per hammer angle the grid point with the most rows over that session's chains
    (ties: the smallest |r|, then the smallest scale)."""
    g = grid()
    tot = {}
    for e in S.load_index(root)["chains"]:
        if e["file"] in scans and scans[e["file"]][0]:
            k = angle_key(e)
            tot[k] = tot.get(k, 0) + scans[e["file"]][1]
    out = {}
    for k, s in tot.items():
        best = [i for i in range(len(g)) if s[i] == s.max()]
        out[k] = min((g[i] for i in best), key=lambda p: (abs(p[2]), p[0] + p[1]))
    return out


def chain_matrices(root, mode, scans=None, other_root=None, other_scans=None):
    """{file: (A, label)} per chain of `root`: mode "held-out" (the other session's per-angle table, start A where the
    angle is not in it), "own" (the chain's own best grid point), "geometry" (the session geometry: start A)."""
    g = grid()
    tab = table(other_root, other_scans) if mode == "held-out" else {}
    out = {}
    for e in S.load_index(root)["chains"]:
        if mode == "geometry":
            p = (2, 2, 0)
        elif mode == "own":
            R, s = scans[e["file"]]
            best = [i for i in range(len(g)) if s[i] == s.max()]
            p = min((g[i] for i in best), key=lambda q: (abs(q[2]), q[0] + q[1])) if R else (2, 2, 0)
        else:
            p = tab.get(angle_key(e), (2, 2, 0))
        out[e["file"]] = (PL.bake_matrix(*p), p)
    return out


def run_harness(root, mats, chains=None, settings=None, batch=512, capacity=96):
    """goi_jax.parity.harness run with each chain's Tip baked through its matrix."""
    from goi_jax.parity import harness as H
    root = Path(root)
    level = L.load_level(root)
    points = L.tip_points(level)
    settings = settings or H.engine_settings()
    runner = H.Runner(level, settings)
    entries = [e for e in S.load_index(root)["chains"]
               if not chains or any(Path(e["file"]).stem.startswith(c) for c in chains)]
    parts, t0 = [], time.time()
    for e in entries:
        A, p = mats[e["file"]]
        lv = L.with_tip_bake(level, A, points)
        runner.level = lv
        chain = S.load_chain(root / e["file"])
        for steps in H._batches(len(chain), batch):
            jm, st, tgt, mask, info = H.prepare(lv, chain, steps, settings, capacity)
            k = len(steps)
            out = H._cut(runner(*H._pad((jm, st, tgt, mask), batch)), k)
            parts.append(dict(res=H.compare(lv, chain, steps, out), cls=H.classify(lv, chain, steps),
                              chain=np.array([e["file"]] * k), step=steps,
                              loader=dict(not_candidate=np.array([i["not_candidate"] for i in info]),
                                          flipped=np.array([i["flipped"] for i in info]))))
        print(f"{e['file']}: A {p}, {len(chain)} steps ({time.time() - t0:.0f} s)", flush=True)
    return H._concat(parts), level, settings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*")
    ap.add_argument("--scan", help="this session's scan cache (.npz; written when missing)")
    ap.add_argument("--table-from", nargs=2, metavar=("ROOT", "SCAN"), help="other session for the held-out table")
    ap.add_argument("--harness", action="store_true")
    ap.add_argument("--mode", choices=("held-out", "own", "geometry"), default="held-out")
    ap.add_argument("--save", help="harness per-step results .npz")
    ap.add_argument("--worst", type=int, default=20)
    args = ap.parse_args(argv)
    try:
        scans = scan(args.root, args.scan, None if args.harness else args.chains) if args.scan else None
        if scans is not None:
            g = grid()
            tot = sum(r for r, _ in scans.values())
            own = sum(int(s.max()) for r, s in scans.values() if r)
            start = sum(int(s[g.index((2, 2, 0))]) for r, s in scans.values() if r)
            print(f"rows {tot}: start A {start}, each chain's best grid A {own}")
            print("table (this session):", table(args.root, scans))
        other = None
        if args.table_from:
            oroot, oscan = args.table_from
            other = (oroot, scan(oroot, oscan))
            tab = table(*other)
            print("table (held-out source):", tab)
            if scans is not None:
                g = grid()
                ho = 0
                for e in S.load_index(args.root)["chains"]:
                    R, s = scans.get(e["file"], (0, None))
                    if R:
                        ho += int(s[g.index(tab.get(angle_key(e), (2, 2, 0)))])
                print(f"held-out rows matched: {ho} of {sum(r for r, _ in scans.values())}")
        if args.harness:
            from goi_jax.parity import report
            mats = chain_matrices(args.root, args.mode, scans, *(other or (None, None)))
            res, level, settings = run_harness(args.root, mats, args.chains)
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

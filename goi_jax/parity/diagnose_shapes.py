"""The hammer head's (Tip's) baked collider shapes per recording placement.

Finding: Unity bakes the Tip's polygon pieces from its full 3D Transform (the out-of-plane tilt of the Handle/Tip,
docs/understanding/plugin.md "What the determinism work found" 3; plugin/Game.cs:55-58), so every world rebuild (a
chain's placement) leaves the Tip's vertices ~1e-8 (about one float32 ulp of the polygon's size) away from the
session's geometry record (dumps/parity/geometry.bin, read once, at the level-start placement). The game's manifolds
show it directly: on face-A manifolds (a Tip edge is the reference) the stored localNormal / localPoint are that edge's
normal and midpoint bit for bit, and on 44 of 46 such NAT1 manifolds they are not the recorded geometry's. The shape
stays fixed through a chain (material swaps do not re-bake it) and chains from the same placement share it; the pot's
pieces always match. Collide on the stale shape puts manifold points ~1e-7 off, and on stiff hammer-on-rock steps
(300 velocity iterations, joints at their limits) that sends the step elsewhere (c03 step 134: |dv| 4.7 m/s; with the
recorded manifold injected, bit exact: goi_jax/parity/diagnose_inject.py).

`infer_tip_shapes` recovers the Tip pieces from the game's own manifolds: rows are NAT1 manifolds that b2Contact::Update
computed at the NAT0 pose (no TOI on the contact, the Tip not advanced by a TOI event); coordinate descent over vertex,
edge and polygon moves (see _moves) keeps the candidate that reproduces the most rows exactly (count, type, keys, local
normal / point, point local points), normals re-derived as b2PolygonShape::Set does. `fit_all` fits each placement's
chains together (a chain keeps its own fit where that reproduces more of its rows); `run_with_shapes` is the harness run
with each chain's fitted shapes in the Model. 876 of 891 rows reproduced (geometry: 777). The fit is a stand-in: the
exact fix is to record the Tip's fixture shapes once per placement (recorder change).

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_shapes [--chains c03 ...] [--fits F.npz]  # fit table
    ... -m goi_jax.parity.diagnose_shapes --harness --fits F.npz [--chains ...] [--save R.npz]   # harness with them

--fits caches the fitted shapes (written when missing, read when present; the fit takes ~3 min on CPU).
Last line DONE or STOPPED.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import collide as nc
from goi_jax.box2d import shapes as SH
from goi_jax.game import level as L
from goi_jax.parity import states as S

F32 = np.float32


class _M:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@jax.jit
def _eval(verts, normals, count, radius, stype, xa, xb):
    R = xa[0].shape[0]
    mdl = _M(verts=verts, normals=normals, count=count, radius=radius, shape_type=stype)
    return nc.evaluate(mdl, jnp.arange(R), R + jnp.arange(R), xa, xb, F32(0.005))


def _xf(rec):  # recorded (p.x, p.y, q.s, q.c) -> (px, py, qc, qs)
    return rec[..., 0], rec[..., 1], rec[..., 3], rec[..., 2]


def collect_rows(level, chain):
    """NAT1 manifolds of contacts with a Tip piece, computed by Collide at the NAT0 pose. Returns a dict of arrays."""
    order = np.argsort(level.native_slot)[:-1]
    slot_of = {int(e): k for k, e in enumerate(order)}
    tipf = set(int(f) for f in level.tip_fixtures)
    tip_slot = slot_of[int(level.hammer_body)]
    fbody = level.fixture["body"]
    cd, ids, ncn = chain.contacts[1], chain.contact_ids[1], chain.n_contacts[1]
    rows = {k: [] for k in ("step", "fa", "fb", "xa", "xb", "count", "mtype", "keys", "ln", "lp", "pl")}
    for s in range(len(chain)):
        if chain.bodies[1]["alpha0"][s, tip_slot] != 0 or not chain.bodies[0]["flags"][s, tip_slot] & 2:
            continue
        for i in range(int(ncn[s])):
            n = int(cd["pointCount"][s, i])
            if n == 0 or cd["toiCount"][s, i] > 0:
                continue
            ca, pa, cb, pb = (int(x) for x in ids[s, i])
            fa, fb = level.fixture_of[(ca, pa)], level.fixture_of[(cb, pb)]
            if fa not in tipf and fb not in tipf:
                continue
            xs = []
            for f in (fa, fb):
                k = slot_of.get(int(fbody[f]))
                xs.append((F32(0), F32(0), F32(1), F32(0)) if k is None else _xf(chain.bodies[0]["xf"][s, k]))
            rows["step"].append(s)
            rows["fa"].append(fa)
            rows["fb"].append(fb)
            rows["xa"].append(xs[0])
            rows["xb"].append(xs[1])
            rows["count"].append(n)
            rows["mtype"].append(int(cd["manifoldType"][s, i]))
            rows["keys"].append(cd["key"][s, i].astype(np.uint32).view(np.int32))
            rows["ln"].append(cd["localNormal"][s, i])
            rows["lp"].append(cd["localPoint"][s, i])
            rows["pl"].append(cd["pointLocal"][s, i])
    out = {k: np.asarray(v) for k, v in rows.items()}
    if len(out["step"]):
        out["xa"] = tuple(np.asarray(out["xa"], F32).T)
        out["xb"] = tuple(np.asarray(out["xb"], F32).T)
    return out


def match_rows(level, rows, verts, normals):
    """(R,) exact manifold match of every row with fixture shapes `verts` / `normals` (F, 8, 2) (all fixtures)."""
    fa, fb = rows["fa"], rows["fb"]
    fx = level.fixture
    cat = lambda a: jnp.asarray(np.concatenate([a[fa], a[fb]]))
    man = _eval(cat(verts), cat(normals), cat(fx["count"].astype(np.int32)), cat(fx["radius"]),
                cat(fx["shape_type"].astype(np.int32)), tuple(jnp.asarray(x) for x in rows["xa"]),
                tuple(jnp.asarray(x) for x in rows["xb"]))
    man = jax.tree.map(np.asarray, man)
    n = rows["count"]
    ok = (man["count"] == n) & (man["mtype"] == rows["mtype"])
    ok &= (man["local_normal"][0] == rows["ln"][:, 0]) & (man["local_normal"][1] == rows["ln"][:, 1])
    ok &= (man["local_point"][0] == rows["lp"][:, 0]) & (man["local_point"][1] == rows["lp"][:, 1])
    for p in range(2):
        on = n > p
        same = (man["point_id"][p] == rows["keys"][:, p]) & (man["point_local"][p][0] == rows["pl"][:, p, 0]) & \
               (man["point_local"][p][1] == rows["pl"][:, p, 1])
        ok &= ~on | same
    return ok


def _nudge(v, k):
    v = F32(v)
    for _ in range(abs(int(k))):
        v = np.nextafter(v, F32(np.inf) if k > 0 else F32(-np.inf))
    return v


def _edge_normals_batch(v, n):
    """shapes._edge_normals for (C, n, 2) candidate polygons (same float32 op order)."""
    e = np.roll(v, -1, axis=1) - v
    x, y = (F32(1.0) * e[..., 1]).astype(F32), (F32(-1.0) * e[..., 0]).astype(F32)
    ln = np.sqrt((x * x + y * y).astype(F32)).astype(F32)
    inv = (F32(1.0) / np.where(ln < np.finfo(F32).eps, F32(1), ln)).astype(F32)
    small = ln < np.finfo(F32).eps
    return np.stack([np.where(small, x, x * inv), np.where(small, y, y * inv)], -1).astype(F32)


def _score(level, rows, f, cand_v, verts, normals):
    """(C,) rows matched with fixture f's polygon replaced by each candidate (C, n, 2)."""
    C, n = cand_v.shape[:2]
    cn = _edge_normals_batch(cand_v, n)
    pad = lambda a: np.concatenate([a, np.repeat(a[:, :1], 8 - n, 1)], 1)
    cv, cn = pad(cand_v), pad(cn)
    Rr = len(rows["step"])
    fa, fb = np.tile(rows["fa"], C), np.tile(rows["fb"], C)
    ci = np.repeat(np.arange(C), Rr)
    fx = level.fixture

    def shp(fi, side_is_f):
        v = np.where(side_is_f[:, None, None], cv[ci], verts[fi])
        nn = np.where(side_is_f[:, None, None], cn[ci], normals[fi])
        return v, nn
    va, na = shp(fa, fa == f)
    vb, nb = shp(fb, fb == f)
    tile = lambda x: np.tile(x, C)
    big = {k: (tuple(tile(x) for x in v) if isinstance(v, tuple) else
               (np.tile(v, (C,) + (1,) * (v.ndim - 1)))) for k, v in rows.items()}
    man = _eval(jnp.asarray(np.concatenate([va, vb])), jnp.asarray(np.concatenate([na, nb])),
                jnp.asarray(np.concatenate([fx["count"][fa], fx["count"][fb]]).astype(np.int32)),
                jnp.asarray(np.concatenate([fx["radius"][fa], fx["radius"][fb]])),
                jnp.asarray(np.concatenate([fx["shape_type"][fa], fx["shape_type"][fb]]).astype(np.int32)),
                tuple(jnp.asarray(x) for x in big["xa"]), tuple(jnp.asarray(x) for x in big["xb"]))
    ok = _compare(jax.tree.map(np.asarray, man), big)
    return ok.reshape(C, Rr).sum(1)


def _compare(man, rows):
    n = rows["count"]
    ok = (man["count"] == n) & (man["mtype"] == rows["mtype"])
    ok &= (man["local_normal"][0] == rows["ln"][:, 0]) & (man["local_normal"][1] == rows["ln"][:, 1])
    ok &= (man["local_point"][0] == rows["lp"][:, 0]) & (man["local_point"][1] == rows["lp"][:, 1])
    for p in range(2):
        on = n > p
        same = (man["point_id"][p] == rows["keys"][:, p]) & (man["point_local"][p][0] == rows["pl"][:, p, 0]) & \
               (man["point_local"][p][1] == rows["pl"][:, p, 1])
        ok &= ~on | same
    return ok


def _moves(base, n):
    """Candidate polygons around `base` (the unchanged polygon first, so score ties keep it). Unity's bake errors are
    absolute (about one ulp of the polygon's size, ~3e-8), so a small coordinate can move by many of its own ulps,
    and an edge can move rigidly (its normal unchanged). Moves:
      one coordinate by m of its own ulps (|m| <= 64); one vertex by (mx, my) own ulps (|m| <= 6);
      an edge (two adjacent vertices) translated by (mx, my) * 2^-28 (|m| <= 8); the polygon translated likewise."""
    sp = np.spacing(np.abs(base)).astype(F32)
    out = [base[None]]
    m1 = np.r_[-64:0, 1:65].astype(F32)
    for k in range(n):
        for a in range(2):
            c = np.repeat(base[None], len(m1), 0)
            c[:, k, a] = (base[k, a] + m1 * sp[k, a]).astype(F32)
            out.append(c)
    g = np.arange(-6, 7)
    gx, gy = (x.ravel().astype(F32) for x in np.meshgrid(g, g))
    for k in range(n):
        c = np.repeat(base[None], len(gx), 0)
        c[:, k, 0] = (base[k, 0] + gx * sp[k, 0]).astype(F32)
        c[:, k, 1] = (base[k, 1] + gy * sp[k, 1]).astype(F32)
        out.append(c)
    q = F32(2.0 ** -28)
    g = np.arange(-8, 9)
    tx, ty = (x.ravel().astype(F32) * q for x in np.meshgrid(g, g))
    for k in range(n):
        c = np.repeat(base[None], len(tx), 0)
        for kk in (k, (k + 1) % n):
            c[:, kk, 0] = (base[kk, 0] + tx).astype(F32)
            c[:, kk, 1] = (base[kk, 1] + ty).astype(F32)
        out.append(c)
    c = np.repeat(base[None], len(tx), 0)
    c[..., 0] = (base[None, :, 0] + tx[:, None]).astype(F32)
    c[..., 1] = (base[None, :, 1] + ty[:, None]).astype(F32)
    out.append(c)
    return np.concatenate(out).astype(F32)


def infer_tip_shapes(level, chain, sweeps=16, rows=None):
    """(verts, normals) arrays for all fixtures with the chain's Tip pieces fitted, and (rows matched before, after,
    total)."""
    rows = collect_rows(level, chain) if rows is None else rows
    verts = np.array(level.fixture["verts"], F32)
    normals = np.array(level.fixture["normals"], F32)
    R = len(rows["step"])
    if R == 0:
        return verts, normals, (0, 0, 0)
    before = int(match_rows(level, rows, verts, normals).sum())
    for _ in range(sweeps):
        changed = False
        for f in level.tip_fixtures:
            sel = (rows["fa"] == f) | (rows["fb"] == f)
            if not sel.any():
                continue
            n = int(level.fixture["count"][f])
            sub = {k: (tuple(x[sel] for x in v) if isinstance(v, tuple) else v[sel]) for k, v in rows.items()}
            cands = _moves(verts[f, :n], n)
            sc = np.concatenate([_score(level, sub, f, cands[i:i + 4096], verts, normals)
                                 for i in range(0, len(cands), 4096)])
            best = cands[int(np.argmax(sc))]
            if not np.array_equal(best, verts[f, :n]):
                changed = True
                verts[f, :n] = best
                verts[f, n:] = best[0]
                normals[f, :n] = _edge_normals_batch(best[None], n)[0]
                normals[f, n:] = normals[f, 0]
        if not changed:
            break
    after = int(match_rows(level, rows, verts, normals).sum())
    return verts, normals, (before, after, R)


def placement_key(entry):
    """Chains recorded from the same placement share the Tip's baked shape (identical face-A deviations across them)."""
    return (entry["group"], entry.get("along"))


def _concat_rows(parts):
    parts = [p for p in parts if len(p["step"])]
    if not parts:
        return {"step": np.zeros(0, int)}
    out = {}
    for k in parts[0]:
        if isinstance(parts[0][k], tuple):
            out[k] = tuple(np.concatenate([p[k][i] for p in parts]) for i in range(len(parts[0][k])))
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


_FITS = {}


def fit_all(root="dumps/parity", level=None, chains=None, cache=None):
    """{chain file: (verts, normals, (geometry matches, fitted matches, rows), source)} for every chain: the fit over
    all chains of its placement, or the chain's own fit where that reproduces more of its rows."""
    key = (str(root), tuple(chains or ()))
    if key in _FITS:
        return _FITS[key]
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        files = [str(f) for f in z["files"]]
        out = {f: (z["verts"][i], z["normals"][i], tuple(int(x) for x in z["stats"][i]), str(z["source"][i]))
               for i, f in enumerate(files)
               if not chains or any(Path(f).stem.startswith(c) for c in chains)}
        _FITS[key] = out
        return out
    level = level or L.load_level(root)
    entries = S.load_index(root)["chains"]
    if chains:
        entries = [e for e in entries if any(Path(e["file"]).stem.startswith(c) for c in chains)]
    groups = {}
    for e in entries:
        groups.setdefault(placement_key(e), []).append(e)
    out = {}
    for _, es in groups.items():
        rows = {e["file"]: collect_rows(level, S.load_chain(Path(root) / e["file"])) for e in es}
        gv, gn, _ = infer_tip_shapes(level, None, rows=_concat_rows(list(rows.values())))
        for e in es:
            r = rows[e["file"]]
            if not len(r["step"]):
                out[e["file"]] = (gv, gn, (0, 0, 0), "placement")
                continue
            g_ok = int(match_rows(level, r, gv, gn).sum())
            cv, cn, st = infer_tip_shapes(level, None, rows=r)
            if st[1] > g_ok:
                out[e["file"]] = (cv, cn, st, "chain")
            else:
                out[e["file"]] = (gv, gn, (st[0], g_ok, st[2]), "placement")
    _FITS[key] = out
    if cache:
        fs = list(out)
        np.savez_compressed(cache, files=np.array(fs), verts=np.stack([out[f][0] for f in fs]),
                            normals=np.stack([out[f][1] for f in fs]), stats=np.array([out[f][2] for f in fs]),
                            source=np.array([out[f][3] for f in fs]))
    return out


def run_with_shapes(root="dumps/parity", chains=None, settings=None, batch=512, capacity=96, cache=None):
    """goi_jax.parity.harness.run with each chain's fitted Tip shapes in the Model."""
    import dataclasses
    import time
    from goi_jax.parity import harness as H
    root = Path(root)
    level = L.load_level(root)
    fits = fit_all(root, level, chains, cache)
    settings = settings or H.engine_settings()
    runner = H.Runner(level, settings)
    entries = [e for e in S.load_index(root)["chains"] if e["file"] in fits]
    parts, t0 = [], time.time()
    for e in entries:
        v, n, _, _ = fits[e["file"]]
        lv = dataclasses.replace(level, model=level.model._replace(verts=jnp.asarray(v), normals=jnp.asarray(n)))
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
        print(f"{e['file']}: {len(chain)} steps ({time.time() - t0:.0f} s)", flush=True)
    return H._concat(parts), level, settings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*")
    ap.add_argument("--harness", action="store_true", help="run the parity harness with the inferred Tip shapes")
    ap.add_argument("--save", help="harness per-step results .npz")
    ap.add_argument("--fits", help="cache of the fitted shapes (.npz; written when missing, read when present)")
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    args = ap.parse_args(argv)
    try:
        if args.harness:
            from goi_jax.parity import harness as H
            from goi_jax.parity import report
            kw = dict(s.split("=", 1) for s in args.set)
            settings = H.engine_settings(**{k: H._parse_value(v) for k, v in kw.items()})
            res, level, settings = run_with_shapes(args.root, args.chains, settings, cache=args.fits)
            print(report.format_report(res, level, settings, worst=args.worst))
            if args.save:
                report.save(res, args.save)
        else:
            level = L.load_level(args.root)
            fits = fit_all(args.root, level, args.chains, args.fits)
            print(f"{'chain':<22} {'rows':>5} {'geometry':>9} {'fitted':>7} {'fit over':>9}  moved vertices "
                  f"(piece:index (dx, dy) own ulps)")
            tot = np.zeros(3, int)
            for file, (v, n, st, src) in fits.items():
                tot += st
                moved = []
                for f in level.tip_fixtures:
                    for k in range(int(level.fixture["count"][f])):
                        g = level.fixture["verts"][f, k]
                        if not np.array_equal(v[f, k], g):
                            du = [int(np.round((v[f, k, a] - g[a]) / np.spacing(np.abs(g[a])))) for a in range(2)]
                            moved.append(f"{int(level.fixture['piece'][f])}:{k}{tuple(du)}")
                print(f"{Path(file).stem:<22} {st[2]:5d} {st[0]:9d} {st[1]:7d} {src:>9}  {' '.join(moved)}")
            print(f"{'total':<22} {tot[2]:5d} {tot[0]:9d} {tot[1]:7d}")
        print("DONE")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

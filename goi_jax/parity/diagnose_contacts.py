"""Where did the game's post-step manifold come from? For every touching contact after a recorded K=1 step (NAT1),
evaluate b2Contact::Evaluate (goi_jax.box2d.collide) at candidate poses of its two bodies and see which reproduces
the recorded manifold (point count, feature keys, local points):

    pre      the recorded NAT0 transforms (what Collide at the top of b2World::Step sees)
    post     the recorded NAT1 transforms
    c0       NAT1's sweep start (c0, a0): after a TOI event Box2D leaves the TOI bodies' c0 on the solved TOI pose
    alpha    the full-step sweep NAT0 -> NAT1 at alpha on a grid (finds a TOI-like intermediate pose)

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.diagnose_contacts [--chains c01 ...] [--all-bodies]

Prints one summary table (per contact kind: matched pre / post / c0 / alpha / none) and the 'none' cases with their
nearest alpha and residual. Last line DONE or STOPPED.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d import collide as nc
from goi_jax.box2d import math2d as m
from goi_jax.game import level as L
from goi_jax.parity import states as S

F32 = np.float32
ALPHAS = np.linspace(0, 1, 257, dtype=np.float32)


def _pose_from_sweep(c, a, lc):
    """b2Body::SynchronizeTransform from (c, a) in float32 (host): p = c - R(a) lc."""
    qc, qs = (np.asarray(x) for x in m.rot_of(jnp.asarray(a, jnp.float32)))
    rx = qc * lc[..., 0] - qs * lc[..., 1]
    ry = qs * lc[..., 0] + qc * lc[..., 1]
    return np.stack([c[..., 0] - rx, c[..., 1] - ry], -1).astype(F32), qc.astype(F32), qs.astype(F32)


@jax.jit
def _eval(model_v, model_n, model_c, model_r, model_t, fa, fb, xa, xb):
    mdl = type("M", (), dict(verts=model_v, normals=model_n, count=model_c, radius=model_r, shape_type=model_t))
    return nc.evaluate(mdl, fa, fb, xa, xb, F32(0.005))


def evaluate(level, fa, fb, xa, xb):
    md = level.model
    out = _eval(md.verts, md.normals, md.count, md.radius, md.shape_type, jnp.asarray(fa), jnp.asarray(fb),
                tuple(jnp.asarray(x) for x in xa), tuple(jnp.asarray(x) for x in xb))
    return jax.tree.map(np.asarray, out)


def _xf_rec(xf):  # recorded xf (p.x, p.y, q.s, q.c) -> (px, py, qc, qs)
    return xf[..., 0], xf[..., 1], xf[..., 3], xf[..., 2]


def _match(man, i, gcount, gkeys, gpl):
    """(exact match, max |local point error|) of manifold row i against the recorded one."""
    n = int(man["count"][i])
    if n != gcount:
        return False, np.inf
    if n == 0:
        return True, 0.0
    keys = np.array([man["point_id"][k][i] for k in range(n)], np.int64)
    if (keys != gkeys[:n]).any():
        return False, np.inf
    pl = np.array([[man["point_local"][k][0][i], man["point_local"][k][1][i]] for k in range(n)], F32)
    err = np.abs(pl.astype(np.float64) - gpl[:n]).max()
    return bool((pl == gpl[:n]).all()), float(err)


def analyse_chain(level, chain, dyn_only=True, tip_only=False):
    """List of per-contact dicts for every touching NAT1 contact (count > 0) of the chain."""
    order = np.argsort(level.native_slot)[:-1]  # engine body per native slot
    slot_of = np.zeros(level.n_bodies, int) - 1
    slot_of[order] = np.arange(len(order))
    fb_ = level.fixture["body"]
    tipf = set(int(f) for f in level.tip_fixtures)
    rows = []
    cd, ids, ncn = chain.contacts[1], chain.contact_ids[1], chain.n_contacts[1]
    b0, b1 = chain.bodies[0], chain.bodies[1]
    for s in range(len(chain)):
        n = int(ncn[s])
        for i in range(n):
            if not cd["flags"][s, i] & S.CONTACT_TOUCHING or cd["pointCount"][s, i] == 0:
                continue
            ca, pa, cb, pb = (int(x) for x in ids[s, i])
            fa, fbb = level.fixture_of[(ca, pa)], level.fixture_of[(cb, pb)]
            if tip_only and not (fa in tipf or fbb in tipf):
                continue
            ea, eb = int(fb_[fa]), int(fb_[fbb])
            if dyn_only and level.body_type[ea] != 2 and level.body_type[eb] != 2:
                continue
            rows.append(dict(step=s, i=i, fa=fa, fb=fbb, ea=ea, eb=eb, tip=fa in tipf or fbb in tipf,
                             count=int(cd["pointCount"][s, i]), keys=cd["key"][s, i].astype(np.uint32).view(np.int32),
                             pl=cd["pointLocal"][s, i].astype(np.float64), toi_count=int(cd["toiCount"][s, i]),
                             flags=int(cd["flags"][s, i])))
    if not rows:
        return rows
    R = len(rows)
    fa = np.array([r["fa"] for r in rows])
    fbb = np.array([r["fb"] for r in rows])

    def body_xf(which, eng_body, steps):
        out = []
        for e, s in zip(eng_body, steps):
            k = slot_of[e]
            if k < 0:  # ground
                out.append((F32(0), F32(0), F32(1), F32(0)))
            else:
                out.append(_xf_rec(chain.bodies[which]["xf"][s, k]))
        return tuple(np.array(x, F32) for x in zip(*out))

    def body_sweep(eng_body, steps, which, field_c, field_a):
        c = np.zeros((len(steps), 2), F32)
        a = np.zeros(len(steps), F32)
        lc = np.zeros((len(steps), 2), F32)
        for j, (e, s) in enumerate(zip(eng_body, steps)):
            k = slot_of[e]
            if k >= 0:
                c[j], a[j] = chain.bodies[which][field_c][s, k], chain.bodies[which][field_a][s, k]
                lc[j] = chain.bodies[which]["localCenter"][s, k]
        return c, a, lc

    steps = np.array([r["step"] for r in rows])
    ea = np.array([r["ea"] for r in rows])
    eb = np.array([r["eb"] for r in rows])
    cands = {}
    cands["pre"] = (body_xf(0, ea, steps), body_xf(0, eb, steps))
    cands["post"] = (body_xf(1, ea, steps), body_xf(1, eb, steps))

    def sweep_pose(eng, which, fc, fa_):
        c, a, lc = body_sweep(eng, steps, which, fc, fa_)
        p, qc, qs = _pose_from_sweep(c, a, lc)
        static = level.body_type[eng] == 0
        p[static] = 0
        qc[static], qs[static] = 1, 0
        return p[:, 0], p[:, 1], qc, qs
    cands["c0"] = (sweep_pose(ea, 1, "c0", "a0"), sweep_pose(eb, 1, "c0", "a0"))
    res = {k: evaluate(level, fa, fbb, *v) for k, v in cands.items()}
    # alpha grid on the NAT0 -> NAT1 sweep (c, a)
    cA0, aA0, lcA = body_sweep(ea, steps, 0, "c", "a")
    cA1, aA1, _ = body_sweep(ea, steps, 1, "c", "a")
    cB0, aB0, lcB = body_sweep(eb, steps, 0, "c", "a")
    cB1, aB1, _ = body_sweep(eb, steps, 1, "c", "a")
    best = np.full(R, np.inf)
    best_alpha = np.full(R, np.nan)
    exact_alpha = np.zeros(R, bool)
    for al in ALPHAS:
        def at(c0, a0, c1, a1, lc, eng):
            c = ((F32(1) - al) * c0 + al * c1).astype(F32)
            a = ((F32(1) - al) * a0 + al * a1).astype(F32)
            p, qc, qs = _pose_from_sweep(c, a, lc)
            static = level.body_type[eng] == 0
            p[static] = 0
            qc[static], qs[static] = 1, 0
            return p[:, 0], p[:, 1], qc, qs
        man = evaluate(level, fa, fbb, at(cA0, aA0, cA1, aA1, lcA, ea), at(cB0, aB0, cB1, aB1, lcB, eb))
        for j, r in enumerate(rows):
            ok, err = _match(man, j, r["count"], r["keys"], r["pl"])
            exact_alpha[j] |= ok
            if err < best[j]:
                best[j], best_alpha[j] = err, al
    for j, r in enumerate(rows):
        for k in cands:
            ok, err = _match(res[k], j, r["count"], r["keys"], r["pl"])
            r[k], r[k + "_err"] = ok, err
        r["alpha_exact"], r["alpha_best"], r["alpha_err"] = bool(exact_alpha[j]), float(best_alpha[j]), float(best[j])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*")
    ap.add_argument("--tip-only", action="store_true")
    ap.add_argument("--show", type=int, default=40)
    args = ap.parse_args(argv)
    try:
        level = L.load_level(args.root)
        allrows = []
        for e, ch in S.iter_chains(args.root):
            if args.chains and not any(Path(e["file"]).stem.startswith(c) for c in args.chains):
                continue
            for r in analyse_chain(level, ch, tip_only=args.tip_only):
                r["chain"] = Path(e["file"]).stem
                allrows.append(r)
        print(f"{'kind':<10} {'n':>6} {'pre':>6} {'post':>6} {'c0':>6} {'alpha':>6} {'none':>6}")
        for kind, sel in (("tip", lambda r: r["tip"]), ("other", lambda r: not r["tip"]),
                          ("tip,toi", lambda r: r["tip"] and r["toi_count"] > 0)):
            rs = [r for r in allrows if sel(r)]
            none = [r for r in rs if not (r["pre"] or r["post"] or r["c0"] or r["alpha_exact"])]
            print(f"{kind:<10} {len(rs):6d} {sum(r['pre'] for r in rs):6d} {sum(r['post'] for r in rs):6d} "
                  f"{sum(r['c0'] for r in rs):6d} {sum(r['alpha_exact'] for r in rs):6d} {len(none):6d}")
        none = [r for r in allrows if not (r["pre"] or r["post"] or r["c0"] or r["alpha_exact"])]
        print(f"\n'none' contacts: {len(none)} in {len({(r['chain'], r['step']) for r in none})} steps")
        for r in none[:args.show]:
            print(f"  {r['chain']} s{r['step']:3d} tip={int(r['tip'])} n={r['count']} toi={r['toi_count']} "
                  f"fl={r['flags']:#x} pre_err={r['pre_err']:.3g} post_err={r['post_err']:.3g} "
                  f"best_alpha={r['alpha_best']:.3f} err={r['alpha_err']:.3g}")
        print("DONE")
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

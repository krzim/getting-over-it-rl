"""Single-step parity of the JAX engine against the real game: for every recorded K=1 step, load the native world
right before Simulate (NAT0) into goi_jax.box2d (goi_jax/game/level.py), run one jitted, vmapped
PhysicsScene2D.Simulate (goi_jax/game/unity.py), and compare with the native world right after it (NAT1).

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.harness                    # all chains, report to stdout
    ... -m goi_jax.parity.harness --chains c01 c03 --set velocity_iterations=300    # a subset, a settings override
    ... -m goi_jax.parity.harness --t5                                              # also the T5 alternative

Engine options come from goi_jax.box2d.Settings.game() plus `--set name=value` (any Settings field, e.g. the flags
the revolute v2.4 and TOI ports add), so new solver paths are picked up when they land. The last line printed is
DONE or STOPPED (reason).

Comparison (per step, over the 15 recorded bodies; the ground is static): per quantity the max |engine - game|, the
max ulp distance, whether every value is within tolerance (|err| <= max(1e-5, float32 spacing at the game value)),
and whether it is bit-exact. Contacts are matched by (collider, piece) pairs and points by feature key; joints by
name. Classes and diagnosis: goi_jax/parity/report.py.

Oracle path (goi_jax/reference): not used. Its C API cannot load a contact's manifold, feature keys or impulses, so
it cannot start warm from a recorded state; a cold oracle step would measure the warm start, not Box2D vs Unity.
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import box2d as jb
from goi_jax.game import level as L
from goi_jax.game import unity
from goi_jax.parity import states as S

F32 = np.float32
BODY_Q = ("c", "a", "v", "w", "xf_p", "xf_q")  # engine-side names; game: c, a, v, w, xf[:2], (q.c, q.s)
CONTACT_Q = ("normal_impulse", "tangent_impulse")
JOINT_Q = ("joint_impulse", "motor_impulse", "limit_impulse")
QUANTITIES = BODY_Q + ("kinematic",) + CONTACT_Q + JOINT_Q
TOL_ABS = 1e-5


def engine_settings(**overrides):
    """Settings.game() with overrides; unknown names are an error (so a typo does not silently do nothing)."""
    names = {f.name for f in dataclasses.fields(jb.Settings)}
    bad = [k for k in overrides if k not in names]
    if bad:
        raise ValueError(f"unknown Settings fields {bad}; known: {sorted(names)}")
    return jb.Settings.game(**overrides)


def _parse_value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        return float(v)


def ulp_distance(a, b):
    """Float32 values between a and b (0 = identical bits; +0 and -0 equal)."""
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    oa = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ob = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(oa - ob)


def tolerance(ref):
    return np.maximum(F32(TOL_ABS), np.spacing(np.abs(np.asarray(ref, F32))))


# ---------------------------------------------------------------------------------------------------------------------
# Running

class Runner:
    """One jitted, vmapped Simulate over a batch of recorded steps (Model joint fields batched per step)."""

    def __init__(self, level: L.Level, settings, kinematic_move=True):
        self.level, self.settings = level, settings
        s, km = settings, kinematic_move

        def one(model, st, tgt, mask):
            return unity.simulate(model, st, s, jb.GAME_DT, tgt, mask, axis_name="w", kinematic_move=km)
        self._run = jax.jit(jax.vmap(one, in_axes=(L.model_axes(), 0, 0, 0), axis_name="w"))

    def __call__(self, jm, st, tgt, mask):
        model = self.level.model_for(jm)
        out = self._run(model, jax.tree.map(jnp.asarray, st), jnp.asarray(tgt), jnp.asarray(mask))
        return jax.tree.map(np.asarray, out)


def prepare(level, chain, steps, settings, capacity=96, t5="before"):
    """(JointModel, State, MovePosition targets, mask, loader info) for the given steps (pre-Simulate NAT0).
    t5 = "before": contact friction as recorded (Unity refreshed it at the material assignment, before the solve);
    "after": the hammer/pot contacts solve with the previous material's mix (the alternative hypothesis)."""
    jm, st, info = level.native_states(chain, steps, 0, capacity)
    st = level.set_limit_states(jm, st, settings)
    if t5 == "after":
        st = _old_friction(level, chain, steps, st)
    tgt, mask = unity.move_targets(chain, steps, level)
    return jm, st, tgt, mask, info


def _old_friction(level, chain, steps, st):
    fx = st.fixtures.friction.copy()
    fx[:, level.tip_fixtures] = chain.field(0, "hammerFriction")[steps][:, None]
    fx[:, level.pot_fixtures] = chain.field(0, "potBottomFriction")[steps][:, None]
    ct = st.contacts
    mix = unity.contact_friction(fx, ct.fa, ct.fb)
    swapped = np.isin(ct.fa, np.r_[level.tip_fixtures, level.pot_fixtures]) | \
        np.isin(ct.fb, np.r_[level.tip_fixtures, level.pot_fixtures])
    live = ct.pair >= 0
    return st._replace(contacts=ct._replace(friction=np.where(live & swapped, mix, ct.friction).astype(F32)))


def _batches(n, size):
    for i in range(0, n, size):
        yield np.arange(i, min(i + size, n))


def _pad(tree, n):
    """Pad the leading axis of every array to n by repeating the last row."""
    def p(a):
        a = np.asarray(a)
        if a.shape[0] == n:
            return a
        return np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)])
    return jax.tree.map(p, tree)


def _cut(tree, n):
    return jax.tree.map(lambda a: a[:n], tree)


# ---------------------------------------------------------------------------------------------------------------------
# Comparison

def compare(level, chain, steps, out) -> dict:
    """Per-step error arrays for one batch. `out` is the engine State after the step (numpy, leading axis = steps).
    Returns {quantity: dict(abs, ulp, ok, exact)} with (N,) maxima / all-flags, plus body/contact diagnostics."""
    steps = np.asarray(steps)
    N = len(steps)
    exp = level.export(out)
    g1 = chain.bodies[1]
    order = np.argsort(level.native_slot)[:-1]  # engine body per native slot
    btype = level.body_type[order]
    dyn = btype == jb.DYNAMIC
    kin = btype == jb.KINEMATIC
    game = dict(c=g1["c"][steps], a=g1["a"][steps], v=g1["v"][steps], w=g1["w"][steps], xf_p=g1["xf"][steps][..., :2],
                xf_q=np.stack([g1["xf"][steps][..., 3], g1["xf"][steps][..., 2]], -1))
    eb = exp["bodies"]
    eng = dict(c=eb["c"], a=eb["a"], v=eb["v"], w=eb["w"], xf_p=eb["xf"][..., :2],
               xf_q=np.stack([eb["xf"][..., 3], eb["xf"][..., 2]], -1))
    res = {}
    worst_body = {}
    for q in BODY_Q:
        e, gq = eng[q], game[q]
        err = np.abs(e.astype(np.float64) - gq.astype(np.float64))
        ulp = ulp_distance(e, gq)
        ok = err <= tolerance(gq)
        ax = tuple(range(2, err.ndim))
        err_b = err.max(ax) if ax else err
        ulp_b = ulp.max(ax) if ax else ulp
        ok_b = ok.all(ax) if ax else ok
        res[q] = dict(abs=np.where(dyn, err_b, 0).max(1), ulp=np.where(dyn, ulp_b, 0).max(1),
                      ok=np.where(dyn, ok_b, True).all(1), exact=np.where(dyn, ulp_b == 0, True).all(1))
        worst_body[q] = order[np.argmax(np.where(dyn, err_b, -1), 1)]
    # kinematic bodies (cursor, crane): position after the MovePosition finish and velocity
    kerr = np.zeros(N)
    kulp = np.zeros(N, np.int64)
    for q in ("xf_p", "v"):
        e, gq = eng[q][:, kin], game[q][:, kin]
        kerr = np.maximum(kerr, np.abs(e.astype(np.float64) - gq).max((1, 2)))
        kulp = np.maximum(kulp, ulp_distance(e, gq).max((1, 2)))
    res["kinematic"] = dict(abs=kerr, ulp=kulp, ok=kerr <= TOL_ABS, exact=kulp == 0)

    # contacts
    cd, ids, nc = chain.contacts[1], chain.contact_ids[1], chain.n_contacts[1]
    cstats = {q: dict(abs=np.zeros(N), ulp=np.zeros(N, np.int64), ok=np.ones(N, bool), exact=np.ones(N, bool))
              for q in CONTACT_Q}
    missing, extra, order_ok, touch_bad, count_bad, geom_bad, geom_bad_tip = (np.zeros(N, np.int64) for _ in range(7))
    tip_cols = {int(level.fixture["collider"][f]) for f in level.tip_fixtures}
    for t, s in enumerate(steps):
        rows = exp["contacts"][t]
        gids = [tuple(int(x) for x in ids[s, i]) for i in range(nc[s])]
        missing[t] = sum(1 for k in gids if k not in rows)
        extra[t] = len(set(rows) - set(gids))
        common = [k for k in gids if k in rows]
        order_ok[t] = int([rows[k]["pos"] for k in common] == sorted(rows[k]["pos"] for k in common))
        for i, k in enumerate(gids):
            r = rows.get(k)
            if r is None:
                continue
            if r["touching"] != bool(cd["flags"][s, i] & S.CONTACT_TOUCHING):
                touch_bad[t] += 1
            gn = int(cd["pointCount"][s, i])
            if r["count"] != gn:
                count_bad[t] += 1
            gkeys = cd["key"][s, i].astype(np.uint32).view(np.int32)
            if gn and r["count"] == gn and (r["point_local"][:gn] != cd["pointLocal"][s, i, :gn]).any():
                geom_bad[t] += 1  # the game's manifold is not Collide at the NAT0 pose (TOI update or pose)
                geom_bad_tip[t] += int(k[0] in tip_cols or k[2] in tip_cols)
            for p in range(gn):
                hit = [j for j in range(r["count"]) if r["point_id"][j] == gkeys[p]]
                if not hit:
                    continue
                j = hit[0]
                for q, gname in (("normal_impulse", "normalImpulse"), ("tangent_impulse", "tangentImpulse")):
                    ev, gv = F32(r[q][j]), F32(cd[gname][s, i, p])
                    err = abs(float(ev) - float(gv))
                    u = int(ulp_distance(ev, gv))
                    c = cstats[q]
                    c["abs"][t] = max(c["abs"][t], err)
                    c["ulp"][t] = max(c["ulp"][t], u)
                    c["ok"][t] &= err <= float(tolerance(gv))
                    c["exact"][t] &= u == 0
    res.update(cstats)

    # joints (by name; engine index e of native slot k is NJ - 1 - k at that step)
    jidx = chain.joint_index[1][steps]
    NJ = jidx.shape[1]
    jt = exp["joints"]
    jstats = {q: dict(abs=np.zeros(N), ulp=np.zeros(N, np.int64), ok=np.ones(N, bool), exact=np.ones(N, bool))
              for q in JOINT_Q}
    jtype1 = chain.joints[1][1]["type"][steps]
    jidx0 = chain.joint_index[0][steps]
    for t, s in enumerate(steps):
        for k in range(NJ):
            name = jidx[t, k]
            k0 = int(np.nonzero(jidx0[t] == name)[0][0])
            e = NJ - 1 - k0
            typ = int(jtype1[t, k])
            d = {n: v[s, k] for n, v in chain.joints[1][typ].items()}
            imp = jt.impulse[t, e]
            pairs = []
            if typ == jb.REVOLUTE:
                pairs.append(("joint_impulse", imp[:2], d["impulse"][:2]))
                pairs.append(("motor_impulse", [jt.motor_impulse[t, e]], [d["motorImpulse"]]))
                if hasattr(jt, "lower_impulse"):
                    pairs.append(("limit_impulse", [jt.lower_impulse[t, e], jt.upper_impulse[t, e]],
                                  [d["lowerImpulse"], d["upperImpulse"]]))
                else:
                    pairs.append(("limit_impulse", [imp[2]], [F32(d["lowerImpulse"] - d["upperImpulse"])]))
            elif typ == jb.PRISMATIC:
                pairs.append(("joint_impulse", imp[:2], d["impulse"][:2]))
                pairs.append(("limit_impulse", [imp[2]], [d["impulse"][2]]))
                pairs.append(("motor_impulse", [jt.motor_impulse[t, e]], [d["motorImpulse"]]))
            else:
                pairs.append(("joint_impulse", imp, d["impulse"]))
            for q, ev, gv in pairs:
                ev, gv = np.asarray(ev, F32), np.asarray(gv, F32)
                err = np.abs(ev.astype(np.float64) - gv).max()
                u = int(ulp_distance(ev, gv).max())
                c = jstats[q]
                c["abs"][t] = max(c["abs"][t], err)
                c["ulp"][t] = max(c["ulp"][t], u)
                c["ok"][t] &= bool((np.abs(ev.astype(np.float64) - gv) <= tolerance(gv)).all())
                c["exact"][t] &= u == 0
    res.update(jstats)
    res["_diag"] = dict(missing=missing, extra=extra, order_ok=order_ok, touch_bad=touch_bad, count_bad=count_bad,
                        geom_bad=geom_bad, geom_bad_tip=geom_bad_tip,
                        worst_body_c=worst_body["c"], worst_body_v=worst_body["v"],
                        overflow=np.asarray(out.overflow))
    return res


def classify(level, chain, steps) -> dict:
    """Boolean class masks (N,) from the recordings alone (see report.CLASSES)."""
    steps = np.asarray(steps)
    N = len(steps)
    fb, btype = level.fixture["body"], level.body_type
    static_fx = btype[fb] == jb.STATIC
    tip = np.zeros(len(fb), bool)
    tip[level.tip_fixtures] = True
    potb = fb == level.pot_body
    out = {k: np.zeros(N, bool) for k in ("no_contact", "static_contact", "hammer_terrain", "pot_terrain",
                                          "dynamic_pair", "toi", "joint_limit", "friction_swap", "hammer_return",
                                          "joint_order")}
    for t, s in enumerate(steps):
        anyt = False
        for k in range(2):
            cd, ids, nc = chain.contacts[k], chain.contact_ids[k], chain.n_contacts[k][s]
            for i in range(nc):
                if not cd["flags"][s, i] & S.CONTACT_TOUCHING:
                    continue
                anyt = True
                ca, pa, cb, pb = (int(x) for x in ids[s, i])
                fa, fbb = level.fixture_of[(ca, pa)], level.fixture_of[(cb, pb)]
                st_a, st_b = static_fx[fa], static_fx[fbb]
                if (tip[fa] and st_b) or (tip[fbb] and st_a):
                    out["hammer_terrain"][t] = True
                elif (potb[fa] and st_b) or (potb[fbb] and st_a):
                    out["pot_terrain"][t] = True
                if not st_a and not st_b:
                    out["dynamic_pair"][t] = True
        out["no_contact"][t] = not anyt
        out["toi"][t] = (chain.contacts[1]["toiCount"][s, :chain.n_contacts[1][s]] > 0).any()
    out["static_contact"] = ~out["no_contact"] & ~out["hammer_terrain"] & ~out["dynamic_pair"]
    # joint limits: a revolute limit impulse or a prismatic limit state after the step
    j1 = chain.joints[1]
    typ = j1[1]["type"][steps]
    rev = typ == jb.REVOLUTE
    lim_r = rev & ((j1[1]["lowerImpulse"][steps] != 0) | (j1[1]["upperImpulse"][steps] != 0))
    pri = typ == jb.PRISMATIC
    lim_p = pri & ((j1[2]["limitState"][steps] != 0) | (j1[2]["impulse"][steps][..., 2] != 0))
    out["joint_limit"] = (lim_r | lim_p).any(1)
    for f in ("hammerFriction", "potBottomFriction"):
        out["friction_swap"] |= chain.field(0, f)[steps] != chain.field(1, f)[steps]
    ev = chain.events
    hr = ev[ev["kind"] == S.EV_HAMMER_RETURN]["step"]
    out["hammer_return"] = np.isin(steps, hr)
    j0 = chain.joint_index[0][steps]
    # the joint list as it stands after any action (climber joints re-created at its head); a chain's first step can
    # still hold the order of the placement (rope joints first)
    climber_first = np.array([level.names["joints"].index(n) for n in level.names["joints"]
                              if n.startswith("Player")], int)
    out["joint_order"] = ~np.isin(j0[:, 0], climber_first)
    return out


# ---------------------------------------------------------------------------------------------------------------------

def run(root="dumps/parity", chains=None, settings=None, batch=512, capacity=96, t5="before", progress=True,
        body_order="unity"):
    """Every recorded step of the selected chains (file stems or prefixes like 'c01'); returns a result dict with
    concatenated per-step arrays: errors per quantity, classes, diagnostics, chain/step ids."""
    root = Path(root)
    level = L.load_level(root, body_order=body_order)
    settings = settings or engine_settings()
    runner = Runner(level, settings)
    entries = S.load_index(root)["chains"]
    if chains:
        entries = [e for e in entries if any(Path(e["file"]).stem.startswith(c) for c in chains)]
    parts, t0 = [], time.time()
    for e in entries:
        chain = S.load_chain(root / e["file"])
        for steps in _batches(len(chain), batch):
            jm, st, tgt, mask, info = prepare(level, chain, steps, settings, capacity, t5)
            n = len(steps)
            jm_p, st_p, tgt_p, mask_p = _pad((jm, st, tgt, mask), batch)
            out = _cut(runner(jm_p, st_p, tgt_p, mask_p), n)
            res = compare(level, chain, steps, out)
            cls = classify(level, chain, steps)
            parts.append(dict(res=res, cls=cls, chain=np.array([e["file"]] * n), step=steps,
                              loader=dict(not_candidate=np.array([i["not_candidate"] for i in info]),
                                          flipped=np.array([i["flipped"] for i in info]))))
        if progress:
            print(f"{e['file']}: {len(chain)} steps ({time.time() - t0:.0f} s)", flush=True)
    return _concat(parts), level, settings


def _concat(parts):
    def cat(xs):
        if isinstance(xs[0], dict):
            return {k: cat([x[k] for x in xs]) for k in xs[0]}
        return np.concatenate(xs)
    return cat(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*", help="chain file stems or prefixes (default: all)")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="Settings overrides")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--t5", action="store_true", help="also run the 'friction refreshed after the solve' hypothesis")
    ap.add_argument("--worst", type=int, default=20)
    ap.add_argument("--body-order", default="unity", choices=("unity", "box2d"),
                    help="island seed order: unity = oldest recorded body first (matches the game), box2d = stock")
    ap.add_argument("--save", help="write the per-step results to this .npz")
    args = ap.parse_args(argv)
    from goi_jax.parity import report
    try:
        kw = dict(s.split("=", 1) for s in args.set)
        settings = engine_settings(**{k: _parse_value(v) for k, v in kw.items()})
        res, level, settings = run(args.root, args.chains, settings, args.batch, body_order=args.body_order)
        print(report.format_report(res, level, settings, worst=args.worst))
        if args.t5:
            alt, _, _ = run(args.root, args.chains, settings, args.batch, t5="after", progress=False,
                            body_order=args.body_order)
            print(report.format_t5(res, alt))
        if args.save:
            report.save(res, args.save)
        print("DONE")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

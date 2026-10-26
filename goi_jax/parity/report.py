"""Tables for goi_jax/parity/harness.py results: per quantity and per step class, the worst steps with diagnosis
hints, and the T5 comparison.

Classes (a step can be in several; from the recordings only):
    all              every recorded step
    no_contact       no touching contact before or after the step
    static_contact   touching contacts, none hammer-on-terrain and none between two moving bodies
    hammer_terrain   the hammer head (Tip) touches a static piece
    pot_terrain      the pot touches a static piece
    dynamic_pair     a touching contact between two non-static bodies (props)
    toi              a contact had a TOI event in the step (NAT1 toiCount > 0)
    joint_limit      a revolute limit impulse or prismatic limit state after the step
    friction_swap    the hammer or pot material changed in this step's FixedUpdate
    hammer_return    HammerReturn snapped the cursor this step
    joint_order      the joint list is not yet in the per-action order (first step after a placement)
    plain            none of toi / joint_limit / friction_swap / joint_order
"""
from __future__ import annotations

import numpy as np

CLASSES = ("all", "no_contact", "static_contact", "hammer_terrain", "pot_terrain", "dynamic_pair", "toi",
           "joint_limit", "friction_swap", "hammer_return", "joint_order", "plain")
BODY_ROWS = ("c", "a", "v", "w")
ALL_ROWS = ("c", "a", "v", "w", "xf_p", "xf_q", "kinematic", "normal_impulse", "tangent_impulse", "joint_impulse",
            "motor_impulse", "limit_impulse")


def class_masks(res) -> dict:
    cls = dict(res["cls"])
    n = len(res["step"])
    cls["all"] = np.ones(n, bool)
    cls["plain"] = ~(cls["toi"] | cls["joint_limit"] | cls["friction_swap"] | cls["joint_order"])
    return cls


def step_ok(res, rows=BODY_ROWS):
    ok = np.ones(len(res["step"]), bool)
    for q in rows:
        ok &= res["res"][q]["ok"]
    return ok


def quantity_table(res, mask=None) -> str:
    r = res["res"]
    mask = np.ones(len(res["step"]), bool) if mask is None else mask
    lines = [f"{'quantity':<16} {'max abs':>10} {'median abs':>11} {'max ulp':>11} {'in tol':>7} {'exact':>7}"]
    for q in ALL_ROWS:
        a = r[q]["abs"][mask]
        if not len(a):
            continue
        lines.append(f"{q:<16} {a.max():10.3g} {np.median(a):11.3g} {int(r[q]['ulp'][mask].max()):11d} "
                     f"{r[q]['ok'][mask].mean():7.1%} {r[q]['exact'][mask].mean():7.1%}")
    return "\n".join(lines)


def class_table(res) -> str:
    cls = class_masks(res)
    ok_b = step_ok(res)
    ok_pos = step_ok(res, ("c", "a"))
    ok_vel = step_ok(res, ("v", "w"))
    exact = np.ones(len(res["step"]), bool)
    for q in BODY_ROWS:
        exact &= res["res"][q]["exact"]
    r = res["res"]
    lines = [f"{'class':<15} {'steps':>6} {'bodies ok':>9} {'pos ok':>7} {'vel ok':>7} {'exact':>6} "
             f"{'max |dc|':>9} {'med |dc|':>9} {'max |dv|':>9} {'med |dv|':>9} {'max |dw|':>9}"]
    for k in CLASSES:
        m = cls[k]
        if not m.any():
            lines.append(f"{k:<15} {0:6d}")
            continue
        lines.append(f"{k:<15} {m.sum():6d} {ok_b[m].mean():9.1%} {ok_pos[m].mean():7.1%} {ok_vel[m].mean():7.1%} "
                     f"{exact[m].mean():6.1%} {r['c']['abs'][m].max():9.2g} {np.median(r['c']['abs'][m]):9.2g} "
                     f"{r['v']['abs'][m].max():9.2g} {np.median(r['v']['abs'][m]):9.2g} {r['w']['abs'][m].max():9.2g}")
    return "\n".join(lines)


def worst_table(res, level, n=20) -> str:
    r, d = res["res"], res["res"]["_diag"]
    cls = class_masks(res)
    score = np.maximum(r["c"]["abs"] / 1e-5, r["v"]["abs"] / 1e-5)
    idx = np.argsort(-score)[:n]
    names = level.names["body_names"]
    lines = [f"{'chain':<26} {'step':>4} {'|dc|':>9} {'|dv|':>9} {'|dw|':>9} {'worst c':<14} {'worst v':<14} "
             f"classes / hints"]
    for i in idx:
        bc, bv = int(d["worst_body_c"][i]), int(d["worst_body_v"][i])
        nm = lambda b: (names[level.body_game[b]] if level.body_game[b] >= 0 else "ground").split("/")[-1][:14]
        tags = [k for k in CLASSES[1:-1] if cls[k][i]]
        hints = []
        if cls["toi"][i]:
            hints.append("TOI event (toi.py)")
        if cls["joint_limit"][i]:
            hints.append("joint at a limit")
        if d["missing"][i] or d["extra"][i]:
            hints.append(f"contact set -{d['missing'][i]}/+{d['extra'][i]}")
        if d["touch_bad"][i] or d["count_bad"][i]:
            hints.append(f"manifold differs on {max(d['touch_bad'][i], d['count_bad'][i])} contacts")
        if d["geom_bad"][i]:
            hints.append(f"manifold points != Collide at NAT0 on {d['geom_bad'][i]} contacts")
        if not d["order_ok"][i]:
            hints.append("contact order")
        if r["joint_impulse"]["abs"][i] > 1e-3:
            hints.append(f"joint impulse {r['joint_impulse']['abs'][i]:.2g}")
        lines.append(f"{res['chain'][i][-26:]:<26} {int(res['step'][i]):4d} {r['c']['abs'][i]:9.2g} "
                     f"{r['v']['abs'][i]:9.2g} {r['w']['abs'][i]:9.2g} {nm(bc):<14} {nm(bv):<14} "
                     f"{','.join(tags)} | {'; '.join(hints)}")
    return "\n".join(lines)


def format_report(res, level, settings, worst=20) -> str:
    d = res["res"]["_diag"]
    n = len(res["step"])
    out = [f"Single-step parity: {n} recorded K=1 steps; engine Settings.game() "
           f"(velocity {settings.velocity_iterations}, position {settings.position_iterations}, "
           f"continuous_physics={settings.continuous_physics}); proxy order: {level.proxy_source}",
           f"tolerance: |err| <= max(1e-5, float32 spacing at the game value); bodies = the 13 dynamic bodies",
           "",
           "Per quantity (all steps):", quantity_table(res), "",
           "Per class (bodies: c centre, a angle, v linear, w angular velocity):", class_table(res), "",
           f"Contacts after the step vs the game: steps with missing {int((d['missing'] > 0).sum())}, "
           f"extra {int((d['extra'] > 0).sum())} (end-of-step FindNewContacts on estimated fat AABBs), list order "
           f"differs on {int((d['order_ok'] == 0).sum())}, touching/pointCount differs on "
           f"{int(((d['touch_bad'] + d['count_bad']) > 0).sum())}, manifold points differ (same count) on "
           f"{int((d['geom_bad'] > 0).sum())} steps ({int((d['geom_bad_tip'] > 0).sum())} involving the hammer head); loader: recorded contacts not in the candidate "
           f"list {int(res['loader']['not_candidate'].sum())}, A/B orientation differs {int(res['loader']['flipped'].sum())}; "
           f"contact overflow {int(d['overflow'].sum())}", "",
           f"Worst {worst} steps:", worst_table(res, level, worst)]
    return "\n".join(out)


def format_t5(before, after) -> str:
    m = before["cls"]["friction_swap"]
    out = ["T5 (contact friction after a material swap): body error on the friction_swap steps",
           f"{'hypothesis':<34} {'steps':>6} {'bodies ok':>9} {'max |dv|':>10} {'med |dv|':>10} {'max |dc|':>10}"]
    for name, r in (("refreshed before the solve (NAT0)", before), ("refreshed after the solve", after)):
        ok = step_ok(r)[m]
        out.append(f"{name:<34} {m.sum():6d} {ok.mean() if m.any() else 0:9.1%} "
                   f"{r['res']['v']['abs'][m].max() if m.any() else 0:10.3g} "
                   f"{np.median(r['res']['v']['abs'][m]) if m.any() else 0:10.3g} "
                   f"{r['res']['c']['abs'][m].max() if m.any() else 0:10.3g}")
    if m.any():
        b, a = before["res"]["v"]["abs"][m], after["res"]["v"]["abs"][m]
        out.append(f"steps where 'before' is closer: {(b < a).sum()}, 'after' closer: {(a < b).sum()}, "
                   f"equal: {(a == b).sum()}")
    return "\n".join(out)


def save(res, path):
    flat = {}

    def walk(prefix, x):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(f"{prefix}{k}/" if prefix or k else k, v)
        else:
            flat[prefix.rstrip("/")] = np.asarray(x)
    walk("", res)
    np.savez_compressed(path, **flat)

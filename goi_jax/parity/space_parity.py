"""Parity of the lockstep sim (goi_jax/game/sim.py with goi_jax/game/space.py) against recorded chains, single step
and short horizon, with the space mechanics' own outputs checked too. Made for the space recordings
(goi_jax/parity/record_space.py, dumps/parity-space) and run the same way on other sections' route chains for the
baseline.

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.space_parity --root dumps/parity-space
    ... --root dumps/parity-route --sections space radio_mast        # route chains of some sections (index "section")

Single step: from every recorded pre-step state (state_from_recording: NAT0 + SCR0, the space state inferred), one
sim.physics_step with the recorded mouse input, compared with the recording right after it: every dynamic body
(origin, angle, centre, velocities) against NAT0 of the next step (teleports and well forces are applied by then; at
an action's last step that also holds the plugin's pose re-sync, < 4e-6), body forces (the wells, bit for bit),
gravity (SCR2), and at action ends Won and the teleporter flags (meta script_after, when recorded). A step is "ok"
when every body value is within max(1e-5, float32 spacing at the game value) (goi_jax/parity/harness.py tolerance)
and forces, gravity and flags are exact.

Horizon: from the first canonical action and then every --every actions, --horizon actions open loop (sim.act, K
steps + end_action); after each action the pot and the hammer head (Tip) body origins against NAT1 of the action's
last step, gravity, Won and teleporter flags. A replayed action is "ok" when both are within the same tolerance
and the flags match.

The last line printed is DONE or STOPPED (reason).
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import level as L
from goi_jax.game import sim as SM
from goi_jax.parity import states as S

F32 = np.float32
BATCH = 64


def tolerance(ref):
    return np.maximum(F32(1e-5), np.spacing(np.abs(np.asarray(ref, F32))))


def within(a, ref):
    return np.abs(np.asarray(a, F32) - np.asarray(ref, F32)) <= tolerance(ref)


def _pad(tree, n):
    def p(a):
        a = np.asarray(a)
        return a if a.shape[0] == n else np.concatenate([a, np.repeat(a[-1:], n - a.shape[0], 0)])
    return jax.tree.map(p, tree)


def _stack(states):
    return jax.tree.map(lambda *a: np.stack(a), *states)


def _flags_after(chain):
    """Per action (won, teleport flags (NT, 3)) from meta script_after, or None."""
    sa = chain.meta.get("script_after")
    if not sa:
        return None
    return np.array([a["won"] for a in sa], bool), np.array([np.reshape(a["teleport"], (-1, 3)) for a in sa], bool)


def _with_flags(sim, s, chain, t):
    """The state at step t with Won and teleporter flags from the recording when t starts an action after one whose
    script state was recorded (meta script_after); else as inferred (space.initial)."""
    fl = _flags_after(chain)
    if fl is None or s.space is None:
        return s
    aos = np.asarray(chain.meta["action_of_step"])
    a = aos[t]
    if a == 0:
        return s
    won, tele = fl
    if t > 0 and aos[t - 1] == a:  # mid-action: teleporter flags not recorded (inferred); Won is sticky
        return s._replace(space=s.space._replace(won=np.bool_(won[a - 1])))
    return s._replace(space=s.space._replace(won=np.bool_(won[a - 1]), teleport=tele[a - 1]))


class Runner:
    def __init__(self, sim, k):
        self.sim, self.k = sim, k
        self.step = jax.jit(jax.vmap(lambda md, s, a: sim.physics_step(md, sim.params, s, a, axis_name="w"),
                                     in_axes=(L.model_axes(), 0, 0), axis_name="w"))
        self.act = jax.jit(jax.vmap(lambda md, s, a: sim.act(md, sim.params, s, a, k=k, axis_name="w"),
                                    in_axes=(L.model_axes(), 0, 0), axis_name="w"))


def single_step(runner, chain):
    sim = runner.sim
    lv = sim.level
    t_c = SM.canonical_step(chain)
    steps = np.arange(t_c, len(chain) - 1)
    aos = np.asarray(chain.meta.get("action_of_step", np.arange(len(chain))))
    last = np.r_[aos[1:] != aos[:-1], True]
    fl = _flags_after(chain)
    rows = []
    for i in range(0, len(steps), BATCH):
        st_ = steps[i:i + BATCH]
        states = [_with_flags(sim, sim.state_from_recording(chain, int(t)), chain, int(t)) for t in st_]
        jm, _, _ = lv.native_states(chain, st_, 0)
        model = lv.model_for(_pad(jm, BATCH))
        out, _ = runner.step(model, _pad(_stack(states), BATCH), jnp.asarray(_pad(chain.action[st_], BATCH)))
        out = jax.tree.map(lambda a: np.asarray(a)[:len(st_)], out)
        _, ref, _ = lv.native_states(chain, st_ + 1, 0)
        bd, rb = out.world.bodies, ref.bodies
        dyn = np.asarray(lv.body_type) == 2
        body_ok = np.ones(len(st_), bool)
        body_exact = np.ones(len(st_), bool)
        for q in ("xf_p", "c", "v"):
            body_ok &= within(getattr(bd, q), getattr(rb, q)).all(-1)[:, dyn].all(1)
            body_exact &= (getattr(bd, q) == getattr(rb, q)).all(-1)[:, dyn].all(1)
        for q in ("a", "w"):
            body_ok &= within(getattr(bd, q), getattr(rb, q))[:, dyn].all(1)
            body_exact &= (getattr(bd, q) == getattr(rb, q))[:, dyn].all(1)
        force = (bd.force == rb.force).all(-1)[:, dyn].all(1)
        g_rec = np.stack([chain.field(2, "gravity.x")[st_], chain.field(2, "gravity.y")[st_]], 1)
        grav = (out.world.gravity == g_rec).all(1)
        flags = np.ones(len(st_), bool)
        if fl is not None and out.space is not None:
            won, tele = fl
            a = aos[st_]
            m = last[st_]
            # at action ends; not on the win action from a mid-action state (whether the exit already happened
            # earlier in the action is not recorded)
            known = (aos[np.maximum(st_ - 1, 0)] != a) | (won[a] == won[np.maximum(a - 1, 0)])
            flags = np.where(m & known, (out.space.won == won[a]) & (out.space.teleport == tele[a]).all((1, 2)), True)
        jump = np.abs(rb.xf_p[:, sim.pot, 0] - np.asarray(chain.body(1, 0)["xf"][st_, 0])) > 5
        rows.append(dict(step=st_, body_ok=body_ok, body_exact=body_exact, force=force, gravity=grav, flags=flags,
                         teleport=jump, g_change=g_rec[:, 1] != chain.field(0, "gravity.y")[st_]))
    return {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}


def horizon(runner, chain, horizon_actions=8, every=4):
    sim, k = runner.sim, runner.k
    lv = sim.level
    aos = np.asarray(chain.meta.get("action_of_step", np.arange(len(chain))))
    first = np.flatnonzero(np.r_[True, aos[1:] != aos[:-1]])
    lastst = np.r_[first[1:] - 1, len(chain) - 1]
    t_c = SM.canonical_step(chain)
    a0 = int(np.searchsorted(first, t_c))
    starts = np.arange(a0, len(first) - 1, every)
    fl = _flags_after(chain)
    acts_req = np.asarray(chain.action)[first]  # the input the controller read (clamped), per action
    H = horizon_actions
    pot_ok, tip_ok, flag_ok, errs = [], [], [], []
    for i in range(0, len(starts), BATCH):
        sa = starts[i:i + BATCH]
        st = _stack([_with_flags(sim, sim.state_from_recording(chain, int(first[a])), chain, int(first[a])) for a in sa])
        n = len(sa)
        st = _pad(st, BATCH)
        jm, _, _ = lv.native_states(chain, first[sa], 0)   # the start's joint frames (horizon.py "sim" mode)
        model = lv.model_for(_pad(jm, BATCH))
        valid = np.zeros((n, H), bool)
        pe, te, fo = np.full((n, H), np.nan), np.full((n, H), np.nan), np.ones((n, H), bool)
        po, to_ = np.zeros((n, H), bool), np.zeros((n, H), bool)
        s = jax.tree.map(jnp.asarray, st)
        for h in range(H):
            idx = np.minimum(sa + h, len(first) - 1)
            a = _pad(acts_req[idx], BATCH)
            s, _ = runner.act(model, s, jnp.asarray(a))
            bd = jax.tree.map(np.asarray, s.world.bodies)
            v = (sa + h) < len(first)
            valid[:, h] = v
            tl = lastst[idx]
            pot_ref = np.asarray(chain.body(1, 0)["xf"])[tl, :2]
            tip_ref = np.asarray(chain.body(1, lv.names["body_names"].index(
                "Player/Hub/Slider/Handle/PoleMiddle/Tip"))["xf"])[tl, :2]
            pot, tip = bd.xf_p[:n, sim.pot], bd.xf_p[:n, sim.tip]
            po[:, h] = within(pot, pot_ref).all(-1)
            to_[:, h] = within(tip, tip_ref).all(-1)
            pe[:, h] = np.abs(pot - pot_ref).max(-1)
            te[:, h] = np.abs(tip - tip_ref).max(-1)
            g = np.asarray(s.world.gravity)[:n]
            g_rec = np.stack([chain.field(2, "gravity.x")[tl], chain.field(2, "gravity.y")[tl]], 1)
            ok = (g == g_rec).all(1)
            if fl is not None and s.space is not None:
                won, tele = fl
                ok &= (np.asarray(s.space.won)[:n] == won[idx]) & (np.asarray(s.space.teleport)[:n] == tele[idx]).all((1, 2))
            fo[:, h] = ok
        pot_ok.append(np.where(valid, po, True))
        tip_ok.append(np.where(valid, to_, True))
        flag_ok.append(np.where(valid, fo, True))
        errs.append(np.where(valid, np.maximum(pe, te), np.nan))
    return dict(pot_ok=np.concatenate(pot_ok), tip_ok=np.concatenate(tip_ok), flags=np.concatenate(flag_ok),
                err=np.concatenate(errs))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity-space")
    ap.add_argument("--sim-root", default=None, help="recording session for Sim.from_recordings (default: --root)")
    ap.add_argument("--chains", nargs="*", help="chain file stems (default: all)")
    ap.add_argument("--sections", nargs="*", help="only chains whose index 'section' is one of these")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--no-horizon", action="store_true")
    ap.add_argument("--pose-nudge", action="store_true")
    ap.add_argument("--save", help="write the per-chain table as JSON here")
    args = ap.parse_args(argv)
    status = "DONE"
    table = []
    try:
        t0 = time.perf_counter()
        sim = SM.Sim.from_recordings(args.sim_root or args.root, pose_nudge=args.pose_nudge)
        entries = S.load_index(args.root)["chains"]
        if args.sections:
            secs = {x.replace("_", " ") for x in args.sections}
            entries = [e for e in entries if e.get("section") in secs]
        if args.chains:
            entries = [e for e in entries if any(Path(e["file"]).stem.startswith(c) for c in args.chains)]
        k = int(entries[0].get("k", 4))
        runner = Runner(sim, k)
        print(f"{len(entries)} chains, sim built in {time.perf_counter() - t0:.1f} s", flush=True)
        hdr = (f"{'chain':34s} {'steps':>5s} {'ok':>6s} {'exact':>6s} {'force':>6s} {'grav':>6s} {'flags':>6s} "
               f"{'tp':>3s} {'g-sw':>4s} | {'h-acts':>6s} {'h-ok':>6s} {'h-flag':>6s} {'h-1e-3':>6s} {'max err':>8s}")
        print(hdr, flush=True)
        for e in entries:
            chain = S.load_chain(Path(args.root) / e["file"])
            r = single_step(runner, chain)
            ok = r["body_ok"] & r["force"] & r["gravity"] & r["flags"]
            row = dict(chain=e["file"], steps=int(len(ok)), ok=int(ok.sum()), exact=int(r["body_exact"].sum()),
                       force=int(r["force"].sum()), gravity=int(r["gravity"].sum()), flags=int(r["flags"].sum()),
                       teleports=int(r["teleport"].sum()), g_switches=int(r["g_change"].sum()),
                       bad_steps=[int(x) for x in r["step"][~ok][:12]])
            line = (f"{Path(e['file']).stem:34s} {len(ok):5d} {ok.mean():6.3f} {r['body_exact'].mean():6.3f} "
                    f"{r['force'].mean():6.3f} {r['gravity'].mean():6.3f} {r['flags'].mean():6.3f} "
                    f"{r['teleport'].sum():3d} {r['g_change'].sum():4d} |")
            if not args.no_horizon:
                h = horizon(runner, chain, args.horizon, args.every)
                v = ~np.isnan(h["err"])
                hok = h["pot_ok"] & h["tip_ok"] & h["flags"]
                row.update(h_actions=int(v.sum()), h_ok=int((hok & v).sum()), h_flags=int((h["flags"] & v).sum()),
                           h_1e3=int((np.nan_to_num(h["err"], nan=9) <= 1e-3).sum()),
                           h_max_err=float(np.nanmax(h["err"])) if v.any() else 0.0,
                           h_err_by_action=[float(x) for x in np.nanmax(h["err"], 0)])
                line += (f" {v.sum():6d} {row['h_ok'] / max(1, v.sum()):6.3f} {row['h_flags'] / max(1, v.sum()):6.3f} "
                         f"{row['h_1e3'] / max(1, v.sum()):6.3f} {row['h_max_err']:8.2e}")
            print(line + (f"  bad {row['bad_steps']}" if row["bad_steps"] else ""), flush=True)
            table.append(row)
        tot = {q: sum(r[q] for r in table) for q in ("steps", "ok", "exact", "force", "gravity", "flags", "teleports",
                                                      "g_switches")}
        line = (f"{'TOTAL':34s} {tot['steps']:5d} {tot['ok'] / tot['steps']:6.3f} {tot['exact'] / tot['steps']:6.3f} "
                f"{tot['force'] / tot['steps']:6.3f} {tot['gravity'] / tot['steps']:6.3f} {tot['flags'] / tot['steps']:6.3f} "
                f"{tot['teleports']:3d} {tot['g_switches']:4d} |")
        if not args.no_horizon:
            ha = sum(r["h_actions"] for r in table)
            line += (f" {ha:6d} {sum(r['h_ok'] for r in table) / ha:6.3f} {sum(r['h_flags'] for r in table) / ha:6.3f} "
                     f"{sum(r['h_1e3'] for r in table) / ha:6.3f} {max(r['h_max_err'] for r in table):8.2e}")
        print(line, flush=True)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    if args.save:
        Path(args.save).write_text(json.dumps(table, indent=1))
    print(status)


if __name__ == "__main__":
    main()

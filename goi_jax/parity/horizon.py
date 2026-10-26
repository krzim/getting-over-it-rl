"""How long do simulated trajectories stay with the game's? Open-loop replay of the recorded chains, at any K.

    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.horizon                          # K=1 chains, all modes
    ... -m goi_jax.parity.horizon --root dumps/parity-k4                                   # the K=4 chains
    ... -m goi_jax.parity.horizon --chains c01 c24 --modes sim oracle --save horizon.npz

Each chain (goi_jax.parity.record; K from its meta "action_of_step") starts from the first recorded action whose
first step has the canonical joint order (action 0, or 1 after a placement): the native world NAT0 plus the scripts'
SCR0. From there the JAX lockstep game (goi_jax/game/sim.py: per action K physics_steps, then end_action, the joint
re-creation, as Game.Step) runs the recorded mouse inputs open loop, carrying its own contacts, impulses, controller,
reaction torque and friction-switch state. After every action the pot and the hammer head (Tip) body origins are
compared with the game's NAT1 of the action's last step. Per chain: the first action at which max(|pot error|,
|tip error|) exceeds 1e-5, 1e-3, 1e-2 and 0.1 m (censored at the chain length), and the first action whose
friction-switch flags, tracked collider sets or HammerReturn snap differ from the recording (a branch flip).

Three modes:
- "sim": what an RL rollout gets (anchors and Tip z fixed at the start values, no pose nudge).
- "model": Sim(pose_nudge=True): the Transform round trip modelled by goi_jax/game/pose.py (Transform positions for
  the controller, the nudge at end_action). The Transform write is bit-exact (pose.py); the re-created
  joints' anchors (Unity anchor * lossyScale, pose.joint_anchor) are not yet applied.
- "oracle": "sim" plus the recorded per-action pose round trip injected after each action (NAT0 of the next action's
  first step minus NAT1 of this action's last step, every body), the re-created joints' recorded anchors and the
  Tip Transform z. The gap between sim and oracle is what the unmodelled Transform re-sync costs (sim.py).
The last line printed is DONE or STOPPED (reason).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import level as L
from goi_jax.game import sim as SM
from goi_jax.parity import states as S

THRESHOLDS = (1e-5, 1e-3, 1e-2, 0.1)
F32 = np.float32


def _chain_inputs(sim, chain, capacity, max_actions=None, transforms=None):
    """Start state, per-action inputs, oracle tables and references for one chain (numpy). Actions are the
    recorder's (meta action_of_step; one step each at K = 1)."""
    lv = sim.level
    aos = np.asarray(chain.meta.get("action_of_step", np.arange(len(chain))))
    first = np.flatnonzero(np.r_[True, aos[1:] != aos[:-1]])  # first step of every action
    last = np.r_[first[1:] - 1, len(chain) - 1]
    k = int(np.bincount(aos).max())
    if not (np.diff(first) == k).all():
        raise ValueError("actions of unequal length in the chain")
    t_canon = SM.canonical_step(chain)
    a0 = int(np.searchsorted(first, t_canon))  # the first action starting at or after the canonical step
    acts = np.arange(a0, len(first) if max_actions is None else min(len(first), a0 + max_actions))
    n = len(acts)
    f, l = first[acts], last[acts]
    s0 = sim.state_from_recording(chain, int(f[0]), capacity, transforms)
    jm, _, _ = lv.native_states(chain, f, 0)  # the climber joints are re-created once per action
    jm = jax.tree.map(np.asarray, jm)
    # per-action nudge: NAT0 of the next action's first step minus NAT1 of this action's last step (zero at the end)
    order = np.argsort(lv.native_slot)[:-1]  # engine body per native slot
    B = lv.n_bodies
    d_p = np.zeros((n, B, 2), F32)
    d_c = np.zeros((n, B, 2), F32)
    nxt = np.minimum(l + 1, len(chain) - 1)
    for name, out in (("xf", d_p), ("c", d_c)):
        a0_ = chain.bodies[0][name][..., :2].astype(F32)
        a1_ = chain.bodies[1][name][..., :2].astype(F32)
        out[:-1, order] = (a0_[nxt[:-1]] - a1_[l[:-1]]).astype(F32)
    tip_z = chain.scr[0]["T.z"][nxt].astype(F32)  # Tip z for the NEXT action
    bn = lv.names["body_names"]
    ref = {key: chain.body(1, bn.index(name))["xf"][l][:, :2].astype(F32)
           for key, name in (("pot", "Player"), ("tip", "Player/Hub/Slider/Handle/PoleMiddle/Tip"))}
    ncol = sim.n_colliders
    ht = np.zeros((n, ncol), bool)
    pt = np.zeros((n, ncol), bool)
    for i, s in enumerate(l):
        ht[i, chain.hammer_keys[2][s]] = True
        pt[i, chain.pot_keys[2][s]] = True
    ev = chain.events
    snaps = ev[(ev["kind"] == S.EV_HAMMER_RETURN) & (ev["script"] == 1) & ((ev["flags"] & 1) == 1)]["step"]
    snapped = np.array([np.isin(np.arange(a, b + 1), snaps).any() for a, b in zip(f, l)])
    rec = dict(hammer_slide=chain.scr[2]["hammerSlide"][l] > 0, pot_slide=chain.scr[2]["potSlide"][l] > 0,
               hammer_tracked=ht, pot_tracked=pt, snapped=snapped)
    return dict(t0=int(f[0]), k=k, n=n, s0=s0, action=chain.action[f].astype(F32), jm=jm, d_p=d_p, d_c=d_c,
                tip_z=tip_z, ref=ref, rec=rec)


def _pad(a, T):
    a = np.asarray(a)
    if a.shape[0] == T:
        return a
    return np.concatenate([a, np.repeat(a[-1:], T - a.shape[0], 0)])


def make_runner(sim, k):
    """jit(vmap over replays) of the open-loop scan over actions (K physics steps each, then end_action). Inputs per
    replay: start state, (T, 2) actions, (T,) JointModel fields, nudges, Tip z, and `oracle` (bool) selecting
    whether the recorded per-action tables are used. Positions are compared before end_action (sim.pose_nudge
    applies its modelled nudge there)."""
    params = sim.params

    def replay(s0, actions, jm, d_p, d_c, tip_z, oracle):
        jm0 = jax.tree.map(lambda a: a[0], jm)

        def body(s, x):
            a, jm_t, dp, dc, tz = x
            jm_use = jax.tree.map(lambda u, v: jnp.where(oracle, u, v), jm_t, jm0)
            model = sim.level.model_for(L.JointModel(*jm_use))
            a = SM.clamp_magnitude(jnp.asarray(a, jnp.float32), jnp.float32(100))
            s, info = jax.lax.scan(lambda st, _: sim.physics_step(model, params, st, a, "w"), s, None, length=k)
            bd = s.world.bodies
            out = dict(pot=bd.xf_p[sim.pot], tip=bd.xf_p[sim.tip], hammer_slide=s.fric.hammer_slide,
                       pot_slide=s.fric.pot_slide, hammer_tracked=s.fric.hammer_tracked,
                       pot_tracked=s.fric.pot_tracked, snapped=info.snapped.any(), toi=info.toi_events.sum())
            s = sim.end_action(s)
            bd = s.world.bodies
            # the recorded pose round trip, between this action and the next
            bd = bd._replace(xf_p=jnp.where(oracle, bd.xf_p + dp, bd.xf_p), c=jnp.where(oracle, bd.c + dc, bd.c))
            s = s._replace(world=s.world._replace(bodies=bd), tip_z=jnp.where(oracle, tz, s.tip_z))
            return s, out
        _, out = jax.lax.scan(body, s0, (actions, jm, d_p, d_c, tip_z))
        return out
    return jax.jit(jax.vmap(replay, axis_name="w"))


MODES = ("sim", "model", "oracle")


def run(root="dumps/parity", chains=None, capacity=96, progress=True, max_actions=None, modes=MODES):
    """Replay every selected chain in each mode (at most max_actions actions each); returns a dict of per-replay
    results (numpy). Modes: sim (no nudge), model (Sim(pose_nudge=True): pose.py's modelled Transform round trip),
    oracle (no model; the recorded nudge, joint anchors and Tip z injected between actions)."""
    from goi_jax.game import pose as P
    root = Path(root)
    t_start = time.time()
    entries = S.load_index(root)["chains"]
    if chains:
        entries = [e for e in entries if any(Path(e["file"]).stem.startswith(c) for c in chains)]
    sims = {}
    if any(md != "model" for md in modes):
        sims[False] = SM.Sim.from_recordings(root)
    if "model" in modes:
        sims[True] = SM.Sim.from_recordings(root, pose_nudge=True)
    loaded = [(e["file"], S.load_chain(root / e["file"]), P.load_transforms(root / e["file"])) for e in entries]
    rows, outs = [], {}
    k = None
    for nudge in sims:
        sim = sims[nudge]
        items = [(f, _chain_inputs(sim, c, capacity, max_actions, tf)) for f, c, tf in loaded]
        ks = {it["k"] for _, it in items}
        if len(ks) != 1:
            raise ValueError(f"chains with different K: {sorted(ks)}")
        k = ks.pop()
        T = max(it["n"] for _, it in items)
        mode_list = [md for md in modes if (md == "model") == nudge]
        sub = [(f, it, md) for md in mode_list for f, it in items]

        def stack(get):
            return jax.tree.map(lambda *a: np.stack(a), *[get(it) for _, it, _ in sub])
        s0 = stack(lambda it: it["s0"])
        actions = stack(lambda it: _pad(it["action"], T))
        jm = stack(lambda it: L.JointModel(*(_pad(a, T) for a in it["jm"])))
        d_p = stack(lambda it: _pad(it["d_p"], T))
        d_c = stack(lambda it: _pad(it["d_c"], T))
        tip_z = stack(lambda it: _pad(it["tip_z"], T))
        oracle = np.array([md == "oracle" for _, _, md in sub])
        if progress:
            print(f"K={k}: {len(items)} chains, {T} actions max, {len(sub)} replays ({'/'.join(mode_list)}); "
                  f"compiling + running ...", flush=True)
        out = jax.tree.map(np.asarray, make_runner(sim, k)(s0, actions, jm, d_p, d_c, tip_z, oracle))
        for r, row in enumerate(sub):
            rows.append(row)
            outs[len(rows) - 1] = jax.tree.map(lambda a: a[r], out)
    if progress:
        print(f"replayed in {time.time() - t_start:.0f} s", flush=True)
    res = dict(file=np.array([f for f, _, _ in rows]), mode=np.array([md for _, _, md in rows]), k=k,
               n=np.array([it["n"] for _, it, _ in rows]), t0=np.array([it["t0"] for _, it, _ in rows]),
               first={th: [] for th in THRESHOLDS}, first_flip=[], err=[], toi=[])
    for r, (f, it, md) in enumerate(rows):
        n, o = it["n"], outs[r]
        err = np.maximum(np.abs(o["pot"][:n] - it["ref"]["pot"]).max(1), np.abs(o["tip"][:n] - it["ref"]["tip"]).max(1))
        res["err"].append(err)
        for th in THRESHOLDS:
            hit = np.flatnonzero(err > th)
            res["first"][th].append(int(hit[0]) if len(hit) else n)
        flip = np.zeros(n, bool)
        for key in ("hammer_slide", "pot_slide", "snapped"):
            flip |= o[key][:n] != it["rec"][key]
        for key in ("hammer_tracked", "pot_tracked"):
            flip |= (o[key][:n] != it["rec"][key]).any(1)
        hit = np.flatnonzero(flip)
        res["first_flip"].append(int(hit[0]) if len(hit) else n)
        res["toi"].append(int(np.flatnonzero(o["toi"][:n] > 0)[0]) if (o["toi"][:n] > 0).any() else n)
    res["first"] = {th: np.array(v) for th, v in res["first"].items()}
    res["first_flip"] = np.array(res["first_flip"])
    res["first_toi"] = np.array(res["toi"])
    del res["toi"]
    return res


def format_report(res) -> str:
    k = res["k"]
    lines = [f"Open-loop horizon at K={k} ({k} physics step{'s' if k > 1 else ''} per action): first ACTION (0 = the first "
             "replayed) after which", "max(|pot - game|, |tip - game|) exceeds each threshold (positions after the action's "
             f"last step, before the pose update); '>n' = never within the chain's n actions. Physics steps = actions x {k}.",
             "flip = first friction-switch / tracked set / HammerReturn difference; toi = first action with a TOI event.", ""]
    hdr = f"{'chain':28s} {'mode':6s} {'n':>4s} " + " ".join(f"{th:>7g}" for th in THRESHOLDS) + \
        f" {'flip':>6s} {'toi':>5s}"
    lines.append(hdr)

    def cell(v, n):
        return f">{n}" if v >= n else str(v)
    for r in range(len(res["file"])):
        n = res["n"][r]
        lines.append(f"{Path(res['file'][r]).stem:28s} {res['mode'][r]:6s} {n:4d} " +
                     " ".join(f"{cell(res['first'][th][r], n):>7s}" for th in THRESHOLDS) +
                     f" {cell(res['first_flip'][r], n):>6s} {cell(res['first_toi'][r], n):>5s}")
    lines.append("")
    lines.append(f"Distribution over chains, in actions (x{k} = physics steps; censored chains count at their length; "
                 "median [min, quartiles, max]):")
    for mode in dict.fromkeys(res["mode"]):
        msk = res["mode"] == mode
        lines.append(f"  mode {mode}:")
        for th in THRESHOLDS + ("flip",):
            v = res["first_flip"][msk] if th == "flip" else res["first"][th][msk]
            never = int((v >= res["n"][msk]).sum())
            q = np.percentile(v, [0, 25, 50, 75, 100])
            lines.append(f"    {str(th):>6s}: median {q[2]:6.1f}  [min {q[0]:.0f}, q25 {q[1]:.0f}, q75 {q[3]:.0f}, "
                         f"max {q[4]:.0f}]  never: {never}/{int(msk.sum())}")
    return "\n".join(lines)


def save(res, path):
    flat = {("chain" if key == "file" else key): v for key, v in res.items() if key not in ("first", "err")}
    for th, v in res["first"].items():
        flat[f"first_{th:g}"] = v
    T = max(len(e) for e in res["err"])
    flat["err"] = np.stack([np.pad(e, (0, T - len(e)), constant_values=np.nan) for e in res["err"]])
    np.savez(path, **flat)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="dumps/parity")
    ap.add_argument("--chains", nargs="*", help="chain file stems or prefixes (default: all)")
    ap.add_argument("--save", help="write per-replay results to this .npz")
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    ap.add_argument("--max-actions", type=int)
    args = ap.parse_args(argv)
    try:
        res = run(args.root, args.chains, modes=tuple(args.modes), max_actions=args.max_actions)
        print(format_report(res))
        if args.save:
            save(res, args.save)
        print("DONE")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

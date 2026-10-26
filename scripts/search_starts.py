"""Go-Explore start-state search in the JAX sim (goi_jax/training/search.py): grow an archive of the best settled state
per (2 m route-progress bin x hammer-direction sector) by branching thousands of envs from archive cells, all on
device. Writes a start bank that training loads with --start-group all+searched (or searched; starts.py).

Resumable: rerunning the same command continues from --out (the bank file is also the resume point). Last line
DONE (--max-rounds reached) or STOPPED (time budget, signal, error).

    # compile without the GPU lock, then search under it (RTX 3090 shared; cuda_run.sh with GOI_LATE_LOCK=1)
    GOI_LATE_LOCK=1 scripts/cuda_run.sh python -u scripts/search_starts.py --name s1 \\
        --policy runs/route-fast/checkpoints/update_0000704.pkl --seed-group all --max-minutes 12
    # train from it: scripts/train_jax.py ... --start-group all+searched  (file: runs/search/bank.pkl by default,
    #   or GOI_SEARCHED_BANK=<file>)

--bench "SEEDS:MODE,..." runs short fresh searches one after another with one compiled program and prints the
frontier over time per phase (SEEDS: start | all | section names joined by '+'; MODE: random | policy).
Output: one JSON line per device call, a coverage table per section, then DONE / STOPPED.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="search", help="runs/search/<name>/ (log)")
    ap.add_argument("--out", help="bank file (default runs/search/bank.pkl; also the resume point)")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing --out")
    ap.add_argument("--policy", help="training checkpoint whose policy mean drives the branches (+ OU noise)")
    ap.add_argument("--use-policy", type=int, choices=(0, 1), help="default 1 with --policy")
    ap.add_argument("--seed-group", default="all", help="recorded start group the seeds come from (starts.GROUPS)")
    ap.add_argument("--seed-sections", default="", help="comma-separated sections to seed from (default: all but space)")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), help="only branch from cells in [LO, HI)")
    ap.add_argument("--branches", type=int, default=8192)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--rounds", type=int, default=1, help="rounds per device call")
    ap.add_argument("--max-rounds", type=int, default=10**9)
    ap.add_argument("--max-minutes", type=float, default=12.0, help="GPU (lock held) time budget; STOPPED after")
    ap.add_argument("--save-minutes", type=float, default=3.0)
    ap.add_argument("--bin-size", type=float, default=2.0)
    ap.add_argument("--angle-cells", type=int, default=4)
    ap.add_argument("--max-progress", type=float, default=760.0)
    ap.add_argument("--min-held", type=int, default=2)
    ap.add_argument("--max-speed", type=float, default=6.0)
    ap.add_argument("--frontier-bonus", type=float, default=3.0)
    ap.add_argument("--pick", choices=("section", "cell"), default="section")
    ap.add_argument("--solver", default="pallas", choices=("xla", "pallas"))
    ap.add_argument("--compact", type=int, default=1, choices=(0, 1))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", help="phases SEEDS:MODE,... (fresh archive each, no bank written)")
    ap.add_argument("--phase-minutes", type=float, default=2.0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    import jax
    import numpy as np
    jax.config.update("jax_compilation_cache_dir", str((ROOT / "runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    from goi_jax import gpu_lock
    from goi_jax.training import backend
    from goi_jax.training import search as S
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig

    wanted = os.environ.get("JAX_PLATFORMS", "").split(",")[0]
    if wanted and wanted != "cpu" and jax.default_backend() == "cpu":
        print(f"STOPPED: JAX_PLATFORMS asks for {wanted} but JAX came up on the CPU", flush=True)
        return 1
    say = lambda *a: print(*a, flush=True)  # noqa: E731

    # ---- env (the policy's env settings if a policy is given)
    pol_cfg = None
    if args.policy:
        pol, pol_cfg = S.load_policy(args.policy)
    cfg = TrainConfig.from_dict(pol_cfg) if pol_cfg else TrainConfig()
    cfg = dataclasses.replace(cfg, name="search", start_group=args.seed_group, solver=args.solver,
                              compact_fixtures=bool(args.compact))
    t0 = time.perf_counter()
    env = St.make_env(cfg)
    backend.select_solver(env, cfg.solver)
    bank = env.default_params.starts
    bi = env.bank_index
    say(f"env built in {time.perf_counter() - t0:.1f} s: {len(bi.section)} seed candidates ({cfg.start_group}), "
        f"backend {jax.default_backend()}")
    if not args.policy:
        pol = S.no_policy(env.obs_size, cfg.hidden)
    if args.use_policy is not None:
        pol = S.with_use(pol, args.use_policy)
    sc = S.SearchConfig(bin_size=args.bin_size, angle_cells=args.angle_cells, max_progress=args.max_progress,
                        branches=args.branches, horizon=args.horizon, rounds=args.rounds, min_held=args.min_held,
                        max_speed=args.max_speed, frontier_bonus=args.frontier_bonus, pick=args.pick)
    se = S.Searcher(env, sc, bounded_mean=cfg.bounded_mean, clip_obs=cfg.clip_obs, norm_obs=cfg.norm_obs)
    names = {n: i for i, (_, n) in enumerate(St.SECTIONS)}

    def seeds_of(spec):
        """Bank states of the sections in spec ('start', 'all' or names joined by '+' / ',')."""
        if spec in ("", "all"):
            m = bi.progress < sc.max_progress
        else:
            want = [s.strip() for s in spec.replace("+", ",").split(",") if s.strip()]
            bad = [w for w in want if w not in names]
            if bad:
                raise ValueError(f"unknown sections {bad}; known: {list(names)}")
            m = np.isin(bi.section, [names[w] for w in want])
        idx = np.nonzero(m)[0]
        if not len(idx):
            raise ValueError(f"no seed states for {spec!r} in start group {cfg.start_group!r}")
        return jax.tree.map(lambda a: np.asarray(a)[idx], bank), len(idx)

    one = jax.tree.map(lambda a: a[0], bank)

    def summary(ar):
        h = {k: np.asarray(getattr(ar, k)) for k in ("filled", "progress", "root")}
        return h, S.coverage(h, sc)

    def fresh_archive(spec):
        st, n = seeds_of(spec)
        ar = se.seed(S.archive_init(sc, one), st)
        return ar, n

    out = Path(args.out or St.SEARCHED_BANK)
    logdir = ROOT / "runs" / "search" / args.name
    logdir.mkdir(parents=True, exist_ok=True)
    log = open(logdir / ("bench.jsonl" if args.bench else "search.jsonl"), "a")

    # ---- phases: the bench list, or one resumable search
    if args.bench:
        phases = []
        for item in args.bench.split(","):
            spec, _, mode = item.rpartition(":")
            if mode not in ("random", "policy") or not spec:
                raise ValueError(f"--bench item {item!r}: SEEDS:random|policy")
            if mode == "policy" and not args.policy:
                raise ValueError("--bench ...:policy needs --policy")
            phases.append((spec, mode))
    else:
        phases = [(args.seed_sections or "all", "policy" if float(pol.use) > 0 else "random")]

    meta0, round0, wall0 = {}, 0, 0.0
    ar = None
    if not args.bench and out.exists() and not args.fresh:
        b = S.load_bank(out)
        ar = S.archive_from_bank(b, sc)
        meta0 = b.get("meta", {})
        round0, wall0 = int(meta0.get("rounds", 0)), float(meta0.get("wall_s", 0.0))
        say(f"resumed {out}: {len(b['cell'])} cells, {round0} rounds, {wall0 / 60:.1f} min")
    if ar is None:
        ar, n_seed = fresh_archive(phases[0][0])
        say(f"seeded {int(np.asarray(ar.filled).sum())} cells from {n_seed} bank states ({phases[0][0]})")

    tc = time.perf_counter()
    se.compile(ar, pol)   # no device work: fills the persistent cache before the lock
    say(f"compiled in {time.perf_counter() - tc:.1f} s ({sc.branches} branches x {sc.horizon} actions x "
        f"{sc.rounds} rounds per call, {sc.n_cells} cells)")

    stop = {"why": None}
    signal.signal(signal.SIGTERM, lambda s, f: stop.update(why="signal SIGTERM"))
    signal.signal(signal.SIGINT, lambda s, f: stop.update(why="signal SIGINT"))
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), round0)
    status = None
    lo, hi = (args.window or (-np.inf, np.inf))

    def save(ar, rounds, wall):
        if args.bench:
            return
        S.save_bank(out, ar, sc, dict(meta0, rounds=rounds, wall_s=wall, policy=args.policy,
                                     use_policy=float(pol.use), seed_group=cfg.start_group,
                                     compact=bool(args.compact), contact_capacity=cfg.contact_capacity,
                                     config=cfg.to_dict()))

    hold = gpu_lock.held()
    say(f"GPU lock taken after {hold.__enter__():.1f} s wait")
    rounds = round0
    try:
        for pi, (spec, mode) in enumerate(phases):
            if pi > 0:
                ar, n_seed = fresh_archive(spec)
            p_use = S.with_use(pol, 1.0 if mode == "policy" else 0.0)
            budget = 60 * (args.phase_minutes if args.bench else args.max_minutes)
            t_start = time.perf_counter()
            t_save = t_start
            r_phase = 0
            h, cov = summary(ar)
            say(json.dumps(dict(phase=f"{spec}:{mode}", t=0.0, cells=int(h["filled"].sum()),
                                frontier=float(h["progress"][h["filled"]].max()))))
            while True:
                if stop["why"]:
                    break
                el = time.perf_counter() - t_start
                if el > budget:
                    if not args.bench:
                        stop["why"] = f"time budget of {args.max_minutes:g} min used"
                    break
                if rounds >= args.max_rounds:
                    status = "DONE"
                    break
                tk = time.perf_counter()
                ar, st = se.run(ar, jax.random.fold_in(key, rounds), rounds, p_use, lo, hi)
                st = jax.device_get(st)
                dt = time.perf_counter() - tk
                rounds += sc.rounds
                r_phase += sc.rounds
                h, cov = summary(ar)
                f = h["filled"]
                top = {c["section"]: c["reached_from"] for c in cov if c["reached_from"] is not None}
                row = dict(phase=f"{spec}:{mode}", round=rounds, t=round(time.perf_counter() - t_start, 1),
                           call_s=round(dt, 2), env_steps_s=round(sc.branches * sc.horizon * sc.rounds / dt),
                           cells=int(f.sum()), bins=int(f.reshape(sc.n_bins, -1).any(-1).sum()),
                           frontier=float(h["progress"][f].max()), new=int(st["new"]), deaths=int(st["deaths"]),
                           invalid=int(st["invalid"]), wins=int(st["wins"]), best_seen=float(st["best"]),
                           reached_from=top)
                say(json.dumps(row))
                log.write(json.dumps(row) + "\n")
                log.flush()
                if not args.bench and time.perf_counter() - t_save > 60 * args.save_minutes:
                    save(ar, rounds, wall0 + time.perf_counter() - t_start)
                    t_save = time.perf_counter()
            h, cov = summary(ar)
            say(f"--- coverage {spec}:{mode} after {r_phase} rounds, {(time.perf_counter() - t_start) / 60:.1f} min")
            for c in cov:
                say(json.dumps(dict(phase=f"{spec}:{mode}", **c)))
                log.write(json.dumps(dict(phase=f"{spec}:{mode}", **c)) + "\n")
            if not args.bench:
                save(ar, rounds, wall0 + time.perf_counter() - t_start)
                say(f"bank: {out} ({int(h['filled'].sum())} states)")
            if stop["why"] or status:
                break
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    finally:
        hold.__exit__(None, None, None)
        log.close()
    if status is None:
        status = f"STOPPED: {stop['why']}; rerun to resume from {out}" if stop["why"] else "DONE"
    if status == "DONE" and not args.bench:
        status = f"DONE: {rounds} rounds; bank {out}"
    print(status, flush=True)
    return 0 if status.startswith("DONE") else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

"""A/B runs of learning-speed options: each variant is a short scripts/train_jax.py run warm-started from one checkpoint
with the same wall budget on the GPU (--max-minutes, counted while the shared GPU lock is held; compiles run before
the lock is taken), run one after another, then compared from their metrics.jsonl and final checkpoints.

    # all default variants, 20 min each, from route-fast's latest checkpoint (resumable: rerun the same command)
    .venv/bin/python scripts/ab_runs.py --tag ab1 --base runs/route-fast/checkpoints/latest.pkl --minutes 20
    # a subset, or only the table
    .venv/bin/python scripts/ab_runs.py --tag ab1 --base ... --variants base ent min-std --minutes 20
    .venv/bin/python scripts/ab_runs.py --tag ab1 --report

Runs are runs/<tag>-<variant>/ (log: runs/<tag>-<variant>/ab.log). A variant is finished when its train_jax.py run
ended with the time budget (runs/<tag>-<variant>/ab_done.json); a rerun skips finished variants and continues an
interrupted one from its own latest.pkl with what is left of its budget. All variants import the same code: a snapshot
of goi_jax taken on the first launch (runs/<tag>.code), copied into every variant's run folder (same compile cache).

Comparison (over the last --tail updates of each run; "start" = the base checkpoint's own last --tail updates):
episode return, bank_crossed (share of bank-started episodes, recent window, whose held best reached the next
section), mean held best of bank episodes, policy std, approx KL, gradient steps per update (80 = all epochs ran),
share of updates stopped by target_kl, env steps/s, updates done; then per-section crossing rates (from each final
checkpoint's archive: decayed recent counts) for sections with enough tries. Last line: DONE or STOPPED.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"

# name -> train_jax.py flags (on top of the base checkpoint's config). Keep names short: they become run folders.
VARIANTS = {
    "base": [],
    "ent": ["--ent-coef", "0.005"],
    "min-std": ["--min-log-std", f"{math.log(0.15):.4f}"],                  # floor std 0.15 (raised at the start)
    "std-reset": ["--log-std-reset", f"{math.log(0.15):.4f}"],              # raise to 0.15 once, then free
    "gamma": ["--gamma", "0.998"],
    "wide": ["--hidden", "512", "512"],                                      # exact widening of 2x256
    "deep": ["--hidden", "512", "512", "512"],                               # widening + one added layer (s 0.05: KL ~0.001)
    "section": ["--section-input", "1"],
    "epochs": ["--epochs", "15"],
}
DEFAULT = ["base", "ent", "min-std", "gamma", "deep", "section", "epochs"]


def say(*a):
    print(*a, flush=True)


def rows(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def base_info(base: Path) -> dict:
    with open(base, "rb") as f:
        ck = pickle.load(f)
    return dict(wall_s=float(ck.get("wall_s", 0.0)), update=int(ck["update"]), name=ck["config"].get("name"))


def snapshot(tag: str) -> Path:
    snap = RUNS / f"{tag}.code"
    if not (snap / "goi_jax" / "__init__.py").exists():
        tmp = RUNS / f"{tag}.code.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(REPO / "goi_jax", tmp / "goi_jax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        tmp.rename(snap)
    return snap


def run_variant(args, name, flags, info, child):
    run = RUNS / f"{args.tag}-{name}"
    done = run / "ab_done.json"
    if done.exists():
        say(f"[{name}] finished earlier ({json.loads(done.read_text()).get('minutes', '?')} min): skipped")
        return "done"
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "code" / "goi_jax" / "__init__.py").exists():
        shutil.copytree(snapshot(args.tag) / "goi_jax", run / "code" / "goi_jax")
    latest = run / "checkpoints" / "latest.pkl"
    used = 0.0
    if latest.exists():
        rs = rows(run / "metrics.jsonl")
        used = (rs[-1]["wall_s"] - info["wall_s"]) / 60 if rs else 0.0
    left = args.minutes - used
    if left < 0.5:
        done.write_text(json.dumps(dict(minutes=args.minutes, flags=flags)))
        return "done"
    cmd = [str(REPO / "scripts" / "cuda_run.sh"), "python", "-u", str(REPO / "scripts" / "train_jax.py"),
           "--name", run.name, "--max-minutes", f"{left:.2f}", *args.common, *flags]
    if not latest.exists():
        cmd += ["--resume", str(args.base)]
    say(f"[{name}] {time.strftime('%H:%M:%S')} {'resuming' if latest.exists() else 'starting'} with {left:.1f} min "
        f"left: {' '.join(flags) or '(base checkpoint config)'}")
    env = dict(os.environ, GOI_LATE_LOCK="1")
    env.pop("JAX_PLATFORMS", None)   # this process sets cpu for itself (line below main); the GPU runs must not inherit it
    with open(run / "ab.log", "a") as log:
        log.write(f"=== {time.strftime('%F %T')} {' '.join(cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
        child["p"] = p
        p.wait()
        child["p"] = None
    last = [ln for ln in (run / "ab.log").read_text().splitlines() if ln.startswith(("DONE", "STOPPED"))]
    last = last[-1] if last else f"(no DONE/STOPPED line, exit {p.returncode})"
    say(f"[{name}] {time.strftime('%H:%M:%S')} {last[:160]}")
    if "time budget" in last or last.startswith("DONE"):
        done.write_text(json.dumps(dict(minutes=args.minutes, flags=flags, end=last)))
        return "done"
    return last


# ---------------------------------------------------------------- report


def _mean(rs, k):
    xs = [r[k] for r in rs if k in r and isinstance(r[k], (int, float)) and math.isfinite(r[k])]
    return sum(xs) / len(xs) if xs else float("nan")


def summary(rs, tail):
    t = rs[-tail:]
    return dict(updates=len(rs), ret=_mean(t, "ep_return"), crossed=_mean(t, "bank_crossed"),
                bank_held=_mean(t, "win_bank_held"), std=_mean(t, "std"), kl=_mean(t, "approx_kl"),
                gs=_mean(t, "grad_steps"), stop=_mean(t, "kl_stopped"), sps=_mean(t, "env_steps_per_s"),
                lr=_mean(t, "lr"), ev=_mean(t, "explained_var"))


def section_rates(ck_path: Path):
    """{section index: (tries, crossed share)} from a checkpoint's archive (decayed recent counts)."""
    if not ck_path.exists():
        return {}
    with open(ck_path, "rb") as f:
        ck = pickle.load(f)
    ar = ck.get("archive")
    if ar is None or len(ar) < 6 or ar[4] is None:
        return {}
    tries, wins = ar[4], ar[5]
    return {i: (float(tries[i]), float(wins[i]) / max(float(tries[i]), 1e-6)) for i in range(len(tries))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="sweep name: runs are runs/<tag>-<variant>")
    ap.add_argument("--base", default="runs/route-fast/checkpoints/latest.pkl", help="checkpoint every variant starts from")
    ap.add_argument("--variants", nargs="+", default=DEFAULT, help=f"from {sorted(VARIANTS)} (default {DEFAULT})")
    ap.add_argument("--minutes", type=float, default=20.0, help="GPU wall budget per variant (lock held, compile excluded)")
    ap.add_argument("--common", nargs=argparse.REMAINDER, default=[],
                    help="train_jax.py flags for every variant (last on the line), e.g. --common --save-every-updates 25")
    ap.add_argument("--tail", type=int, default=20, help="updates averaged in the table")
    ap.add_argument("--min-tries", type=float, default=20.0)
    ap.add_argument("--report", action="store_true", help="only print the table")
    args = ap.parse_args(argv)
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        ap.error(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")
    args.base = Path(args.base).resolve()
    if not args.base.exists():
        say(f"STOPPED: no base checkpoint {args.base}")
        return 1
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")   # this process only reads pickles; the runs set their own
    # a frozen copy of the base checkpoint: route-fast keeps overwriting latest.pkl while the sweep runs
    frozen = RUNS / f"{args.tag}.base.pkl"
    if args.report and not frozen.exists():   # a report before the first launch must not freeze the base early
        frozen = args.base
    elif not frozen.exists():
        shutil.copy2(args.base, frozen.with_suffix(".tmp"))
        os.replace(frozen.with_suffix(".tmp"), frozen)
    base_run_metrics = args.base.parent.parent
    args.base = frozen
    info = base_info(frozen)
    args.base_run = base_run_metrics
    if args.report:
        report(args)
        say("DONE: report")
        return 0
    say(f"A/B {args.tag}: {len(args.variants)} variants x {args.minutes:.0f} min from {frozen} "
        f"({info['name']} update {info['update']})")
    child, stop = {"p": None}, {"why": None}

    def on_signal(sig, _f):
        stop["why"] = f"signal {signal.Signals(sig).name}"
        if child["p"] is not None:
            child["p"].send_signal(signal.SIGTERM)   # train_jax.py saves and stops (resumable)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    for name in args.variants:
        if stop["why"]:
            break
        st = run_variant(args, name, VARIANTS[name], info, child)
        if st != "done":
            stop["why"] = stop["why"] or f"variant {name}: {st}"
            break
    report(args)
    if stop["why"]:
        say(f"STOPPED: {stop['why']}; rerun the same command to continue")
        return 1
    say(f"DONE: {len(args.variants)} variants")
    return 0


def report(args):
    from goi_jax.training.starts import SECTIONS
    base_rows = rows(args.base_run / "metrics.jsonl")
    upd = base_info(args.base)["update"]
    cols = [("start", [r for r in base_rows if r.get("update", 0) <= upd], args.base)]
    for n in args.variants:
        run = RUNS / f"{args.tag}-{n}"
        cols.append((n, rows(run / "metrics.jsonl"), run / "checkpoints" / "latest.pkl"))
    say(f"\nA/B {args.tag}: means over the last {args.tail} updates of each run ('start': {args.base_run.name} "
        f"up to update {upd})")
    say(f"{'variant':<12}{'upd':>5}{'return':>8}{'crossed':>8}{'bank held':>10}{'std':>7}{'kl':>8}{'gs':>5}"
        f"{'kl stop':>8}{'lr':>9}{'ev':>6}{'st/s':>8}")
    for n, rs, _ in cols:
        if not rs:
            say(f"{n:<12}  (no metrics yet)")
            continue
        s = summary(rs, args.tail)
        say(f"{n:<12}{s['updates']:>5}{s['ret']:>8.2f}{s['crossed']:>8.3f}{s['bank_held']:>10.1f}{s['std']:>7.3f}"
            f"{s['kl']:>8.4f}{s['gs']:>5.0f}{s['stop']:>8.2f}{s['lr']:>9.2e}{s['ev']:>6.2f}{s['sps']:>8.0f}")
    rates = [(n, section_rates(p)) for n, _, p in cols]
    secs = sorted({i for _, r in rates for i, (t, _) in r.items() if t >= args.min_tries})
    if secs:
        say(f"\nper-section crossing: share of recent bank episodes started in the section whose held best reached "
            f"the next one (decayed counts in each final checkpoint; '-': < {args.min_tries:.0f} tries)")
        say(f"{'section':<18}" + "".join(f"{n[:9]:>10}" for n, _ in rates))
        for i in secs:
            cells = []
            for _, r in rates:
                t, c = r.get(i, (0.0, float("nan")))
                cells.append(f"{c:>10.2f}" if t >= args.min_tries else f"{'-':>10}")
            say(f"{SECTIONS[i][1][:17]:<18}" + "".join(cells))


if __name__ == "__main__":
    sys.exit(main())

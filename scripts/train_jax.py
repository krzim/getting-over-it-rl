"""Train PPO on the JAX Getting Over It env, all on device (goi_jax/training). Resumable: rerunning the same command
continues from runs/<name>/checkpoints/latest.pkl. The last line printed is DONE (total steps reached) or STOPPED
(with the reason: signal, --max-minutes budget, non-finite loss, error).

GPU (NVIDIA RTX 3090; scripts/cuda_run.sh holds the shared GPU lock; train_chain.sh runs it in resumable chunks):
    scripts/cuda_run.sh python -u scripts/train_jax.py --name coffee-1 --max-minutes 30
CPU (one XLA host device per --cpu-devices, envs split across them):
    JAX_PLATFORMS=cpu .venv/bin/python scripts/train_jax.py --name coffee-cpu --cpu-devices 16 --num-envs 64

Logs: runs/<name>/metrics.jsonl (one line per update), runs/<name>/config.json, optional TensorBoard (--tensorboard)
in runs/<name>/tb. Checkpoints: runs/<name>/checkpoints/update_<n>.pkl and latest.pkl (goi_jax/training/checkpoint.py).

Code snapshot: a run imports goi_jax from runs/<name>/code/goi_jax, a copy of the package taken when the run started
(--fresh, or the first time a run without one is launched), imported through runs/.code/<hash>/goi_jax so runs on
identical code share compiled programs (Pallas kernels put source paths into the cache key). Resumed chunks then trace exactly the same program, so
XLA's persistent cache (runs/.jax-cache) hits and the first device call takes seconds instead of a 4-6 minute
compile, and edits other people make to the working tree mid-run cannot change the physics under a running
training. --refresh-code re-copies the working tree (one recompile); --code live imports the working tree directly.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _code_files(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                  and p.suffix not in (".pyc", ".tmp"))


def code_hash(root: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    for p in _code_files(root):
        h.update(str(p.relative_to(root)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()[:12]


def use_code_snapshot(run: Path, refresh: bool) -> str:
    """Import goi_jax from run/code/goi_jax (copied from the working tree if missing or refresh). Must run before
    anything imports goi_jax. Returns a one-line description for the log."""
    import shutil
    if "goi_jax" in sys.modules:
        raise RuntimeError("goi_jax was imported before the code snapshot was put on sys.path")
    live, snap = REPO / "goi_jax", run / "code"
    taken = refresh or not (snap / "goi_jax" / "__init__.py").exists()
    if taken:
        tmp = run / "code.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(live, tmp / "goi_jax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.rmtree(snap, ignore_errors=True)
        tmp.rename(snap)
    h_snap, h_live = code_hash(snap / "goi_jax"), code_hash(live)
    # Import from a content-addressed copy, runs/.code/<hash>/goi_jax, not from runs/<name>/code: Pallas/Triton kernels
    # carry source file paths into XLA's cache key, so the same code imported from two different folders compiles twice.
    # With one folder per code version, every run (and every probe or sweep) on the same code shares the cache.
    shared = run.parent / ".code" / h_snap
    if not (shared / "goi_jax" / "__init__.py").exists():
        tmp = run.parent / ".code" / f"{h_snap}.tmp{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(snap / "goi_jax", tmp / "goi_jax", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        try:
            tmp.rename(shared)
        except OSError:  # another run created it first
            shutil.rmtree(tmp, ignore_errors=True)
    sys.path.insert(0, str(shared.resolve()))
    note = "taken now" if taken else ("same as the working tree" if h_snap == h_live else
                                     "differs from the working tree (--refresh-code to pick that up, one recompile)")
    return f"code: snapshot {snap}/goi_jax {h_snap} (imported from {shared}/goi_jax), {note}"

FLAGS = {  # CLI flag -> TrainConfig field (type from the dataclass default)
    "k": int, "episode_steps": int, "stall_seconds": float, "time_penalty": float, "action_cost": float,
    "mouse": float, "recovery": float, "regress": float, "air_bonus": float, "speed_bonus": float,
    "start_group": str, "start_sampling": str, "start_every": int, "saved_starts": float, "bin_size": float, "archive_bins": int,
    "num_envs": int, "num_steps": int, "minibatch_size": int, "num_minibatches": int, "epochs": int, "lr": float, "lr_end": float,
    "gamma": float, "gae_lambda": float, "clip": float, "ent_coef": float, "vf_coef": float,
    "max_grad_norm": float, "target_kl": float, "log_std_init": float, "total_steps": int,
    "updates_per_call": int, "devices": int, "solver": str, "seed": int, "save_every_updates": int,
    "episode_window": int, "parity_dir": str, "route": str, "contact_capacity": int,
    "min_log_std": float, "log_std_reset": float, "deepen_scale": float, "bank_skip": str,
    "teachers": str, "distill_coef": float, "distill_coef_end": float, "distill_updates": int, "distill_start": int,
}
BOOLS = ("snake_ends", "held_records", "bounded_mean", "norm_obs", "norm_reward", "compact_fixtures", "section_input")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 argument_default=argparse.SUPPRESS)
    ap.add_argument("--name", required=True, help="run folder under runs/")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint of this run (overwrites it)")
    ap.add_argument("--resume", help="a checkpoint .pkl to continue from (default: runs/<name>/checkpoints/latest.pkl)")
    ap.add_argument("--code", choices=("snapshot", "live"), help="import goi_jax from the run's snapshot "
                    "(runs/<name>/code, default; keeps resumes on the compilation cache) or the working tree")
    ap.add_argument("--refresh-code", action="store_true", help="re-copy the working tree into the run's snapshot")
    ap.add_argument("--cpu-devices", type=int, help="CPU only: XLA host devices to split the envs over (default 16)")
    ap.add_argument("--max-minutes", type=float, help="stop (STOPPED, resumable) after this much wall time "
                    "with the GPU (lock) held: compile and lock wait not counted")
    ap.add_argument("--duty", type=float, help="share of wall time the device may be busy (0-1, default 1): after each "
                    "device call, idle for call_time * (1/duty - 1) so a GPU shared with the desktop stays responsive")
    ap.add_argument("--tensorboard", action="store_true", help="also write TensorBoard scalars to runs/<name>/tb")
    ap.add_argument("--quiet", action="store_true")
    for f, t in FLAGS.items():
        ap.add_argument("--" + f.replace("_", "-"), type=t)
    ap.add_argument("--hidden", type=int, nargs="+")
    ap.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"))
    for f in BOOLS:
        ap.add_argument("--" + f.replace("_", "-"), type=int, choices=(0, 1), help="0 or 1")
    return ap.parse_args(argv)


def overrides(args) -> dict:
    d = {f: getattr(args, f) for f in list(FLAGS) + ["hidden", "window"] if hasattr(args, f)}
    d.update({f: bool(getattr(args, f)) for f in BOOLS if hasattr(args, f)})
    if "minibatch_size" in d and "num_minibatches" not in d:
        d["num_minibatches"] = 0   # an explicit --minibatch-size means a fixed minibatch
    if "hidden" in d:
        d["hidden"] = tuple(d["hidden"])
    if "window" in d:
        d["window"] = tuple(d["window"])
    return d


def main(argv=None):
    args = parse(argv)
    on_cpu = os.environ.get("JAX_PLATFORMS", "").split(",")[0] == "cpu"
    if on_cpu:
        n = getattr(args, "cpu_devices", 16)
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + f" --xla_force_host_platform_device_count={n}").strip()
    run = Path("runs") / args.name
    code_note = "code: working tree (--code live)"
    if getattr(args, "code", "snapshot") == "snapshot":
        refresh = getattr(args, "refresh_code", False) or (
            getattr(args, "fresh", False) and getattr(args, "resume", None) is None)
        run.mkdir(parents=True, exist_ok=True)
        code_note = use_code_snapshot(run, refresh)
    import dataclasses
    import json
    import math
    import signal

    import jax
    import numpy as np

    from goi_jax.training import backend
    from goi_jax.training import checkpoint as C
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig

    run = Path("runs") / args.name
    ckdir = run / "checkpoints"
    jax.config.update("jax_compilation_cache_dir", str(Path("runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)

    resume = getattr(args, "resume", None)
    if resume is None and not getattr(args, "fresh", False) and (ckdir / "latest.pkl").exists():
        resume = str(ckdir / "latest.pkl")
    ck = C.load(resume) if resume else None
    cfg = TrainConfig.from_dict(ck["config"]) if ck else TrainConfig()
    cfg = dataclasses.replace(cfg, name=args.name, **overrides(args))
    if cfg.teachers and not hasattr(args, "distill_start") and (
            cfg.distill_start < 0 or not ck or ck["config"].get("name") != cfg.name
            or not ck["config"].get("teachers")):
        # the distillation schedule counts from the update this run starts distilling at (a warm start keeps the
        # base run's update counter); chained chunks of the same run keep the saved value
        cfg = dataclasses.replace(cfg, distill_start=int(ck["update"]) if ck else 0)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    say = (lambda *a: None) if getattr(args, "quiet", False) else (lambda *a: print(*a, flush=True))
    say(code_note)

    wanted = os.environ.get("JAX_PLATFORMS", "").split(",")[0]
    if wanted and wanted != "cpu" and jax.default_backend() == "cpu":
        # e.g. the CUDA plugin failed to initialise (its traceback is above): JAX silently falls back to the CPU
        print(f"STOPPED: JAX_PLATFORMS asks for {wanted} but JAX came up on the CPU (see the plugin error above)",
              flush=True)
        return 1

    devices = jax.local_devices()
    D = cfg.devices or len(devices)
    say(f"backend {jax.default_backend()}, {D} of {len(devices)} devices ({devices[0].device_kind}); "
        f"{cfg.num_envs} envs x {cfg.num_steps} steps per update, solver {cfg.solver}")
    t0 = time.perf_counter()
    env = St.make_env(cfg)
    backend.select_solver(env, cfg.solver)
    nbank = jax.tree.leaves(env.default_params.starts)[0].shape[0]
    say(f"env built in {time.perf_counter() - t0:.1f} s: obs {env.obs_size}, {nbank} bank states ({cfg.start_group})")
    tr = ppo.Trainer(env, cfg, D)
    for i, t in enumerate(tr.teachers):
        say(f"teacher {i}: {t['path']} from progress {t['lo']:g} (bounded_mean {t['bounded_mean']}, norm_obs "
            f"{t['norm_obs']}); distill_coef {cfg.distill_coef} -> {cfg.distill_coef_end} over "
            f"{cfg.distill_updates} updates from update {cfg.distill_start}")
    key = jax.random.PRNGKey(cfg.seed)
    update0, steps0, wall0 = 0, 0, 0.0
    if ck:
        key = jax.random.fold_in(jax.numpy.asarray(ck["key"]), ck["update"])
        arch = St.Archive(*ck["archive"]) if ck.get("archive") is not None else None
        if arch is not None and arch.filled.shape[0] != tr.nb:
            arch = None
        if arch is not None:  # a full-layout archive (saved before --compact-fixtures 1) in the env's fixture layout
            arch = arch._replace(states=env.sim.compact_state(arch.states))
        # another architecture / section input / log_std reset / floor (goi_jax/training/nets.py warm_start)
        from goi_jax.training import nets as N
        params, opt, notes = N.warm_start(jax.random.fold_in(key, 7), ck["params"], ck["opt"], cfg, env.obs_size,
                                          same_run=ck["config"].get("name") == cfg.name, sections=St.NS)
        for note in notes:
            say(f"warm start: {note}")
        runner = tr.init(key, params, opt, ck["obs_rms"], ck["ret_rms"], arch, ck["update"], ck["record"])
        update0, steps0, wall0 = ck["update"], ck["env_steps"], ck.get("wall_s", 0.0)
        say(f"resumed {resume}: update {update0}, {steps0} env steps")
    else:
        runner = tr.init(key)
    total_updates = math.ceil(cfg.total_steps / tr.steps_per_update)

    tb = None
    if getattr(args, "tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                SummaryWriter = None
                say("no TensorBoard writer installed (torch or tensorboardX): jsonl only")
        tb = SummaryWriter(str(run / "tb")) if SummaryWriter else None

    stop = {"why": None}

    def on_signal(sig, _frame):
        stop["why"] = f"signal {signal.Signals(sig).name}"
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    max_s = getattr(args, "max_minutes", None)
    max_s = None if max_s is None else 60 * max_s

    log = open(run / "metrics.jsonl", "a" if ck else "w")   # a fresh run starts its log over
    savers = []
    update, t_start, status = update0, time.perf_counter(), None
    pending = None             # (metrics, runner, update after it) of the chunk dispatched last time
    t_first = None
    first_done = [update0]
    last_saved = update0

    def wall():
        return wall0 + time.perf_counter() - t_start

    def write_metrics(m, upd_end):
        m = jax.device_get(m)      # waits for that call only; the next one is already queued on the device
        t_now = time.perf_counter()
        U = cfg.updates_per_call
        rows = []
        for u in range(U):
            row = {k: float(np.asarray(v)[0, u]) for k, v in m.items()}
            row["update"] = int(row["update"])
            row["env_steps"] = row["update"] * tr.steps_per_update
            rows.append(row)
        # env steps per second since the first call finished (compile excluded); NaN for that first call
        done_since = upd_end - first_done[0]
        sps = done_since * tr.steps_per_update / max(t_now - t_mark, 1e-9) if done_since > 0 else float("nan")
        for row in rows:
            row["wall_s"] = wall()
            row["env_steps_per_s"] = sps
            log.write(json.dumps(row) + "\n")
            if tb:
                for k, v in row.items():
                    if isinstance(v, float) and math.isfinite(v):
                        tb.add_scalar(k, v, row["env_steps"])
        log.flush()
        r = rows[-1]
        say(f"upd {r['update']:6d} steps {r['env_steps']:>11,d} | {sps:6.0f} st/s | ep {r['ep_return']:7.2f} "
            f"best {r['ep_best']:6.1f} (win {r['win_best']:6.1f}, bank {r['win_bank_best']:6.1f}, rec "
            f"{r['record']:6.1f}) | archive {r['archive_bins']:3.0f} top {r['archive_top']:6.1f} | kl "
            f"{r['approx_kl']:.4f} stop {r['kl_stopped']:.0f} gs {r['grad_steps']:3.0f} | vl {r['value_loss']:.3f} "
            f"ev {r['explained_var']:.2f} std {r['std']:.3f}"
            + (f" | dkl {r['distill_kl']:.3f} (" + "/".join(f"{r[f'distill_kl_{i}']:.3f}" for i in range(len(tr.teachers)))
               + f") x{r['distill_coef']:.3f}" if tr.teachers else ""))
        bad = [k for k in ("pg_loss", "value_loss", "approx_kl") if not all(math.isfinite(x[k]) for x in rows)]
        return bad

    def checkpoint(runner_done, upd, final=False):
        first = tr.first(runner_done)
        ckpt = C.pack(cfg, first, upd, upd * tr.steps_per_update, wall())
        path = ckdir / f"update_{upd:07d}.pkl"
        if final:
            C.save(path, ckpt)
        else:
            savers.append(C.save_async(path, ckpt))
        return path

    duty = min(max(getattr(args, "duty", 1.0), 0.05), 1.0)
    # GOI_LATE_LOCK=1 (scripts/cuda_run.sh): compile ahead of time with the shared GPU lock free, then hold the lock only
    # while training, so a cold compile never leaves the card idle for the other users (goi_jax/gpu_lock.py).
    from goi_jax import gpu_lock
    gpu_hold = gpu_lock.held()
    if os.environ.get("GOI_LATE_LOCK") == "1":
        tc = time.perf_counter()
        tr._chunk.lower(runner, *tr.device_args()).compile()   # fills the persistent cache; the first call loads it
        say(f"compiled ahead of the GPU lock in {time.perf_counter() - tc:.1f} s")
    say(f"GPU lock taken after {gpu_hold.__enter__():.1f} s wait")
    t_mark = t_start = time.perf_counter()   # --max-minutes and wall time count from here (the lock is held)
    last_path = None
    try:
        while True:
            if update >= total_updates and pending is None:
                status = "DONE"
                break
            if stop["why"] is None and max_s is not None and time.perf_counter() - t_start > max_s:
                stop["why"] = f"time budget of {max_s / 60:.0f} min used"
            dispatched = None
            if update < total_updates and stop["why"] is None:
                tc = time.perf_counter()
                compiling = t_first is None  # don't pay idle time for the compile
                if compiling:
                    # Python handlers only run once XLA's compile returns, so a SIGTERM (timeout, chain stop) during
                    # a long compile would be ignored for the whole compile: let it kill the process instead (nothing
                    # to save yet; the cache entry is only written when the compile finishes)
                    signal.signal(signal.SIGTERM, signal.SIG_DFL)
                    signal.signal(signal.SIGINT, signal.SIG_DFL)   # same for Ctrl-C
                new_runner, metrics = tr.chunk(runner)
                if t_first is None:
                    jax.block_until_ready(metrics)
                    signal.signal(signal.SIGTERM, on_signal)
                    signal.signal(signal.SIGINT, on_signal)
                    t_first = time.perf_counter() - tc
                    say(f"first device call (compile + {cfg.updates_per_call} updates): {t_first:.1f} s")
                    try:
                        ms = jax.local_devices()[0].memory_stats() or {}
                    except Exception:  # noqa: BLE001
                        ms = {}
                    if ms.get("peak_bytes_in_use"):
                        say(f"device 0 memory: peak {ms['peak_bytes_in_use'] / 2**30:.2f} GiB in use "
                            f"(limit {ms.get('bytes_limit', 0) / 2**30:.1f} GiB)")
                    t_mark = time.perf_counter()
                    first_done[0] = update + cfg.updates_per_call
                if duty < 1 and not compiling:
                    jax.block_until_ready(metrics)
                    time.sleep((time.perf_counter() - tc) * (1 / duty - 1))
                update += cfg.updates_per_call
                dispatched = (metrics, new_runner, update)
                runner = new_runner
            if pending is not None:   # the previous call: log it while this one runs
                m, r_prev, upd_prev = pending
                bad = write_metrics(m, upd_prev)
                if bad:
                    status = f"STOPPED: non-finite {', '.join(bad)} at update {upd_prev} (not saved; resume from {last_path})"
                    break
                if upd_prev - last_saved >= cfg.save_every_updates or (dispatched is None):
                    last_path = checkpoint(r_prev, upd_prev)
                    last_saved = upd_prev
            pending = dispatched
            if pending is None and stop["why"] is not None:
                status = f"STOPPED: {stop['why']}; rerun the same command to resume from {last_path}"
                break
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}; rerun to resume from {last_path}"
    finally:
        gpu_hold.__exit__(None, None, None)
        for t in savers:
            t.join()
        log.close()
        if tb:
            tb.close()
    if status == "DONE":
        status = (f"DONE: {update} updates, {update * tr.steps_per_update:,d} env steps in {wall() / 60:.1f} min; "
                  f"checkpoint {last_path}")
    print(status, flush=True)
    return 0 if status.startswith("DONE") else 1


if __name__ == "__main__":
    sys.exit(main())

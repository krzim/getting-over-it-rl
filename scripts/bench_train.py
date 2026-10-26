"""Training throughput on the GPU: one PPO update (rollout + GAE + minibatch epochs) and the rollout alone, per env
count and kernel block size, with the same env and trainer as scripts/train_jax.py. Compiles every configuration first
with the shared GPU lock free (GOI_LATE_LOCK=1), then times them under the lock; --trace profiles ONE update call per
configuration (jax.profiler; CUPTI drops events past ~2M kernels, so never more) and prints the per-kernel table.

    GOI_LATE_LOCK=1 scripts/cuda_run.sh python -u scripts/bench_train.py --num-envs 2048 8192 16384 --parts full,rollout
    GOI_LATE_LOCK=1 scripts/cuda_run.sh python -u scripts/bench_train.py --num-envs 16384 --trace --top 40

One JSON line per configuration (env steps/s of the full update and of the rollout, the PPO part as the difference,
peak device memory); last line DONE or STOPPED.
"""
import argparse
import collections
import dataclasses
import glob
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, nargs="+", default=[2048])
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--num-minibatches", type=int, default=8)
    ap.add_argument("--parts", default="full,rollout", help="full (one PPO update) and/or rollout (num_steps actions)")
    ap.add_argument("--block", type=int, nargs="+", default=[0], help="worlds per Pallas program (0: the default)")
    ap.add_argument("--warps", type=int, default=0)
    ap.add_argument("--island-kernel", type=int, nargs="+", default=[-1],
                    help="gpu/islands.py search kernel off (0) / on (1) per configuration (-1: the module default)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help="speed options per configuration, each 'key=v,key=v' (keys: isl = gpu/islands.ENABLED, "
                         "sort = gpu/kernel.SORT_LANES, small = gpu/toi_loop.SMALL_DIVISOR, idle = "
                         "gpu/pairs.SKIP_IDLE_START, place = gpu/pairs.SLOT_PLACE, joints = gpu/kernel.BRANCH_SPLIT, sync = gpu/pairs.SYNC_ROT, rays = game/observe.RAYS_KERNEL, proxy = "
                         "gpu/toi_search.SHARED_PROXIES, graph = gpu/graph.ENABLED; cb=noloop: no loop command "
                         "buffers for this configuration (true kernel time in a trace); trace=1: trace this one; '-' = "
                         "module defaults); "
                         "replaces --island-kernel")
    ap.add_argument("--start-group", default="all")
    ap.add_argument("--compact", type=int, default=1)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--no-toi", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--trace", action="store_true", help="profile one call per configuration (per-kernel table)")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--code", help="import goi_jax from this folder instead of the working tree")
    ap.add_argument("--dump", help="dump each configuration's thunk sequence under this folder and print its host-driven "
                                   "loops (scripts/graph_check.py)")
    ap.add_argument("--compile-only", action="store_true", help="compile (and --dump) only; never takes the GPU lock")
    ap.add_argument("--hash", action="store_true",
                    help="bit-identity check: sha1 of the env state and of the params after one full update from "
                         "PRNGKey(0), done twice (a determinism control); compare the hashes across XLA flag sets")
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.code).resolve()) if args.code else str(ROOT))
    import jax
    import numpy as np
    jax.config.update("jax_compilation_cache_dir", str((ROOT / "runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    from goi_jax import gpu_lock
    from goi_jax.box2d.gpu import graph as GG
    from goi_jax.box2d.gpu import islands as GI
    from goi_jax.box2d.gpu import kernel as GK
    from goi_jax.box2d.gpu import pairs as GP
    from goi_jax.box2d.gpu import solve as G
    from goi_jax.box2d.gpu import toi_loop as GT
    from goi_jax.box2d.gpu import toi_search as GS
    from goi_jax.game import observe as GO
    from goi_jax.training import backend, ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    print(jax.devices(), os.environ.get("XLA_FLAGS", ""), flush=True)
    wanted = os.environ.get("JAX_PLATFORMS", "").split(",")[0]
    if wanted and wanted != "cpu" and jax.default_backend() == "cpu":
        print(f"STOPPED: JAX_PLATFORMS asks for {wanted} but JAX came up on the CPU (see the plugin error above)")
        return

    base = TrainConfig(name="bench", start_group=args.start_group, solver=args.solver, bounded_mean=True,
                       num_steps=args.num_steps, num_minibatches=args.num_minibatches, updates_per_call=1,
                       compact_fixtures=bool(args.compact))
    t = time.perf_counter()
    env = St.make_env(base)
    backend.select_solver(env, base.solver)
    if args.no_toi:
        env.sim.settings = dataclasses.replace(env.sim.settings, continuous_physics=False)
    print(f"env built in {time.perf_counter() - t:.1f} s", flush=True)
    default_block, default_warps = G.BLOCK, G.NUM_WARPS
    flags = {"isl": (GI, "ENABLED", lambda v: v == "1"), "sort": (GK, "SORT_LANES", lambda v: v == "1"),
             "small": (GT, "SMALL_DIVISOR", int), "idle": (GP, "SKIP_IDLE_START", lambda v: v == "1"),
             "place": (GP, "SLOT_PLACE", lambda v: v == "1"), "joints": (GK, "BRANCH_SPLIT", lambda v: v == "1"),
             "sync": (GP, "SYNC_ROT", lambda v: v == "1"), "rays": (GO, "RAYS_KERNEL", lambda v: v == "1"),
             "proxy": (GS, "SHARED_PROXIES", lambda v: v == "1"), "graph": (GG, "ENABLED", lambda v: v == "1")}
    flags = {k: v for k, v in flags.items() if hasattr(v[0], v[1])}  # (an older tree via --code may lack some)
    defaults = {k: getattr(mod, a) for k, (mod, a, _) in flags.items()}
    if args.variants:
        variants = [{} if v == "-" else dict(kv.split("=") for kv in v.split(",")) for v in args.variants]
    else:
        variants = [{} if i < 0 else {"isl": str(i)} for i in args.island_kernel]

    def set_flags(var):
        for k, (mod, a, conv) in flags.items():
            setattr(mod, a, conv(var[k]) if k in var else defaults[k])
        return {k: getattr(mod, a) for k, (mod, a, _) in flags.items()}

    jobs = []
    for n, blk, var in [(n, b, v) for n in args.num_envs for b in args.block for v in variants]:
            opts = set_flags(var)
            G.BLOCK = blk or default_block
            G.NUM_WARPS = args.warps or (default_warps if not blk else max(1, min(4, G.BLOCK // 32)))
            cfg = dataclasses.replace(base, num_envs=n)
            tr = ppo.Trainer(env, cfg, 1)
            p = tr.replicate_params()   # cached on the trainer: before eval_shape, or it caches tracers
            r_abs = jax.eval_shape(lambda k: tr.init(k), jax.random.PRNGKey(0))
            fns = {}
            for part in args.parts.split(","):
                tc = time.perf_counter()
                if part == "full":
                    f = tr._chunk
                else:
                    def roll(r, p_, tr=tr, cfg=cfg):
                        keys = jax.random.split(r.key, cfg.num_steps)
                        r, _ = jax.lax.scan(lambda c, k: tr._env_step(c, p_, k), r, keys)
                        return r
                    f = jax.pmap(roll, axis_name=ppo.DA)
                copts = {}
                if var.get("cb") == "noloop":   # loops host-driven: CUPTI then sees every kernel (true kernel time)
                    copts["xla_gpu_enable_command_buffer"] = "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL"
                if args.dump:
                    dd = Path(args.dump) / f"{part}_{n}_{len(jobs)}"
                    copts.update(xla_dump_to=str(dd), xla_dump_hlo_module_re="jit__fun|.*roll.*",   # keeps cb=noloop
                                 xla_dump_hlo_pass_re="NONE", xla_dump_hlo_as_text=True)
                fns[part] = f.lower(r_abs, p).compile(compiler_options=copts) if copts else f.lower(r_abs, p).compile()
                if args.dump:
                    sys.path.insert(0, str(ROOT / "scripts"))
                    import graph_check
                    for tf in sorted(dd.glob("*thunk_sequence_after_thunk_passes.txt")):
                        loops, split = graph_check.analyse(tf)
                        print(f"graph check {part} {n} {opts}: {len(loops)} host-driven loops "
                              f"{[t for t, _ in loops]}, outside command buffers {dict(split)} ({tf.name})", flush=True)
                print(f"compiled {part} {n} envs block {G.BLOCK}/{G.NUM_WARPS} warps {opts} in "
                      f"{time.perf_counter() - tc:.1f} s (no GPU lock)", flush=True)
            jobs.append((n, G.BLOCK, G.NUM_WARPS, dict(opts, **{k: var[k] for k in ("cb", "trace") if k in var}),
                         tr, fns))

    if args.compile_only:
        print("DONE (compile only)")
        return
    with gpu_lock.held() as waited:
        print(f"GPU lock after {waited:.1f} s wait", flush=True)
        for n, blk, warps, opts, tr, fns in jobs:
            out = dict(num_envs=n, num_steps=args.num_steps, block=blk, warps=warps, **opts,
                       grad_steps=10 * tr.n_mb,
                       minibatch=tr.mb, toi=not args.no_toi, compact=bool(args.compact))
            p = tr.replicate_params()
            runner = tr.init(jax.random.PRNGKey(0))
            for part, f in fns.items():
                call = (lambda r: f(r, p)[0]) if part == "full" else (lambda r: f(r, p))
                t0 = time.perf_counter()
                runner = jax.block_until_ready(call(runner))
                first = time.perf_counter() - t0
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    runner = jax.block_until_ready(call(runner))
                    ts.append(time.perf_counter() - t0)
                dt = float(np.median(ts))
                out[part] = dict(first_s=round(first, 1), s=round(dt, 3), min_s=round(min(ts), 3),
                                 env_steps_s=round(n * args.num_steps / dt))
                if args.trace or opts.get("trace") == "1":
                    out[part]["trace"] = trace_one(jax, call, runner, args.top, n * args.num_steps,
                                                   part + (" [cb=noloop]" if opts.get("cb") == "noloop" else ""))
            if args.hash and "full" in fns:
                f = fns["full"]
                out["hash"] = [state_hash(jax, np, f(tr.init(jax.random.PRNGKey(0)), p)[0]) for _ in range(2)]
            if "full" in out and "rollout" in out:
                out["ppo_s"] = round(out["full"]["s"] - out["rollout"]["s"], 3)
            try:
                ms = jax.local_devices()[0].memory_stats() or {}
                out["peak_gib"] = round(ms.get("peak_bytes_in_use", 0) / 2**30, 2)
            except Exception:  # noqa: BLE001
                pass
            print(json.dumps(out), flush=True)
            del runner
    print("DONE")


def state_hash(jax, np, runner):
    """sha1 (first 12 hex digits) of the env-side leaves and of the params of a Runner, bit for bit."""
    import hashlib
    hs = {}
    for key, tree in (("env", (runner.env, runner.obs)), ("params", runner.params)):
        h = hashlib.sha1()
        for leaf in jax.tree.leaves(tree):
            h.update(np.ascontiguousarray(np.asarray(leaf)).tobytes())
        hs[key] = h.hexdigest()[:12]
    return hs


def trace_one(jax, call, runner, top, steps, part):
    d = tempfile.mkdtemp(prefix="bench_train_")
    t0 = time.perf_counter()
    with jax.profiler.trace(d, create_perfetto_trace=True):
        jax.block_until_ready(call(runner))
    wall = time.perf_counter() - t0
    files = glob.glob(os.path.join(d, "**", "perfetto_trace.json.gz"), recursive=True)
    if not files:
        return "no trace"
    with gzip.open(files[0], "rt") as fh:
        tr = json.load(fh)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    dev = {pid for pid, nm in pname.items() if "/device:GPU" in nm}
    tot, cnt, iv = collections.Counter(), collections.Counter(), []
    for e in ev:
        if e.get("ph") == "X" and e.get("pid") in dev and e.get("dur", 0) > 0:
            tot[e["name"]] += e["dur"]
            cnt[e["name"]] += 1
            iv.append((e["ts"], e["ts"] + e["dur"]))
    iv.sort()
    busy, (cs, ce) = 0.0, iv[0]
    for s, e in iv[1:]:
        if s > ce:
            busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    window = iv[-1][1] - iv[0][0]
    allk = sum(tot.values())
    print(f"--- trace {part}: {files[0]}", flush=True)
    if "noloop" not in part and ("CONDITIONAL" in os.environ.get("XLA_FLAGS", "")
                                 or "WHILE" in os.environ.get("XLA_FLAGS", "")):
        print("--- NOTE: loop command buffers are on; CUPTI does not record kernels of CUDA-graph conditional (while) "
              "bodies past their first iteration, so busy and kernel time below UNDERCOUNT (their time shows as "
              "gaps). For the true kernel time trace with --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUBLASLT,"
              "CUSTOM_CALL.", flush=True)
    print(f"--- trace {part}: wall {wall:.2f} s (profiled), GPU window {window / 1e6:.2f} s, busy "
          f"{100 * busy / window:.1f}%, kernel time {allk / 1e6:.2f} s, {sum(cnt.values()):,} launches", flush=True)
    for name, us in tot.most_common(top):
        print(f"  {us / 1e3:9.1f} ms {100 * us / allk:5.1f}% {cnt[name]:8d} launches  {name[:100]}", flush=True)
    # by launch count per call (a proxy for where in the program a kernel sits: per step, per TOI iteration, per
    # island-search iteration, ...), Pallas kernels apart
    grp = collections.defaultdict(lambda: [0.0, 0])
    for name, us in tot.items():
        key = "pallas " + name.split("__")[0] if name.startswith("box2d_") else f"xla x{cnt[name]}"
        grp[key][0] += us
        grp[key][1] += 1
    print("  by group (ms, kernels):", ", ".join(f"{k} {v[0] / 1e3:.0f} ({v[1]})" for k, v in
                                               sorted(grp.items(), key=lambda kv: -kv[1][0])[:25]), flush=True)
    return dict(busy=round(busy / window, 3), window_s=round(window / 1e6, 2), kernel_s=round(allk / 1e6, 2),
                launches=sum(cnt.values()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}")

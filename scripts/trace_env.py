"""GPU kernel-time breakdown of the training env step (K physics steps per action) from a jax.profiler trace.

    scripts/cuda_run.sh python scripts/trace_env.py --envs 2048 --actions 5

Builds the training env (goi_jax/training/starts.make_env), runs a few random actions, then traces `--actions` more
and sums device time per kernel name (Perfetto trace parsed here; no TensorBoard needed). Prints the top kernels,
the number of kernel launches per action, and the device-busy and wall time per action. Last line DONE.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--actions", type=int, default=5)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--no-toi", action="store_true")
    ap.add_argument("--xla-flags", default="")
    args = ap.parse_args()
    if args.xla_flags:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + args.xla_flags).strip()
    import dataclasses
    import jax
    import jax.numpy as jnp
    import numpy as np
    from goi_jax.training import backend
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    jax.config.update("jax_compilation_cache_dir", str(Path("runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    print(jax.devices(), flush=True)
    cfg = TrainConfig(name="trace-env", num_envs=args.envs, solver=args.solver)
    env = St.make_env(cfg)
    backend.select_solver(env, cfg.solver)
    if args.no_toi:
        env.sim.settings = dataclasses.replace(env.sim.settings, continuous_physics=False)
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), args.envs)
    _, st = jax.vmap(env.reset, (0, None))(keys, p)
    f = jax.jit(jax.vmap(env.step_env, (0, 0, 0, None), axis_name="w"))
    rng = np.random.default_rng(0)
    act = lambda: jnp.asarray(rng.normal(0, 1, (args.envs, 2)).astype(np.float32))  # noqa: E731
    t0 = time.perf_counter()
    out = jax.block_until_ready(f(keys, st, act(), p))
    print(f"first call {time.perf_counter() - t0:.1f} s", flush=True)
    st = out[1]
    for _ in range(3):
        st = jax.block_until_ready(f(keys, st, act(), p))[1]
    d = tempfile.mkdtemp(prefix="trace_env_")
    acts = [act() for _ in range(args.actions)]
    t0 = time.perf_counter()
    with jax.profiler.trace(d, create_perfetto_trace=True):
        for a in acts:
            st = jax.block_until_ready(f(keys, st, a, p))[1]
    wall = (time.perf_counter() - t0) / args.actions
    files = glob.glob(os.path.join(d, "**", "*.trace.json.gz"), recursive=True)
    if not files:
        print("no perfetto trace written", files, flush=True)
        print("DONE")
        return
    with gzip.open(files[0], "rt") as fh:
        tr = json.load(fh)
    ev = tr["traceEvents"] if isinstance(tr, dict) else tr
    # device tracks: thread/process names containing a GPU stream
    pname = {e["pid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "process_name"}
    dev_pids = {pid for pid, n in pname.items() if "GPU" in n or "/device:" in n}
    tot, cnt = collections.Counter(), collections.Counter()
    for e in ev:
        if e.get("ph") == "X" and e.get("pid") in dev_pids:
            tot[e["name"]] += e.get("dur", 0)
            cnt[e["name"]] += 1
    busy = sum(tot.values()) / 1e3 / args.actions
    print(f"{args.envs} envs, toi={'off' if args.no_toi else 'on'}: wall {1e3 * wall:.1f} ms per action, device busy "
          f"{busy:.1f} ms per action, {sum(cnt.values()) / args.actions:.0f} kernel launches per action", flush=True)
    for name, us in tot.most_common(args.top):
        print(f"{us / 1e3 / args.actions:9.2f} ms/action {cnt[name] / args.actions:8.1f} launches  {name[:110]}")
    print("DONE")


if __name__ == "__main__":
    main()

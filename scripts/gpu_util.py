"""GPU utilization of the training loop, measured with the JAX profiler (nvidia-smi / NVML are off-limits here).

Builds the same env and trainer as scripts/train_jax.py, runs one device call to compile, then traces --calls more and
prints the share of wall time in which a kernel was running (goi_jax/training/gpu_util.py), plus env steps/s.

    scripts/cuda_run.sh python scripts/gpu_util.py --num-envs 8192 --solver pallas --start-group coffee

Last line: DONE (with the numbers) or STOPPED.
"""
import argparse
import dataclasses
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=2048)
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--minibatch-size", type=int, default=0, help="fixed minibatch (0: batch / --num-minibatches)")
    ap.add_argument("--num-minibatches", type=int, default=8)
    ap.add_argument("--updates-per-call", type=int, default=1)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--start-group", default="coffee")
    ap.add_argument("--calls", type=int, default=3)
    ap.add_argument("--out", default="runs/.gpu-util")
    ap.add_argument("--compact", type=int, default=1, help="TrainConfig.compact_fixtures (static fixtures stored once)")
    ap.add_argument("--code", help="import goi_jax from this folder (e.g. a run's code snapshot) instead of the tree")
    args = ap.parse_args()
    if args.code:
        sys.path.insert(0, str(Path(args.code).resolve()))
    import jax
    jax.config.update("jax_compilation_cache_dir", str(Path("runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    from goi_jax import gpu_lock
    from goi_jax.training import backend, gpu_util, ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig

    cfg = dataclasses.replace(TrainConfig(), name="gpu-util", num_envs=args.num_envs, num_steps=args.num_steps,
                              minibatch_size=args.minibatch_size or 512,
                              num_minibatches=0 if args.minibatch_size else args.num_minibatches, updates_per_call=args.updates_per_call,
                              solver=args.solver, start_group=args.start_group)
    if hasattr(cfg, "compact_fixtures"):
        cfg = dataclasses.replace(cfg, compact_fixtures=bool(args.compact))
    env = St.make_env(cfg)
    backend.select_solver(env, cfg.solver)
    tr = ppo.Trainer(env, cfg, len(jax.local_devices()))
    runner = tr.init(jax.random.PRNGKey(0))
    t = time.perf_counter()
    params = tr.replicate_params()
    compiled = tr._chunk.lower(runner, params).compile()   # the pmapped update loop, compiled ahead of time
    chunk = lambda r: compiled(r, params)  # noqa: E731
    print(f"compile (no GPU lock held): {time.perf_counter() - t:.1f} s", flush=True)
    with gpu_lock.held() as waited:
        print(f"GPU lock after {waited:.1f} s wait", flush=True)
        t = time.perf_counter()
        runner, m = chunk(runner)
        jax.block_until_ready(m)
        print(f"first call: {time.perf_counter() - t:.1f} s", flush=True)
        _measure(jax, gpu_util, chunk, runner, args, tr)


def _measure(jax, gpu_util, chunk, runner, args, tr):
    with gpu_util.Trace(args.out) as tracer:
        t = time.perf_counter()
        for _ in range(args.calls):
            runner, m = chunk(runner)   # dispatched back to back, as the trainer does
        jax.block_until_ready(m)
        wall = time.perf_counter() - t
    steps = args.calls * tr.steps_per_update
    print(f"DONE: {tracer.report()}; {steps / wall:,.0f} env steps/s at {args.num_envs} envs")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"STOPPED: {type(e).__name__}: {e}")

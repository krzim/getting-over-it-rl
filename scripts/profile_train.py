"""Where a PPO update's time goes on the GPU: full update vs rollout only vs rollout without TOI (and the PPO
gradient part as the difference), at one or more env counts. Same env / trainer as scripts/train_jax.py.

    scripts/cuda_run.sh python scripts/profile_train.py --num-envs 2048 8192 --solver pallas

Prints one JSON line per env count; last line DONE.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, nargs="+", default=[2048])
    ap.add_argument("--num-steps", type=int, default=32)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--parts", default="full,rollout,rollout_notoi")
    ap.add_argument("--xla-flags", default="", help="extra XLA_FLAGS (set before jax import)")
    args = ap.parse_args()
    import os
    if args.xla_flags:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + args.xla_flags).strip()
    import jax
    import numpy as np
    jax.config.update("jax_compilation_cache_dir", str(Path("runs/.jax-cache").resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 5.0)
    from goi_jax.training import backend
    from goi_jax.training import ppo
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    print(jax.devices(), flush=True)
    parts = args.parts.split(",")
    for n in args.num_envs:
        out = dict(num_envs=n, num_steps=args.num_steps, solver=args.solver, xla_flags=args.xla_flags)
        for part in parts:
            cfg = TrainConfig(name="profile", num_envs=n, num_steps=args.num_steps, minibatch_size=min(8192, n * 4),
                              solver=args.solver, bounded_mean=True)
            env = St.make_env(cfg)
            backend.select_solver(env, cfg.solver)
            if part == "rollout_notoi":
                env.sim.settings = dataclasses.replace(env.sim.settings, continuous_physics=False)
            tr = ppo.Trainer(env, cfg, 1)
            runner = tr.init(jax.random.PRNGKey(0))
            if part == "full":
                f = lambda r: tr.chunk(r)[0]  # noqa: E731
            else:
                def roll(r, p):
                    keys = jax.random.split(r.key, cfg.num_steps)
                    r, (t, _) = jax.lax.scan(lambda c, k: tr._env_step(c, p, k), r, keys)
                    return r
                g = jax.pmap(roll, axis_name=ppo.DA)
                f = lambda r: g(r, tr.replicate_params())  # noqa: E731
            t0 = time.perf_counter()
            runner = jax.block_until_ready(f(runner))
            comp = time.perf_counter() - t0
            ts = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                runner = jax.block_until_ready(f(runner))
                ts.append(time.perf_counter() - t0)
            dt = float(np.median(ts))
            out[part] = dict(first_s=round(comp, 1), s=round(dt, 3), env_steps_s=round(n * cfg.num_steps / dt),
                             physics_ms=round(1e3 * dt / (cfg.num_steps * cfg.k), 2))
            print(json.dumps(out), flush=True)
            del runner, tr, env
        if "full" in out and "rollout" in out:
            out["ppo_update_s"] = round(out["full"]["s"] - out["rollout"]["s"], 3)
        if "rollout" in out and "rollout_notoi" in out:
            out["toi_share_of_rollout"] = round(1 - out["rollout_notoi"]["s"] / out["rollout"]["s"], 3)
        print(json.dumps(out), flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

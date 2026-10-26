"""Compile time of one vmapped training-env step (K physics steps) and how many Pallas kernel calls the compiled
program holds, per kernel name (a K-step scan that XLA unrolls repeats each kernel). Last line DONE.

    scripts/cuda_run.sh python scripts/compile_env.py --envs 256 --solver pallas
"""
import argparse
import collections
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--solver", default="pallas")
    ap.add_argument("--xla-flags", default="", help="extra XLA_FLAGS (set before jax import)")
    args = ap.parse_args()
    import os
    if args.xla_flags:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + args.xla_flags).strip()
    import jax
    import jax.numpy as jnp
    from goi_jax.training import backend
    from goi_jax.training import starts as St
    from goi_jax.training.config import TrainConfig
    print(jax.devices(), os.environ.get("XLA_FLAGS", ""), flush=True)
    cfg = TrainConfig(name="compile-env", num_envs=args.envs, solver=args.solver)
    env = St.make_env(cfg)
    backend.select_solver(env, cfg.solver)
    p = env.default_params
    keys = jax.random.split(jax.random.PRNGKey(0), args.envs)
    obs, st = jax.vmap(env.reset, (0, None))(keys, p)
    f = jax.jit(jax.vmap(env.step_env, (0, 0, 0, None), axis_name="w"))
    a = jnp.zeros((args.envs, 2), jnp.float32)
    t0 = time.perf_counter()
    lowered = f.lower(keys, st, a, p)
    t1 = time.perf_counter()
    c = lowered.compile()
    t2 = time.perf_counter()
    txt = c.as_text()
    names = collections.Counter(re.findall(r'name\s*=\s*\\?"(box2d_[a-z_]+)', txt))
    if not names:
        names = collections.Counter(re.findall(r"(box2d_[a-z_]+)", txt))
    print(f"lower {t1 - t0:.1f} s, compile {t2 - t1:.1f} s; kernel mentions {dict(names)}", flush=True)
    out = jax.block_until_ready(c(keys, st, a, p))
    t0 = time.perf_counter()
    for _ in range(5):
        out = jax.block_until_ready(c(keys, out[1], a, p))
    print(f"step {1e3 * (time.perf_counter() - t0) / 5:.1f} ms per action ({env.k} physics steps), {args.envs} envs",
          flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

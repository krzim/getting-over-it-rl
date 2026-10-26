"""Compile-only probe (never runs anything on the GPU): which XLA compile options keep a scan with stacked outputs and
a data-dependent inner while loop -- the shape of the training update (rollout scan over num_steps, a fixed loop
of K physics sub-steps, physics while loops inside, GAE scan) -- inside one CUDA graph, i.e. no host-driven loops.

    GOI_LATE_LOCK=1 XLA_PYTHON_CLIENT_PREALLOCATE=false scripts/cuda_run.sh python scripts/graph_probe.py

One line per option set: host-driven loops and the thunks outside command buffers (scripts/graph_check.py on a
thunk dump per set under $TMPDIR). Last line DONE.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import graph_check  # noqa: E402

CB = "FUSION,CUBLAS,CUBLASLT,CUSTOM_CALL,CONDITIONAL,WHILE"
SETS = {
    "current": {},
    "cb_dsf": {"xla_gpu_enable_command_buffer": CB + ",DYNAMIC_SLICE_FUSION"},
    "cb_dsf_coll": {"xla_gpu_enable_command_buffer": CB + ",DYNAMIC_SLICE_FUSION,COLLECTIVES"},
    "cb_dsf_conc": {"xla_gpu_enable_command_buffer": CB + ",DYNAMIC_SLICE_FUSION",
                    "xla_gpu_graph_enable_concurrent_region": True},
    "sched0": {"xla_gpu_command_buffer_scheduling_mode": 0},
    "sched1": {"xla_gpu_command_buffer_scheduling_mode": 1},
    "sched2": {"xla_gpu_command_buffer_scheduling_mode": 2},
}


def update(x):
    def substep(_, s):
        def cond(c):
            return jnp.any(c > 1.0)
        return jax.lax.while_loop(cond, lambda c: jnp.where(c > 1.0, c * 0.5, c), s * 3.0 + 1.0)

    def env_step(s, _):
        s = jax.lax.fori_loop(0, 4, substep, s)
        return s, (s * 2.0, s.sum(), s > 0.5)

    s, (o1, o2, o3) = jax.lax.scan(env_step, x, None, length=32)
    adv = jax.lax.scan(lambda c, v: (c * 0.9 + v, c), jnp.zeros_like(x), o1, reverse=True)[1]
    return s, adv, o2, o3


def main():
    print(jax.devices(), os.environ.get("XLA_FLAGS", ""), flush=True)
    x = jnp.ones((16384,), jnp.float32)
    lowered = jax.jit(update).lower(x)
    root = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "graph_probe"
    for name, opts in SETS.items():
        d = root / name
        try:
            lowered.compile(compiler_options=dict(opts, xla_dump_to=str(d), xla_dump_hlo_module_re="jit_update",
                                                  xla_dump_hlo_pass_re="NONE", xla_dump_hlo_as_text=True))
        except Exception as e:  # noqa: BLE001
            print(f"{name}: compile failed: {str(e)[:200]}", flush=True)
            continue
        files = sorted(d.glob("*jit_update*thunk_sequence_after_thunk_passes.txt"))
        if not files:
            print(f"{name}: no thunk dump in {d}", flush=True)
            continue
        loops, split = graph_check.analyse(files[-1])
        print(f"{name}: {len(loops)} host-driven loops, outside command buffers {dict(split)}", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()

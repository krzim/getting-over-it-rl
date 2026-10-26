"""Physics backend selection for training: the Box2D island solver ("xla", the default, or "pallas", the GPU kernel
in goi_jax/box2d/gpu, selected by Settings.solver).

goi_jax.box2d.step.step dispatches on Settings.solver, but goi_jax/game/sim.py composes the Box2D step itself
(Sim._box2d_step, to keep the bodies between the solve and SolveTOI) and calls step.solve directly. Until sim.py
dispatches too, `select_solver` routes that call through the Pallas solver by replacing the `box2d_solve` name sim.py
looks up at trace time (a no-op for "xla"). Call it before the first jit trace.
"""
from __future__ import annotations

import dataclasses


def select_solver(env, solver: str = "xla"):
    from goi_jax.game import sim as SM
    if solver not in ("xla", "pallas"):
        raise ValueError(f"unknown solver {solver!r} (xla | pallas)")
    env.sim.settings = dataclasses.replace(env.sim.settings, solver=solver)
    if solver == "xla":
        return env
    from goi_jax.box2d.gpu.solve import solve as solve_pallas   # ImportError here: the GPU backend has not landed
    original = getattr(SM, "_xla_box2d_solve", SM.box2d_solve)
    SM._xla_box2d_solve = original

    def dispatch(model, state, settings, *a, **kw):
        if getattr(settings, "solver", "xla") == "pallas":
            return solve_pallas(model, state, settings, *a, **kw)
        return original(model, state, settings, *a, **kw)

    SM.box2d_solve = dispatch
    return env

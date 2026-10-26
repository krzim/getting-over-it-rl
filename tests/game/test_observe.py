"""The JAX observation, route progress and reward (goi_jax/game/observe.py, route.py, reward.py) against upstream and
the game.

Without recordings: the field layout against goi_env.client, and the reward / episode rules against goi_env.env.
With recordings (dumps/obs, gitignored; record with `.venv/bin/python -m goi_jax.parity.obs_parity record`, unsandboxed:
it launches the game): the game's own spline queries, the plugin's progress guard, and whole observations, bit for bit.
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/game/test_observe.py -q
"""
from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_OBS_DIR", ROOT / "dumps" / "obs"))
need_data = pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no observation recordings in {DATA}")


def test_layout_matches_upstream():
    from goi_env.client import AHEAD, OBS_FIELDS
    from goi_jax.game import observe as O
    assert tuple(O.OBS_FIELDS) == tuple(OBS_FIELDS[:O.N_FIELDS])
    assert len(OBS_FIELDS) == O.N_FIELDS + 2 * len(AHEAD)
    assert O.ObsConfig(n_rays=32, head_rays=8).size == len(OBS_FIELDS) + 40


def test_ray_directions():
    """2 pi i / n in double stored as float, then (float)Math.Cos / Sin; normalized as b2Vec2::Normalize."""
    from goi_jax.game import observe as O
    d = O.ray_directions(32)
    assert d.dtype == np.float32 and d[0].tolist() == [1.0, 0.0]
    u = O.unit_directions(d)
    assert np.abs(np.hypot(u[:, 0], u[:, 1]) - 1).max() < 2e-7


def _upstream_rewards(moves, held, **weights):
    """goi_env.env.GoiEnv.step on a stand-in game (as env.py's own self-test does)."""
    from goi_env.env import MOUSE, OBS_FIELDS, PREV_ACTIONS, GoiEnv
    P, POT, HEAD = (OBS_FIELDS.index(f) for f in ("progress", "pot_contact", "head_contact"))

    class Steps:
        on_snake, obs_size = False, len(OBS_FIELDS)

        def __init__(self, moves):
            self.moves = iter(moves)

        def step(self, dx, dy):
            raw = np.zeros(len(OBS_FIELDS), np.float32)
            raw[P], raw[POT], raw[HEAD] = next(self.moves)
            return raw

    env = GoiEnv.__new__(GoiEnv)
    w = dict(recovery=0.0, regress=0.0, air_bonus=0.0, speed_bonus=0.0)
    w.update(weights)
    env.__dict__.update(time_penalty=0.001, stall_steps=3, stall_ends=True, snake_ends=False, record_reset=None,
                        action_cost=0.0, mouse=MOUSE, weights=w, held_records=held, _realtime=False,
                        client=Steps(moves), best_progress=428.0, held_best=428.0, _progress=428.0, _stall_mark=428.0,
                        stalled=0, air=0, _actions=deque([(0.0, 0.0)] * PREV_ACTIONS, maxlen=PREV_ACTIONS))
    out = []
    for _ in moves:
        obs, r, term, trunc, _ = env.step(np.array([0.3, -0.2]))
        out.append((obs, r, term, trunc))
    return out


@pytest.mark.parametrize("held", [False, True])
@pytest.mark.parametrize("weights", [{}, dict(recovery=0.5, regress=1.0, air_bonus=0.3, speed_bonus=0.2)])
def test_reward_matches_upstream(held, weights):
    import jax.numpy as jnp
    from goi_jax.game import reward as W
    moves = [(431, 0, 0), (438, 0, 0), (433, 0, 0), (426, 1, 0), (434, 0, 0), (443, 0, 1), (446, 1, 0), (446, 1, 0),
             (446, 1, 0), (446.5, 1, 0)]
    up = _upstream_rewards(moves, held, **weights)
    params = W.RewardParams(held_records=held, stall_steps=3, **weights)
    raw0 = np.zeros(63, np.float32)
    raw0[W.PROGRESS] = 428.0
    ep = W.reset_episode(jnp.asarray(raw0))
    for (p, pot, head), (obs_u, r_u, term_u, trunc_u) in zip(moves, up):
        raw = raw0.copy()
        raw[W.PROGRESS], raw[W.POT], raw[W.HEAD] = p, pot, head
        ep, r, term, trunc = W.step_episode(params, ep, jnp.asarray(raw), False, jnp.asarray([0.3, -0.2]))
        assert float(r) == float(r_u)  # NumPy 2: upstream's reward is float32 arithmetic too
        assert bool(term) == term_u and bool(trunc) == trunc_u
        obs = np.asarray(W.policy_obs(params, ep, jnp.asarray(raw)))
        assert np.array_equal(obs, obs_u)


# ------------------------------------------------------------------------------------------------- recorded data

@pytest.fixture(scope="module")
def route_row():
    return json.loads((DATA / "route.json").read_text())


@need_data
def test_route_queries_bit_exact(route_row):
    """The game's GetNearestPointTF, TFToDistance, DistanceToTF and Interpolate on the route row's probes."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import route as R
    rm = R.route_model(route_row)
    pn = np.array(route_row["probeNearest"], np.float32).reshape(-1, 5)
    pd = np.array(route_row["probeDistance"], np.float32).reshape(-1, 5)
    tf = jax.jit(jax.vmap(R.nearest_tf, in_axes=(None, 0)))(rm, pn[:, :3])
    assert np.array_equal(np.asarray(tf), pn[:, 3])
    d = jax.jit(jax.vmap(R.tf_to_distance, in_axes=(None, 0)))(rm, pn[:, 3])
    assert np.array_equal(np.asarray(d), pn[:, 4])
    t2 = jax.jit(jax.vmap(R.distance_to_tf, in_axes=(None, 0)))(rm, pd[:, 0])
    assert np.array_equal(np.asarray(t2), pd[:, 1])
    q = jax.jit(jax.vmap(lambda rm, t: jnp.stack(R.interpolate(rm, t)), in_axes=(None, 0)))(rm, pd[:, 1])
    assert np.array_equal(np.asarray(q), pd[:, 2:])


@need_data
def test_progress_guard_bit_exact(route_row):
    """The plugin's Progress after place(): from progress = along at the placed pot (many walk the guard)."""
    import jax
    import jax.numpy as jnp
    from goi_jax.game import route as R
    if not (DATA / "guard.npz").exists():
        pytest.skip("no guard probes (obs_parity record-extra)")
    rm = R.route_model(route_row)
    rows = np.load(DATA / "guard.npz")["rows"]
    f = jax.jit(jax.vmap(lambda rm, a, x, y: R.guarded_progress(rm, a, (x, y, rm.player_z)), in_axes=(None, 0, 0, 0)))
    got = np.asarray(f(rm, *(jnp.asarray(rows[:, k]) for k in range(3))))
    assert np.array_equal(got, rows[:, 3])


@need_data
def test_observations_bit_exact():
    """Whole observations (fields, route ahead, 32 pot rays, 8 head rays, Snake status, stepped lake / won flags) of
    a sample of recorded chains, including the lake and Snake ones."""
    from goi_jax.parity import obs_parity as P
    index = json.loads((DATA / "index.json").read_text())
    files = [e["file"] for e in index["chains"]]
    pick = [f for f in files if f.startswith("chains/o0")][:4] + [f for f in files if "lake" in f][:1] + \
        [f for f in files if "snake" in f and f.startswith("chains/y")][:2]
    res = P.compare(DATA, chains=pick, progress=False)
    first = res["t0"]
    J, G = res["jax"], res["game"]
    mask = np.ones_like(J, bool)
    mask[first, 15:63] = False  # progress and ahead need the previous observation's progress
    assert np.array_equal(J[mask], G[mask])
    assert np.array_equal(res["snake_jax"], res["snake_game"])
    assert np.array_equal(res["lake_jax"], G[:, 13] > 0) and np.array_equal(res["won_jax"], G[:, 14] > 0)

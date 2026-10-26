"""Getting Over It as a gymnax-style JAX environment: the plugin's lockstep game (goi_jax/game/sim.py), the bridge's
observation (goi_jax/game/observe.py) and upstream's reward and episode rules (goi_env/env.py GoiEnv), all on device.

    env = GoiEnv.from_recordings()                        # host, once: level, constants, start states, route
    params = env.default_params                           # EnvParams pytree (pass it as a jit argument)
    obs, state = jax.vmap(env.reset, (0, None))(keys, params)
    obs, state, reward, done, info = jax.vmap(env.step, (0, 0, 0, None))(keys, state, actions, params)

Fixed shapes, no host sync in step; batch with vmap (pass `axis_name` to the constructor and vmap with the same
`axis_name` to share the solver's loop trip counts across the batch, which is faster; see goi_jax/box2d/loops.py).

Action: (2,) in [-1, 1], as upstream: clipped, times `mouse` (12, goi_env.env.MOUSE), then the plugin's
ClampMagnitude(., 100); the controller reads it on each of the K physics steps (K = 4 default, the plugin's).
Observation (goi_env.env.GoiEnv._observe): the bridge's fields (OBS_FIELDS, 20 route points ahead, n_rays pot rays,
head_rays head rays), the last 4 actions (newest first), the stall share (stalled / stall_steps, 0 without a cutoff)
and best_progress - progress.
Reward: goi_env.env.progress_reward(prev, now, best, air = 1 - pot_contact, recovery, regress, air_bonus,
speed_bonus) - time_penalty - action_cost * mean(a^2), with upstream's held_records and record_reset options.
Episode: terminated on the lake, the win or (snake_ends) the Snake; truncated after stall_steps without headway (if
> 0; stall_ends=False restarts the count instead, as upstream) or after max_steps actions (if > 0, a TimeLimit).
done = terminated | truncated; the env then auto-resets (gymnax style): the returned obs/state are a fresh episode's,
info["terminated"] / info["truncated"] say why, and info["final_progress"] keeps the finished episode's progress.
Resets draw uniformly from `starts`, a bank of recorded pre-step states (Sim.state_from_recording): the level start
chains by default.
Limits: see goi_jax/game/sim.py "Not modelled" (pose round trip, the gift). Space (zero gravity, wells, teleports,
the win): goi_jax/game/space.py; the observation's won flag is the sim's (SimState.space.won) when the state has one.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.game import observe as O
from goi_jax.game import route as R
from goi_jax.game import sim as SM

f32, i32 = jnp.float32, jnp.int32
MOUSE = 12.0          # goi_env.env.MOUSE
PREV_ACTIONS = 4      # goi_env.env.PREV_ACTIONS
SPEED_REF = 0.2       # goi_env.env.SPEED_REF
MAX_INPUT = 100.0     # plugin/Game.cs Step: Vector2.ClampMagnitude(action, 100f)
F = {name: i for i, name in enumerate(O.OBS_FIELDS)}


class EnvParams(NamedTuple):
    model: object                  # goi_jax.box2d Model (canonical joint order)
    sim: SM.SimParams
    obs: O.ObsModel
    starts: SM.SimState            # (N, ...) reset bank
    mouse: jax.Array = f32(MOUSE)
    time_penalty: jax.Array = f32(0.001)
    action_cost: jax.Array = f32(0.0)
    recovery: jax.Array = f32(0.0)
    regress: jax.Array = f32(0.0)
    air_bonus: jax.Array = f32(0.0)
    speed_bonus: jax.Array = f32(0.0)
    stall_steps: jax.Array = i32(0)     # 0: no stall cutoff (upstream None)
    record_reset: jax.Array = f32(-1)   # < 0: off (upstream None)
    max_steps: jax.Array = i32(0)       # 0: no time limit


class EnvState(NamedTuple):
    sim: SM.SimState
    carry: O.ObsCarry
    best_progress: jax.Array
    held_best: jax.Array
    progress: jax.Array
    stall_mark: jax.Array
    stalled: jax.Array
    air: jax.Array
    t: jax.Array
    actions: jax.Array             # (PREV_ACTIONS, 2) newest first


def progress_reward(prev, now, best, air, recovery, regress, air_bonus, speed_bonus):
    """goi_env.env.progress_reward in float32."""
    new = jnp.maximum(f32(0), now - best)
    forward, back = jnp.maximum(f32(0), now - prev), jnp.maximum(f32(0), prev - now)
    bonus = 1 + air_bonus * air + speed_bonus * jnp.minimum(new / f32(SPEED_REF), f32(2))
    return new * bonus + recovery * (forward - new) - regress * back


class GoiEnv:
    """Static configuration (hashable): the Sim tables, K, observation config and the boolean episode options."""

    def __init__(self, sim: SM.Sim, default_params: EnvParams, k=4, obs_config=O.ObsConfig(32, 15.0, 0, 4.0),
                 held_records=False, stall_ends=True, snake_ends=False, axis_name=None):
        self.sim, self.k = sim, int(k)
        self.default_params = default_params
        self.cfg = obs_config
        self.held_records, self.stall_ends, self.snake_ends = held_records, stall_ends, snake_ends
        self.axis_name = axis_name

    @property
    def default_params(self) -> EnvParams:
        return self._default_params

    @default_params.setter
    def default_params(self, p: EnvParams):
        """Assigning params puts their start bank in the Sim's fixture layout (a no-op unless the Sim's Model is
        compact and the bank is not, e.g. a bank cached before the compact layout existed), adds the space state to a
        bank without one, and sets gravity 0 for start states already inside the zero-gravity zone at -30 (Sim.start_state)."""
        if p is not None and p.starts is not None:
            p = p._replace(starts=self.sim.start_state(p.starts))  # layout, space state, zone gravity (sim.py)
        self._default_params = p

    @classmethod
    def from_recordings(cls, root="dumps/parity", route="dumps/obs/route.json", starts=None, k=4,
                        obs_config=O.ObsConfig(32, 15.0, 0, 4.0), capacity=96, compact_fixtures=False, **kw):
        """Build from a recording session. starts: list of (chain file relative to root, step) for the reset bank;
        default: the canonical first state of every chain in the "start" group (the level start).
        compact_fixtures: the compact fixture layout (Sim.from_recordings; same results, ~36 KB less state per env and
        cheaper fixture kernels). Every SimState used with this env (starts, a checkpoint's saved-starts archive)
        must then be compact too: env.sim.compact_state converts a full-layout one."""
        from goi_jax.parity import states as S
        root = Path(root)
        sim = SM.Sim.from_recordings(root, compact_fixtures=compact_fixtures)
        if starts is None:
            starts = [(e["file"], None) for e in S.load_index(root)["chains"] if e["group"] == "start"]
        bank = []
        for f, t in starts:
            chain = S.load_chain(root / f)
            bank.append(sim.state_from_recording(chain, SM.canonical_step(chain) if t is None else t, capacity))
        bank = jax.tree.map(lambda *a: np.stack(a), *bank)
        import json
        from goi_jax.game import level as L
        route_row = json.loads(Path(route).read_text())
        jm = L.JointModel(*(np.asarray(getattr(sim.model, f)) for f in L.JOINT_MODEL_FIELDS))
        om = O.build_obs_model(sim.level, route_row, jm, reach=max(obs_config.ray_length, obs_config.head_length))
        env_kw = {k_: kw.pop(k_) for k_ in ("held_records", "stall_ends", "snake_ends", "axis_name") if k_ in kw}
        params = EnvParams(model=sim.model, sim=sim.params, obs=om, starts=jax.tree.map(jnp.asarray, bank),
                           **{k_: jnp.asarray(v, f32 if k_ not in ("stall_steps", "max_steps") else i32)
                              for k_, v in kw.items()})
        return cls(sim, params, k, obs_config, **env_kw)

    # ------------------------------------------------------------------ spaces

    @property
    def obs_size(self):
        return self.cfg.size + 2 * PREV_ACTIONS + 2

    @property
    def action_shape(self):
        return (2,)

    # ------------------------------------------------------------------ reset / step

    def _observe(self, p: EnvParams, s: SM.SimState, carry):
        raw, carry, snake = O.observe(p.obs, self.cfg, s.world, s.ctrl.mouse_input, s.ctrl.mva, carry)
        return raw, carry, snake

    def _full_obs(self, p, raw, st: EnvState):
        stuck = jnp.where(p.stall_steps > 0, st.stalled.astype(f32) / jnp.maximum(p.stall_steps, 1).astype(f32), f32(0))
        return jnp.concatenate([raw, st.actions.reshape(-1), jnp.stack([stuck, st.best_progress - raw[F["progress"]]])])

    def reset(self, key, params: EnvParams = None):
        p = params if params is not None else self.default_params
        n = jax.tree.leaves(p.starts)[0].shape[0]
        i = jax.random.randint(key, (), 0, n)
        s = jax.tree.map(lambda a: a[i], p.starts)
        rm = p.obs.route
        bd = s.world.bodies
        prog = R.raw_progress(rm, (bd.xf_p[self.sim.pot, 0], bd.xf_p[self.sim.pot, 1], rm.player_z))
        carry = O.init_carry(p.obs, p.model, s.world, prog)
        raw, carry, _ = self._observe(p, s, carry)
        pr = raw[F["progress"]]
        st = EnvState(sim=s, carry=carry, best_progress=pr, held_best=pr, progress=pr, stall_mark=pr,
                      stalled=i32(0), air=i32(0), t=i32(0), actions=jnp.zeros((PREV_ACTIONS, 2), f32))
        return self._full_obs(p, raw, st), st

    def _act(self, p: EnvParams, s: SM.SimState, carry, a):
        """One plugin action: K physics steps (lake / win flags after each), then the joint re-creation."""
        def one(c, _):
            s, carry = c
            s1, _ = self.sim.physics_step(p.model, p.sim, s, a, self.axis_name)
            carry = O.step_flags(p.obs, p.model, s.world, s1.world, carry)
            if s1.space is not None:  # the sim's own Patches.Won (space.py: the zone trigger's exit through the top)
                carry = carry._replace(won=s1.space.won)   # replaces observe.step_flags' AABB estimate
            return (s1, carry), None
        (s, carry), _ = jax.lax.scan(one, (s, carry), None, length=self.k)
        return self.sim.end_action(s), carry

    def step_env(self, key, st: EnvState, action, params: EnvParams = None):
        """One step without auto-reset: (obs, state, reward, terminated, truncated, snake)."""
        p = params if params is not None else self.default_params
        a = jnp.clip(jnp.asarray(action, f32), -1, 1)
        actions = jnp.concatenate([a[None], st.actions[:-1]])
        mouse = SM.clamp_magnitude(a * p.mouse, f32(MAX_INPUT))
        s, carry = self._act(p, st.sim, st.carry, mouse)
        raw, carry, snake = self._observe(p, s, carry)
        prog = raw[F["progress"]]
        pot_c, head_c = raw[F["pot_contact"]] > 0, raw[F["head_contact"]] > 0
        held = pot_c | head_c
        best = st.best_progress
        if self.held_records:
            best = jnp.where(held, st.held_best, jnp.maximum(st.held_best, prog))
        held_best = jnp.where(held, jnp.maximum(st.held_best, prog), st.held_best)
        air = jnp.where(held, 0, st.air + 1)
        reward = (progress_reward(st.progress, prog, best, f32(1) - raw[F["pot_contact"]], p.recovery, p.regress,
                                  p.air_bonus, p.speed_bonus) - p.time_penalty - p.action_cost * jnp.mean(a * a))
        best_progress = jnp.maximum(st.best_progress, prog)
        rr = (p.record_reset >= 0) & (best_progress - prog > p.record_reset)
        best_progress = jnp.where(rr, prog, best_progress)
        held_best = jnp.where(rr, prog, held_best)
        stalled = st.stalled + 1
        mark = jnp.minimum(st.stall_mark, prog)
        moved = prog >= mark + 1
        mark, stalled = jnp.where(moved, prog, mark), jnp.where(moved, 0, stalled)
        terminated = (raw[F["fell_in_lake"]] > 0) | (raw[F["won"]] > 0)
        if self.snake_ends:
            terminated = terminated | snake
        truncated = (p.stall_steps > 0) & (stalled >= p.stall_steps)
        if not self.stall_ends:
            mark, stalled = jnp.where(truncated, prog, mark), jnp.where(truncated, 0, stalled)
            truncated = jnp.zeros_like(truncated)
        t = st.t + 1
        truncated = truncated | ((p.max_steps > 0) & (t >= p.max_steps))
        new = EnvState(sim=s, carry=carry, best_progress=best_progress, held_best=held_best, progress=prog,
                       stall_mark=mark, stalled=stalled, air=air, t=t, actions=actions)
        return self._full_obs(p, raw, new), new, reward, terminated, truncated, snake

    def step(self, key, st: EnvState, action, params: EnvParams = None):
        """gymnax-style step with auto-reset: (obs, state, reward, done, info)."""
        p = params if params is not None else self.default_params
        k_step, k_reset = jax.random.split(key)
        obs, new, reward, term, trunc, snake = self.step_env(k_step, st, action, p)
        obs_r, st_r = self.reset(k_reset, p)
        done = term | trunc
        final = new.progress
        obs = jnp.where(done, obs_r, obs)
        new = jax.tree.map(lambda r, s: jnp.where(done, r, s), st_r, new)
        info = dict(terminated=term, truncated=trunc, snake=snake, final_progress=jnp.where(done, final, f32(0)),
                    best_progress=new.best_progress)
        return obs, new, reward, done, info

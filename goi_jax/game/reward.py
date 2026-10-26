"""Upstream's reward, episode end and policy observation (goi_env/env.py GoiEnv.step / reset / _observe and
progress_reward) as pure JAX functions over the bridge observation (goi_jax.game.observe.observe).

    params = RewardParams(time_penalty=0.001, stall_steps=..., snake_ends=True, ...)   # GoiEnv's constructor args
    ep     = reset_episode(raw)                            # after a reset / load / placement observation
    ep, reward, terminated, truncated = step_episode(params, ep, raw, snake, action)
    x      = policy_obs(params, ep, raw)                   # what the policy sees (GoiEnv._observe)

`raw` is the bridge observation of the state after the action (observe's vector), `snake` its status (the pot touches
the Snake), `action` the policy's action in [-1, 1]^2 before scaling by `mouse` (GoiEnv clips it, the controller gets
action * mouse). Under NumPy 2 (NEP 50) upstream's arithmetic on the float32 observation values stays float32 (Python
float constants are weak), so rewards match upstream bit for bit (tests/game/test_observe.py); only a nonzero
action_cost can differ in the last bit (upstream multiplies by the double constant).

RewardParams holds Python values that select code paths (`if params.held_records`, `max(stall_steps, 1)`): pass it as
a static argument (closure or static_argnums; it is a hashable NamedTuple), never as a traced jit argument.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

f32, i32 = jnp.float32, jnp.int32
PROGRESS, HEAD, POT, LAKE, WON = 15, 11, 12, 13, 14  # OBS_FIELDS indices
PREV_ACTIONS = 4     # goi_env.env.PREV_ACTIONS
SPEED_REF = 0.2      # goi_env.env.SPEED_REF
MOUSE = 12.0         # goi_env.env.MOUSE


class RewardParams(NamedTuple):
    """GoiEnv's constructor arguments that shape reward and episodes (defaults as GoiEnv). None-able ones: stall_steps
    0 = no stall cutoff; record_reset inf = off."""
    time_penalty: float = 0.001
    action_cost: float = 0.0
    recovery: float = 0.0
    regress: float = 0.0
    air_bonus: float = 0.0
    speed_bonus: float = 0.0
    held_records: bool = False
    record_reset: float = float("inf")
    stall_steps: int = 0
    stall_ends: bool = True
    snake_ends: bool = False
    mouse: float = MOUSE


class Episode(NamedTuple):
    best: jax.Array        # best_progress: the episode's record
    held_best: jax.Array   # the record made while the pot or the hammer touched something
    progress: jax.Array    # _progress: last step's progress
    stall_mark: jax.Array  # _stall_mark
    stalled: jax.Array     # int32 steps without headway
    air: jax.Array         # int32 steps since the pot or the hammer last touched something
    actions: jax.Array     # (PREV_ACTIONS, 2) the last actions, newest first


def progress_reward(prev, now, best, air, recovery=0.0, regress=0.0, air_bonus=0.0, speed_bonus=0.0):
    """goi_env.env.progress_reward: new ground beyond `best` (times the air / speed bonuses), plus `recovery` per unit
    climbed back below it, minus `regress` per unit lost."""
    new = jnp.maximum(f32(0), now - best)
    forward, back = jnp.maximum(f32(0), now - prev), jnp.maximum(f32(0), prev - now)
    bonus = 1 + air_bonus * air + speed_bonus * jnp.minimum(new / f32(SPEED_REF), f32(2))
    return new * bonus + recovery * (forward - new) - regress * back


def reset_episode(raw, best=None) -> Episode:
    """GoiEnv.reset / load_state: the record starts at the observation's progress (or a saved state's `best`)."""
    p = jnp.asarray(raw[PROGRESS], f32)
    b = p if best is None else jnp.asarray(best, f32)
    return Episode(best=b, held_best=b, progress=p, stall_mark=p, stalled=jnp.asarray(0, i32),
                   air=jnp.asarray(0, i32), actions=jnp.zeros((PREV_ACTIONS, 2), f32))


def clip_action(action):
    """GoiEnv.step's clip; the controller then gets clip(action) * params.mouse."""
    return jnp.clip(jnp.asarray(action, f32), -1.0, 1.0)


def step_episode(params: RewardParams, ep: Episode, raw, snake, action):
    """GoiEnv.step after the bridge step: (episode, reward, terminated, truncated)."""
    a = clip_action(action)
    actions = jnp.concatenate([a[None], ep.actions[:-1]], 0)
    now = jnp.asarray(raw[PROGRESS], f32)
    pot = raw[POT] > 0
    held = pot | (raw[HEAD] > 0)
    best = ep.best
    if params.held_records:
        best = jnp.where(held, ep.held_best, jnp.maximum(ep.held_best, now))
    held_best = jnp.where(held, jnp.maximum(ep.held_best, now), ep.held_best)
    air = jnp.where(held, 0, ep.air + 1)
    reward = (progress_reward(ep.progress, now, best, 1.0 - raw[POT], params.recovery, params.regress,
                              params.air_bonus, params.speed_bonus)
              - params.time_penalty - params.action_cost * jnp.mean(a * a))
    new_best = jnp.maximum(ep.best, now)
    reset_rec = (new_best - now) > params.record_reset
    new_best = jnp.where(reset_rec, now, new_best)
    held_best = jnp.where(reset_rec, now, held_best)
    stalled = ep.stalled + 1
    mark = jnp.minimum(ep.stall_mark, now)
    headway = now >= mark + 1.0
    mark = jnp.where(headway, now, mark)
    stalled = jnp.where(headway, 0, stalled)
    terminated = (raw[LAKE] > 0) | (raw[WON] > 0) | (jnp.asarray(snake, bool) & params.snake_ends)
    truncated = jnp.asarray(params.stall_steps > 0) & (stalled >= max(params.stall_steps, 1))
    if not params.stall_ends:
        mark = jnp.where(truncated, now, mark)
        stalled = jnp.where(truncated, 0, stalled)
        truncated = jnp.zeros_like(truncated)
    ep = Episode(best=new_best, held_best=held_best, progress=now, stall_mark=mark, stalled=stalled, air=air,
                 actions=actions)
    return ep, reward.astype(f32), terminated, truncated


def policy_obs(params: RewardParams, ep: Episode, raw):
    """GoiEnv._observe: the bridge observation, the last PREV_ACTIONS actions (newest first), the stall share and how
    far the pot is behind the episode's record."""
    stuck = (ep.stalled.astype(f32) / f32(params.stall_steps)) if params.stall_steps else jnp.zeros((), f32)
    raw = jnp.asarray(raw, f32)
    return jnp.concatenate([raw, ep.actions.reshape(-1), jnp.stack([stuck, ep.best - raw[PROGRESS]])])


def policy_obs_size(obs_size: int) -> int:
    return obs_size + 2 * PREV_ACTIONS + 2

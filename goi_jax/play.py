"""Play a goi_jax.training checkpoint in the real game (through goi_jax.bridge) or in the JAX sim, and report how far
it climbs. The plugin's observation is bit-identical to the sim's observe(), so the checkpoint's observation
normaliser and network apply unchanged; the episode bookkeeping (last 4 actions, stall share, gap to the record,
reward, stall / time-limit / lake / win ends) is GoiEnv.step_env's, mirrored on the host for the real game.

    # real game (unsandboxed: it launches the game headless and muted), Coffee Rock placements, sim alongside
    .venv/bin/python -m goi_jax.play runs/coffee-1/checkpoints/latest.pkl --backend real --along 70 90 110 --side-by-side
    # full-climb attempt: a section specialist below 204 m, the route policy above, no stall / time-limit ends
    .venv/bin/python -m goi_jax.play runs/chimney-1/checkpoints/latest.pkl --then runs/route-v2/checkpoints/latest.pkl \\
        --switch-at 204 --no-ends --steps 60000 --video runs/climb.mp4
    # sim only (also the dry run the tests use)
    JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.play runs/coffee-1/checkpoints/latest.pkl --backend sim --episodes 2

Starts: --along D places the climber on the route at distance D (the plugin's place command, as the parity recorder
does: lift / angle tried in its order) and lets it settle for --settle still actions; --along omitted: the level
start (real) or the checkpoint's start bank (sim). --side-by-side (real backend): the start is recorded (bridge 'w',
3 still physics steps) and the same state is loaded into the sim (Sim.state_from_recording + the last step), then
the policy plays both closed loop with the same noise draws: the report gives both progress curves.
Action: mean + noise * std * N(0, 1), clipped to [-1, 1] (upstream train.py act(); noise 0.5 as its Player).

Output: a table per episode (start / best / held best / final progress, return, steps, how it ended) and
runs/<name>/play-<backend>-<time>.json with the progress curves. Last line: DONE or STOPPED.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import deque
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax import env as E
from goi_jax.training import checkpoint as C
from goi_jax.training import nets as N
from goi_jax.training import normalize as Nz
from goi_jax.training.config import TrainConfig

F = E.F
PROGRESS, LAKE, WON, POT_C, HEAD_C = (F[f] for f in ("progress", "fell_in_lake", "won", "pot_contact", "head_contact"))


class Policy:
    """A checkpoint's network + observation normaliser: full observation (raw, 105 floats) -> action in [-1, 1]."""

    def __init__(self, ckpt_path, noise=0.5, seed=0):
        ck = C.load(ckpt_path)
        self.cfg = TrainConfig.from_dict(ck["config"])
        self.params = jax.tree.map(jnp.asarray, ck["params"])
        self.rms = Nz.RMS(*(jnp.asarray(x) for x in ck["obs_rms"]))
        self.noise = float(noise)
        self.key = jax.random.PRNGKey(seed)
        cfg = self.cfg

        @jax.jit
        def act(params, rms, obs, key, noise):
            x = Nz.normalize_obs(rms, obs, cfg.clip_obs) if cfg.norm_obs else obs
            mean, log_std, v = N.apply(params, x, cfg.bounded_mean, N.section_of(params, obs))   # sec: section_input nets
            a = mean + noise * jnp.exp(log_std) * jax.random.normal(key, mean.shape)
            return jnp.clip(a, -1, 1), v
        self._act = act

    def next_key(self):
        self.key, k = jax.random.split(self.key)
        return k

    def __call__(self, obs, key):
        a, v = self._act(self.params, self.rms, jnp.asarray(obs, jnp.float32), key, jnp.float32(self.noise))
        return np.asarray(a), float(v)


class Handoff:
    """Two policies split by route progress: `low` below `at` metres, `high` from there up (a section specialist and
    the route policy, say). Same observation layout and config (low's) for both."""

    def __init__(self, low: Policy, high: Policy, at: float):
        self.low, self.high, self.at = low, high, float(at)
        self.cfg, self.params, self.noise, self.next_key = low.cfg, low.params, low.noise, low.next_key

    def __call__(self, obs, key):
        return (self.low if obs[PROGRESS] < self.at else self.high)(obs, key)


class HostEpisode:
    """GoiEnv.step_env's bookkeeping on the host for a raw plugin observation stream (float32 like the env)."""

    def __init__(self, cfg: TrainConfig, raw0):
        self.cfg = cfg
        self.stall_steps = cfg.stall_steps if cfg.stall_seconds > 0 else 0
        p = np.float32(raw0[PROGRESS])
        self.best = self.held_best = self.progress = self.mark = p
        self.stalled, self.t, self.ret = 0, 0, 0.0
        self.actions = deque([np.zeros(2, np.float32)] * E.PREV_ACTIONS, maxlen=E.PREV_ACTIONS)
        self.raw = np.asarray(raw0, np.float32)

    def obs(self):
        stuck = np.float32(self.stalled / max(self.stall_steps, 1)) if self.stall_steps > 0 else np.float32(0)
        return np.concatenate([self.raw, np.concatenate(list(self.actions)),
                               np.array([stuck, self.best - self.raw[PROGRESS]], np.float32)]).astype(np.float32)

    def step(self, a, raw, on_snake=False):
        cfg = self.cfg
        a = np.clip(np.asarray(a, np.float32), -1, 1)
        self.actions.appendleft(a)
        raw = np.asarray(raw, np.float32)
        prog = raw[PROGRESS]
        held = raw[POT_C] > 0 or raw[HEAD_C] > 0
        best = self.best
        if cfg.held_records:
            best = self.held_best if held else max(self.held_best, prog)
        if held:
            self.held_best = max(self.held_best, prog)
        new = max(0.0, prog - best)
        fwd, back = max(0.0, prog - self.progress), max(0.0, self.progress - prog)
        bonus = 1 + cfg.air_bonus * (1 - raw[POT_C]) + cfg.speed_bonus * min(new / E.SPEED_REF, 2.0)
        r = new * bonus + cfg.recovery * (fwd - new) - cfg.regress * back
        r -= cfg.time_penalty + cfg.action_cost * float(np.mean(a * a))
        self.ret += float(r)
        self.progress = prog
        self.best = max(self.best, prog)
        self.stalled += 1
        self.mark = min(self.mark, prog)
        if prog >= self.mark + 1:
            self.mark, self.stalled = prog, 0
        self.t += 1
        self.raw = raw
        end = None
        if raw[LAKE] > 0:
            end = "lake"
        elif raw[WON] > 0:
            end = "won"
        elif cfg.snake_ends and on_snake:
            end = "snake"
        elif self.stall_steps > 0 and self.stalled >= self.stall_steps:
            end = "stall"
        elif cfg.episode_steps > 0 and self.t >= cfg.episode_steps:
            end = "timeout"
        return r, end


# ---------------------------------------------------------------- sim side


class SimRunner:
    """One JAX env (no batch) stepping the policy's actions."""

    def __init__(self, cfg: TrainConfig, solver="xla"):
        from goi_jax.training import backend
        from goi_jax.training import starts as St
        self.St, self.cfg = St, cfg
        self.env = St.make_env(cfg, axis_name=None)
        backend.select_solver(self.env, solver)
        self.p = self.env.default_params
        env = self.env
        self._step = jax.jit(lambda st, a, p: env.step_env(jax.random.PRNGKey(0), st, a, p))
        self._reset_from = jax.jit(lambda s, p: St.reset_from(env, p, s))
        self._act1 = jax.jit(lambda s, a, p: env.sim.end_action(env.sim.physics_step(p.model, p.sim, s, a)[0]))

    def bank_state(self, i):
        return jax.tree.map(lambda a: a[i], self.p.starts)

    def bank_progress(self):
        bd = self.p.starts.world.bodies
        from goi_jax.game import route as R
        rm = self.p.obs.route
        pot = self.env.sim.pot
        f = jax.jit(jax.vmap(lambda x, y: R.raw_progress(rm, (x, y, rm.player_z))))
        return np.asarray(f(bd.xf_p[:, pot, 0], bd.xf_p[:, pot, 1]))

    def start(self, s):
        obs, st = self._reset_from(s, self.p)
        return np.asarray(obs), st

    def step(self, st, a):
        obs, st, r, term, trunc, snake = self._step(st, jnp.asarray(a, jnp.float32), self.p)
        return np.asarray(obs), st, float(r), bool(term), bool(trunc), bool(snake)

    def state_from_records(self, records, names, action=(0.0, 0.0)):
        """SimState after the last of `records` (bridge 'w' K=1 records): the pre-step state of the last record,
        stepped once in the sim with its action, then the end-of-action joint re-creation."""
        from goi_jax.parity import states as S
        chain = S.decode_chain(records, {}, names)
        t = len(records) - 1
        s = self.env.sim.state_from_recording(chain, t, self.cfg.contact_capacity)
        s = jax.tree.map(jnp.asarray, s)
        mouse = jnp.asarray(chain.action[t], jnp.float32)
        return self._act1(s, mouse, self.p)


def run_sim_episode(policy: Policy, sim: SimRunner, s0, steps, keys=None):
    obs, st = sim.start(s0)
    prog0 = float(obs[PROGRESS])
    curve, ret, end = [prog0], 0.0, None
    for t in range(steps):
        k = keys[t] if keys is not None else policy.next_key()
        a, _ = policy(obs, k)
        obs, st, r, term, trunc, snake = sim.step(st, a)
        ret += r
        curve.append(float(obs[PROGRESS]))
        if term or trunc:
            end = ("lake" if obs[LAKE] > 0 else "won" if obs[WON] > 0 else "snake") if term else \
                ("timeout" if int(st.t) >= policy.cfg.episode_steps else "stall")
            break
    return dict(start=prog0, best=float(st.best_progress), held_best=float(st.held_best), final=curve[-1],
                ret=ret, steps=len(curve) - 1, end=end or "steps", curve=curve)


# ---------------------------------------------------------------- real side


def place(client, along, rng):
    for lift in (0.5, 1.0, 0.0, 2.0):
        for angle in rng.permutation([90, 60, 120, 30, 150, 0, 180]):
            raw = client.place(float(along), lift, float(angle))
            if raw is not None:
                return raw
    return None


def run_real_episode(policy: Policy, client, raw0, steps, keys, frames=None, realtime=False, after_win=0.0):
    """frames: a callable given each rendered frame (RGB array) after every action; realtime: pace actions at the
    game's own rate (k physics steps of 1/120 s each) so a visible window plays at normal speed; after_win: seconds of
    still actions played (and recorded) after a win, for the game's ending (launched with -goi-ending)."""
    ep = HostEpisode(policy.cfg, raw0)
    tick, t_next = policy.cfg.k / 120.0, time.perf_counter()
    n_in = int(np.shape(policy.params["pi"][0][0])[0])
    if ep.obs().shape[0] != n_in:   # the game was configured with other rays than the training env
        raise ValueError(f"real observation has {ep.obs().shape[0]} floats, the network takes {n_in}")
    mouse = policy.cfg.mouse
    curve, end = [float(raw0[PROGRESS])], None
    for t in range(steps):
        a, _ = policy(ep.obs(), keys[t])
        raw = client.step(*(np.clip(a, -1, 1) * mouse).astype(float))
        if frames is not None:
            frames(client.frame()[2])
        if realtime:
            t_next += tick
            time.sleep(max(0.0, t_next - time.perf_counter()))
        _, end = ep.step(a, raw, client.on_snake)
        curve.append(float(raw[PROGRESS]))
        if (t + 1) % 1000 == 0:   # long (--no-ends) climbs: a stuck run shows in the log, not only at the end
            print(f"  action {t + 1}: progress {curve[-1]:.1f}, best {ep.best:.1f}", flush=True)
        if end:
            break
    if end == "won":
        for _ in range(int(after_win / tick)):
            client.step(0.0, 0.0)
            if frames is not None:
                frames(client.frame()[2])
            if realtime:
                t_next += tick
                time.sleep(max(0.0, t_next - time.perf_counter()))
    return dict(start=curve[0], best=float(ep.best), held_best=float(ep.held_best), final=curve[-1], ret=ep.ret,
                steps=ep.t, end=end or "steps", curve=curve)


def table(rows):
    lines = [f"{'episode':<16}{'start':>8}{'best':>8}{'held':>8}{'final':>8}{'return':>9}{'steps':>7}  end"]
    for name, r in rows:
        lines.append(f"{name:<16}{r['start']:8.1f}{r['best']:8.1f}{r['held_best']:8.1f}{r['final']:8.1f}"
                     f"{r['ret']:9.2f}{r['steps']:7d}  {r['end']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint")
    ap.add_argument("--backend", choices=("real", "sim"), default="real")
    ap.add_argument("--along", type=float, nargs="*", default=None, help="route distances to place the climber at")
    ap.add_argument("--episodes", type=int, default=1, help="episodes per start")
    ap.add_argument("--steps", type=int, default=900, help="actions per episode at most (900 = 30 s at k=4)")
    ap.add_argument("--settle", type=int, default=15, help="still actions after placing")
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side-by-side", action="store_true", help="real backend: also play the same start in the sim")
    ap.add_argument("--solver", default="xla", help="sim island solver: xla | pallas")
    ap.add_argument("--out", help="JSON report path (default runs/<name>/play-<backend>-<time>.json)")
    ap.add_argument("--watch", action="store_true", help="real backend: show the game window and play in real time")
    ap.add_argument("--video", help="real backend: also write the rendered episodes to this .mp4 (30 fps)")
    ap.add_argument("--then", help="a second checkpoint that takes over from --switch-at metres up (Handoff)")
    ap.add_argument("--switch-at", type=float, default=204.0, help="route progress where --then takes over")
    ap.add_argument("--no-ends", action="store_true", help="play on through stalls, the time limit and the snake (a "
                    "full-climb attempt; the lake and the win still end an episode)")
    ap.add_argument("--ending", type=float, default=0.0, metavar="SECONDS", help="real backend: let the game's own "
                    "ending run after a win (credits, the float into space; plugin -goi-ending, which deletes the save "
                    "as a real win does) and keep playing and recording it for SECONDS; use with --watch --video")
    ap.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"), help="window size with "
                    "--watch/--video")
    args = ap.parse_args(argv)

    policy = Policy(args.checkpoint, args.noise, args.seed)
    if args.no_ends:
        policy.cfg = dataclasses.replace(policy.cfg, stall_seconds=0.0, episode_steps=0, snake_ends=False)
    if args.then:
        policy = Handoff(policy, Policy(args.then, args.noise, args.seed + 1), args.switch_at)
    cfg = policy.cfg
    rows, curves, status = [], {}, "DONE"
    rng = np.random.default_rng(args.seed)
    sim = SimRunner(cfg, args.solver) if (args.backend == "sim" or args.side_by_side) else None
    try:
        if args.backend == "sim":
            prog = sim.bank_progress()
            starts = []
            if args.along:
                for d in args.along:
                    starts.append((f"sim@{d:.0f}", int(np.argmin(np.abs(prog - d)))))
            else:
                starts = [(f"sim#{i}", int(i)) for i in rng.choice(len(prog), size=1, replace=False)]
            for name, i in starts:
                for e in range(args.episodes):
                    r = run_sim_episode(policy, sim, sim.bank_state(i), args.steps)
                    rows.append((f"{name}/{e}", r))
                    curves[f"{name}/{e}"] = r["curve"]
                    print(table(rows[-1:]).splitlines()[-1], flush=True)
        else:
            from goi_jax import bridge
            from goi_jax.game import observe as O
            oc = sim.env.cfg if sim else O.ObsConfig(32, 15.0, 0, 4.0)
            shown = bool(args.watch or args.video)
            proc, client = bridge.start(k=cfg.k, n_rays=oc.n_rays, ray_length=oc.ray_length, head_rays=oc.head_rays,
                                        head_length=oc.head_length, headless=not shown, render=shown,
                                        width=args.size[0], height=args.size[1],
                                        args=("-goi-ending",) if args.ending > 0 else ())
            writer = None
            if args.video:
                import imageio.v2 as imageio
                writer = imageio.get_writer(args.video, fps=round(120 / cfg.k), codec="libx264", quality=7,
                                            macro_block_size=1)
            try:
                names = None
                if args.side_by_side:
                    from goi_jax.parity import states as S
                    g = S.parse_geometry(client.geometry())
                    names = {k: g[k] for k in ("colliders", "body_names", "joints")}
                for d in (args.along or [None]):
                    for e in range(args.episodes):
                        raw = client.reset() if d is None else place(client, d, rng)
                        if raw is None:
                            print(f"the climber fits nowhere at {d}; skipped", flush=True)
                            continue
                        for _ in range(args.settle):
                            raw = client.step(0.0, 0.0)
                        s0 = None
                        if args.side_by_side:
                            recs = [client.record_step(0.0, 0.0) for _ in range(3)]
                            raw = client.observe()
                            s0 = sim.state_from_records(recs, names)
                        keys = [policy.next_key() for _ in range(args.steps)]
                        name = "level" if d is None else f"real@{d:.0f}"
                        r = run_real_episode(policy, client, np.asarray(raw, np.float32), args.steps, keys,
                                             frames=writer.append_data if writer else None, realtime=args.watch,
                                             after_win=args.ending)
                        rows.append((f"{name}/{e}", r))
                        curves[f"{name}/{e}"] = r["curve"]
                        print(table(rows[-1:]).splitlines()[-1], flush=True)
                        if s0 is not None:
                            rs = run_sim_episode(policy, sim, s0, args.steps, keys)
                            rows.append((f"  sim/{e}", rs))
                            curves[f"{name}/{e}/sim"] = rs["curve"]
                            print(table(rows[-1:]).splitlines()[-1], flush=True)
                client.close()
            finally:
                if writer is not None:
                    writer.close()
                    print(f"video: {args.video}", flush=True)
                time.sleep(2)
                bridge.stop_all()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        status = f"STOPPED: {type(e).__name__}: {e}"
    print(table(rows))
    out = Path(args.out) if args.out else Path("runs") / cfg.name / f"play-{args.backend}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint), backend=args.backend, noise=args.noise,
                                   episodes=[dict(name=n, **{k: v for k, v in r.items() if k != "curve"})
                                             for n, r in rows], curves=curves), indent=1))
    if status == "DONE":
        best = max((r["best"] for _, r in rows), default=float("nan"))
        status = f"DONE: {len(rows)} episodes, best progress {best:.1f}; report {out}"
    print(status, flush=True)
    return 0 if status.startswith("DONE") else 1


if __name__ == "__main__":
    raise SystemExit(main())

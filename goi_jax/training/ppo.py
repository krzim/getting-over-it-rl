"""PPO on the JAX env, PureJaxRL style: everything between two host calls runs on the device.

One device call (`Trainer.chunk`) = `updates_per_call` PPO updates in a lax.scan; each update =
  rollout: num_steps x vmapped GoiEnv.step_env (K physics steps per action) with this module's own auto-reset (so
           the episode's final observation is still there to bootstrap truncations, as SB3 does), observation and
           return normalisation statistics updated every step (VecNormalize), the saved-starts archive updated
           every step, finished episodes pushed into a metrics window;
  GAE(gamma, lambda);
  epochs x minibatches of the clipped PPO loss, SB3's way: per-minibatch advantage normalisation, MSE value loss,
           approx-KL early stop (the first minibatch with KL > 1.5 target_kl ends the update's epochs, unapplied).
Devices: pmap over all local devices (axis "d"; on CPU, XLA's host-platform device count makes one device per core
group, see scripts/train_jax.py), envs vmapped inside each device (axis "w", which the Box2D solver uses to share
loop trip counts). Gradients, statistics and early-stop decisions are averaged over "d" so all devices stay identical.
Resets run under one batch-level lax.cond per step, so the reset path (observation ray casts) only runs on steps
where some env finished.

Deviations from SB3 (small, deliberate): the bootstrap value of a truncated episode's final observation uses the
observation statistics before that step's update (SB3: after); the minibatch is spread over devices (global size
batch / num_minibatches, or cfg.minibatch_size); approx-KL and advantage statistics are global over that minibatch.

Policy distillation (TrainConfig.teachers, off by default): frozen teacher checkpoints, each responsible from a route
progress `lo` up (load_teachers), passed to the device call as replicated arguments after the env params
(Trainer.device_args). Every rollout step evaluates every teacher on the raw observation (normalised with that
teacher's own frozen obs_rms) and keeps, per env, the mean and log_std of the one responsible for the env's progress
(Transition.t_mean / t_log_std); the loss adds coef * mean KL(teacher || student) (gauss_kl), coef scheduled by
Trainer.distill_coef. Without teachers those fields are None and the device program is exactly the one before.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from goi_jax.box2d.gpu import graph as G
from goi_jax import env as E
from goi_jax.training import checkpoint as C
from goi_jax.training import nets as N
from goi_jax.training import normalize as Nz
from goi_jax.training import starts as St

F = E.F
DA = "d"   # device axis
WA = "w"   # env axis inside a device


class Window(NamedTuple):
    """Ring of the last W finished episodes on this device."""
    ret: jax.Array
    best: jax.Array
    held: jax.Array
    length: jax.Array
    bank: jax.Array        # started from the bank (not the archive)
    ptr: jax.Array
    count: jax.Array


class Runner(NamedTuple):
    params: dict
    opt: N.AdamState
    obs_rms: Nz.RMS
    ret_rms: Nz.RMS
    env: E.EnvState        # (n, ...)
    obs: jax.Array         # (n, obs) raw observation of env
    ret: jax.Array         # (n,) discounted return for reward normalisation
    ep_ret: jax.Array      # (n,) raw episode return so far
    ep_len: jax.Array
    ep_bank: jax.Array     # (n,) bool
    archive: St.Archive
    win: Window
    record: jax.Array      # best progress any finished episode reached
    key: jax.Array
    update: jax.Array      # updates done (int32)
    ep_sec: jax.Array = None   # (n,) int32 section of the bank state the episode started from (-1: archive / unknown)


class Transition(NamedTuple):
    obs: jax.Array
    action: jax.Array
    logp: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    sec: jax.Array = None   # section ids of obs (cfg.section_input), else None
    t_mean: jax.Array = None      # (n, 2) the responsible teacher's action mean (cfg.teachers), else None
    t_log_std: jax.Array = None   # (n, 2) its log_std


class Teacher(NamedTuple):
    """The device-side part of a distillation teacher (load_teachers): frozen params and observation statistics."""
    params: dict
    obs_rms: Nz.RMS


def parse_teachers(spec: str):
    """"path:lo,path:lo" -> [(path, lo)] sorted by lo (the path may itself contain ':')."""
    out = []
    for item in (s_.strip() for s_ in spec.split(",")):
        if not item:
            continue
        path, sep, lo = item.rpartition(":")
        if not sep or not path:
            raise ValueError(f"teacher {item!r}: expected path:lo (lo = route progress it takes over from)")
        out.append((path, float(lo)))
    out.sort(key=lambda t: t[1])
    if len({lo for _, lo in out}) != len(out):
        raise ValueError(f"teachers {spec!r}: two teachers start at the same progress")
    return out


def load_teachers(spec: str):
    """The teachers of TrainConfig.teachers, loaded once on the host: [dict(path, lo, params, obs_rms, bounded_mean,
    norm_obs, clip_obs)] sorted by lo, each with its own checkpoint's observation statistics and network options."""
    from goi_jax.training.config import TrainConfig
    out = []
    for path, lo in parse_teachers(spec):
        ck = C.load(path)
        tc = TrainConfig.from_dict(ck["config"])
        out.append(dict(path=path, lo=lo, params=jax.tree.map(lambda a: np.asarray(a, np.float32), ck["params"]),
                        obs_rms=Nz.RMS(*(np.asarray(a, np.float32) for a in ck["obs_rms"])),
                        bounded_mean=bool(tc.bounded_mean), norm_obs=bool(tc.norm_obs), clip_obs=float(tc.clip_obs)))
    return out


def gauss_kl(m_t, ls_t, m_s, ls_s):
    """KL(teacher || student) of diagonal Gaussians (means m, log stds ls), summed over the last (action) axis:
    log(s_s / s_t) + (s_t^2 + (m_t - m_s)^2) / (2 s_s^2) - 1/2 per dim, written with exp(2 (ls_t - ls_s)) so it is
    exactly 0 for identical inputs."""
    d = ls_t - ls_s
    return jnp.sum(0.5 * jnp.exp(2 * d) + 0.5 * (m_t - m_s) ** 2 * jnp.exp(-2 * ls_s) - d - 0.5, axis=-1)


def _sel(mask, a, b):
    return jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - mask.ndim)), a, b)


def tree_sel(mask, a, b):
    return jax.tree.map(lambda x, y: _sel(mask, x, y), a, b)


class Trainer:
    """Builds the device functions for one (env, config, device count)."""

    def __init__(self, env: E.GoiEnv, cfg, n_devices: int, teachers=None):
        """teachers: load_teachers' list (default: loaded from cfg.teachers; tests pass their own)."""
        self.env, self.cfg, self.D = env, cfg, int(n_devices)
        if cfg.num_envs % self.D:
            raise ValueError(f"num_envs {cfg.num_envs} is not a multiple of the {self.D} devices")
        self.n = cfg.num_envs // self.D
        self.batch = self.n * cfg.num_steps                         # samples per device per update
        nmb = getattr(cfg, "num_minibatches", 0)
        if nmb:   # minibatch scales with the batch: a fixed number of gradient steps per update
            if self.batch % nmb:
                raise ValueError(f"per-device batch {self.batch} (envs x steps) is not a multiple of num_minibatches {nmb}")
            self.mb = self.batch // nmb
        else:
            if cfg.minibatch_size % self.D:
                raise ValueError(f"minibatch_size {cfg.minibatch_size} is not a multiple of the {self.D} devices")
            self.mb = cfg.minibatch_size // self.D
        self.mb_global = self.mb * self.D
        if self.batch % self.mb:
            raise ValueError(f"per-device batch {self.batch} (envs x steps) is not a multiple of the per-device "
                             f"minibatch {self.mb}")
        self.n_mb = self.batch // self.mb
        self.nb = cfg.archive_bins
        self.obs_size = env.obs_size
        self.bins_ok = St.archive_bins_ok(cfg, self.nb)
        self.bi = getattr(env, "bank_index", None)   # sections of the bank states (starts.make_env)
        if cfg.start_sampling not in ("entry", "section", "weak"):
            raise ValueError(f"start_sampling {cfg.start_sampling!r}: entry | section | weak")
        self.steps_per_update = cfg.num_envs * cfg.num_steps
        spec = getattr(cfg, "teachers", "")
        self.teachers = list(load_teachers(spec) if teachers is None and spec else (teachers or []))
        for t in self.teachers:
            n_in = np.shape(t["params"]["pi"][0][0])[0]
            if n_in != env.obs_size:
                raise ValueError(f"teacher {t.get('path')} takes {n_in} observations, the env gives {env.obs_size}")
        self._t_rep = None
        self._init = jax.pmap(self._init_device, axis_name=DA)
        self._chunk = jax.pmap(self._chunk_device, axis_name=DA)

    # ------------------------------------------------------------------ init

    def init(self, key, params=None, opt=None, obs_rms=None, ret_rms=None, archive=None, update=0, record=-np.inf):
        """Fresh device state (replicated learner state, per-device keys and envs). Optional pieces resume a
        checkpoint (single copies, replicated here)."""
        cfg = self.cfg
        k_par, k_dev = jax.random.split(key)
        if params is None:
            params = N.init_params(k_par, self.obs_size, 2, cfg.hidden, cfg.log_std_init,
                                   St.NS if getattr(cfg, "section_input", False) else 0)
        params = N.floor_log_std(params, getattr(cfg, "min_log_std", None))
        if opt is None:
            opt = N.adam_init(params)
        obs_rms = obs_rms if obs_rms is not None else Nz.rms_init((self.obs_size,))
        ret_rms = ret_rms if ret_rms is not None else Nz.rms_init(())
        rep = lambda t: jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (self.D,) + jnp.shape(a)), t)  # noqa
        keys = jax.random.split(k_dev, self.D)
        p_rep = self.replicate_params()
        if archive is not None:
            archive = St.archive_complete(archive)
        return self._init(keys, rep(params), rep(opt), rep(obs_rms), rep(ret_rms),
                          rep(archive) if archive is not None else rep(St.archive_init(self.env.default_params.starts, self.nb)),
                          rep(jnp.int32(update)), rep(jnp.float32(record)), p_rep)

    def replicate_params(self):
        if getattr(self, "_p_rep", None) is None:
            p = self.env.default_params
            self._p_rep = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (self.D,) + jnp.shape(a)), p)
        return self._p_rep

    def device_args(self):
        """The device call's arguments after the runner: (env params,), plus the teachers when distilling
        (replicated like the env params; frozen, never trained)."""
        if not self.teachers:
            return (self.replicate_params(),)
        if self._t_rep is None:
            ts = tuple(Teacher(t["params"], Nz.RMS(*t["obs_rms"])) for t in self.teachers)
            self._t_rep = jax.tree.map(lambda a: jnp.broadcast_to(jnp.asarray(a), (self.D,) + jnp.shape(a)), ts)
        return (self.replicate_params(), self._t_rep)

    def distill_coef(self, update):
        """The distillation weight at update `update` (int32): distill_coef, decayed linearly to distill_coef_end
        over distill_updates updates from distill_start."""
        cfg = self.cfg
        c0, c1, nu = float(cfg.distill_coef), cfg.distill_coef_end, int(cfg.distill_updates)
        if c1 is None or nu <= 0:
            return jnp.float32(c0)
        frac = (update - max(int(cfg.distill_start), 0)).astype(jnp.float32) / nu
        return jnp.float32(c0) + jnp.clip(frac, 0.0, 1.0) * jnp.float32(float(c1) - c0)

    def _init_device(self, key, params, opt, obs_rms, ret_rms, archive, update, record, p):
        cfg, n = self.cfg, self.n
        k_reset, key = jax.random.split(key)
        keys = jax.random.split(k_reset, n)
        if self.bi is not None:
            bank_i, ep_sec = St.bank_draw(self.bi, k_reset, n, cfg.start_sampling, archive.sec_tries, archive.sec_wins)
            obs, st = jax.vmap(lambda s_: St.reset_from(self.env, p, s_))(jax.tree.map(lambda a_: a_[bank_i], p.starts))
        else:
            obs, st = jax.vmap(self.env.reset, (0, None))(keys, p)
            ep_sec = jnp.full(n, -1, jnp.int32)
        if cfg.norm_obs:
            obs_rms = Nz.rms_update(obs_rms, obs, DA)
        W = cfg.episode_window
        z = jnp.zeros(W, jnp.float32)
        win = Window(z, z, z, z, jnp.zeros(W, bool), jnp.int32(0), jnp.int32(0))
        return Runner(params=params, opt=opt, obs_rms=obs_rms, ret_rms=ret_rms, env=st, obs=obs,
                      ret=jnp.zeros(n, jnp.float32), ep_ret=jnp.zeros(n, jnp.float32),
                      ep_len=jnp.zeros(n, jnp.int32), ep_bank=jnp.ones(n, bool), archive=archive, win=win,
                      record=record, key=key, update=update, ep_sec=ep_sec.astype(jnp.int32))

    # ------------------------------------------------------------------ rollout

    def _norm(self, rms, obs):
        return Nz.normalize_obs(rms, obs, self.cfg.clip_obs) if self.cfg.norm_obs else obs

    def _teacher(self, tp, raw):
        """Per env, the mean and log_std (n, 2) of the teacher responsible for the raw observation's progress, and
        its index (n,). Every teacher is evaluated (a few small MLPs) and the responsible one selected."""
        prog = raw[:, F["progress"]]
        out = None
        for i, (t, tt) in enumerate(zip(self.teachers, tp)):
            x = Nz.normalize_obs(tt.obs_rms, raw, t["clip_obs"]) if t["norm_obs"] else raw
            m, ls, _ = N.apply(tt.params, x, t["bounded_mean"], N.section_of(tt.params, raw))
            ls = jnp.broadcast_to(ls, m.shape)
            if out is None:   # the lowest teacher also covers progress below every lo
                out = (m, ls, jnp.zeros(prog.shape, jnp.int32))
            else:             # sorted by lo: the last one with lo <= progress wins
                on = prog >= t["lo"]
                out = (_sel(on, m, out[0]), _sel(on, ls, out[1]), jnp.where(on, jnp.int32(i), out[2]))
        return out

    def _env_step(self, r: Runner, p: E.EnvParams, key, tp=None):
        cfg, env, n, nb = self.cfg, self.env, self.n, self.nb
        k_act, k_step, k_reset = jax.random.split(key, 3)
        nobs = self._norm(r.obs_rms, r.obs)
        sec = N.section_of(r.params, r.obs)
        mean, log_std, v = N.apply(r.params, nobs, cfg.bounded_mean, sec)
        t_mean = t_log_std = None
        if tp is not None:   # distillation: the responsible teacher on this step's raw observation
            t_mean, t_log_std, t_i = self._teacher(tp, r.obs)
        a = mean + jnp.exp(log_std) * jax.random.normal(k_act, mean.shape)
        logp = N.log_prob(mean, log_std, a)
        step = jax.vmap(env.step_env, (0, 0, 0, None), axis_name=WA)
        obs2, st2, rew, term, trunc, snake = step(jax.random.split(k_step, n), r.env, a, p)
        done = term | trunc
        held = (obs2[:, F["pot_contact"]] > 0) | (obs2[:, F["head_contact"]] > 0)
        archive = r.archive
        if cfg.saved_starts > 0:
            archive = St.archive_save(archive, cfg, nb, st2.sim, st2.progress, held, ~term)
        archive = St.section_update(archive, r.ep_sec, done, st2.held_best, cfg.sec_decay)
        # bootstrap truncations (SB3: reward += gamma * V(terminal obs) where truncated and not terminated)
        v_final = N.value(r.params, self._norm(r.obs_rms, obs2), N.section_of(r.params, obs2))
        boot = trunc & ~term
        # episode bookkeeping and the metrics window
        ep_ret = r.ep_ret + rew
        ep_len = r.ep_len + 1
        win = r.win
        W = win.ret.shape[0]
        pos = (win.ptr + jnp.cumsum(done.astype(jnp.int32)) - 1) % W
        idx = jnp.where(done, pos, W)
        put = lambda arr, val: arr.at[idx].set(val.astype(arr.dtype), mode="drop")  # noqa: E731
        nd = jnp.sum(done.astype(jnp.int32))
        win = Window(put(win.ret, ep_ret), put(win.best, st2.best_progress), put(win.held, st2.held_best),
                     put(win.length, ep_len.astype(jnp.float32)), put(win.bank, r.ep_bank),
                     (win.ptr + nd) % W, jnp.minimum(win.count + nd, W))
        lake = obs2[:, F["fell_in_lake"]] > 0
        won = obs2[:, F["won"]] > 0
        timeout = trunc & (st2.t >= p.max_steps) & (p.max_steps > 0)
        ep_stats = dict(episodes=nd.astype(jnp.float32),
                        ret_sum=jnp.sum(jnp.where(done, ep_ret, 0)),
                        best_sum=jnp.sum(jnp.where(done, st2.best_progress, 0)),
                        best_max=jnp.max(jnp.where(done, st2.best_progress, -jnp.inf)),
                        len_sum=jnp.sum(jnp.where(done, ep_len, 0)).astype(jnp.float32),
                        lake=jnp.sum(done & lake).astype(jnp.float32), won=jnp.sum(done & won).astype(jnp.float32),
                        snake=jnp.sum(term & snake & ~lake & ~won).astype(jnp.float32),
                        stall=jnp.sum(trunc & ~term & ~timeout).astype(jnp.float32),
                        timeout=jnp.sum(timeout & ~term).astype(jnp.float32),
                        reward_sum=jnp.sum(rew), progress_sum=jnp.sum(st2.progress))
        if tp is not None:   # KL(teacher || behaviour policy) on the rollout, overall and per teacher
            dkl = gauss_kl(t_mean, t_log_std, mean, log_std)
            ep_stats["dkl_sum"] = jnp.sum(dkl)
            for i in range(len(self.teachers)):
                ep_stats[f"dkl_sum_{i}"] = jnp.sum(jnp.where(t_i == i, dkl, 0.0))
                ep_stats[f"dkl_n_{i}"] = jnp.sum(t_i == i).astype(jnp.float32)

        # auto-reset: bank state, or (saved_starts share) an archive bin
        def do_reset(_):
            kb, ka, kc = jax.random.split(k_reset, 3)
            if self.bi is not None:
                bank_i, sec = St.bank_draw(self.bi, kb, n, cfg.start_sampling, archive.sec_tries, archive.sec_wins)
            else:
                nbank = jax.tree.leaves(p.starts)[0].shape[0]
                bank_i, sec = jax.random.randint(kb, (n,), 0, nbank), jnp.full(n, -1, jnp.int32)
            s = jax.tree.map(lambda a_: a_[bank_i], p.starts)
            use_ar = jnp.zeros(n, bool)
            starts_ct = archive.starts
            if cfg.saved_starts > 0:
                bins, valid = St.archive_draw(archive, ka, self.bins_ok, n)
                bins = jnp.clip(bins, 0, nb - 1)
                use_ar = valid & (jax.random.uniform(kc, (n,)) < cfg.saved_starts)
                s_ar = jax.tree.map(lambda a_: a_[bins], archive.states)
                s = tree_sel(use_ar, s_ar, s)
                starts_ct = starts_ct.at[jnp.where(done & use_ar, bins, nb)].add(1.0, mode="drop")
            obs_r, st_r = jax.vmap(lambda s_: St.reset_from(env, p, s_))(s)
            ep_sec = jnp.where(done, jnp.where(use_ar, -1, sec), r.ep_sec).astype(jnp.int32)
            return (_sel(done, obs_r, obs2), tree_sel(done, st_r, st2), jnp.where(done, ~use_ar, r.ep_bank),
                    starts_ct, ep_sec)

        def no_reset(_):
            return obs2, st2, r.ep_bank, archive.starts, r.ep_sec

        obs_n, st_n, ep_bank, starts_ct, ep_sec = jax.lax.cond(jnp.any(done), do_reset, no_reset, None)
        archive = archive._replace(starts=starts_ct)
        obs_rms = Nz.rms_update(r.obs_rms, obs_n, DA) if cfg.norm_obs else r.obs_rms
        ret = r.ret * cfg.gamma + rew
        if cfg.norm_reward:
            ret_rms = Nz.rms_update(r.ret_rms, ret, DA)
            rew_n = Nz.scale_reward(ret_rms, rew, cfg.clip_reward)
        else:
            ret_rms, rew_n = r.ret_rms, rew
        rew_n = rew_n + cfg.gamma * v_final * boot
        ret = jnp.where(done, 0.0, ret)
        record = jnp.maximum(r.record, ep_stats["best_max"])
        r = r._replace(obs_rms=obs_rms, ret_rms=ret_rms, env=st_n, obs=obs_n, ret=ret,
                       ep_ret=jnp.where(done, 0.0, ep_ret), ep_len=jnp.where(done, 0, ep_len), ep_bank=ep_bank,
                       archive=archive, win=win, record=record, ep_sec=ep_sec)
        # GOI_GPU_SPEED=graph: the stacked outputs go through a no-op XOR so their writes are not copy-only
        # dynamic-update-slices (async side-stream memcpys keep the rollout loop on the host; box2d/gpu/graph.py)
        return r, G.opaque(G.runtime_zero(r.update),
                           (Transition(nobs, a, logp, v, rew_n, done, sec, t_mean, t_log_std), ep_stats))

    # ------------------------------------------------------------------ update

    def _loss(self, params, mb, coef=None):
        """coef: the distillation weight (with mb["t_mean"] / mb["t_log_std"], cfg.teachers)."""
        cfg = self.cfg
        mean, log_std, v = N.apply(params, mb["obs"], cfg.bounded_mean, mb.get("sec"))
        logp = N.log_prob(mean, log_std, mb["action"])
        log_ratio = logp - mb["logp"]
        ratio = jnp.exp(log_ratio)
        adv = mb["adv"]
        m = jax.lax.pmean(jnp.mean(adv), DA)
        sq = jax.lax.pmean(jnp.mean(adv * adv), DA)
        nm = self.mb_global
        std = jnp.sqrt(jnp.maximum(sq - m * m, 0.0) * (nm / max(nm - 1, 1)))
        adv = (adv - m) / (std + 1e-8)
        pg = -jnp.mean(jnp.minimum(adv * ratio, adv * jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip)))
        vl = jnp.mean((mb["ret"] - v) ** 2)
        ent = N.entropy(log_std)
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent
        kl = jax.lax.stop_gradient(jnp.mean((ratio - 1) - log_ratio))
        clipfrac = jax.lax.stop_gradient(jnp.mean((jnp.abs(ratio - 1) > cfg.clip).astype(jnp.float32)))
        aux = dict(pg=pg, vl=vl, ent=ent, kl=kl, clipfrac=clipfrac)
        if "t_mean" in mb:   # distillation: mean KL(teacher || student) over the minibatch
            distill = jnp.mean(gauss_kl(mb["t_mean"], mb["t_log_std"], mean, log_std))
            loss = loss + coef * distill
            aux["distill"] = distill
        return loss, aux

    def _update(self, r: Runner, p: E.EnvParams, tp=None):
        cfg = self.cfg
        key, k_roll, k_ep = jax.random.split(r.key, 3)
        r = r._replace(key=key)
        r, (tr, ep) = jax.lax.scan(lambda c, k: self._env_step(c, p, k, tp), r,
                                   jax.random.split(k_roll, cfg.num_steps))
        last_v = N.value(r.params, self._norm(r.obs_rms, r.obs), N.section_of(r.params, r.obs))

        z = G.runtime_zero(r.update)

        def gae(c, t):
            t = G.opaque(z, t)   # GOI_GPU_SPEED=graph: identity (box2d/gpu/graph.py)
            adv_next, v_next = c
            nd = 1.0 - t.done.astype(jnp.float32)
            delta = t.reward + cfg.gamma * v_next * nd - t.value
            adv = delta + cfg.gamma * cfg.gae_lambda * nd * adv_next
            return (adv, t.value), G.opaque(z, adv)

        _, adv = jax.lax.scan(gae, (jnp.zeros_like(last_v), last_v), tr, reverse=True)
        rets = adv + tr.value
        flat = lambda x: x.reshape((-1,) + x.shape[2:])  # noqa: E731
        data = dict(obs=flat(tr.obs), action=flat(tr.action), logp=flat(tr.logp), adv=flat(adv), ret=flat(rets))
        if tr.sec is not None:
            data["sec"] = flat(tr.sec)
        coef = None
        if tr.t_mean is not None:
            data["t_mean"], data["t_log_std"] = flat(tr.t_mean), flat(tr.t_log_std)
            coef = self.distill_coef(r.update)
        left = 1.0 - (r.update.astype(jnp.float32) * self.steps_per_update) / float(cfg.total_steps)
        lr = cfg.lr if cfg.lr_end is None else cfg.lr_end + jnp.clip(left, 0, 1) * (cfg.lr - cfg.lr_end)
        grad_fn = jax.value_and_grad(self._loss, has_aux=True)
        target = jnp.inf if cfg.target_kl is None else 1.5 * cfg.target_kl

        def minibatch(c, idx):
            params, opt, stopped, acc = c
            mb = jax.tree.map(lambda x: x[idx], data)
            (_, aux), g = grad_fn(params, mb) if coef is None else grad_fn(params, mb, coef)
            g = jax.lax.pmean(g, DA)
            aux = jax.lax.pmean(aux, DA)
            stopped = stopped | (aux["kl"] > target)
            p2, o2, gn = N.adam_step(params, g, opt, lr * opt.lr_scale, cfg.max_grad_norm)
            p2 = N.floor_log_std(p2, getattr(cfg, "min_log_std", None))
            go = ~stopped
            params = jax.tree.map(lambda a_, b_: jnp.where(go, a_, b_), p2, params)
            opt = jax.tree.map(lambda a_, b_: jnp.where(go, a_, b_), o2, opt)
            w = go.astype(jnp.float32)
            acc = dict(acc, **{k_: acc[k_] + w * v_ for k_, v_ in dict(aux, gn=gn, n=jnp.float32(1)).items()})
            acc["kl_last"] = jnp.where(go, aux["kl"], acc["kl_last"])
            acc["kl_stop"] = jnp.where(stopped & (acc["kl_stop"] == 0), aux["kl"], acc["kl_stop"])
            return (params, opt, stopped, acc), None

        def epoch(c, k):
            perm = jax.random.permutation(k, self.batch).reshape(self.n_mb, self.mb)
            c, _ = jax.lax.scan(minibatch, c, perm)
            return c, None

        acc0 = {k_: jnp.float32(0) for k_ in ("pg", "vl", "ent", "kl", "clipfrac", "gn", "n", "kl_last", "kl_stop")}
        if coef is not None:
            acc0["distill"] = jnp.float32(0)
        (params, opt, stopped, acc), _ = jax.lax.scan(epoch, (r.params, r.opt, jnp.bool_(False), acc0),
                                                      jax.random.split(k_ep, cfg.epochs))
        steps = jnp.maximum(acc["n"], 1.0)
        if getattr(cfg, "kl_adapt", False) and cfg.target_kl is not None:
            # the update's KL: the value that stopped it, else the mean over the applied minibatches
            kl_u = jnp.where(stopped, acc["kl_stop"], acc["kl"] / steps)
            sc = opt.lr_scale
            # an update cut short by the target_kl rule means the LR overshoots: lower it until updates run their epochs
            sc = jnp.where(stopped | (kl_u > 2 * cfg.target_kl), sc / 1.5,
                           jnp.where(kl_u < 0.5 * cfg.target_kl, sc * 1.5, sc))
            opt = opt._replace(lr_scale=jnp.clip(sc, 1e-3, 1.0).astype(jnp.float32))
        var_ret = jax.lax.pmean(jnp.var(data["ret"]), DA)
        var_res = jax.lax.pmean(jnp.var(data["ret"] - flat(tr.value)), DA)
        r = r._replace(params=params, opt=opt, update=r.update + 1)
        ep = {k_: (jnp.max(v_) if k_ == "best_max" else jnp.sum(v_, 0)) for k_, v_ in ep.items()}
        metrics = self._metrics(r, ep, acc, steps, stopped, var_ret, var_res, lr)
        if coef is not None:
            metrics.update(self._distill_metrics(ep, acc, steps, coef))
        return r, metrics

    def _distill_metrics(self, ep, acc, steps, coef):
        """distill_kl: KL(teacher || student) on the update's rollout (before its gradient steps), overall and per
        teacher (distill_kl_<i>, with distill_share_<i> of the samples); distill_loss: the loss term's mean over the
        applied minibatches; distill_coef: its weight."""
        psum = lambda x: jax.lax.psum(x, DA)  # noqa: E731
        samples = float(self.n * self.cfg.num_steps * self.D)
        out = dict(distill_kl=psum(ep["dkl_sum"]) / samples, distill_loss=acc["distill"] / steps,
                   distill_coef=jnp.asarray(coef, jnp.float32))
        for i in range(len(self.teachers)):
            cnt = psum(ep[f"dkl_n_{i}"])
            out[f"distill_kl_{i}"] = psum(ep[f"dkl_sum_{i}"]) / jnp.maximum(cnt, 1.0)
            out[f"distill_share_{i}"] = cnt / samples
        return out

    def _metrics(self, r, ep, acc, steps, stopped, var_ret, var_res, lr):
        cfg = self.cfg
        psum = lambda x: jax.lax.psum(x, DA)  # noqa: E731
        tot = {k_: psum(v_) for k_, v_ in ep.items() if k_ != "best_max" and not k_.startswith("dkl_")}
        n_ep = jnp.maximum(tot["episodes"], 1.0)
        win = r.win
        valid = jnp.arange(win.ret.shape[0]) < win.count
        bank = valid & win.bank
        cnt = psum(jnp.sum(valid).astype(jnp.float32))
        cnt_b = psum(jnp.sum(bank).astype(jnp.float32))
        wmean = lambda x, m: psum(jnp.sum(jnp.where(m, x, 0))) / jnp.maximum(psum(jnp.sum(m).astype(jnp.float32)), 1.0)  # noqa
        ar = r.archive
        filled_prog = jnp.where(ar.filled, ar.progress, -jnp.inf)
        samples = float(self.n * cfg.num_steps * self.D)
        return dict(
            update=r.update, lr=jnp.asarray(lr * r.opt.lr_scale, jnp.float32),
            reward_mean=tot["reward_sum"] / samples, progress_mean=tot["progress_sum"] / samples,
            episodes=tot["episodes"], ep_return=tot["ret_sum"] / n_ep, ep_best=tot["best_sum"] / n_ep,
            ep_len=tot["len_sum"] / n_ep, ep_best_max=jax.lax.pmax(ep["best_max"], DA),
            ends_lake=tot["lake"], ends_won=tot["won"], ends_snake=tot["snake"], ends_stall=tot["stall"],
            ends_timeout=tot["timeout"],
            win_episodes=cnt, win_return=wmean(win.ret, valid), win_best=wmean(win.best, valid),
            win_held=wmean(win.held, valid), win_len=wmean(win.length, valid),
            win_bank_episodes=cnt_b, win_bank_best=wmean(win.best, bank), win_bank_held=wmean(win.held, bank),
            record=jax.lax.pmax(r.record, DA),
            archive_bins=jax.lax.pmax(jnp.sum(ar.filled).astype(jnp.float32), DA),
            archive_top=jax.lax.pmax(jnp.max(filled_prog), DA),
            bank_crossed=psum(jnp.sum(ar.sec_wins)) / jnp.maximum(psum(jnp.sum(ar.sec_tries)), 1e-6),
            pg_loss=acc["pg"] / steps, value_loss=acc["vl"] / steps, entropy=acc["ent"] / steps,
            approx_kl=acc["kl"] / steps, clip_frac=acc["clipfrac"] / steps, grad_norm=acc["gn"] / steps,
            grad_steps=acc["n"], kl_stopped=stopped.astype(jnp.float32), kl_at_stop=acc["kl_stop"],
            explained_var=1.0 - var_res / jnp.maximum(var_ret, 1e-8),
            std=jnp.mean(jnp.exp(r.params["log_std"])), ret_rms_std=jnp.sqrt(r.ret_rms.var),
        )

    def _chunk_device(self, r: Runner, p: E.EnvParams, tp=None):
        return jax.lax.scan(lambda c, _: self._update(c, p, tp), r, None, length=self.cfg.updates_per_call)

    def chunk(self, runner):
        """updates_per_call updates on every device; returns (runner, metrics with leading (D, U))."""
        return self._chunk(runner, *self.device_args())

    # ------------------------------------------------------------------ host helpers

    @staticmethod
    def first(tree):
        """Device 0's copy of a replicated pytree, as numpy."""
        return jax.tree.map(lambda a: np.asarray(a[0]), tree)

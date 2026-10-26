"""The actor-critic network and its optimiser in plain JAX (no flax/optax in the venvs), mirroring upstream's SB3
setup (scripts/train.py, PPO("MlpPolicy", net_arch=[256, 256])):

- separate policy and value trunks, each Linear -> tanh per hidden layer (SB3's default activation is Tanh);
- action head Linear(256, 2) -> Gaussian mean (tanh on it with bounded_mean: scripts/policy.py BoundedMeanPolicy);
  a state-independent log_std vector, initialised to log_std_init (SB3 default 0);
- value head Linear(256, 1);
- orthogonal init (SB3 ortho_init): trunks gain sqrt(2), action head 0.01, value head 1, biases 0;
- Adam(eps=1e-5) (SB3's PPO optimizer) after clipping the gradient's global norm (max_grad_norm 0.5).

Params are a dict pytree {"pi": [(W, b), ...], "vf": [...], "mu": (W, b), "v": (W, b), "log_std": (2,)}; the same
layout is what checkpoint.py saves and play.py loads. Optional (TrainConfig flags, off by default): "pi_sec"/"vf_sec"
(section_input: a learned per-section first-layer bias, from the raw progress), a log_std floor (min_log_std), and
warm_start's surgery to resume a checkpoint into a wider / deeper network.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

LOG_2PI = math.log(2 * math.pi)


def _dense(key, n_in, n_out, gain):
    w = jax.nn.initializers.orthogonal(gain)(key, (n_in, n_out), jnp.float32)
    return w, jnp.zeros((n_out,), jnp.float32)


def init_params(key, obs_size, act_size=2, hidden=(256, 256), log_std_init=0.0, sections=0):
    """sections > 0 (TrainConfig.section_input): also "pi_sec" / "vf_sec", (sections, hidden[0]) zeros, a learned
    per-section bias of each trunk's first layer (= a one-hot of the section appended to the input; see apply)."""
    keys = iter(jax.random.split(key, 2 * len(hidden) + 2))
    params = {}
    for trunk in ("pi", "vf"):
        layers, n = [], obs_size
        for h in hidden:
            layers.append(_dense(next(keys), n, h, math.sqrt(2)))
            n = h
        params[trunk] = layers
    params["mu"] = _dense(next(keys), hidden[-1], act_size, 0.01)
    params["v"] = _dense(next(keys), hidden[-1], 1, 1.0)
    params["log_std"] = jnp.full((act_size,), log_std_init, jnp.float32)
    if sections:
        params["pi_sec"] = jnp.zeros((sections, hidden[0]), jnp.float32)
        params["vf_sec"] = jnp.zeros((sections, hidden[0]), jnp.float32)
    return params


def _trunk(layers, x, sec_bias=None):
    for i, (w, b) in enumerate(layers):
        z = x @ w + b
        if i == 0 and sec_bias is not None:
            z = z + sec_bias
        x = jnp.tanh(z)
    return x


def _sec_bias(params, name, sec):
    """The section term of a trunk's first layer: params[name][sec] (one_hot(sec) @ W_sec), or None without one."""
    if name not in params:
        return None
    if sec is None:
        raise ValueError(f"params have {name!r} (section_input): pass sec=section_of(params, raw_obs)")
    return params[name][sec]


def section_of(params, raw_obs):
    """Section ids (goi_jax/training/starts.py SECTIONS, by the observation's route progress, raw, not normalised) for
    a section-aware network, None for one without section weights. The real game's observation carries the same
    progress float, so play.py computes the same feature."""
    if "pi_sec" not in params:
        return None
    from goi_jax import env as E
    from goi_jax.training import starts as St
    return St.section_index(jnp.asarray(raw_obs)[..., E.F["progress"]]).astype(jnp.int32)


def apply(params, obs, bounded_mean=False, sec=None):
    """obs (..., obs_size) normalised -> (mean (..., 2), log_std (2,), value (...)). sec (...) int: section ids
    (section_of on the raw observation), required when params carry section weights."""
    mean = _trunk(params["pi"], obs, _sec_bias(params, "pi_sec", sec)) @ params["mu"][0] + params["mu"][1]
    if bounded_mean:
        mean = jnp.tanh(mean)
    value = (_trunk(params["vf"], obs, _sec_bias(params, "vf_sec", sec)) @ params["v"][0] + params["v"][1])[..., 0]
    return mean, params["log_std"], value


def value(params, obs, sec=None):
    return (_trunk(params["vf"], obs, _sec_bias(params, "vf_sec", sec)) @ params["v"][0] + params["v"][1])[..., 0]


def floor_log_std(params, min_log_std):
    """params with log_std >= min_log_std (TrainConfig.min_log_std; None: unchanged). ppo.py applies it after every
    Adam step, so the floor holds in apply / log_prob / entropy and in the saved params play.py reads."""
    if min_log_std is None:
        return params
    return dict(params, log_std=jnp.maximum(params["log_std"], jnp.float32(min_log_std)))


# ---------------------------------------------------------------- warm starts into another architecture


def _widen_trunk(key, layers, obs_size, hidden, deepen_scale):
    """Function-preserving surgery of one tanh trunk to `hidden` (each layer at least as wide, at least as many
    layers). Returns (layers, head_row_scale, n_old_last): the head's rows for the old last-layer units must be
    multiplied by head_row_scale; rows of new units are zero.

    Widening (exact): a new unit gets random incoming weights (orthogonal, gain sqrt 2, as init) and zero outgoing
    weights (zero rows in the next layer / head), so every output is unchanged and all weights still get gradients.
    Deepening (approximate, tanh has no identity): a new layer before the head with s * I on the old units
    (s = deepen_scale) and the head's rows scaled by 1 / s: tanh(s h) / s = h - s^2 h^3 / 3 + ..., so |error| <=
    s^2 / 3 per unit (s 0.05: <= 0.08%). The error is measured against the policy std, which is small in a trained
    run: route-fast update 704 (std 0.02 / 0.05) into 3x512 moves the mean by 0.34 std on average at s 0.2 (KL 0.19,
    10x target_kl), 0.02 std at s 0.05 (KL 0.0008); hence the default 0.05."""
    old = [np.asarray(w).shape[1] for w, _ in layers]
    if len(hidden) < len(old) or any(h < o for h, o in zip(hidden, old)):
        raise ValueError(f"cannot shrink the network from {tuple(old)} to {tuple(hidden)}")
    keys = iter(jax.random.split(key, len(hidden)))
    out, n_in_old, n_in = [], obs_size, obs_size
    for i, h in enumerate(hidden):
        k = next(keys)
        fresh = np.asarray(jax.nn.initializers.orthogonal(math.sqrt(2))(k, (n_in, h), jnp.float32))
        w = fresh.copy()
        b = np.zeros(h, np.float32)
        if i < len(old):
            w0, b0 = (np.asarray(a, np.float32) for a in layers[i])
            o = old[i]
            w[:, :o] = 0.0                      # old units: rows from new inputs stay zero
            w[:n_in_old, :o] = w0
            b[:o] = b0
            n_old_units = o
        else:                                   # new layer: s * identity on the previous layer's old units
            o = n_in_old
            if h < o:
                raise ValueError(f"an added layer ({h}) must be at least as wide as the old last layer ({o})")
            w[:, :o] = 0.0
            w[np.arange(o), np.arange(o)] = deepen_scale
            n_old_units = o
        out.append((jnp.asarray(w), jnp.asarray(b)))
        n_in_old, n_in = n_old_units, h
    scale = 1.0 if len(hidden) == len(old) else 1.0 / deepen_scale
    return out, scale, n_in_old


def widen_params(key, params, obs_size, hidden, deepen_scale=0.05):
    """A checkpoint's params resized to `hidden` (see _widen_trunk). Section weights get zero columns for new
    first-layer units."""
    hidden = tuple(int(h) for h in hidden)
    k_pi, k_vf = jax.random.split(key)
    out = dict(params)
    for trunk, head, k in (("pi", "mu", k_pi), ("vf", "v", k_vf)):
        layers, scale, n_old = _widen_trunk(k, params[trunk], obs_size, hidden, deepen_scale)
        hw, hb = (np.asarray(a, np.float32) for a in params[head])
        w = np.zeros((hidden[-1], hw.shape[1]), np.float32)
        w[:n_old] = hw * scale
        out[trunk], out[head] = layers, (jnp.asarray(w), jnp.asarray(hb))
        if trunk + "_sec" in params:
            s0 = np.asarray(params[trunk + "_sec"], np.float32)
            s = np.zeros((s0.shape[0], hidden[0]), np.float32)
            s[:, :s0.shape[1]] = s0
            out[trunk + "_sec"] = jnp.asarray(s)
    return out


def hidden_of(params):
    return tuple(int(np.asarray(w).shape[1]) for w, _ in params["pi"])


def warm_start(key, params, opt, cfg, obs_size, same_run=True, sections=0):
    """Adapt a loaded checkpoint's (params, opt) to cfg. Returns (params, opt, notes):
      - cfg.hidden differs from the checkpoint's: widen_params (function-preserving widening; deepening ~1%);
      - cfg.section_input and no section weights: zero ones (the policy is unchanged until they learn);
      - cfg.log_std_reset (only when warm-starting from another run, same_run False, so chained chunks of the same
        run do not reset again): log_std raised to at least that value;
      - cfg.min_log_std: the floor.
    Adam: widening and section input keep the checkpoint's moments and step count for the old weights (new weights
    get zero moments; every old weight sits in the top-left block of its new array), so a warm-started run's first
    steps match the base run's; adding layers restarts the moments (count 0). lr_scale is kept either way."""
    notes = []
    changed = False
    old_h = hidden_of(params)
    if tuple(cfg.hidden) != old_h:
        params = widen_params(key, params, obs_size, cfg.hidden, getattr(cfg, "deepen_scale", 0.05))
        notes.append(f"network {old_h} -> {tuple(cfg.hidden)} (function-preserving surgery)")
        changed = True
    if getattr(cfg, "section_input", False) and "pi_sec" not in params:
        h0 = int(cfg.hidden[0])
        params = dict(params, pi_sec=jnp.zeros((sections, h0), jnp.float32),
                      vf_sec=jnp.zeros((sections, h0), jnp.float32))
        notes.append(f"section input added ({sections} sections, zero weights)")
        changed = True
    reset = getattr(cfg, "log_std_reset", None)
    if reset is not None and not same_run:
        before = np.asarray(params["log_std"]).copy()
        params = dict(params, log_std=jnp.maximum(jnp.asarray(params["log_std"]), jnp.float32(reset)))
        notes.append(f"log_std {np.round(before, 3).tolist()} -> {np.round(np.asarray(params['log_std']), 3).tolist()}")
    params = floor_log_std(params, getattr(cfg, "min_log_std", None))
    params = jax.tree.map(jnp.asarray, params)
    deepened = len(tuple(cfg.hidden)) != len(old_h)
    if opt is None or (changed and deepened):
        scale = 1.0 if opt is None else opt.lr_scale
        opt = adam_init(params)._replace(lr_scale=jnp.float32(scale))
        if changed:
            notes.append("Adam moments restarted (layers added)")
    elif changed:
        opt = opt._replace(m=_pad_moments(params, opt.m), v=_pad_moments(params, opt.v))
        notes.append("Adam moments kept for the old weights (zero for the new ones)")
    return params, opt, notes


def _pad_moments(params, old):
    """Adam moments shaped like params: the old moments in the top-left block (where widen_params / the section
    weights keep the old weights), zeros elsewhere and for new leaves."""
    def pad(new, o):
        z = np.zeros(np.shape(new), np.float32)
        o = np.asarray(o, np.float32)
        z[tuple(slice(0, n) for n in o.shape)] = o
        return jnp.asarray(z)
    out = {}
    for k, v in params.items():
        out[k] = (jax.tree.map(pad, v, old[k]) if k in old
                  else jax.tree.map(lambda a: jnp.zeros(np.shape(a), jnp.float32), v))
    return out


def log_prob(mean, log_std, a):
    z = (a - mean) * jnp.exp(-log_std)
    return jnp.sum(-0.5 * z * z - log_std - 0.5 * LOG_2PI, axis=-1)


def entropy(log_std):
    return jnp.sum(log_std + 0.5 + 0.5 * LOG_2PI)


# ---------------------------------------------------------------- Adam with global-norm clipping


class AdamState(NamedTuple):
    m: dict
    v: dict
    count: jax.Array
    lr_scale: jax.Array = 1.0  # KL-adaptive learning-rate multiplier (ppo.py); checkpoints without it start at 1


def adam_init(params):
    z = jax.tree.map(jnp.zeros_like, params)
    return AdamState(z, jax.tree.map(jnp.zeros_like, params), jnp.zeros((), jnp.int32), jnp.float32(1.0))


def global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(x * x) for x in jax.tree.leaves(tree)))


def adam_step(params, grads, st: AdamState, lr, max_grad_norm=0.5, b1=0.9, b2=0.999, eps=1e-5):
    """torch.nn.utils.clip_grad_norm_ then torch.optim.Adam (bias-corrected, eps outside the sqrt)."""
    gn = global_norm(grads)
    scale = jnp.minimum(1.0, max_grad_norm / (gn + 1e-6))
    grads = jax.tree.map(lambda g: g * scale, grads)
    count = st.count + 1
    m = jax.tree.map(lambda m_, g: b1 * m_ + (1 - b1) * g, st.m, grads)
    v = jax.tree.map(lambda v_, g: b2 * v_ + (1 - b2) * g * g, st.v, grads)
    c1 = 1 - b1 ** count.astype(jnp.float32)
    c2 = 1 - b2 ** count.astype(jnp.float32)
    params = jax.tree.map(lambda p, m_, v_: p - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + eps), params, m, v)
    return params, AdamState(m, v, count, st.lr_scale), gn

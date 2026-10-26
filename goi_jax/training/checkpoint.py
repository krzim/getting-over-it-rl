"""Checkpoints as plain pickles of numpy pytrees (no orbax in the venvs).

    runs/<name>/checkpoints/update_<n>.pkl     every save (policy, optimiser, normalisers; no archive)
    runs/<name>/checkpoints/latest.pkl         the newest (written atomically: tmp + rename)

Contents: config (dict), update, env_steps, params, opt (Adam m, v, count), obs_rms, ret_rms, archive (device 0's
saved-starts archive, or None), record, key, wall_s. Everything play.py needs is params + obs_rms + config.
Environment states are not saved: a resumed run starts fresh episodes.
"""
from __future__ import annotations

import os
import pickle
import threading
from pathlib import Path

import jax
import numpy as np

from goi_jax.training import nets as N
from goi_jax.training import normalize as Nz


def to_numpy(tree):
    return jax.tree.map(lambda a: np.asarray(a), tree)


def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def save(path, ckpt: dict, keep_latest=True):
    """path (update_<n>.pkl) gets the policy without the archive (~0.5 MB); latest.pkl next to it gets everything
    (the archive is ~25 MB) and is what a rerun resumes from."""
    path = Path(path)
    _write(path, dict(ckpt, archive=None) if keep_latest else ckpt)
    if keep_latest and path.name != "latest.pkl":
        _write(path.parent / "latest.pkl", ckpt)


def save_async(path, ckpt: dict) -> threading.Thread:
    """Write in a background thread (the arrays are already on the host)."""
    t = threading.Thread(target=save, args=(path, ckpt), daemon=False)
    t.start()
    return t


def load(path) -> dict:
    with open(path, "rb") as f:
        ck = pickle.load(f)
    if ck.get("opt") is not None:
        opt = N.AdamState(*ck["opt"])
        ck["opt"] = opt._replace(lr_scale=np.float32(opt.lr_scale))   # older checkpoints: 1.0
    else:
        ck["opt"] = None
    ck["obs_rms"] = Nz.RMS(*ck["obs_rms"])
    ck["ret_rms"] = Nz.RMS(*ck["ret_rms"])
    return ck


def pack(cfg, runner_first, update, env_steps, wall_s, with_archive=True) -> dict:
    """runner_first: device 0's Runner as numpy (ppo.Trainer.first)."""
    r = runner_first
    return dict(config=cfg.to_dict(), update=int(update), env_steps=int(env_steps), params=r.params,
                opt=tuple(r.opt), obs_rms=tuple(r.obs_rms), ret_rms=tuple(r.ret_rms),
                archive=(tuple(r.archive) if with_archive else None), record=float(r.record), key=np.asarray(r.key),
                wall_s=float(wall_s))

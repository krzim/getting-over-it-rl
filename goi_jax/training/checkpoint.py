"""Checkpoints as plain pickles of numpy pytrees (no orbax in the venvs).

    runs/<name>/checkpoints/update_<n>.pkl     every save (policy, optimiser, normalisers; no archive)
    runs/<name>/checkpoints/latest.pkl         the newest (written atomically: tmp + rename)

Contents: config (dict), update, env_steps, params, opt (Adam m, v, count), obs_rms, ret_rms, archive (device 0's
saved-starts archive, or None), record, key, wall_s. Everything play.py needs is params + obs_rms + config.
Environment states are not saved: a resumed run starts fresh episodes.

A policy can also be exported as plain arrays with no pickle (save_npz): params as pi{i}_w / pi{i}_b, vf{i}_*, mu_*,
v_*, log_std (and pi_sec / vf_sec for section-input nets), obs_mean / obs_var / obs_count, and config as a JSON
string. load() reads either file, so play.py takes a .npz in place of a .pkl.

    .venv/bin/python -m goi_jax.training.checkpoint runs/<name>/checkpoints/latest.pkl policy.npz
"""
from __future__ import annotations

import json
import os
import pickle
import sys
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
    if Path(path).suffix == ".npz":
        return load_npz(path)
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


def save_npz(path, ckpt: dict):
    """The policy of a checkpoint (as load returns it) as plain arrays: params, observation normaliser, config."""
    p = ckpt["params"]
    out = {"log_std": p["log_std"], "mu_w": p["mu"][0], "mu_b": p["mu"][1], "v_w": p["v"][0], "v_b": p["v"][1]}
    for trunk in ("pi", "vf"):
        for i, (w, b) in enumerate(p[trunk]):
            out[f"{trunk}{i}_w"], out[f"{trunk}{i}_b"] = w, b
        if f"{trunk}_sec" in p:
            out[f"{trunk}_sec"] = p[f"{trunk}_sec"]
    rms = ckpt["obs_rms"]
    out.update(obs_mean=rms.mean, obs_var=rms.var, obs_count=rms.count, config=np.array(json.dumps(ckpt["config"])))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{k: np.asarray(v) for k, v in out.items()})


def load_npz(path) -> dict:
    """save_npz's file as a checkpoint dict with params, obs_rms and config (no optimiser or return normaliser). A
    file without a config array (the first published policy.npz) takes it from config.json beside it."""
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if "config" in a:
        config = json.loads(str(a["config"]))
    else:
        config = json.loads((Path(path).parent / "config.json").read_text())
    params = {"log_std": a["log_std"], "mu": (a["mu_w"], a["mu_b"]), "v": (a["v_w"], a["v_b"])}
    for trunk in ("pi", "vf"):
        n = sum(1 for k in a if k.startswith(trunk) and k[len(trunk):-2].isdigit() and k.endswith("_w"))
        params[trunk] = [(a[f"{trunk}{i}_w"], a[f"{trunk}{i}_b"]) for i in range(n)]
        if f"{trunk}_sec" in a:
            params[f"{trunk}_sec"] = a[f"{trunk}_sec"]
    obs_rms = Nz.RMS(a["obs_mean"], a["obs_var"], a.get("obs_count", np.float32(1.0)))
    return dict(config=config, params=params, obs_rms=obs_rms, opt=None, ret_rms=None)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m goi_jax.training.checkpoint CHECKPOINT.pkl OUT.npz")
    save_npz(sys.argv[2], load(sys.argv[1]))
    print(f"wrote {sys.argv[2]}")

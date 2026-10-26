"""On-device PPO for the JAX Getting Over It env (goi_jax/env.py), PureJaxRL style.

    config.py       TrainConfig: upstream scripts/train.py's settings (network, PPO, reward, episode rules, starts)
    nets.py         the actor-critic (SB3 MlpPolicy layout), Adam with global-norm clipping, in plain JAX
    normalize.py    running mean/variance (SB3 VecNormalize semantics) for observations and returns
    starts.py       reset banks from recorded states (Coffee Rock, level start) and the saved-starts archive
    ppo.py          rollout + GAE + minibatch epochs, all in one jitted, device-sharded lax.scan
    checkpoint.py   plain pickles of params, optimiser, normalisers, archive

Launch with scripts/train_jax.py; play a checkpoint in the real game with goi_jax/play.py.
"""

"""Training configuration. Defaults mirror upstream scripts/train.py (SB3 PPO + VecNormalize + Starts + TimeLimit),
so a policy trained here sees the same observation, action scale, reward and episode rules as one trained there and
transfers to the real game through goi_jax/play.py.

Upstream -> here:
  --k 4, --episode-steps 3000 (TimeLimit), --stall-seconds 10 (stall_steps = 10 * 120 / k = 300), snake_ends=True,
  time_penalty 0.001, action_cost/mouse/recovery/regress/air_bonus/speed_bonus/held_records: same names and defaults.
  PPO: lr 3e-4 (optional linear decay to lr_end), gamma 0.995, gae_lambda 0.95 (SB3), clip 0.2 (SB3), ent_coef 0,
  vf_coef 0.5 (SB3), max_grad_norm 0.5 (SB3), n_epochs 10 (SB3), batch_size 512 (minibatch), target_kl 0.02 (SB3's
  rule: stop the update's epochs at the first minibatch whose approx KL > 1.5 * target_kl).
  VecNormalize: obs and reward normalisation, clip 10, gamma for the return statistics.
  Network: 2 x 256 tanh trunks, separate actor/critic, state-independent log_std (init 0); --bounded-mean as upstream.
  (Optional, off by default, not upstream: min_log_std floor, log_std_reset, section_input, wider/deeper warm starts,
  policy distillation from frozen teachers.)
  Starts (train.py Starts): saved_starts 0.5 share from an archive of states saved per bin_size (2) of progress the
  first time an episode gets there touching something, drawn with weight 1/sqrt(1 + starts); `window` limits them.

Deliberate deviations (see also ppo.py):
  - Rollout shape: num_envs x num_steps per update (default 256 x 32 = 8192 samples; upstream 5 x 1024 = 5120).
    Epochs and the target_kl rule are upstream's; the minibatch is batch / num_minibatches (default 8), so the number
    of gradient steps per update (80) stays near upstream's 100 at any env count instead of growing with the batch
    (num_minibatches 0 restores a fixed minibatch_size).
  - The "bottom" of an episode is a bank of recorded game states (start_group: "coffee" = Coffee Rock, route
    progress ~60-120, the recorded parity chains every start_every steps; "start" = the level start; "route" = the
    climber placed every 5 m along the whole route with the plugin's place command, as upstream --place does, and
    recorded; "all" = all three) instead of the level start only. start_sampling picks the bank state: uniformly
    over sections (default), weighted toward sections the policy fails to cross ("weak"), or over states ("entry").
  - The saved-starts archive is one per device (shared by that device's envs), not one per game process.
  - No rewinds, traps, demos or curricula (upstream's defaults have them off too); no gSDE (upstream --gsde off by
    default).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    name: str = "jax-ppo"
    # ---- environment (goi_jax/env.py EnvParams + episode rules)
    k: int = 4
    episode_steps: int = 3000          # TimeLimit in actions (100 s at k=4)
    stall_seconds: float = 10.0        # truncate after this long without headway (0: never)
    snake_ends: bool = True
    held_records: bool = False
    time_penalty: float = 0.001
    action_cost: float = 0.0
    mouse: float = 12.0
    recovery: float = 0.0
    regress: float = 0.0
    air_bonus: float = 0.0
    speed_bonus: float = 0.0
    # ---- start states
    parity_dir: str = "dumps/parity"
    route: str = "dumps/obs/route.json"
    start_group: str = "coffee"        # recorded chains to draw "bottom" starts from: coffee | start | route | all
    route_dir: str = "dumps/parity-route"   # the route-wide recordings (goi_jax/parity/record_route.py)
    start_sampling: str = "section"    # bank draw: entry (uniform over states) | section (uniform over sections) | weak
    sec_decay: float = 0.999           # "weak": decay of the per-section episode counts per step that ends episodes
    route_max_air_speed: float = 3.0   # drop route bank states in free flight (pot and hammer head touching nothing,
                                       # pot faster than this, m/s: placements that fell off); <= 0 keeps them
    bank_skip: str = ""                # comma-separated sections left out of the bank ("" keeps all; space is modelled
                                       # since goi_jax/game/space.py: zone, wells, teleports, win; was "space")
    start_every: int = 10              # recorded physics steps between bank states along each chain
    saved_starts: float = 0.5          # share of episodes started from the saved-starts archive (once it has states)
    bin_size: float = 2.0
    window: tuple | None = None        # (lo, hi) route progress the archive may start episodes from
    archive_bins: int = 440            # route length ~870 / bin_size
    contact_capacity: int = 96
    compact_fixtures: bool = False     # static fixtures' AABBs/materials stored once in the Model, not per env
                                       # (goi_jax/box2d/world.py compact_model): same physics, ~35 KB less per env
    # ---- PPO
    num_envs: int = 256                # total over devices
    num_steps: int = 32                # actions per env per update
    minibatch_size: int = 512          # used only when num_minibatches == 0
    num_minibatches: int = 8           # minibatches per epoch: the minibatch scales with the batch (num_envs x
                                       # num_steps / num_minibatches), so an update is epochs x 8 = 80 gradient steps
                                       # at any env count (upstream: 5120 / 512 x 10 = 100). 0: fixed minibatch_size
    epochs: int = 10
    lr: float = 3e-4
    lr_end: float | None = None
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02
    kl_adapt: bool = True              # scale the LR by the update's KL (/1.5 if stopped early or above 2x target_kl, x1.5 below half,
                                       # capped at 1): with 16k-env batches the policy std shrank until a single Adam
                                       # step at the base LR overshot target_kl and every update stopped after 1 step
    norm_obs: bool = True
    norm_reward: bool = True
    clip_obs: float = 10.0
    clip_reward: float = 10.0
    # ---- network
    hidden: tuple = (256, 256)
    bounded_mean: bool = False
    log_std_init: float = 0.0
    # ---- learning-speed options (all off by default; nets.py warm_start / floor_log_std, ppo.py)
    min_log_std: float | None = None   # floor on the policy's log_std, held after every Adam step (e.g. log(0.1) = -2.3)
    log_std_reset: float | None = None # warm start from ANOTHER run's checkpoint: raise log_std to at least this
                                       # (e.g. log(0.3) = -1.2); chained chunks of the same run are not reset again
    section_input: bool = False        # section-aware nets: a learned per-section first-layer bias in both trunks
                                       # (= the section one-hot appended to the input), section from the raw progress
                                       # observation (starts.py SECTIONS; play.py computes the same); zero-initialised
    deepen_scale: float = 0.05         # warm start into more layers: the added layer is s * identity (nets.py)
    # resuming into another `hidden` widens / deepens the checkpoint's network function-preservingly (nets.warm_start)
    # ---- policy distillation (off by default; ppo.py load_teachers / _loss): several frozen teacher policies merged
    # into this one network by a KL(teacher || student) term on the rollout's observations, each step's teacher
    # picked by its raw route progress (the one-network replacement for play.py's Handoff)
    teachers: str = ""                 # "path:lo,path:lo,...": checkpoint .pkl and the progress (m) it takes over
                                       # from; a step's teacher is the one with the largest lo <= its progress (the
                                       # lowest one below every lo). "" = no distillation (the program is unchanged)
    distill_coef: float = 0.0          # weight of the mean KL(teacher || student) in the loss ...
    distill_coef_end: float | None = None   # ... decayed linearly to this (None: constant distill_coef) ...
    distill_updates: int = 0           # ... over this many updates counted from distill_start (0: constant)
    distill_start: int = -1            # update the run started distilling; scripts/train_jax.py sets it to the
                                       # resumed update when a run (re)starts distilling from another run (-1: 0)
    # ---- run
    total_steps: int = 10_000_000      # env steps (actions) over all envs
    updates_per_call: int = 4          # PPO updates per device call (lax.scan)
    devices: int = 0                   # 0: all local devices of the default backend
    solver: str = "xla"                # Box2D island solver: xla | pallas (goi_jax/box2d/gpu)
    seed: int = 0
    save_every_updates: int = 50
    episode_window: int = 256          # finished episodes kept per device for the windowed metrics
    extra: dict = field(default_factory=dict)

    @property
    def stall_steps(self) -> int:
        return int(self.stall_seconds * 120 / self.k)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["hidden"] = list(self.hidden)
        d["window"] = list(self.window) if self.window else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        legacy = "num_minibatches" not in d   # configs saved before num_minibatches: fixed minibatch_size
        d = {k: v for k, v in d.items() if k in names}
        if legacy:
            d["num_minibatches"] = 0
        if d.get("hidden") is not None:
            d["hidden"] = tuple(d["hidden"])
        if d.get("window") is not None:
            d["window"] = tuple(d["window"])
        return cls(**d)

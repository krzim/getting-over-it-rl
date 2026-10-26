"""observe.zone_overlap / step_flags with a compact Model (only the non-static fixtures tested) against the
all-fixture path, bit for bit, on bank states moved into and around the space zone.

    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/game/test_zone_compact.py -q
"""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from goi_jax.game import observe as O

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def env():
    if not (ROOT / "dumps/parity").exists():
        pytest.skip("needs dumps/parity")
    from goi_jax.env import GoiEnv
    return GoiEnv.from_recordings(root=ROOT / "dumps/parity", route=ROOT / "dumps/obs/route.json",
                                  compact_fixtures=True)


def test_zone_compact_matches_full(env):
    p = env.default_params
    model, om = p.model, p.obs
    assert model.slot_fixture is not None
    full = model._replace(slot_fixture=None)
    # zone_fix only selects fixtures on non-static bodies (what the compact path relies on)
    zf = np.asarray(om.zone_fix)
    rows = np.asarray(model.slot_fixture)
    assert not zf[np.setdiff1d(np.arange(zf.shape[0]), rows)].any()
    rng = np.random.default_rng(0)
    z = np.asarray(om.zone)
    st = jax.tree.map(lambda x: x[0], p.starts).world
    ref = jnp.asarray(st.bodies.xf_p[np.asarray(om.pot)])
    zone_c = jax.jit(lambda s: O.zone_overlap(om, model, s))
    zone_f = jax.jit(lambda s: O.zone_overlap(om, full, s))
    flags_c = jax.jit(lambda a, b, c: O.step_flags(om, model, a, b, c))
    flags_f = jax.jit(lambda a, b, c: O.step_flags(om, full, a, b, c))
    hits = 0
    for k in range(24):
        # shift the whole world so the player lands around the zone's top edge (both in and out of it)
        tgt = np.array([rng.uniform(z[0] - 2, z[2] + 2), rng.uniform(z[3] - 6, z[3] + 3)], np.float32)
        d = jnp.asarray(tgt) - ref
        b = st.bodies
        dyn = (b.type != 0)[:, None]
        s0 = st._replace(bodies=b._replace(xf_p=jnp.where(dyn, b.xf_p + d, b.xf_p)))
        s1 = s0._replace(bodies=s0.bodies._replace(
            xf_p=jnp.where(dyn, s0.bodies.xf_p + jnp.asarray(rng.normal(0, 1.5, 2), jnp.float32), s0.bodies.xf_p)))
        a, b_ = zone_c(s0), zone_f(s0)
        assert np.array_equal(np.asarray(a), np.asarray(b_)), k
        hits += int(np.asarray(a).any())
        carry = O.ObsCarry(progress=jnp.float32(0), lake=jnp.asarray(False), won=jnp.asarray(False), in_zone=a)
        cc, cf = flags_c(s1, s1, carry), flags_f(s1, s1, carry)
        for x, y in zip(cc, cf):
            assert np.array_equal(np.asarray(x), np.asarray(y)), k
    assert hits > 0

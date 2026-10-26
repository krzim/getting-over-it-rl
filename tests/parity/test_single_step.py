"""Single-step parity of the JAX engine against the game's recorded native Box2D world (goi_jax/parity/harness.py).

Skipped when the recordings are absent (dumps/ is gitignored; record with goi_jax.parity.record). The engine part
runs a few chains (about 1-3 minutes on CPU):
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/parity -x -q
Full table: JAX_PLATFORMS=cpu .venv/bin/python -m goi_jax.parity.harness
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("GOI_PARITY_DIR", ROOT / "dumps" / "parity"))
pytestmark = pytest.mark.skipif(not (DATA / "index.json").exists(), reason=f"no recordings in {DATA}")
CHAINS = ("c01", "c13", "c24", "c28")  # Coffee Rock random / smooth, level start smooth / still


@pytest.fixture(scope="module")
def level():
    from goi_jax.game import level as L
    return L.load_level(DATA)


@pytest.fixture(scope="module")
def chains():
    from goi_jax.parity import states
    return [(e, c) for e, c in states.iter_chains(DATA) if any(Path(e["file"]).stem.startswith(p) for p in CHAINS)]


@pytest.fixture(scope="module")
def result():
    from goi_jax.parity import harness, report
    res, lv, settings = harness.run(DATA, CHAINS, progress=False)
    print(report.format_report(res, lv, settings, worst=5))
    return res


def test_level_loads_every_recorded_contact(level, chains):
    """Proxy ids matched for every piece; every recorded contact is a candidate pair, oriented as Box2D does."""
    assert (level.fixture["proxy"] >= 0).all()
    for _, c in chains:
        _, _, info = level.native_states(c, np.arange(len(c)), 0)
        assert sum(i["not_candidate"] for i in info) == 0
        assert sum(i["flipped"] for i in info) == 0


def test_export_round_trip(level, chains):
    """Loading NAT1 and exporting it gives back the recorded bodies and contacts bit for bit."""
    from goi_jax.parity import harness
    _, c = chains[0]
    steps = np.arange(10)
    _, st, _ = level.native_states(c, steps, 1)
    out = level.export(st)
    order_ok = harness.ulp_distance(out["bodies"]["c"], c.bodies[1]["c"][steps]) == 0
    assert order_ok.all()
    assert (out["bodies"]["xf"] == c.bodies[1]["xf"][steps]).all()
    for t, s in enumerate(steps):
        ids = [tuple(int(x) for x in c.contact_ids[1][s, i]) for i in range(c.n_contacts[1][s])]
        assert [out["contacts"][t][k]["pos"] for k in ids] == list(range(len(ids)))


def test_t5_contact_friction_is_refreshed_before_the_solve(level, chains):
    """On every material-swap step, the contacts of the swapped collider already hold the new mix in NAT0."""
    from goi_jax.game import unity
    n_swap = 0
    for _, c in chains:
        steps = np.arange(len(c))
        _, st, _ = level.native_states(c, steps, 0)
        swap = np.zeros(len(c), bool)
        for f in ("hammerFriction", "potBottomFriction"):
            swap |= c.field(0, f) != c.field(1, f)
        ct = st.contacts
        mix = unity.contact_friction(st.fixtures.friction, ct.fa, ct.fb)
        live = ct.pair >= 0
        assert (mix[live] == ct.friction[live]).all()
        n_swap += swap.sum()
    assert n_swap > 0


def test_single_step_parity(result):
    """Thresholds on the step classes the engine already reproduces (bodies within max(1e-5, 1 ulp))."""
    from goi_jax.parity import report
    cls = report.class_masks(result)
    ok = report.step_ok(result)
    exact = np.ones(len(ok), bool)
    for q in report.BODY_ROWS:
        exact &= result["res"][q]["exact"]
    for k, min_ok, min_exact in (("plain", 0.99, 0.95), ("static_contact", 0.99, 0.95), ("pot_terrain", 0.98, 0.95),
                                 ("all", 0.97, 0.90)):
        m = cls[k]
        if m.sum() < 20:
            continue
        assert ok[m].mean() >= min_ok, (k, ok[m].mean())
        assert exact[m].mean() >= min_exact, (k, exact[m].mean())
    # contacts and joints after the step (impulses within tolerance) on plain steps
    m = cls["plain"]
    for q in ("normal_impulse", "tangent_impulse", "joint_impulse", "motor_impulse"):
        assert result["res"][q]["ok"][m].mean() >= 0.98, q
    # kinematic bodies (cursor, crane) on every step: MovePosition targets incl. HammerReturn's callback snap, which
    # the next step's Simulate consumes (c01 step 13)
    assert result["res"]["kinematic"]["ok"].all()

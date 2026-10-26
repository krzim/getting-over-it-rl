"""The hammer head's per-rebuild collider bake (goi_jax/game/placement.py, goi_jax/parity/diagnose_bake.py).

Skipped when the recordings are absent (dumps/ is gitignored). Runs in well under a minute on CPU:
    JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/parity/test_tip_bake.py -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
K1, K4 = ROOT / "dumps" / "parity", ROOT / "dumps" / "parity-k4"
pytestmark = pytest.mark.skipif(not (K1 / "index.json").exists() or not (K4 / "index.json").exists(),
                                reason="no recordings in dumps/parity and dumps/parity-k4")


@pytest.mark.parametrize("root", [K1, K4])
def test_start_bake_is_pure_scale(root):
    """Both sessions' geometry (baked at the level start) is the authored points times 1 + 2 ulp, bit for bit."""
    from goi_jax.game import level as L, placement as PL
    lv = L.load_level(root)
    rebuilt = L.with_tip_bake(lv, PL.bake_matrix())
    assert np.array_equal(rebuilt.fixture["verts"], lv.fixture["verts"])
    assert np.array_equal(rebuilt.fixture["normals"], lv.fixture["normals"])


def test_sessions_share_authored_points():
    from goi_jax.game import level as L
    p1, p4 = L.tip_points(L.load_level(K1)), L.tip_points(L.load_level(K4))
    assert sorted(map(tuple, np.concatenate(list(p1.values())).tolist())) == \
        sorted(map(tuple, np.concatenate(list(p4.values())).tolist()))


def test_lossy_scale_bit_exact():
    """Transform.lossyScale of every chain member from the recorded local rotations (first K=4 chain, all records)."""
    from goi_jax.game import placement as PL, pose as PO
    from goi_jax.parity import states as S
    e = S.load_index(K4)["chains"][0]
    T = PO.load_transforms(K4 / e["file"]).reshape(-1, len(PO.CHAIN), 20)
    T = T[~np.isnan(T[:, 0, 0])]
    for i in range(len(PO.CHAIN)):
        got = PL.lossy_scale(T[:, :i + 1, PO.LQ], T[:, i, PO.WQ])
        assert np.array_equal(got, T[:, i, PO.WS]), PO.CHAIN[i]


def test_rebuild_rotation_reproduces_manifolds():
    """K=4 chain c00 (placed with the hammer at 0 deg): the start bake misses Tip manifolds that a rotation of
    -112 * 2^-30 rad reproduces, all of them."""
    from goi_jax.game import level as L, placement as PL
    from goi_jax.parity import diagnose_bake as DB, diagnose_shapes as D, states as S
    lv = L.load_level(K4)
    e = next(e for e in S.load_index(K4)["chains"] if Path(e["file"]).stem.startswith("c00"))
    rows = D.collect_rows(lv, S.load_chain(K4 / e["file"]))
    start, rotated = DB._score_batch(lv, rows, [PL.bake_matrix(), PL.bake_matrix(2, 2, -112)], L.tip_points(lv))
    assert rotated == len(rows["step"]) > start

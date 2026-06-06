"""
Phase-3 tests: GeoTox-Adaptive scaling and the results-analysis helpers.

Run locally (requires torch):
    python3 -m pytest tests/test_phase3.py -v
or:
    python3 tests/test_phase3.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.server import FederatedServer
from server.aggregation import aggregate
import analyze_results as ar


def _server(method, max_scale=5.0):
    """A FederatedServer with only the attrs _apply_adaptive_scaling needs."""
    s = FederatedServer.__new__(FederatedServer)
    s.aggregation_method = method
    s.attack_adaptive_max_scale = max_scale
    return s


def _u(value, dim=6):
    return {"w": torch.full((dim,), float(value))}


def test_adaptive_scales_to_cap_when_defense_always_accepts():
    """Against mean (no filtering) adaptive should reach the max scale."""
    updates = [_u(1.0), _u(1.0), _u(0.2)]
    base_mal = updates[2]["w"].clone()
    srv = _server("mean", max_scale=5.0)

    srv._apply_adaptive_scaling(updates, [2], [1, 1, 1], server_update=None)
    assert torch.allclose(updates[2]["w"], base_mal * 5.0)


def test_adaptive_keeps_base_when_outlier_is_filtered():
    """Against Krum a lone far outlier is rejected, so scale stays at 1x."""
    updates = [_u(1.0), _u(1.0), _u(1.0), _u(1.0), _u(100.0)]
    base_mal = updates[4]["w"].clone()
    srv = _server("krum", max_scale=5.0)

    srv._apply_adaptive_scaling(updates, [4], [1] * 5, server_update=None)
    assert torch.allclose(updates[4]["w"], base_mal)  # unchanged (1x)

    # And it is indeed not selected by Krum.
    _, info = aggregate(updates, [1] * 5, "krum", num_byzantine=1)
    assert info.selected[4] is False


def test_adaptive_result_is_accepted_at_boundary():
    """Adaptive scaling against multi_krum should push the malicious update up
    to the acceptance boundary and leave it accepted (without over-shooting).

    Setup: a tight cluster at 1.0 that includes the malicious client (idx 0),
    plus one benign outlier at 3.0 which multi_krum drops first. As the
    malicious update is scaled up it eventually becomes the outlier, so the
    acceptance boundary lies strictly inside (1, max_scale).
    """
    updates = [_u(1.0), _u(1.0), _u(1.0), _u(1.0), _u(1.0), _u(3.0)]
    srv = _server("multi_krum", max_scale=5.0)
    srv._apply_adaptive_scaling(updates, [0], [1] * 6, server_update=None)

    _, info = aggregate(updates, [1] * 6, "multi_krum", num_byzantine=1)
    assert info.selected[0] is True
    # Scaled up beyond the base (1x) but not to the cap (a real boundary).
    val = updates[0]["w"][0].item()
    assert 1.0 < val < 5.0, val


def test_analyze_collects_final_metrics():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        stem = "mnist_mean_geotox_iid_m4_t0.5_s42"
        (d / f"{stem}.csv").write_text(
            "round,loss,accuracy,asr,evasion_rate,timestamp\n"
            "1,0.5,90.0,10.0,100.0,t\n"
            "2,0.4,92.0,20.0,100.0,t\n"
            "3,0.3,94.0,30.0,50.0,t\n"
        )
        (d / f"{stem}_config.txt").write_text(
            "==========\n"
            "dataset: mnist\n"
            "aggregation: mean\n"
            "attack: geotox\n"
            "partition: iid\n"
            "malicious: 4\n"
            "seed: 42\n"
            "tau: 0.5\n"
            "==========\n"
        )
        records = ar.collect(d, last=2)
        assert len(records) == 1
        r = records[0]
        assert r["dataset"] == "mnist" and r["attack"] == "geotox"
        assert r["tau"] == "0.5" and r["malicious"] == "4"
        # last=2 -> mean of rounds 2 and 3
        assert abs(r["accuracy"] - 93.0) < 1e-6
        assert abs(r["asr"] - 25.0) < 1e-6
        assert abs(r["evasion"] - 75.0) < 1e-6
        assert r["rounds"] == 3


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-3 tests passed.")

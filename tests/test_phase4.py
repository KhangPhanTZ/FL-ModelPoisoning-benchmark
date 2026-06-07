"""
Phase-4 tests: durability (attacker leaves) scheduling, the durability metric
in the analysis, and the durability-aware result filename.

These are torch-free.

Run:
    python3 -m pytest tests/test_phase4.py -v
or:
    python3 tests/test_phase4.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.schedule import attack_active
from utils.logger import FLLogger
import analyze_results as ar


def test_attack_active_schedule():
    # No early stop: always active.
    assert all(attack_active(r, 0) for r in range(1, 10))
    # Active up to and including attack_until, inactive after.
    assert attack_active(1, 5) is True
    assert attack_active(5, 5) is True
    assert attack_active(6, 5) is False
    assert attack_active(50, 5) is False


def test_durability_filename_encodes_until_and_tau():
    with tempfile.TemporaryDirectory() as d:
        lg = FLLogger("mean", "geotox", "iid", 4, results_dir=d,
                      dataset="mnist", seed=42, tau=0.0, attack_until=25)
        assert lg.filename == "mnist_mean_geotox_iid_m4_u25_t0.0_s42.csv", lg.filename


def test_durability_metric_retention():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        stem = "mnist_mean_geotox_iid_m4_u3_t0.0_s42"
        (d / f"{stem}.csv").write_text(
            "round,loss,accuracy,asr,evasion_rate,timestamp\n"
            "1,0.5,90,10,100,t\n"
            "2,0.4,91,50,100,t\n"
            "3,0.3,92,80,100,t\n"   # attacker leaves here -> ASR@stop = 80
            "4,0.3,93,60,,t\n"
            "5,0.3,94,30,,t\n"
            "6,0.3,95,20,,t\n"      # final ASR = 20 -> retention = 25%
        )
        (d / f"{stem}_config.txt").write_text(
            "dataset: mnist\naggregation: mean\nattack: geotox\n"
            "partition: iid\nmalicious: 4\nseed: 42\ntau: 0.0\n"
            "attack_until: 3\n"
        )
        recs = ar.collect(d, last=3)
        assert len(recs) == 1
        r = recs[0]
        assert r["attack_until"] == 3
        assert abs(r["asr_at_stop"] - 80.0) < 1e-6
        assert abs(r["retention"] - 25.0) < 1e-6  # 100 * 20 / 80


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-4 tests passed.")

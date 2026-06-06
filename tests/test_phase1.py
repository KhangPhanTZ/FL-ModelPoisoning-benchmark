"""
Phase-1 tests: complete FLTrust (with a trusted server update), multi-dataset
support, and the dataset/seed-aware result filename scheme.

Run locally (requires torch / torchvision):
    PYTHONPATH=. python3 tests/test_phase1.py
or:
    python3 -m pytest tests/test_phase1.py -v
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.aggregation import aggregate
from data.datasets import available_datasets, get_transforms, DATASET_INFO
from utils.logger import FLLogger


def _u(value, shape=(4,)):
    return {"w": torch.full(shape, float(value))}


def test_fltrust_rejects_update_opposing_server_direction():
    """
    With a trusted server update pointing in +direction, benign (aligned)
    clients get positive trust while a client pointing the opposite way gets
    zero trust (ReLU of a negative cosine) and is not selected.
    """
    server_update = _u(1.0)              # reference direction: +
    updates = [
        _u(1.0),                         # benign, cos = +1
        _u(2.0),                         # benign, cos = +1
        _u(-5.0),                        # malicious, cos = -1 -> trust 0
    ]
    malicious = [2]

    delta, info = aggregate(
        updates, [1, 1, 1], "fltrust", server_update=server_update
    )

    assert info.selected[2] is False, info.selected
    assert info.selected[0] and info.selected[1]

    # Evasion rate of the malicious client should be 0% (it was filtered).
    accepted = sum(1 for i in malicious if info.selected[i])
    assert 100.0 * accepted / len(malicious) == 0.0

    # Aggregated direction stays aligned with the trusted server update.
    dot = sum((delta["w"] * server_update["w"]).tolist())
    assert dot > 0, dot


def test_fltrust_normalises_to_server_norm():
    """An aligned client is rescaled to the server-update magnitude."""
    server_update = _u(1.0)              # ||.|| = 2.0 over 4 coords
    updates = [_u(10.0)]                 # same direction, much larger norm
    delta, info = aggregate(updates, [1], "fltrust", server_update=server_update)

    server_norm = torch.norm(server_update["w"])
    assert torch.allclose(torch.norm(delta["w"]), server_norm, atol=1e-4)


def test_available_datasets_and_transforms():
    names = available_datasets()
    assert "mnist" in names and "fashion_mnist" in names
    for name in names:
        t = get_transforms(name)
        assert t is not None
        assert DATASET_INFO[name]["num_classes"] == 10
    try:
        get_transforms("not_a_dataset")
        assert False, "expected ValueError for unknown dataset"
    except ValueError:
        pass


def test_logger_filename_scheme():
    with tempfile.TemporaryDirectory() as d:
        iid = FLLogger("fltrust", "lie", "iid", 2,
                       results_dir=d, dataset="fashion_mnist", seed=7)
        assert iid.filename == "fashion_mnist_fltrust_lie_iid_m2_s7.csv", iid.filename

        noniid = FLLogger("krum", "minmax", "noniid", 4,
                          results_dir=d, dataset="mnist", alpha=0.3, seed=1)
        assert noniid.filename == "mnist_krum_minmax_noniid_a0.3_m4_s1.csv", noniid.filename


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-1 tests passed.")

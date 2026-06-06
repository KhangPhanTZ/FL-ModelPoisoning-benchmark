"""
Phase-0 correctness tests for the update-space refactor.

Run locally (requires torch):
    python -m pytest tests/test_phase0.py -v
or simply:
    python tests/test_phase0.py
"""

import torch

from server.attacks import apply_attack
from server.aggregation import aggregate


def _make_update(value, shape=(3,)):
    return {"w": torch.full(shape, float(value))}


def test_fedavg_update_space_matches_weight_space():
    """FedAvg on updates + reconstruction must equal FedAvg on weights."""
    w_global = {"w": torch.tensor([1.0, 1.0, 1.0])}
    local = [
        {"w": torch.tensor([2.0, 2.0, 2.0])},
        {"w": torch.tensor([4.0, 4.0, 4.0])},
    ]
    sizes = [1, 3]

    # Weight-space FedAvg
    total = sum(sizes)
    weight_space = sum(local[i]["w"] * (sizes[i] / total) for i in range(2))

    # Update-space FedAvg + reconstruction
    updates = [{"w": local[i]["w"] - w_global["w"]} for i in range(2)]
    delta, info = aggregate(updates, sizes, "mean")
    reconstructed = w_global["w"] + delta["w"]

    assert torch.allclose(reconstructed, weight_space), (reconstructed, weight_space)
    assert info.selected == [True, True]


def test_lie_attack_sets_mean_minus_z_std():
    """LIE malicious update = mean_benign - z * std_benign (per coordinate)."""
    updates = [
        _make_update(1.0),
        _make_update(3.0),
        _make_update(999.0),  # malicious, value will be overwritten
    ]
    malicious = [2]
    z = 2.0

    benign = torch.stack([updates[0]["w"], updates[1]["w"]])
    expected = benign.mean(0) - z * torch.clamp(benign.std(0), min=1e-8)

    out = apply_attack(updates, malicious, "lie", z=z)
    assert torch.allclose(out[2]["w"], expected), (out[2]["w"], expected)


def test_model_replacement_scales_update():
    """Model replacement scales the malicious update by total/malicious data."""
    updates = [_make_update(1.0), _make_update(1.0), _make_update(0.5)]
    sizes = [10, 10, 5]
    malicious = [2]

    out = apply_attack(
        updates, malicious, "model_replacement",
        z=1.0, client_data_sizes=sizes,
    )
    scale = sum(sizes) / sizes[2]  # 25 / 5 = 5
    assert torch.allclose(out[2]["w"], torch.full((3,), 0.5 * scale))


def test_krum_rejects_lone_outlier_and_reports_evasion():
    """Krum should pick a benign update; the malicious one is not selected."""
    updates = [
        _make_update(1.0),
        _make_update(1.1),
        _make_update(0.9),
        _make_update(100.0),  # malicious outlier
    ]
    malicious = [3]
    delta, info = aggregate(updates, [1, 1, 1, 1], "krum", num_byzantine=1)

    assert info.selected[3] is False, info.selected
    accepted = sum(1 for i in malicious if info.selected[i])
    evasion = 100.0 * accepted / len(malicious)
    assert evasion == 0.0


def test_median_is_coordinate_wise():
    updates = [_make_update(1.0), _make_update(2.0), _make_update(100.0)]
    delta, info = aggregate(updates, [1, 1, 1], "median")
    assert torch.allclose(delta["w"], torch.full((3,), 2.0))
    assert info.selection_type == "coordinate_wise"


def test_none_attack_is_noop():
    updates = [_make_update(1.0), _make_update(2.0)]
    before = [u["w"].clone() for u in updates]
    out = apply_attack(updates, [0], "none")
    for b, u in zip(before, out):
        assert torch.allclose(b, u["w"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-0 tests passed.")

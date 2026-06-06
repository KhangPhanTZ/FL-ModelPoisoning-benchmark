"""
Phase-2 tests: the GeoTox attack contracts.

GeoTox shapes a malicious (backdoor-carrying) update so that, on the flattened
update vector:
  - directional stealth: cos(final, benign_mean) >= tau,
  - magnitude stealth:    ||final|| == median(benign update norms),
  - stealth is monotone in tau (higher tau -> more benign-aligned).

These are the contracts that make tau a principled Evasion<->ASR trade-off knob.

Run locally (requires torch):
    python3 -m pytest tests/test_phase2.py -v
or:
    python3 tests/test_phase2.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import torch

from server.attacks import compute_geotox_attack, _blend_lambda, _flatten_update


def _make_updates(dim=40, n_benign=6, seed=0):
    """A tight benign cluster plus one malicious update pointing elsewhere."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(dim, generator=g)
    updates = [{"w": base + 0.02 * torch.randn(dim, generator=g)} for _ in range(n_benign)]
    updates.append({"w": 5.0 * torch.randn(dim, generator=g)})  # malicious "raw"
    return updates


def _benign_ref(updates, benign_idx):
    flats = torch.stack([_flatten_update(updates[i]) for i in benign_idx])
    mu = flats.mean(0)
    return mu / mu.norm(), flats.norm(dim=1).median()


def test_geotox_satisfies_cosine_constraint():
    for tau in (0.0, 0.3, 0.6, 0.9):
        updates = _make_updates()
        benign_idx = list(range(6))
        mu_hat, _ = _benign_ref(updates, benign_idx)

        compute_geotox_attack(updates, malicious_indices=[6], tau=tau, mask_ratio=0.7)
        final = _flatten_update(updates[6])
        cos = torch.dot(final / final.norm(), mu_hat).item()
        assert cos >= tau - 1e-2, (tau, cos)


def test_geotox_matches_median_benign_norm():
    updates = _make_updates()
    benign_idx = list(range(6))
    _, B = _benign_ref(updates, benign_idx)

    compute_geotox_attack(updates, malicious_indices=[6], tau=0.5, mask_ratio=0.7)
    final = _flatten_update(updates[6])
    assert torch.allclose(final.norm(), B, atol=1e-4), (final.norm().item(), B.item())


def test_geotox_stealth_is_monotone_in_tau():
    base = _make_updates()
    benign_idx = list(range(6))
    mu_hat, _ = _benign_ref(base, benign_idx)

    cos_low = _run_and_cos(copy.deepcopy(base), mu_hat, tau=0.1)
    cos_high = _run_and_cos(copy.deepcopy(base), mu_hat, tau=0.9)
    assert cos_high >= cos_low, (cos_low, cos_high)


def _run_and_cos(updates, mu_hat, tau):
    compute_geotox_attack(updates, malicious_indices=[6], tau=tau, mask_ratio=0.7)
    final = _flatten_update(updates[6])
    return torch.dot(final / final.norm(), mu_hat).item()


def test_blend_lambda_contract():
    # Already aligned -> no blending needed.
    assert _blend_lambda(0.8, 0.5) == 0.0
    # Opposed (but not exactly anti-parallel) direction -> blended cosine
    # must reach tau. c = -1 is a degenerate (zero-vector) edge case and is
    # excluded on purpose.
    c, tau = -0.5, 0.6
    lam = _blend_lambda(c, tau)
    num = lam + (1 - lam) * c
    den = (lam ** 2 + (1 - lam) ** 2 + 2 * lam * (1 - lam) * c) ** 0.5
    assert den > 1e-9 and num / den >= tau - 1e-3, (lam, num / den)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-2 tests passed.")

"""
Phase-5 tests: extra defenses (FLAME-style, trimmed mean, norm clipping).

Run locally (requires torch):
    python3 -m pytest tests/test_phase5.py -v
or:
    python3 tests/test_phase5.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from server.aggregation import aggregate, _flatten


def _u(value, dim=6):
    return {"w": torch.full((dim,), float(value))}


def _vec(values):
    return {"w": torch.tensor([float(v) for v in values])}


def test_flame_filters_cosine_outlier():
    """A malicious update pointing opposite the benign majority is filtered."""
    updates = [_u(1.0), _u(1.0), _u(1.0), _u(1.0), _u(-1.0)]  # idx4 opposite dir
    delta, info = aggregate(updates, [1] * 5, "flame",
                            num_byzantine=1, noise_lambda=0.0)
    assert info.selected[4] is False, info.selected
    assert all(info.selected[i] for i in range(4))


def test_flame_clips_to_median_norm():
    """Admitted updates are clipped to the median admitted norm."""
    updates = [_u(1.0), _u(1.0), _u(1.0), _u(50.0)]  # one large-norm (same dir)
    delta, info = aggregate(updates, [1] * 4, "flame",
                            num_byzantine=1, noise_lambda=0.0)
    S = info.extra["median_norm"]
    # Averaged clipped updates cannot exceed the clip bound S.
    assert torch.norm(_flatten(delta)).item() <= S + 1e-4


def test_trimmed_mean_drops_extremes():
    updates = [_vec([1, 1]), _vec([2, 2]), _vec([3, 3]), _vec([4, 4]), _vec([100, 100])]
    delta, info = aggregate(updates, [1] * 5, "trimmed_mean", num_byzantine=1)
    # beta=1 -> drop min and max per coord -> mean of {2,3,4} = 3
    assert torch.allclose(delta["w"], torch.tensor([3.0, 3.0]))
    assert info.selection_type == "coordinate_wise"


def test_norm_clip_limits_but_never_rejects():
    updates = [_u(1.0), _u(1.0), _u(1.0), _u(100.0)]
    delta, info = aggregate(updates, [1] * 4, "norm_clip")
    assert all(info.selected)                       # never rejects
    assert info.extra["clipped"][3] is True         # outlier was clipped
    assert info.extra["clipped"][0] is False


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(fns)} Phase-5 tests passed.")

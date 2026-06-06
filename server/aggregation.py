"""
Byzantine-robust aggregation operating in UPDATE SPACE.

Phase-0 refactor.  Every aggregator now consumes client *updates*
    u_i = w_local_i - w_global
and returns the aggregated update  Delta  (the server reconstructs the new
global model as  w_global + Delta).  This keeps norms / distances / cosine
similarities defined on gradients, which is what robust aggregators assume.

Each aggregator additionally returns an ``AggregationInfo`` describing, per
client, the weight it received in the final aggregate and whether it was
"accepted".  This instrumentation is what lets us measure the **Evasion Rate**
of an attack: the fraction of malicious client-rounds whose update was accepted
(not filtered out) by the defense.

For coordinate-wise defenses (Median, Trimmed-Mean inside Bulyan) there is no
per-client accept/reject decision; such clients are reported as accepted with a
``selection_type`` of ``"coordinate_wise"`` so downstream code can treat their
evasion rate appropriately.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class AggregationInfo:
    """Per-client bookkeeping returned by every aggregator."""

    selected: List[bool]
    weights: List[float]
    # "selection": defense explicitly accepts/rejects clients (Krum, FLTrust...)
    # "coordinate_wise": no per-client decision (Median); selected is all-True.
    selection_type: str = "selection"
    extra: Dict[str, object] = field(default_factory=dict)


Update = Dict[str, torch.Tensor]


def _zeros_like_update(reference: Update) -> Update:
    return {k: torch.zeros_like(v, dtype=torch.float32) for k, v in reference.items()}


def _flatten(update: Update) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in update.values()])


def _krum_scores(flat_updates: List[torch.Tensor], k: int) -> List[float]:
    """Krum score: sum of the k smallest squared distances to other clients."""
    n = len(flat_updates)
    scores = []
    for i in range(n):
        distances = [
            torch.sum((flat_updates[i] - flat_updates[j]) ** 2).item()
            for j in range(n)
            if i != j
        ]
        distances.sort()
        scores.append(sum(distances[:k]))
    return scores


def fedavg(updates: List[Update], data_sizes: List[int]) -> Tuple[Update, AggregationInfo]:
    """Weighted average of updates (data-size weighted)."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    total = sum(data_sizes)
    weights = [size / total for size in data_sizes]

    aggregated = _zeros_like_update(updates[0])
    for key in aggregated:
        for update, w in zip(updates, weights):
            aggregated[key] += update[key].float() * w

    info = AggregationInfo(selected=[True] * len(updates), weights=weights)
    return aggregated, info


def median(updates: List[Update], data_sizes: List[int]) -> Tuple[Update, AggregationInfo]:
    """Coordinate-wise median of updates."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    aggregated = {}
    for key in updates[0]:
        stacked = torch.stack([u[key].float() for u in updates])
        aggregated[key] = stacked.median(dim=0)[0]

    n = len(updates)
    info = AggregationInfo(
        selected=[True] * n,
        weights=[1.0 / n] * n,
        selection_type="coordinate_wise",
    )
    return aggregated, info


def krum(
    updates: List[Update],
    data_sizes: List[int],
    num_byzantine: int = 0,
) -> Tuple[Update, AggregationInfo]:
    """Krum: select the single update closest to its n-f-2 nearest neighbours."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    if n == 1:
        return updates[0], AggregationInfo(selected=[True], weights=[1.0])

    f = min(num_byzantine, n - 2)
    k = max(1, n - f - 2)

    flat = [_flatten(u) for u in updates]
    scores = _krum_scores(flat, k)
    selected_idx = scores.index(min(scores))

    selected = [i == selected_idx for i in range(n)]
    weights = [1.0 if i == selected_idx else 0.0 for i in range(n)]
    info = AggregationInfo(selected=selected, weights=weights,
                           extra={"selected_idx": selected_idx})
    return updates[selected_idx], info


def multi_krum(
    updates: List[Update],
    data_sizes: List[int],
    num_byzantine: int = 0,
    num_select: int = 0,
) -> Tuple[Update, AggregationInfo]:
    """Multi-Krum: average the m updates with the lowest Krum scores."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    if n == 1:
        return updates[0], AggregationInfo(selected=[True], weights=[1.0])

    f = min(num_byzantine, n - 2)
    k = max(1, n - f - 2)
    m = num_select if num_select > 0 else max(1, n - f)

    flat = [_flatten(u) for u in updates]
    scores = _krum_scores(flat, k)
    selected_indices = sorted(range(n), key=lambda i: scores[i])[:m]
    selected_set = set(selected_indices)

    aggregated = _zeros_like_update(updates[0])
    for key in aggregated:
        stacked = torch.stack([updates[i][key].float() for i in selected_indices])
        aggregated[key] = stacked.mean(dim=0)

    selected = [i in selected_set for i in range(n)]
    weights = [1.0 / m if i in selected_set else 0.0 for i in range(n)]
    info = AggregationInfo(selected=selected, weights=weights,
                           extra={"selected_indices": selected_indices})
    return aggregated, info


def bulyan(
    updates: List[Update],
    data_sizes: List[int],
    num_byzantine: int = 0,
) -> Tuple[Update, AggregationInfo]:
    """Bulyan: Multi-Krum selection followed by a coordinate-wise trimmed mean."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    if n == 1:
        return updates[0], AggregationInfo(selected=[True], weights=[1.0])

    f = min(num_byzantine, (n - 3) // 2)  # Bulyan requires n >= 4f + 3
    k = max(1, n - f - 2)
    m = max(1, n - 2 * f)

    flat = [_flatten(u) for u in updates]
    scores = _krum_scores(flat, k)
    selected_indices = sorted(range(n), key=lambda i: scores[i])[:m]
    selected_set = set(selected_indices)

    beta = max(1, f) if len(selected_indices) > 2 else 0

    aggregated = {}
    for key in updates[0]:
        stacked = torch.stack([updates[i][key].float() for i in selected_indices])
        if beta > 0 and stacked.size(0) > 2 * beta:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            aggregated[key] = sorted_vals[beta:-beta].mean(dim=0)
        else:
            aggregated[key] = stacked.mean(dim=0)

    selected = [i in selected_set for i in range(n)]
    weights = [1.0 / m if i in selected_set else 0.0 for i in range(n)]
    info = AggregationInfo(
        selected=selected,
        weights=weights,
        selection_type="coordinate_wise",  # trimmed mean acts per coordinate
        extra={"selected_indices": selected_indices, "trim": beta},
    )
    return aggregated, info


def _unflatten_like(reference: Update, flat: torch.Tensor) -> Update:
    """Reshape a flat vector back into an update dict matching ``reference``."""
    out, offset = {}, 0
    for key, ref in reference.items():
        numel = ref.numel()
        out[key] = flat[offset:offset + numel].reshape(ref.shape).clone()
        offset += numel
    return out


def trimmed_mean(
    updates: List[Update],
    data_sizes: List[int],
    num_byzantine: int = 0,
) -> Tuple[Update, AggregationInfo]:
    """Coordinate-wise trimmed mean (Yin et al., 2018): drop beta=f extremes."""
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    beta = min(num_byzantine, (n - 1) // 2)

    aggregated = {}
    for key in updates[0]:
        stacked = torch.stack([u[key].float() for u in updates])
        if beta > 0 and n > 2 * beta:
            sorted_vals, _ = torch.sort(stacked, dim=0)
            aggregated[key] = sorted_vals[beta:-beta].mean(dim=0)
        else:
            aggregated[key] = stacked.mean(dim=0)

    info = AggregationInfo(
        selected=[True] * n,
        weights=[1.0 / n] * n,
        selection_type="coordinate_wise",
        extra={"trim": beta},
    )
    return aggregated, info


def norm_clip(
    updates: List[Update],
    data_sizes: List[int],
    clip_norm: float = None,
) -> Tuple[Update, AggregationInfo]:
    """
    Norm-bounded FedAvg: clip each update to ``clip_norm`` (default = median
    update norm), then take the weighted average.

    This is the natural defense to test GeoTox's magnitude stealth: GeoTox sits
    at the median benign norm, so it is (by construction) not clipped.
    """
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    norms = [torch.norm(_flatten(u)) for u in updates]
    bound = clip_norm if clip_norm is not None else torch.stack(norms).median().item()
    total = sum(data_sizes)

    aggregated = _zeros_like_update(updates[0])
    clipped = []
    weights = []
    for u, size, norm in zip(updates, data_sizes, norms):
        scale = min(1.0, bound / (norm.item() + 1e-12))
        clipped.append(scale < 1.0)
        w = size / total
        weights.append(w)
        for key in aggregated:
            aggregated[key] += w * scale * u[key].float()

    info = AggregationInfo(
        selected=[True] * len(updates),  # norm-clip limits but never rejects
        weights=weights,
        extra={"clip_norm": bound, "clipped": clipped},
    )
    return aggregated, info


def flame(
    updates: List[Update],
    data_sizes: List[int],
    num_byzantine: int = 0,
    noise_lambda: float = 0.001,
) -> Tuple[Update, AggregationInfo]:
    """
    FLAME-style defense (Nguyen et al., USENIX Security 2022).

    Three ingredients: (1) cosine-distance outlier filtering to admit the
    benign majority, (2) clip admitted updates to the median admitted norm,
    (3) add Gaussian noise scaled by that median norm.

    Faithfulness note: the paper uses HDBSCAN for step (1). To stay
    dependency-free we admit the ``n - f`` updates with the smallest total
    cosine distance to the others (the dense majority core); this preserves
    FLAME's cosine-clustering + clip + noise structure. Set ``noise_lambda=0``
    for deterministic tests.
    """
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    if n == 1:
        return updates[0], AggregationInfo(selected=[True], weights=[1.0])

    flat = torch.stack([_flatten(u) for u in updates])
    unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cos_dist = 1.0 - unit @ unit.t()          # pairwise cosine distance
    centrality = cos_dist.sum(dim=1)          # smaller = more central

    f = min(num_byzantine, n - 1)
    k_admit = max(1, n - f)
    admitted = torch.argsort(centrality)[:k_admit].tolist()
    admitted_set = set(admitted)

    # Clip admitted updates to their median norm, then average.
    admitted_norms = torch.stack([flat[i].norm() for i in admitted])
    S = admitted_norms.median()
    agg_flat = torch.zeros_like(flat[0])
    for i in admitted:
        norm = flat[i].norm().clamp_min(1e-12)
        scale = torch.clamp(S / norm, max=1.0)
        agg_flat = agg_flat + scale * flat[i]
    agg_flat = agg_flat / len(admitted)

    if noise_lambda and noise_lambda > 0:
        agg_flat = agg_flat + noise_lambda * S * torch.randn_like(agg_flat)

    aggregated = _unflatten_like(updates[0], agg_flat)
    selected = [i in admitted_set for i in range(n)]
    weights = [1.0 / k_admit if i in admitted_set else 0.0 for i in range(n)]
    info = AggregationInfo(
        selected=selected,
        weights=weights,
        extra={"admitted": admitted, "median_norm": S.item()},
    )
    return aggregated, info


def fltrust(
    updates: List[Update],
    data_sizes: List[int],
    server_update: Optional[Update] = None,
) -> Tuple[Update, AggregationInfo]:
    """
    FLTrust: weight each client update by ReLU(cosine(u_i, u_server)).

    Each accepted update is normalised to the server-update magnitude before the
    trust-weighted average.  When no trusted server update is available (no root
    dataset yet, Phase 0), we fall back to the coordinate-wise median of the
    updates as a trusted reference direction.
    """
    if not updates:
        raise ValueError("No client updates provided for aggregation")

    n = len(updates)
    fallback_reference = server_update is None
    if fallback_reference:
        server_update, _ = median(updates, data_sizes)

    server_flat = _flatten(server_update)
    server_norm = torch.norm(server_flat)
    if server_norm < 1e-8:
        agg, info = fedavg(updates, data_sizes)
        info.extra["fltrust_fallback"] = "zero_server_update"
        return agg, info

    trust_scores = []
    client_norms = []
    for u in updates:
        flat = _flatten(u)
        norm = torch.norm(flat)
        client_norms.append(norm)
        if norm < 1e-8:
            trust_scores.append(0.0)
        else:
            cos = torch.dot(flat, server_flat) / (norm * server_norm)
            trust_scores.append(max(0.0, cos.item()))

    total_trust = sum(trust_scores)
    if total_trust < 1e-8:
        agg, info = median(updates, data_sizes)
        info.extra["fltrust_fallback"] = "all_untrusted"
        return agg, info

    normalized_trust = [t / total_trust for t in trust_scores]

    aggregated = _zeros_like_update(updates[0])
    for i, u in enumerate(updates):
        if normalized_trust[i] <= 0 or client_norms[i] < 1e-8:
            continue
        scale = (server_norm / client_norms[i]).item()
        for key in aggregated:
            aggregated[key] += normalized_trust[i] * scale * u[key].float()

    selected = [t > 0 for t in trust_scores]
    info = AggregationInfo(
        selected=selected,
        weights=normalized_trust,
        extra={
            "trust_scores": trust_scores,
            "fallback_reference": fallback_reference,
        },
    )
    return aggregated, info


def aggregate(
    updates: List[Update],
    data_sizes: List[int],
    aggregation_method: str = "mean",
    **kwargs,
) -> Tuple[Update, AggregationInfo]:
    """
    Aggregate client updates with the given method.

    Returns:
        (aggregated_update, info) where ``aggregated_update`` is Delta such that
        the new global model is  w_global + Delta, and ``info`` carries the
        per-client selection/weight bookkeeping used for Evasion-Rate metrics.
    """
    default_byz = max(1, len(updates) // 5)

    if aggregation_method == "mean":
        return fedavg(updates, data_sizes)
    if aggregation_method == "median":
        return median(updates, data_sizes)
    if aggregation_method == "krum":
        return krum(updates, data_sizes, kwargs.get("num_byzantine", default_byz))
    if aggregation_method == "multi_krum":
        return multi_krum(updates, data_sizes, kwargs.get("num_byzantine", default_byz))
    if aggregation_method == "bulyan":
        return bulyan(updates, data_sizes, kwargs.get("num_byzantine", default_byz))
    if aggregation_method == "fltrust":
        return fltrust(updates, data_sizes, kwargs.get("server_update"))
    if aggregation_method == "trimmed_mean":
        return trimmed_mean(updates, data_sizes, kwargs.get("num_byzantine", default_byz))
    if aggregation_method == "norm_clip":
        return norm_clip(updates, data_sizes, kwargs.get("clip_norm"))
    if aggregation_method == "flame":
        return flame(updates, data_sizes, kwargs.get("num_byzantine", default_byz),
                     noise_lambda=kwargs.get("noise_lambda", 0.001))

    raise ValueError(
        f"Unknown aggregation method: {aggregation_method}. Available: "
        f"['mean', 'median', 'krum', 'multi_krum', 'bulyan', 'fltrust', "
        f"'trimmed_mean', 'norm_clip', 'flame']"
    )

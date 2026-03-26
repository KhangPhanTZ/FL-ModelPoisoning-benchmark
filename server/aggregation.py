import torch
import torch.nn as nn
from typing import List, Dict


def fedavg(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """
    FedAvg aggregation: weighted average of client models based on data size.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has

    Returns:
        Aggregated model state dict
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    total_samples = sum(client_data_sizes)

    aggregated_weights = {}
    for key in client_weights[0].keys():
        weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)

        for client_weight, data_size in zip(client_weights, client_data_sizes):
            weight_factor = data_size / total_samples
            weighted_sum += client_weight[key].float() * weight_factor

        aggregated_weights[key] = weighted_sum

    return aggregated_weights


def median(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Coordinate-wise median aggregation (robust to outliers).

    For each parameter, compute the median across all clients.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has (unused but kept for API consistency)

    Returns:
        Aggregated model state dict
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    aggregated_weights = {}
    for key in client_weights[0].keys():
        # Stack all client weights for this parameter
        stacked = torch.stack([w[key].float() for w in client_weights])
        # Compute coordinate-wise median
        aggregated_weights[key] = stacked.median(dim=0)[0]

    return aggregated_weights


def krum(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
    num_byzantine: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Krum aggregation: select the update closest to others.

    Krum selects the client whose update has minimum sum of squared distances
    to its (n - f - 2) nearest neighbors, where f is the number of Byzantine clients.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has (unused)
        num_byzantine: Expected number of Byzantine/malicious clients

    Returns:
        Selected model state dict (single best client)
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    n = len(client_weights)
    if n == 1:
        return client_weights[0]

    # Number of neighbors to consider: n - f - 2 (at least 1)
    f = min(num_byzantine, n - 2)
    k = max(1, n - f - 2)

    # Flatten each client's weights into a single vector for distance computation
    flat_weights = []
    for weights in client_weights:
        flat = torch.cat([w.float().flatten() for w in weights.values()])
        flat_weights.append(flat)

    # Compute pairwise distances (memory efficient: compute row by row)
    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = torch.sum((flat_weights[i] - flat_weights[j]) ** 2).item()
                distances.append(dist)

        # Sort distances and sum the k smallest
        distances.sort()
        score = sum(distances[:k])
        scores.append(score)

    # Select client with minimum score
    selected_idx = scores.index(min(scores))

    return client_weights[selected_idx]


def multi_krum(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
    num_byzantine: int = 0,
    num_select: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Multi-Krum: select top-m clients and average their updates.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has
        num_byzantine: Expected number of Byzantine/malicious clients
        num_select: Number of clients to select (0 = auto: n - f)

    Returns:
        Averaged model from selected clients
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    n = len(client_weights)
    if n == 1:
        return client_weights[0]

    f = min(num_byzantine, n - 2)
    k = max(1, n - f - 2)
    m = num_select if num_select > 0 else max(1, n - f)

    # Flatten weights
    flat_weights = []
    for weights in client_weights:
        flat = torch.cat([w.float().flatten() for w in weights.values()])
        flat_weights.append(flat)

    # Compute scores
    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = torch.sum((flat_weights[i] - flat_weights[j]) ** 2).item()
                distances.append(dist)
        distances.sort()
        score = sum(distances[:k])
        scores.append((score, i))

    # Select top-m clients with lowest scores
    scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:m]]

    # Average selected clients
    aggregated_weights = {}
    for key in client_weights[0].keys():
        stacked = torch.stack([client_weights[i][key].float() for i in selected_indices])
        aggregated_weights[key] = stacked.mean(dim=0)

    return aggregated_weights


def bulyan(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
    num_byzantine: int = 0
) -> Dict[str, torch.Tensor]:
    """
    Bulyan aggregation: Multi-Krum selection + coordinate-wise trimmed mean.

    1. Use Multi-Krum to select (n - 2f) most trustworthy clients
    2. Apply coordinate-wise trimmed mean on selected updates

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has
        num_byzantine: Expected number of Byzantine/malicious clients

    Returns:
        Aggregated model state dict
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    n = len(client_weights)
    if n == 1:
        return client_weights[0]

    f = min(num_byzantine, (n - 3) // 2)  # Bulyan requires n >= 4f + 3
    k = max(1, n - f - 2)
    m = max(1, n - 2 * f)  # Select n - 2f clients

    # Step 1: Multi-Krum selection
    flat_weights = []
    for weights in client_weights:
        flat = torch.cat([w.float().flatten() for w in weights.values()])
        flat_weights.append(flat)

    scores = []
    for i in range(n):
        distances = []
        for j in range(n):
            if i != j:
                dist = torch.sum((flat_weights[i] - flat_weights[j]) ** 2).item()
                distances.append(dist)
        distances.sort()
        score = sum(distances[:k])
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])
    selected_indices = [idx for _, idx in scores[:m]]

    # Step 2: Coordinate-wise trimmed mean on selected clients
    # Trim beta values from each end (beta = f for Bulyan)
    beta = max(1, f) if len(selected_indices) > 2 else 0

    aggregated_weights = {}
    for key in client_weights[0].keys():
        stacked = torch.stack([client_weights[i][key].float() for i in selected_indices])

        if beta > 0 and stacked.size(0) > 2 * beta:
            # Sort along client dimension and trim
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[beta:-beta]
            aggregated_weights[key] = trimmed.mean(dim=0)
        else:
            aggregated_weights[key] = stacked.mean(dim=0)

    return aggregated_weights


def clip_updates(
    client_weights: List[Dict[str, torch.Tensor]],
    global_weights: Dict[str, torch.Tensor],
    clip_norm: float = 10.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Clip update norms to limit influence of any single client.

    Args:
        client_weights: List of client model weights
        global_weights: Current global model weights
        clip_norm: Maximum L2 norm for updates

    Returns:
        Clipped client weights
    """
    clipped = []
    for weights in client_weights:
        # Compute update (difference from global)
        update_flat = torch.cat([
            (weights[key].float() - global_weights[key].float()).flatten()
            for key in weights.keys()
        ])

        norm = torch.norm(update_flat)
        scale = min(1.0, clip_norm / (norm.item() + 1e-8))

        if scale < 1.0:
            clipped_weights = {}
            for key in weights.keys():
                update = weights[key].float() - global_weights[key].float()
                clipped_weights[key] = global_weights[key].float() + scale * update
            clipped.append(clipped_weights)
        else:
            clipped.append(weights)

    return clipped


def fltrust(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
    server_update: Dict[str, torch.Tensor] = None,
    global_weights: Dict[str, torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    FLTrust aggregation: use server's trusted update to compute trust scores.

    Trust score = max(0, cosine_similarity(client_update, server_update))
    Final = normalized weighted average using trust scores.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has
        server_update: Server's trusted update (computed on clean data)
        global_weights: Current global model weights

    Returns:
        Aggregated model state dict
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")

    n = len(client_weights)

    # If no server update provided, fall back to computing trust from median
    if server_update is None or global_weights is None:
        # Use median as trusted reference
        server_update = median(client_weights, client_data_sizes)
        # Compute pseudo global_weights as mean of all
        global_weights = {}
        for key in client_weights[0].keys():
            stacked = torch.stack([w[key].float() for w in client_weights])
            global_weights[key] = stacked.mean(dim=0)

    # Flatten server update
    server_flat = torch.cat([
        (server_update[key].float() - global_weights[key].float()).flatten()
        for key in server_update.keys()
    ])
    server_norm = torch.norm(server_flat)

    if server_norm < 1e-8:
        # Fallback to simple average if server update is zero
        return fedavg(client_weights, client_data_sizes)

    # Compute trust scores for each client
    trust_scores = []
    client_updates_flat = []

    for weights in client_weights:
        client_flat = torch.cat([
            (weights[key].float() - global_weights[key].float()).flatten()
            for key in weights.keys()
        ])
        client_updates_flat.append(client_flat)

        # Cosine similarity
        client_norm = torch.norm(client_flat)
        if client_norm < 1e-8:
            trust_scores.append(0.0)
        else:
            cos_sim = torch.dot(client_flat, server_flat) / (client_norm * server_norm)
            # ReLU: only positive correlations contribute
            trust = max(0.0, cos_sim.item())
            trust_scores.append(trust)

    # Normalize trust scores
    total_trust = sum(trust_scores)
    if total_trust < 1e-8:
        # All clients untrusted, use median
        return median(client_weights, client_data_sizes)

    normalized_trust = [t / total_trust for t in trust_scores]

    # Aggregate with trust-weighted average, normalizing each client update
    aggregated_weights = {}
    for key in client_weights[0].keys():
        weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)

        for i, weights in enumerate(client_weights):
            if normalized_trust[i] > 0:
                # Normalize client update to server update magnitude
                client_update = weights[key].float() - global_weights[key].float()
                client_norm = torch.norm(client_updates_flat[i])

                if client_norm > 1e-8:
                    # Scale client update to have same norm as server
                    scale = server_norm / client_norm
                    normalized_update = scale * client_update
                    weighted_sum += normalized_trust[i] * normalized_update

        # Add to global weights
        aggregated_weights[key] = global_weights[key].float() + weighted_sum

    return aggregated_weights


def aggregate(
    client_weights: List[Dict[str, torch.Tensor]],
    client_data_sizes: List[int],
    aggregation_method: str = "mean",
    **kwargs
) -> Dict[str, torch.Tensor]:
    """
    Aggregate client model weights using specified method.

    Args:
        client_weights: List of state dicts from each client
        client_data_sizes: Number of samples each client has
        aggregation_method: Aggregation strategy
        **kwargs: Additional parameters (global_weights, server_update for fltrust)

    Returns:
        Aggregated model state dict
    """
    if aggregation_method == "mean":
        return fedavg(client_weights, client_data_sizes)

    if aggregation_method == "median":
        return median(client_weights, client_data_sizes)

    if aggregation_method == "krum":
        num_byzantine = kwargs.get("num_byzantine", max(1, len(client_weights) // 5))
        return krum(client_weights, client_data_sizes, num_byzantine)

    if aggregation_method == "multi_krum":
        num_byzantine = kwargs.get("num_byzantine", max(1, len(client_weights) // 5))
        return multi_krum(client_weights, client_data_sizes, num_byzantine)

    if aggregation_method == "bulyan":
        num_byzantine = kwargs.get("num_byzantine", max(1, len(client_weights) // 5))
        return bulyan(client_weights, client_data_sizes, num_byzantine)

    if aggregation_method == "fltrust":
        global_weights = kwargs.get("global_weights", None)
        server_update = kwargs.get("server_update", None)
        return fltrust(client_weights, client_data_sizes, server_update, global_weights)

    raise ValueError(
        f"Unknown aggregation method: {aggregation_method}. "
        f"Available: ['mean', 'median', 'krum', 'multi_krum', 'bulyan', 'fltrust']"
    )

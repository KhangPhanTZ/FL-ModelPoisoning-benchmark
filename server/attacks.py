"""
Model-poisoning attacks operating in UPDATE SPACE.

Phase-0 refactor: every attack manipulates the client *update*
    u_i = w_local_i - w_global
instead of the raw weights w_local_i.  This matches the threat model used by
Byzantine-robust aggregators (Krum, Median, FLTrust, ...), which all measure
norms / cosine similarities on the update (gradient) rather than on the
absolute weight values.  Working in update space keeps attacks and defenses on
the same footing and is the foundation the later GeoTox attack is built on.
"""

import torch
from typing import List, Dict


def _benign_indices(num_clients: int, malicious_indices: List[int]) -> List[int]:
    """Return the indices of benign clients."""
    malicious = set(malicious_indices)
    return [i for i in range(num_clients) if i not in malicious]


def compute_lie_attack(
    client_updates: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    z: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    Little is Enough (LIE) attack (Baruch et al., 2019), in update space.

    Malicious clients send: mu_benign - z * sigma_benign, where the mean and
    std are computed coordinate-wise over the BENIGN client *updates* only
    (the attacker estimates the benign update distribution).  The negative
    direction pushes the aggregated model away from the correct optimum.

    Args:
        client_updates: List of update dicts (w_local - w_global) per client.
        malicious_indices: Indices of malicious clients.
        z: Attack strength multiplier (higher = stronger attack).

    Returns:
        The (mutated) list of client updates with the attack applied.
    """
    if not malicious_indices or len(client_updates) <= 1:
        return client_updates

    benign_indices = _benign_indices(len(client_updates), malicious_indices)
    if not benign_indices:
        return client_updates

    keys = client_updates[0].keys()
    for key in keys:
        stacked = torch.stack([client_updates[i][key].float() for i in benign_indices])
        mean = stacked.mean(dim=0)
        std = torch.clamp(stacked.std(dim=0), min=1e-8)
        malicious_value = mean - z * std
        for mal_idx in malicious_indices:
            client_updates[mal_idx][key] = malicious_value.clone()

    return client_updates


def compute_minmax_attack(
    client_updates: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    gamma: float = 1.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    Min-Max style attack (Shejwalkar & Houmansadr, 2021), in update space.

    The malicious update deviates from the benign mean update in the opposite
    direction, scaled by gamma times the maximum coordinate-wise deviation
    observed among benign updates.  This is an approximation: the full paper
    binary-searches gamma to find the largest perturbation that still satisfies
    a distance constraint; here gamma is a direct hyper-parameter.

    Args:
        client_updates: List of update dicts per client.
        malicious_indices: Indices of malicious clients.
        gamma: Attack strength multiplier.

    Returns:
        The (mutated) list of client updates with the attack applied.
    """
    if not malicious_indices or len(client_updates) <= 1:
        return client_updates

    benign_indices = _benign_indices(len(client_updates), malicious_indices)
    if not benign_indices:
        return client_updates

    keys = client_updates[0].keys()
    for key in keys:
        stacked = torch.stack([client_updates[i][key].float() for i in benign_indices])
        benign_mean = stacked.mean(dim=0)
        # Maximum coordinate-wise deviation among benign updates bounds the attack.
        max_deviation = (stacked - benign_mean).abs().max(dim=0)[0]

        direction = -torch.sign(benign_mean)
        direction[direction == 0] = 1.0  # break ties on zero coordinates
        malicious_value = benign_mean + direction * gamma * max_deviation

        for mal_idx in malicious_indices:
            client_updates[mal_idx][key] = malicious_value.clone()

    return client_updates


def compute_model_replacement_attack(
    client_updates: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    client_data_sizes: List[int] = None,
    scale_factor: float = 1.0,
    clip_value: float = 100.0,
) -> List[Dict[str, torch.Tensor]]:
    """
    Model Replacement attack (Bagdasaryan et al., 2020), in update space.

    For weighted FedAvg the aggregated update is  Delta = sum_i (n_i/N) * u_i.
    To make the global model adopt the malicious local model, each malicious
    client scales its own update so that, after aggregation, the benign mass is
    overwritten:
        u_send = (N / n_malicious) * scale_factor * u_local

    Args:
        client_updates: List of update dicts per client.
        malicious_indices: Indices of malicious clients.
        client_data_sizes: Data sizes per client (for weighted scaling).
        scale_factor: Additional scaling multiplier.
        clip_value: Symmetric clamp on the scaled update to avoid NaN/Inf.

    Returns:
        The (mutated) list of client updates with the attack applied.
    """
    if not malicious_indices:
        return client_updates

    num_clients = len(client_updates)
    if client_data_sizes is not None:
        total_data = sum(client_data_sizes)
        malicious_data = sum(client_data_sizes[i] for i in malicious_indices)
        denominator = malicious_data if malicious_data > 0 else len(malicious_indices)
        numerator = total_data if malicious_data > 0 else num_clients
        scale = (numerator / denominator) * scale_factor
    else:
        scale = (num_clients / len(malicious_indices)) * scale_factor

    keys = client_updates[0].keys()
    for mal_idx in malicious_indices:
        for key in keys:
            scaled = scale * client_updates[mal_idx][key].float()
            client_updates[mal_idx][key] = torch.clamp(scaled, -clip_value, clip_value)

    return client_updates


def apply_attack(
    client_updates: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    attack_type: str,
    **kwargs,
) -> List[Dict[str, torch.Tensor]]:
    """
    Apply the specified attack to client updates (in update space).

    Args:
        client_updates: List of update dicts (w_local - w_global) per client.
        malicious_indices: Indices of malicious clients.
        attack_type: One of 'none', 'lie', 'minmax', 'model_replacement'.
        **kwargs: Attack-specific parameters (z, client_data_sizes).

    Returns:
        The (mutated) list of client updates.
    """
    if attack_type == "none" or not malicious_indices:
        return client_updates

    if attack_type == "lie":
        return compute_lie_attack(client_updates, malicious_indices, kwargs.get("z", 1.0))

    if attack_type == "minmax":
        gamma = kwargs.get("gamma", kwargs.get("z", 1.0))
        return compute_minmax_attack(client_updates, malicious_indices, gamma)

    if attack_type == "model_replacement":
        return compute_model_replacement_attack(
            client_updates,
            malicious_indices,
            client_data_sizes=kwargs.get("client_data_sizes"),
            scale_factor=kwargs.get("z", 1.0),
        )

    raise ValueError(
        f"Unknown attack type: {attack_type}. "
        f"Available: ['none', 'lie', 'minmax', 'model_replacement']"
    )

import torch
from typing import List, Dict


def compute_lie_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    z: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Little is Enough (LIE) attack (Baruch et al., 2019).

    Malicious clients send: μ_benign - z * σ_benign
    Statistics are computed from BENIGN clients only (attacker observes
    the benign gradient distribution). The negative direction pushes the
    aggregated model away from the correct optimum.

    Args:
        client_weights: List of model weights from all clients
        malicious_indices: Indices of malicious clients in the list
        z: Attack strength multiplier (higher = stronger attack)

    Returns:
        Modified client weights with attack applied
    """
    if not malicious_indices or len(client_weights) <= 1:
        return client_weights

    # Get benign client indices
    benign_indices = [i for i in range(len(client_weights)) if i not in malicious_indices]

    if not benign_indices:
        return client_weights

    # Compute mean and std from BENIGN clients only (Baruch et al., 2019)
    # The attacker observes/estimates the benign update distribution
    keys = client_weights[0].keys()

    mean_weights = {}
    std_weights = {}

    for key in keys:
        # Stack only benign client weights for this parameter
        stacked = torch.stack([client_weights[i][key].float() for i in benign_indices])

        mean_weights[key] = stacked.mean(dim=0)
        std_weights[key] = stacked.std(dim=0)

        # Prevent division issues with zero std
        std_weights[key] = torch.clamp(std_weights[key], min=1e-8)

    # Apply attack: malicious clients send μ - z * σ (negative direction)
    for mal_idx in malicious_indices:
        for key in keys:
            client_weights[mal_idx][key] = mean_weights[key] - z * std_weights[key]

    return client_weights


def compute_minmax_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    gamma: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Min-Max style attack: maximize perturbation bounded by benign deviation.

    Approximation of the Min-Max attack (Shejwalkar & Houmansadr, 2021):
    the malicious update deviates from the benign mean in the opposite
    direction, scaled by gamma times the maximum benign deviation.
    The constraint ensures the malicious update stays within gamma times
    the maximum pairwise distance among benign clients.

    Note: the full paper uses a binary search over gamma to find the
    largest perturbation satisfying the distance constraint. This
    implementation uses gamma as a direct hyperparameter.

    Args:
        client_weights: List of model weights from all clients
        malicious_indices: Indices of malicious clients in the list
        gamma: Attack strength multiplier (higher = stronger perturbation)

    Returns:
        Modified client weights with attack applied
    """
    if not malicious_indices or len(client_weights) <= 1:
        return client_weights

    benign_indices = [i for i in range(len(client_weights)) if i not in malicious_indices]

    if not benign_indices:
        return client_weights

    keys = client_weights[0].keys()

    # Compute mean of benign client weights
    benign_mean = {}
    for key in keys:
        stacked = torch.stack([client_weights[i][key].float() for i in benign_indices])
        benign_mean[key] = stacked.mean(dim=0)

    # Compute the deviation of each benign client from the mean
    # Find the maximum deviation to bound the attack
    max_deviation = {}
    for key in keys:
        deviations = []
        for i in benign_indices:
            dev = (client_weights[i][key].float() - benign_mean[key]).abs()
            deviations.append(dev)
        if deviations:
            stacked_dev = torch.stack(deviations)
            max_deviation[key] = stacked_dev.max(dim=0)[0]
        else:
            max_deviation[key] = torch.zeros_like(benign_mean[key])

    # Malicious update: move in opposite direction of benign mean
    # Scaled by gamma * max_deviation to maximize distance
    for mal_idx in malicious_indices:
        for key in keys:
            # Direction: opposite of where benign clients are moving
            # Perturbation bounded by max deviation seen in benign clients
            perturbation = gamma * max_deviation[key]

            # Alternate direction based on sign of mean to maximize divergence
            direction = -torch.sign(benign_mean[key])
            direction[direction == 0] = 1.0  # Handle zeros

            # Apply attack: mean - gamma * max_dev * sign(mean)
            client_weights[mal_idx][key] = benign_mean[key] + direction * perturbation

    return client_weights


def compute_model_replacement_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    global_weights: Dict[str, torch.Tensor],
    client_data_sizes: List[int] = None,
    scale_factor: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Model Replacement attack (Bagdasaryan et al., 2020).

    Scale malicious update so that after weighted FedAvg aggregation,
    the global model is replaced by the malicious local model.

    For weighted FedAvg: w_new = sum_i (n_i/N) * w_i
    To achieve w_new ≈ w_mal_local, each malicious client sends:
        w_scaled = global + (N / sum_mal_data) * (local - global)

    Args:
        client_weights: List of model weights from all clients
        malicious_indices: Indices of malicious clients in the list
        global_weights: Current global model weights
        client_data_sizes: Data sizes for each client (for weighted scaling)
        scale_factor: Additional scaling multiplier (default: 1.0)

    Returns:
        Modified client weights with attack applied
    """
    if not malicious_indices or not global_weights:
        return client_weights

    num_malicious = len(malicious_indices)
    if num_malicious == 0:
        return client_weights

    # Compute scale for weighted FedAvg: total_data / malicious_data
    if client_data_sizes is not None:
        total_data = sum(client_data_sizes)
        malicious_data = sum(client_data_sizes[i] for i in malicious_indices)
        if malicious_data > 0:
            scale = (total_data / malicious_data) * scale_factor
        else:
            scale = (len(client_weights) / num_malicious) * scale_factor
    else:
        # Fallback to uniform assumption
        scale = (len(client_weights) / num_malicious) * scale_factor

    keys = client_weights[0].keys()

    for mal_idx in malicious_indices:
        for key in keys:
            local_w = client_weights[mal_idx][key].float()
            global_w = global_weights[key].float()

            # Scaled update: global + scale * (local - global)
            # After weighted averaging, this should result in approximately the local model
            scaled_update = global_w + scale * (local_w - global_w)

            # Clip to prevent extreme values that cause NaN
            client_weights[mal_idx][key] = torch.clamp(scaled_update, min=-100.0, max=100.0)

    return client_weights


def apply_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    attack_type: str,
    **kwargs
) -> List[Dict[str, torch.Tensor]]:
    """
    Apply specified attack to client weights.

    Args:
        client_weights: List of model weights from all clients
        malicious_indices: Indices of malicious clients
        attack_type: Type of attack ('none', 'lie', 'minmax', 'model_replacement')
        **kwargs: Additional attack parameters

    Returns:
        Modified client weights
    """
    if attack_type == "none" or not malicious_indices:
        return client_weights

    if attack_type == "lie":
        z = kwargs.get("z", 1.0)
        return compute_lie_attack(client_weights, malicious_indices, z)

    if attack_type == "minmax":
        gamma = kwargs.get("gamma", kwargs.get("z", 1.0))
        return compute_minmax_attack(client_weights, malicious_indices, gamma)

    if attack_type == "model_replacement":
        global_weights = kwargs.get("global_weights", {})
        client_data_sizes = kwargs.get("client_data_sizes", None)
        scale_factor = kwargs.get("z", 1.0)
        return compute_model_replacement_attack(
            client_weights, malicious_indices, global_weights, client_data_sizes, scale_factor
        )

    raise ValueError(f"Unknown attack type: {attack_type}. Available: ['none', 'lie', 'minmax', 'model_replacement']")

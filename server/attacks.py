import torch
from typing import List, Dict


def compute_lie_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    z: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Little is Enough (LIE) attack.

    Malicious clients send: mean + z * std
    This pushes the aggregated model in a harmful direction.

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

    # Compute mean and std from ALL clients (as per original LIE paper)
    # This simulates attackers knowing the distribution
    keys = client_weights[0].keys()

    mean_weights = {}
    std_weights = {}

    for key in keys:
        # Stack all weights for this parameter
        stacked = torch.stack([client_weights[i][key].float() for i in range(len(client_weights))])

        mean_weights[key] = stacked.mean(dim=0)
        std_weights[key] = stacked.std(dim=0)

        # Prevent division issues with zero std
        std_weights[key] = torch.clamp(std_weights[key], min=1e-8)

    # Apply attack: malicious clients send mean + z * std
    for mal_idx in malicious_indices:
        for key in keys:
            client_weights[mal_idx][key] = mean_weights[key] + z * std_weights[key]

    return client_weights


def compute_minmax_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    gamma: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Min-Max attack: maximize distance from benign updates.

    Malicious clients compute the mean of benign updates and send
    perturbation in the opposite direction to maximize damage.

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

            # Clip to prevent extreme values
            client_weights[mal_idx][key] = torch.clamp(
                client_weights[mal_idx][key],
                min=-10.0,
                max=10.0
            )

    return client_weights


def compute_model_replacement_attack(
    client_weights: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    global_weights: Dict[str, torch.Tensor],
    num_clients: int,
    scale_factor: float = 1.0
) -> List[Dict[str, torch.Tensor]]:
    """
    Model Replacement attack: scale malicious update to replace global model.

    After FedAvg aggregation, the global model should be replaced by the
    malicious local model. Uses scaling: w = global + scale * (local - global)

    Args:
        client_weights: List of model weights from all clients
        malicious_indices: Indices of malicious clients in the list
        global_weights: Current global model weights
        num_clients: Total number of clients in this round
        scale_factor: Additional scaling multiplier (default: 1.0)

    Returns:
        Modified client weights with attack applied
    """
    if not malicious_indices or not global_weights:
        return client_weights

    num_malicious = len(malicious_indices)
    if num_malicious == 0:
        return client_weights

    # Scale factor: ensures malicious model dominates after averaging
    # scale = (num_clients / num_malicious) * scale_factor
    scale = (num_clients / num_malicious) * scale_factor

    keys = client_weights[0].keys()

    for mal_idx in malicious_indices:
        for key in keys:
            local_w = client_weights[mal_idx][key].float()
            global_w = global_weights[key].float()

            # Scaled update: global + scale * (local - global)
            # After averaging, this should result in approximately the local model
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
        num_clients = kwargs.get("num_clients", len(client_weights))
        scale_factor = kwargs.get("z", 1.0)
        return compute_model_replacement_attack(
            client_weights, malicious_indices, global_weights, num_clients, scale_factor
        )

    raise ValueError(f"Unknown attack type: {attack_type}. Available: ['none', 'lie', 'minmax', 'model_replacement']")

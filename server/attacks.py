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


def _flatten_update(update: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([v.float().flatten() for v in update.values()])


def _assign_flat(update: Dict[str, torch.Tensor], flat: torch.Tensor) -> None:
    """Write a flat vector back into an update dict in-place (by key order)."""
    offset = 0
    for key in update:
        numel = update[key].numel()
        update[key] = flat[offset:offset + numel].reshape(update[key].shape).clone()
        offset += numel


def _blend_lambda(c: float, tau: float, iters: int = 40) -> float:
    """
    Smallest lambda in [0, 1] such that the unit blend
        v = lambda * mu_hat + (1 - lambda) * d_hat
    satisfies cos(v, mu_hat) >= tau, given c = <d_hat, mu_hat>.

    cos(v, mu_hat) = (lambda + (1-lambda)c) / ||v|| is monotonically increasing
    in lambda (from c at lambda=0 to 1 at lambda=1), so we binary-search.
    """
    if c >= tau:
        return 0.0

    def cos_of(lam: float) -> float:
        num = lam + (1.0 - lam) * c
        den = (lam * lam + (1.0 - lam) ** 2 + 2.0 * lam * (1.0 - lam) * c) ** 0.5
        return num / den if den > 1e-12 else 1.0

    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cos_of(mid) >= tau:
            hi = mid
        else:
            lo = mid
    return hi


def compute_geotox_attack(
    client_updates: List[Dict[str, torch.Tensor]],
    malicious_indices: List[int],
    tau: float = 0.5,
    mask_ratio: float = 0.7,
    eps: float = 1e-8,
) -> List[Dict[str, torch.Tensor]]:
    """
    GeoTox: a multi-constraint stealthy + durable model-poisoning attack.

    The malicious clients are assumed to have trained on backdoored data, so
    each malicious update carries a backdoor signal.  GeoTox then shapes that
    raw update to evade several defense families simultaneously:

      1. Durability (Neurotoxin-style): keep the backdoor in the coordinates
         that benign clients move the LEAST (low |mean benign update|), so
         honest updates are unlikely to overwrite it.  ``mask_ratio`` is the
         fraction of (lowest-importance) coordinates retained.
      2. Directional stealth: blend the masked direction toward the benign mean
         direction until cos(update, benign_mean) >= ``tau`` -- this is the
         knob that trades stealth (high tau) against backdoor strength.
      3. Magnitude stealth: rescale to the MEDIAN benign update norm, so the
         update sits inside the benign point-cloud (evading norm clipping).

    NOTE (honest framing): tau induces a trade-off -- larger tau aligns the
    update with benign clients (higher evasion) but dilutes the backdoor
    (lower ASR).  Sweeping tau is how we characterise that frontier; GeoTox is
    not claimed to defeat every defense at every tau.

    Args:
        client_updates: Per-client update dicts (mutated in place).
        malicious_indices: Indices of malicious clients.
        tau: Target cosine alignment with the benign mean direction (0..1).
        mask_ratio: Fraction of lowest-importance coordinates kept for the
            backdoor (0..1); the rest are zeroed for durability.

    Returns:
        The mutated list of client updates.
    """
    if not malicious_indices or len(client_updates) <= 1:
        return client_updates

    benign_indices = _benign_indices(len(client_updates), malicious_indices)
    if not benign_indices:
        return client_updates

    benign_flat = torch.stack([_flatten_update(client_updates[i]) for i in benign_indices])
    mu_b = benign_flat.mean(dim=0)
    mu_norm = torch.norm(mu_b)
    if mu_norm < eps:
        return client_updates  # no benign direction to align to / hide behind
    mu_hat = mu_b / mu_norm

    # Magnitude budget: median benign update norm.
    B = benign_flat.norm(dim=1).median()

    # Durability mask: zero the highest-importance (most-moved) benign coords,
    # keeping the backdoor in the lowest-importance ``mask_ratio`` fraction.
    importance = mu_b.abs()
    dim = importance.numel()
    keep = max(1, int(mask_ratio * dim))
    num_zero = dim - keep
    mask = torch.ones(dim)
    if num_zero > 0:
        high_idx = torch.topk(importance, num_zero, largest=True).indices
        mask[high_idx] = 0.0

    for mal_idx in malicious_indices:
        raw = _flatten_update(client_updates[mal_idx])
        dur = raw * mask
        if torch.norm(dur) < eps:
            dur = raw  # mask wiped everything; fall back to the raw update
        dur_norm = torch.norm(dur)
        if dur_norm < eps:
            continue
        d_hat = dur / dur_norm

        c = torch.dot(d_hat, mu_hat).item()
        lam = _blend_lambda(c, tau)
        blended = lam * mu_hat + (1.0 - lam) * d_hat
        blended = blended / torch.clamp(torch.norm(blended), min=eps)

        final = B * blended
        _assign_flat(client_updates[mal_idx], final)

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

    if attack_type == "geotox":
        return compute_geotox_attack(
            client_updates,
            malicious_indices,
            tau=kwargs.get("tau", 0.5),
            mask_ratio=kwargs.get("mask_ratio", 0.7),
        )

    raise ValueError(
        f"Unknown attack type: {attack_type}. "
        f"Available: ['none', 'lie', 'minmax', 'model_replacement', 'geotox']"
    )

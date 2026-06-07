"""Attack scheduling helpers (durability experiments)."""


def attack_active(round_num: int, attack_until: int) -> bool:
    """
    Whether the attack is active on a given round.

    Durability study: the attacker participates only up to ``attack_until``,
    then leaves (stops poisoning / shaping), and we watch the backdoor decay.

    Args:
        round_num: 1-based current round.
        attack_until: Last round the attack is active. ``<= 0`` means the
            attack is always active (no early stop).

    Returns:
        True if the attack should be applied this round.
    """
    if attack_until <= 0:
        return True
    return round_num <= attack_until

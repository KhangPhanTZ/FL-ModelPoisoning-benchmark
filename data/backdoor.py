"""
Backdoor attack utilities for Federated Learning.

Implements trigger pattern injection for backdoor attacks.
"""

import torch
from typing import Tuple


def add_trigger(
    images: torch.Tensor,
    pattern_size: int = 4,
    trigger_value: float = 1.0
) -> torch.Tensor:
    """
    Add a white square trigger to the bottom-right corner of images.

    Args:
        images: Tensor of shape (N, C, H, W) or (C, H, W)
        pattern_size: Size of the square trigger pattern
        trigger_value: Pixel value for the trigger (1.0 = white)

    Returns:
        Images with trigger pattern added
    """
    triggered = images.clone()

    if triggered.dim() == 3:
        # Single image (C, H, W)
        triggered[:, -pattern_size:, -pattern_size:] = trigger_value
    else:
        # Batch of images (N, C, H, W)
        triggered[:, :, -pattern_size:, -pattern_size:] = trigger_value

    return triggered


def create_poisoned_batch(
    data: torch.Tensor,
    target: torch.Tensor,
    poison_ratio: float = 0.5,
    target_class: int = 7,
    pattern_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a poisoned batch by adding triggers and changing labels.

    Args:
        data: Input images (N, C, H, W)
        target: Original labels (N,)
        poison_ratio: Fraction of batch to poison (0.0 to 1.0)
        target_class: Target label for poisoned samples
        pattern_size: Size of trigger pattern

    Returns:
        Tuple of (poisoned_data, poisoned_targets)
    """
    batch_size = data.size(0)
    num_poison = int(batch_size * poison_ratio)

    if num_poison == 0:
        return data, target

    # Select random samples to poison
    poison_indices = torch.randperm(batch_size)[:num_poison]

    poisoned_data = data.clone()
    poisoned_target = target.clone()

    # Add trigger to selected samples
    poisoned_data[poison_indices] = add_trigger(
        data[poison_indices],
        pattern_size=pattern_size
    )

    # Change labels to target class
    poisoned_target[poison_indices] = target_class

    return poisoned_data, poisoned_target


def create_backdoor_test_set(
    test_loader,
    target_class: int = 7,
    pattern_size: int = 4,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create a triggered test set for ASR evaluation.

    All samples get the trigger pattern added, and we measure
    how many are classified as the target class.

    Args:
        test_loader: DataLoader for test set
        target_class: Target label for backdoor attack
        pattern_size: Size of trigger pattern
        device: Device to use

    Returns:
        Tuple of (all_triggered_data, all_original_labels)
    """
    all_data = []
    all_labels = []

    for data, target in test_loader:
        triggered_data = add_trigger(data, pattern_size=pattern_size)
        all_data.append(triggered_data)
        all_labels.append(target)

    return torch.cat(all_data, dim=0).to(device), torch.cat(all_labels, dim=0).to(device)

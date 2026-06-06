"""
Dataset loading for the FL benchmark.

Supports MNIST and Fashion-MNIST (both 1x28x28, 10 classes, so the same LeNet
works for either).  Also provides a small clean "root dataset" loader used by
FLTrust, where the server holds a tiny trusted set to compute a reference
update each round.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple


# Per-dataset normalisation statistics and metadata.
DATASET_INFO = {
    "mnist": {
        "cls": datasets.MNIST,
        "mean": (0.1307,),
        "std": (0.3081,),
        "num_classes": 10,
    },
    "fashion_mnist": {
        "cls": datasets.FashionMNIST,
        "mean": (0.2860,),
        "std": (0.3530,),
        "num_classes": 10,
    },
}


def available_datasets():
    return list(DATASET_INFO.keys())


def get_transforms(dataset: str) -> transforms.Compose:
    """Standard tensor + per-dataset normalisation transform."""
    if dataset not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {available_datasets()}"
        )
    info = DATASET_INFO[dataset]
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(info["mean"], info["std"]),
    ])


def load_dataset(dataset: str = "mnist", data_dir: str = "./data") -> Tuple:
    """
    Load train and test datasets by name.

    Args:
        dataset: 'mnist' or 'fashion_mnist'.
        data_dir: Download / cache directory.

    Returns:
        (train_dataset, test_dataset)
    """
    if dataset not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset: {dataset}. Available: {available_datasets()}"
        )

    cls = DATASET_INFO[dataset]["cls"]
    transform = get_transforms(dataset)

    train_dataset = cls(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = cls(root=data_dir, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def build_root_loader(
    train_dataset,
    root_size: int = 100,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """
    Build a small clean root-dataset loader for FLTrust.

    Samples ``root_size`` examples (class-balanced where possible) that the
    server treats as a trusted set to compute its reference update each round.

    Args:
        train_dataset: The training dataset to draw the root set from.
        root_size: Number of clean samples in the root set.
        batch_size: Batch size for the root loader.
        seed: RNG seed for reproducible root sampling.

    Returns:
        A DataLoader over the sampled root subset.
    """
    targets = train_dataset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()

    num_classes = len(set(targets))
    per_class = max(1, root_size // num_classes)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(targets), generator=generator).tolist()

    selected = []
    class_counts = {c: 0 for c in range(num_classes)}
    for idx in perm:
        label = int(targets[idx])
        if class_counts.get(label, 0) < per_class:
            selected.append(idx)
            class_counts[label] = class_counts.get(label, 0) + 1
        if len(selected) >= root_size:
            break

    root_subset = Subset(train_dataset, selected)
    return DataLoader(root_subset, batch_size=batch_size, shuffle=True)

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import List, Tuple


def get_mnist_transforms():
    """Get standard MNIST transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def load_mnist(data_dir: str = "./data") -> Tuple[datasets.MNIST, datasets.MNIST]:
    """Load MNIST train and test datasets."""
    transform = get_mnist_transforms()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


def partition_data_iid(dataset: datasets.MNIST, num_clients: int) -> List[Subset]:
    """Partition dataset into IID subsets for each client."""
    num_samples = len(dataset)
    indices = np.random.permutation(num_samples)

    samples_per_client = num_samples // num_clients
    client_indices = []

    for i in range(num_clients):
        start_idx = i * samples_per_client
        if i == num_clients - 1:
            end_idx = num_samples
        else:
            end_idx = start_idx + samples_per_client
        client_indices.append(indices[start_idx:end_idx].tolist())

    client_datasets = [Subset(dataset, idx) for idx in client_indices]
    return client_datasets


def partition_data_noniid(
    dataset: datasets.MNIST,
    num_clients: int,
    alpha: float = 0.5
) -> List[Subset]:
    """
    Partition dataset into non-IID subsets using Dirichlet distribution.

    Args:
        dataset: MNIST dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter (lower = more heterogeneous)

    Returns:
        List of Subset objects, one per client
    """
    labels = np.array(dataset.targets)
    num_classes = 10

    # Group indices by label
    label_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples according to Dirichlet
    for class_idx in range(num_classes):
        class_indices = label_indices[class_idx]
        np.random.shuffle(class_indices)

        # Sample proportions from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Convert proportions to sample counts
        proportions = proportions / proportions.sum()
        counts = (proportions * len(class_indices)).astype(int)

        # Distribute remaining samples to clients with highest proportion
        remainder = len(class_indices) - counts.sum()
        if remainder > 0:
            top_clients = np.argsort(proportions)[-remainder:]
            counts[top_clients] += 1

        # Assign indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(class_indices[start:end].tolist())
            start = end

    # Ensure no client has empty dataset (minimum 10 samples)
    min_samples = 10
    for i in range(num_clients):
        if len(client_indices[i]) < min_samples:
            # Borrow samples from clients with most data
            donors = sorted(range(num_clients), key=lambda x: len(client_indices[x]), reverse=True)
            needed = min_samples - len(client_indices[i])
            for donor in donors:
                if donor == i:
                    continue
                can_give = max(0, len(client_indices[donor]) - min_samples)
                give = min(can_give, needed)
                if give > 0:
                    client_indices[i].extend(client_indices[donor][-give:])
                    client_indices[donor] = client_indices[donor][:-give]
                    needed -= give
                if needed <= 0:
                    break

    # Shuffle each client's data
    for idx_list in client_indices:
        np.random.shuffle(idx_list)

    client_datasets = [Subset(dataset, indices) for indices in client_indices]
    return client_datasets


def partition_data(
    dataset: datasets.MNIST,
    num_clients: int,
    partition: str,
    alpha: float = 0.5
) -> List[Subset]:
    """Partition dataset according to specified strategy."""
    if partition == "iid":
        return partition_data_iid(dataset, num_clients)
    elif partition == "noniid":
        return partition_data_noniid(dataset, num_clients, alpha)
    else:
        raise ValueError(f"Unknown partition strategy: {partition}. Available: ['iid', 'noniid']")


def get_test_loader(test_dataset: datasets.MNIST, batch_size: int = 128) -> DataLoader:
    """Get test data loader."""
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def print_partition_stats(
    client_datasets: List[Subset],
    dataset: datasets.MNIST,
    num_clients_to_show: int = 5
):
    """Print label distribution statistics for each client."""
    labels = np.array(dataset.targets)

    print(f"\nLabel distribution (showing first {num_clients_to_show} clients):")
    print("-" * 60)
    print(f"{'Client':<8} {'Samples':<8} {'Labels (count per class 0-9)'}")
    print("-" * 60)

    for i, subset in enumerate(client_datasets[:num_clients_to_show]):
        client_labels = labels[subset.indices]
        counts = np.bincount(client_labels, minlength=10)
        counts_str = " ".join([f"{c:4d}" for c in counts])
        print(f"{i:<8} {len(subset):<8} [{counts_str}]")

    print("-" * 60)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple
import random
import copy

from client.client import FederatedClient
from server.aggregation import aggregate
from server.attacks import apply_attack


class FederatedServer:
    """Federated Learning server that coordinates training."""

    def __init__(
        self,
        global_model: nn.Module,
        clients: List[FederatedClient],
        device: torch.device,
        aggregation_method: str = "mean",
        attack_type: str = "none",
        attack_z: float = 1.0
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregation_method = aggregation_method
        self.attack_type = attack_type
        self.attack_z = attack_z

    def select_clients(self, num_clients: int) -> List[FederatedClient]:
        """Randomly select clients for a training round."""
        return random.sample(self.clients, min(num_clients, len(self.clients)))

    def train_round(
        self,
        selected_clients: List[FederatedClient],
        local_epochs: int = 1
    ) -> None:
        """Execute one round of federated training."""
        client_weights = []
        client_data_sizes = []
        malicious_indices = []

        # Store global weights for model_replacement attack
        global_weights = {
            name: param.cpu().clone()
            for name, param in self.global_model.state_dict().items()
        }

        for idx, client in enumerate(selected_clients):
            weights = client.train(self.global_model, local_epochs)
            client_weights.append(weights)
            client_data_sizes.append(len(client))

            if client.is_malicious:
                malicious_indices.append(idx)

        # Apply attack if any malicious clients
        if malicious_indices and self.attack_type != "none":
            client_weights = apply_attack(
                client_weights,
                malicious_indices,
                self.attack_type,
                z=self.attack_z,
                global_weights=global_weights,
                num_clients=len(selected_clients)
            )

        aggregated_weights = aggregate(
            client_weights,
            client_data_sizes,
            self.aggregation_method,
            global_weights=global_weights
        )

        self.global_model.load_state_dict(aggregated_weights)

    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)

        avg_loss = test_loss / total
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def compute_asr(
        self,
        test_loader: DataLoader,
        source_class: int = 1,
        target_class: int = 7
    ) -> float:
        """
        Compute Attack Success Rate (ASR).

        ASR measures the percentage of samples from source_class that are
        misclassified as target_class. High ASR indicates successful attack.

        Args:
            test_loader: Test data loader
            source_class: Original class label (default: 1)
            target_class: Target class for misclassification (default: 7)

        Returns:
            ASR percentage (0-100)
        """
        self.global_model.eval()
        source_samples = 0
        misclassified_to_target = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                # Only consider samples from source class
                source_mask = target == source_class
                if source_mask.sum() == 0:
                    continue

                source_data = data[source_mask]
                source_samples += source_mask.sum().item()

                # Get predictions
                output = self.global_model(source_data)
                pred = output.argmax(dim=1)

                # Count how many are misclassified to target class
                misclassified_to_target += (pred == target_class).sum().item()

        if source_samples == 0:
            return 0.0

        asr = 100.0 * misclassified_to_target / source_samples
        return asr

    def get_global_model(self) -> nn.Module:
        """Return a copy of the global model."""
        return copy.deepcopy(self.global_model)

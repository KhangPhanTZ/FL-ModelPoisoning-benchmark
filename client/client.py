import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Dict
import copy


class FederatedClient:
    """Federated Learning client that performs local training."""

    def __init__(
        self,
        client_id: int,
        dataset: Subset,
        device: torch.device,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        is_malicious: bool = False,
        attack_type: str = "none"
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.is_malicious = is_malicious
        self.attack_type = attack_type

        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True
        )

    def train(
        self,
        global_model: nn.Module,
        local_epochs: int = 1
    ) -> Dict[str, torch.Tensor]:
        """Train local model and return model updates."""
        local_model = copy.deepcopy(global_model)
        local_model.to(self.device)
        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(),
            lr=self.learning_rate,
            momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.data_loader):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()

                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), max_norm=10.0)

                optimizer.step()

        local_weights = {
            name: param.cpu().clone()
            for name, param in local_model.state_dict().items()
        }

        return local_weights

    def __len__(self) -> int:
        """Return the number of samples in client's dataset."""
        return len(self.dataset)

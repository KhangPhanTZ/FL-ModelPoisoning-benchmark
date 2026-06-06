import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional
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
        attack_z: float = 1.0,
        root_loader: DataLoader = None,
        learning_rate: float = 0.01,
        attack_tau: float = 0.5,
        attack_mask_ratio: float = 0.7,
        attack_adaptive_max_scale: float = 5.0,
    ):
        self.global_model = global_model.to(device)
        self.clients = clients
        self.device = device
        self.aggregation_method = aggregation_method
        self.attack_type = attack_type
        self.attack_z = attack_z
        # GeoTox stealth knobs.
        self.attack_tau = attack_tau
        self.attack_mask_ratio = attack_mask_ratio
        self.attack_adaptive_max_scale = attack_adaptive_max_scale
        # Trusted clean root set used by FLTrust to compute a reference update.
        self.root_loader = root_loader
        self.learning_rate = learning_rate

    def select_clients(self, num_clients: int) -> List[FederatedClient]:
        """Randomly select clients for a training round."""
        return random.sample(self.clients, min(num_clients, len(self.clients)))

    def _compute_server_update(self, global_weights, local_epochs: int):
        """
        Train the global model on the trusted root set and return its update.

        This is the trusted reference direction FLTrust scores clients against.
        Returns None when no root loader is configured.
        """
        if self.root_loader is None:
            return None

        server_model = copy.deepcopy(self.global_model).to(self.device)
        server_model.train()
        optimizer = torch.optim.SGD(
            server_model.parameters(), lr=self.learning_rate, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for data, target in self.root_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(server_model(data), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=10.0)
                optimizer.step()

        server_state = {
            name: param.detach().cpu().float().clone()
            for name, param in server_model.state_dict().items()
        }
        return {key: server_state[key] - global_weights[key] for key in global_weights}

    def train_round(
        self,
        selected_clients: List[FederatedClient],
        local_epochs: int = 1
    ) -> Optional[float]:
        """
        Execute one round of federated training (in update space).

        Returns:
            The Evasion Rate for this round: the fraction of malicious clients
            whose update was accepted (not filtered) by the aggregator, or None
            if no malicious client participated this round.
        """
        client_updates = []
        client_data_sizes = []
        malicious_indices = []

        # Current global weights (CPU float) used to form per-client updates.
        global_weights = {
            name: param.detach().cpu().float().clone()
            for name, param in self.global_model.state_dict().items()
        }

        for idx, client in enumerate(selected_clients):
            local_weights = client.train(self.global_model, local_epochs)
            # Update space: u_i = w_local_i - w_global
            update = {
                key: local_weights[key].float() - global_weights[key]
                for key in global_weights
            }
            client_updates.append(update)
            client_data_sizes.append(len(client))

            if client.is_malicious:
                malicious_indices.append(idx)

        # GeoTox-Adaptive reuses the GeoTox shaping, then (below) tunes its
        # magnitude against the known defense. Map it to the base attack here.
        base_attack = "geotox" if self.attack_type == "geotox_adaptive" else self.attack_type

        # Apply the attack to the malicious clients' updates.
        if malicious_indices and base_attack != "none":
            client_updates = apply_attack(
                client_updates,
                malicious_indices,
                base_attack,
                z=self.attack_z,
                client_data_sizes=client_data_sizes,
                tau=self.attack_tau,
                mask_ratio=self.attack_mask_ratio,
            )

        # FLTrust needs a trusted server update computed on the clean root set.
        server_update = None
        if self.aggregation_method == "fltrust":
            server_update = self._compute_server_update(global_weights, local_epochs)

        # GeoTox-Adaptive (white-box, omniscient upper bound): scale the shaped
        # malicious update to the defense's acceptance boundary.
        if self.attack_type == "geotox_adaptive" and malicious_indices:
            client_updates = self._apply_adaptive_scaling(
                client_updates, malicious_indices, client_data_sizes, server_update
            )

        # Pass the actual malicious count so robust aggregators (Krum, Bulyan)
        # use the correct f parameter instead of a heuristic.
        aggregated_update, info = aggregate(
            client_updates,
            client_data_sizes,
            self.aggregation_method,
            num_byzantine=len(malicious_indices),
            server_update=server_update,
        )

        # Reconstruct the new global model: w_global + Delta.
        new_state = {
            key: global_weights[key] + aggregated_update[key]
            for key in global_weights
        }
        self.global_model.load_state_dict(new_state)

        return self._compute_evasion_rate(info, malicious_indices)

    def _apply_adaptive_scaling(
        self,
        client_updates,
        malicious_indices: List[int],
        data_sizes: List[int],
        server_update,
        iters: int = 15,
    ):
        """
        GeoTox-Adaptive: scale the shaped malicious update up to the largest
        factor the (known) defense still accepts.

        White-box, omniscient upper bound: the attacker simulates the exact
        aggregator on all updates and binary-searches the scale in
        [1, max_scale] that keeps every malicious client accepted, then applies
        it -- operating right at the defense's acceptance boundary for maximum
        backdoor strength.
        """
        max_scale = self.attack_adaptive_max_scale
        base = [
            {k: client_updates[i][k].clone() for k in client_updates[i]}
            for i in malicious_indices
        ]

        def set_scale(s: float):
            for shaped, i in zip(base, malicious_indices):
                client_updates[i] = {k: v * s for k, v in shaped.items()}

        def accepted(s: float) -> bool:
            set_scale(s)
            _, info = aggregate(
                client_updates, data_sizes, self.aggregation_method,
                num_byzantine=len(malicious_indices), server_update=server_update,
            )
            return all(info.selected[i] for i in malicious_indices)

        if accepted(max_scale):
            set_scale(max_scale)              # defense tolerates the cap
        elif not accepted(1.0):
            set_scale(1.0)                    # even the base update is filtered
        else:
            lo, hi = 1.0, max_scale           # boundary is in (1, max_scale)
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if accepted(mid):
                    lo = mid
                else:
                    hi = mid
            set_scale(lo)
        return client_updates

    @staticmethod
    def _compute_evasion_rate(info, malicious_indices: List[int]) -> Optional[float]:
        """
        Fraction of malicious clients whose update was accepted by the defense.

        For coordinate-wise defenses (Median) there is no per-client rejection,
        so every participating client counts as accepted; this is reported as-is
        and should be interpreted alongside ``info.selection_type`` downstream.
        """
        if not malicious_indices:
            return None
        accepted = sum(1 for i in malicious_indices if info.selected[i])
        return 100.0 * accepted / len(malicious_indices)

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
        target_class: int = 7,
        source_class: Optional[int] = None,
    ) -> float:
        """
        Compute Attack Success Rate (ASR) using triggered samples.

        ASR = (# eligible triggered samples classified as target) /
              (# eligible triggered samples)

        Eligible samples are those whose true label is NOT the target class
        (so we never inflate ASR with samples already in the target class). If
        ``source_class`` is given, only samples of that class are considered,
        matching a single-source backdoor; otherwise all non-target samples are
        used (all-to-target backdoor).

        Args:
            test_loader: Test data loader.
            target_class: Target label the backdoor maps triggered inputs to.
            source_class: If set, restrict ASR to this source class only.

        Returns:
            ASR percentage (0-100).
        """
        from data.backdoor import add_trigger

        self.global_model.eval()
        total_samples = 0
        successful_attacks = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                # Eligible samples: not already in the target class, and (if a
                # source class is specified) belonging to that source class.
                mask = target != target_class
                if source_class is not None:
                    mask = mask & (target == source_class)
                if mask.sum() == 0:
                    continue

                filtered_data = data[mask]

                # Apply trigger to filtered test samples
                triggered_data = add_trigger(filtered_data)

                # Get predictions on triggered samples
                output = self.global_model(triggered_data)
                pred = output.argmax(dim=1)

                # Count how many triggered samples are classified as target class
                successful_attacks += (pred == target_class).sum().item()
                total_samples += filtered_data.size(0)

        if total_samples == 0:
            return 0.0

        asr = 100.0 * successful_attacks / total_samples
        return asr

    def get_global_model(self) -> nn.Module:
        """Return a copy of the global model."""
        return copy.deepcopy(self.global_model)

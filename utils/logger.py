"""Streaming CSV logger for Federated Learning experiments."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class SkipExperiment(Exception):
    """Raised when experiment should be skipped (e.g., result file exists)."""
    pass


class FLLogger:
    """Logger for FL experiments with streaming CSV output."""

    def __init__(
        self,
        aggregation: str,
        attack: str,
        partition: str,
        malicious: int,
        results_dir: str = "results",
        skip_existing: bool = False,
        dataset: str = "mnist",
        alpha: float = None,
        seed: int = None,
        tau: float = None,
        attack_until: int = 0,
    ):
        """
        Initialize the logger.

        Args:
            aggregation: Aggregation method name
            attack: Attack type name
            partition: Data partition strategy
            malicious: Number of malicious clients
            results_dir: Directory to save results
            skip_existing: If True, raise SkipExperiment if file exists
            dataset: Dataset name (encoded in the filename)
            alpha: Dirichlet alpha (encoded only for non-IID partitions)
            seed: Random seed (encoded when provided, to separate repeated runs)
        """
        self.aggregation = aggregation
        self.attack = attack
        self.partition = partition
        self.malicious = malicious
        self.results_dir = Path(results_dir)
        self.skip_existing = skip_existing
        self.dataset = dataset
        self.alpha = alpha
        self.seed = seed
        self.tau = tau
        self.attack_until = attack_until

        # Create results directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        self.filename = self._generate_filename()
        self.filepath = self.results_dir / self.filename

        # Check if file exists and skip_existing is True
        if self.skip_existing and self.filepath.exists():
            raise SkipExperiment(f"Result file already exists: {self.filepath}")

        # For backward compatibility: resolve collisions if not skipping
        if not self.skip_existing:
            self.filepath = self._resolve_unique_path(self.filename)

        # Initialize file with header
        self._write_header()

    def _generate_filename(self) -> str:
        """
        Generate a descriptive filename.

        Scheme: {dataset}_{aggregation}_{attack}_{partition}[_a{alpha}]_m{malicious}[_s{seed}].csv
        Alpha is included only for non-IID partitions; seed only when provided.
        """
        parts = [self.dataset, self.aggregation, self.attack, self.partition]
        base = "_".join(parts)
        if self.partition == "noniid" and self.alpha is not None:
            base += f"_a{self.alpha}"
        base += f"_m{self.malicious}"
        # Durability runs: encode the round the attacker leaves.
        if self.attack_until and self.attack_until > 0:
            base += f"_u{self.attack_until}"
        # GeoTox sweeps tau; encode it so the trade-off runs do not collide.
        if self.attack in ("geotox", "geotox_adaptive") and self.tau is not None:
            base += f"_t{self.tau}"
        if self.seed is not None:
            base += f"_s{self.seed}"
        return f"{base}.csv"

    def _resolve_unique_path(self, filename: str) -> Path:
        """
        Resolve a unique result path to avoid overwriting existing logs.

        If `filename` exists, use incremented suffixes:
        name_1.csv, name_2.csv, ...
        """
        candidate = self.results_dir / filename
        if not candidate.exists():
            return candidate

        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while True:
            candidate = self.results_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def _write_header(self):
        """Write CSV header."""
        with self.filepath.open("x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round",
                "loss",
                "accuracy",
                "asr",
                "evasion_rate",
                "timestamp"
            ])

    def log_round(
        self,
        round_num: int,
        loss: float,
        accuracy: float,
        asr: Optional[float] = None,
        evasion_rate: Optional[float] = None
    ):
        """
        Log a single round's metrics (streaming write).

        Args:
            round_num: Current round number
            loss: Test loss
            accuracy: Test accuracy (%)
            asr: Attack success rate (%), optional
            evasion_rate: Fraction (%) of malicious updates accepted by the
                defense this round, optional
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self.filepath.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                f"{loss:.6f}",
                f"{accuracy:.4f}",
                f"{asr:.4f}" if asr is not None else "",
                f"{evasion_rate:.4f}" if evasion_rate is not None else "",
                timestamp
            ])

    def log_config(self, config: dict):
        """
        Log experiment configuration to a separate file.

        Args:
            config: Dictionary of configuration parameters
        """
        config_path = self.filepath.with_name(f"{self.filepath.stem}_config.txt")
        with config_path.open("w") as f:
            f.write("=" * 50 + "\n")
            f.write("Experiment Configuration\n")
            f.write("=" * 50 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("=" * 50 + "\n")

    def get_filepath(self) -> str:
        """Return the path to the results file."""
        return str(self.filepath)


def create_logger(args, skip_existing: bool = False) -> FLLogger:
    """
    Create a logger from argparse arguments.

    Args:
        args: Parsed command line arguments
        skip_existing: If True, raise SkipExperiment if result file exists

    Returns:
        Configured FLLogger instance
    """
    logger = FLLogger(
        aggregation=args.aggregation,
        attack=args.attack,
        partition=args.partition,
        malicious=args.malicious,
        results_dir="results",
        skip_existing=skip_existing,
        dataset=getattr(args, "dataset", "mnist"),
        alpha=getattr(args, "alpha", None),
        seed=getattr(args, "seed", None),
        tau=getattr(args, "tau", None),
        attack_until=getattr(args, "attack_until", 0),
    )

    # Log configuration
    config = {
        "dataset": getattr(args, "dataset", "mnist"),
        "aggregation": args.aggregation,
        "attack": args.attack,
        "partition": args.partition,
        "num_clients": args.num_clients,
        "clients_per_round": args.clients_per_round,
        "malicious": args.malicious,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "model": args.model,
    }

    if args.attack != "none":
        config["attack_z"] = args.z
    if args.partition == "noniid":
        config["alpha"] = args.alpha
    if args.attack in ("geotox", "geotox_adaptive"):
        config["tau"] = getattr(args, "tau", None)
        config["mask_ratio"] = getattr(args, "mask_ratio", None)
    if getattr(args, "attack_until", 0):
        config["attack_until"] = args.attack_until

    logger.log_config(config)

    return logger

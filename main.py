#!/usr/bin/env python3
"""
Federated Learning with MNIST and LeNet.

Usage:
    python main.py --aggregation mean --attack none --partition iid
"""

import argparse
import sys
import random
import numpy as np
import torch

from models.lenet import get_model
from data.datasets import load_dataset, build_root_loader, available_datasets
from data.mnist import partition_data, get_test_loader, print_partition_stats
from client.client import FederatedClient
from server.server import FederatedServer
from utils.logger import create_logger
from utils.schedule import attack_active


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Learning on MNIST with LeNet"
    )

    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "krum", "multi_krum", "bulyan", "fltrust",
                 "trimmed_mean", "norm_clip", "flame"],
        help="Aggregation method (default: mean)"
    )

    parser.add_argument(
        "--attack",
        type=str,
        default="none",
        choices=["none", "lie", "minmax", "model_replacement", "geotox", "geotox_adaptive"],
        help="Attack type (default: none)"
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
        help="GeoTox stealth: target cosine alignment with benign mean (0..1)"
    )

    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=1.0,
        help="GeoTox durability (opt-in): fraction of low-importance coords "
             "kept; 1.0 = masking off (default). <1.0 boosts persistence but "
             "lowers ASR -- sweep deliberately."
    )

    parser.add_argument(
        "--adaptive_max_scale",
        type=float,
        default=5.0,
        help="GeoTox-Adaptive: max scale searched against the defense (default: 5.0)"
    )

    parser.add_argument(
        "--attack_until",
        type=int,
        default=0,
        help="Durability: last round the attack is active (0 = always; >0 lets "
             "the attacker leave so backdoor decay/durability can be measured)"
    )

    parser.add_argument(
        "--z",
        type=float,
        default=1.0,
        help="LIE attack strength multiplier (default: 1.0)"
    )

    parser.add_argument(
        "--partition",
        type=str,
        default="iid",
        choices=["iid", "noniid"],
        help="Data partition strategy (default: iid)"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Dirichlet alpha for non-IID partition (default: 0.5, lower = more heterogeneous)"
    )

    parser.add_argument(
        "--num_clients",
        type=int,
        default=20,
        help="Total number of clients (default: 20)"
    )

    parser.add_argument(
        "--clients_per_round",
        type=int,
        default=10,
        help="Number of clients sampled per round (default: 10)"
    )

    parser.add_argument(
        "--malicious",
        type=int,
        default=0,
        help="Number of malicious clients (default: 0)"
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=50,
        help="Number of training rounds (default: 50)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lenet",
        choices=["lenet"],
        help="Model architecture (default: lenet)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=available_datasets(),
        help="Dataset (default: mnist)"
    )

    parser.add_argument(
        "--root_size",
        type=int,
        default=500,
        help="Size of the clean root dataset used by FLTrust (default: 500)"
    )

    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs per client (default: 1)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    print("=" * 60)
    print("Federated Learning Configuration")
    print("=" * 60)
    print(f"Dataset:          {args.dataset}")
    print(f"Model:            {args.model}")
    print(f"Aggregation:      {args.aggregation}")
    if args.attack in ("geotox", "geotox_adaptive"):
        attack_detail = f" (tau={args.tau}, mask_ratio={args.mask_ratio})"
        if args.attack == "geotox_adaptive":
            attack_detail += f" (adaptive<=x{args.adaptive_max_scale})"
    elif args.attack != "none":
        attack_detail = f" (z={args.z})"
    else:
        attack_detail = ""
    print(f"Attack:           {args.attack}{attack_detail}")
    print(f"Partition:        {args.partition}" + (f" (alpha={args.alpha})" if args.partition == "noniid" else ""))
    print(f"Total Clients:    {args.num_clients}")
    print(f"Clients/Round:    {args.clients_per_round}")
    print(f"Malicious:        {args.malicious}")
    print(f"Rounds:           {args.rounds}")
    print(f"Local Epochs:     {args.local_epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Seed:             {args.seed}")
    print("=" * 60)

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading {args.dataset} dataset...")
    train_dataset, test_dataset = load_dataset(args.dataset, data_dir="./data")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print(f"\nPartitioning data ({args.partition}) among {args.num_clients} clients...")
    client_datasets = partition_data(train_dataset, args.num_clients, args.partition, args.alpha)

    if args.partition == "noniid":
        print_partition_stats(client_datasets, train_dataset)

    # FLTrust needs a trusted clean root set held by the server.
    root_loader = None
    if args.aggregation == "fltrust":
        root_loader = build_root_loader(
            train_dataset,
            root_size=args.root_size,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        print(f"FLTrust root dataset: {args.root_size} clean samples")

    print("Creating clients...")
    clients = []
    for i in range(args.num_clients):
        is_malicious = i < args.malicious
        client = FederatedClient(
            client_id=i,
            dataset=client_datasets[i],
            device=device,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            is_malicious=is_malicious,
            attack_type=args.attack if is_malicious else "none"
        )
        clients.append(client)

    print(f"\nInitializing {args.model} model...")
    global_model = get_model(args.model)

    server = FederatedServer(
        global_model=global_model,
        clients=clients,
        device=device,
        aggregation_method=args.aggregation,
        attack_type=args.attack,
        attack_z=args.z,
        root_loader=root_loader,
        learning_rate=args.lr,
        attack_tau=args.tau,
        attack_mask_ratio=args.mask_ratio,
        attack_adaptive_max_scale=args.adaptive_max_scale,
    )

    test_loader = get_test_loader(test_dataset)

    print("\n" + "=" * 60)
    print("Starting Federated Training")
    print("=" * 60 + "\n")

    logger = create_logger(args)
    print(f"Logging results to: {logger.get_filepath()}")

    for round_num in range(1, args.rounds + 1):
        selected_clients = server.select_clients(args.clients_per_round)

        active = attack_active(round_num, args.attack_until)
        evasion_rate = server.train_round(selected_clients, args.local_epochs, attack_active=active)

        loss, accuracy = server.evaluate(test_loader)
        asr = None

        # ASR is a backdoor metric and is only meaningful for the backdoor
        # attacks (model_replacement, geotox); untargeted attacks (LIE /
        # Min-Max) are reported via the accuracy drop versus the `none`
        # baseline instead.
        if args.attack in ("model_replacement", "geotox", "geotox_adaptive") and args.malicious > 0:
            asr = server.compute_asr(test_loader, target_class=7)

        line = f"Round {round_num:3d} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%"
        if asr is not None:
            line += f" | ASR: {asr:.2f}%"
        if evasion_rate is not None:
            line += f" | Evasion: {evasion_rate:.1f}%"
        print(line)

        logger.log_round(
            round_num=round_num,
            loss=loss,
            accuracy=accuracy,
            asr=asr,
            evasion_rate=evasion_rate
        )

        if np.isnan(loss):
            print("\nERROR: Loss is NaN! Training failed.")
            print("Possible causes:")
            print("  - Learning rate too high")
            print("  - Data normalization issue")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Final Loss: {loss:.4f}")
    print(f"Final Accuracy: {accuracy:.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())

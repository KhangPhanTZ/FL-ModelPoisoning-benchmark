#!/usr/bin/env python3
"""
Robust experiment scheduler for Federated Learning experiments.

Runs all 54 configurations:
  - aggregation: ["mean", "median", "krum"]
  - attack: ["lie", "minmax", "model_replacement"]
  - partition: ["iid", "noniid"]
  - malicious: [2, 4, 6]

Features:
  - Skips completed experiments (checks if result file exists)
  - Retry logic (2 retries per failed config)
  - Memory cleanup between runs
  - Progress tracking with summary CSV
"""

import argparse
import csv
import gc
import itertools
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict


# Experiment configurations
AGGREGATIONS = ["mean", "median", "krum"]
ATTACKS = ["lie", "minmax", "model_replacement"]
PARTITIONS = ["iid", "noniid"]
MALICIOUS_COUNTS = [2, 4, 6]

# Attack strength parameters (z values)
ATTACK_Z = {
    "lie": 3.0,
    "minmax": 15.0,
    "model_replacement": 1.0,
}

# Non-IID alpha parameter
NONIID_ALPHA = 0.5


def generate_all_configs() -> List[Dict]:
    """Generate all 54 experiment configurations."""
    configs = []
    for agg, attack, partition, mal in itertools.product(
        AGGREGATIONS, ATTACKS, PARTITIONS, MALICIOUS_COUNTS
    ):
        config = {
            "aggregation": agg,
            "attack": attack,
            "partition": partition,
            "malicious": mal,
            "z": ATTACK_Z[attack],
        }
        if partition == "noniid":
            config["alpha"] = NONIID_ALPHA
        configs.append(config)
    return configs


def get_result_filename(config: Dict) -> str:
    """Generate expected result filename for a config."""
    return f"{config['aggregation']}_{config['attack']}_{config['partition']}_m{config['malicious']}.csv"


def get_completed_configs(results_dir: Path) -> set:
    """Get set of completed config filenames from results directory."""
    completed = set()
    if results_dir.exists():
        for f in results_dir.glob("*.csv"):
            # Only consider files without _N suffix as completed
            name = f.stem
            if not any(name.endswith(f"_{i}") for i in range(1, 100)):
                completed.add(f.name)
    return completed


def run_experiment(config: Dict, results_dir: Path) -> Tuple[bool, str]:
    """
    Run a single experiment configuration.

    Returns:
        Tuple of (success: bool, message: str)
    """
    cmd = [
        sys.executable, "main.py",
        "--aggregation", config["aggregation"],
        "--attack", config["attack"],
        "--partition", config["partition"],
        "--malicious", str(config["malicious"]),
        "--z", str(config["z"]),
        "--rounds", "50",
        "--num_clients", "20",
        "--clients_per_round", "10",
    ]

    if config.get("alpha"):
        cmd.extend(["--alpha", str(config["alpha"])])

    config_str = f"{config['aggregation']}/{config['attack']}/{config['partition']}/m{config['malicious']}"

    try:
        print(f"\n{'='*60}")
        print(f"Running: {config_str}")
        print(f"{'='*60}")

        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent,
            capture_output=False,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode == 0:
            return True, "Success"
        else:
            return False, f"Exit code: {result.returncode}"

    except subprocess.TimeoutExpired:
        return False, "Timeout (>10 min)"
    except Exception as e:
        return False, f"Error: {str(e)}"


def update_summary(summary_path: Path, config: Dict, success: bool, message: str):
    """Append experiment result to summary CSV."""
    file_exists = summary_path.exists()

    with summary_path.open("a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp", "aggregation", "attack", "partition",
                "malicious", "z", "success", "message"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            config["aggregation"],
            config["attack"],
            config["partition"],
            config["malicious"],
            config["z"],
            "YES" if success else "NO",
            message
        ])


def main():
    parser = argparse.ArgumentParser(description="Run all FL experiments")
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Number of retries for failed experiments (default: 2)"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Results directory (default: results)"
    )
    parser.add_argument(
        "--sleep", type=float, default=2.0,
        help="Sleep time between experiments in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only show what would be run, don't execute"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = results_dir / "experiment_summary.csv"

    # Generate all configurations
    all_configs = generate_all_configs()
    print(f"Total configurations: {len(all_configs)}")

    # Check completed experiments
    completed = get_completed_configs(results_dir)
    print(f"Already completed: {len(completed)}")

    # Filter to pending configs
    pending_configs = []
    for config in all_configs:
        filename = get_result_filename(config)
        if filename not in completed:
            pending_configs.append(config)

    print(f"Pending experiments: {len(pending_configs)}")

    if args.dry_run:
        print("\n[DRY RUN] Would run the following experiments:")
        for i, config in enumerate(pending_configs, 1):
            cfg_str = f"{config['aggregation']}/{config['attack']}/{config['partition']}/m{config['malicious']}"
            print(f"  {i:2d}. {cfg_str}")
        return 0

    if not pending_configs:
        print("\nAll experiments completed!")
        return 0

    # Run experiments with retry logic
    success_count = 0
    fail_count = 0

    for i, config in enumerate(pending_configs, 1):
        config_str = f"{config['aggregation']}/{config['attack']}/{config['partition']}/m{config['malicious']}"
        print(f"\n[{i}/{len(pending_configs)}] Starting: {config_str}")

        success = False
        message = ""

        for attempt in range(args.retries + 1):
            if attempt > 0:
                print(f"  Retry {attempt}/{args.retries}...")

            success, message = run_experiment(config, results_dir)

            if success:
                break

            # Wait before retry
            if attempt < args.retries:
                time.sleep(1)

        # Update summary
        update_summary(summary_path, config, success, message)

        if success:
            success_count += 1
            print(f"  [OK] {config_str} completed")
        else:
            fail_count += 1
            print(f"  [FAIL] {config_str}: {message}")

        # Memory cleanup
        gc.collect()

        # Sleep between experiments
        if i < len(pending_configs):
            time.sleep(args.sleep)

    # Final summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total pending:    {len(pending_configs)}")
    print(f"Successful:       {success_count}")
    print(f"Failed:           {fail_count}")
    print(f"Summary file:     {summary_path}")
    print("=" * 60)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

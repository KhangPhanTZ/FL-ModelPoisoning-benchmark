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
DATASETS = ["mnist", "fashion_mnist"]
AGGREGATIONS = ["mean", "median", "krum", "fltrust"]
ATTACKS = ["lie", "minmax", "model_replacement"]
MALICIOUS_COUNTS = [2, 4, 6]
SEEDS = [42]  # add more seeds (e.g. [42, 1, 7]) for mean +/- std reporting

# Non-IID heterogeneity sweep (the key variable for RQ2). IID is run as a
# separate single point with no alpha.
NONIID_ALPHAS = [0.1, 0.5]

# Attack strength parameters (z values)
ATTACK_Z = {
    "none": 0.0,
    "lie": 3.0,
    "minmax": 15.0,
    "model_replacement": 1.0,
    "geotox": 0.0,           # GeoTox is controlled by tau, not z
    "geotox_adaptive": 0.0,
}

# GeoTox stealth sweep (the trade-off knob: low tau = strong backdoor / low
# evasion, high tau = stealthy / weak backdoor).
GEOTOX_TAUS = [0.0, 0.5, 0.9]

# Durability study (RQ3): if set (e.g. 25 with --rounds 50), additionally run
# GeoTox / GeoTox-Adaptive where the attacker leaves at this round, so backdoor
# decay can be measured. None disables (keeps the main grid unchanged).
DURABILITY_UNTIL = None
DURABILITY_TAUS = [0.0]  # strongest backdoor is the natural durability probe


def _partition_settings():
    """Yield (partition, alpha) pairs: IID (no alpha) + each non-IID alpha."""
    yield ("iid", None)
    for alpha in NONIID_ALPHAS:
        yield ("noniid", alpha)


def generate_all_configs() -> List[Dict]:
    """
    Generate all experiment configurations.

    Grid = datasets x aggregations x seeds x partition-settings, and for each:
      - a clean ``none`` baseline (malicious=0) to measure attack-induced drop;
      - the attack x malicious grid.

    This can be large; use --dry-run to preview the count and trim the lists
    (DATASETS / AGGREGATIONS / NONIID_ALPHAS / SEEDS) as needed.
    """
    configs = []
    for dataset, agg, seed in itertools.product(DATASETS, AGGREGATIONS, SEEDS):
        for partition, alpha in _partition_settings():
            base = {
                "dataset": dataset,
                "aggregation": agg,
                "partition": partition,
                "seed": seed,
                "alpha": alpha,
            }
            # Clean baseline.
            configs.append({**base, "attack": "none", "malicious": 0,
                            "z": ATTACK_Z["none"], "tau": None})
            # Standard attack grid.
            for attack, mal in itertools.product(ATTACKS, MALICIOUS_COUNTS):
                configs.append({**base, "attack": attack, "malicious": mal,
                                "z": ATTACK_Z[attack], "tau": None})
            # GeoTox + GeoTox-Adaptive: sweep malicious x tau (Evasion<->ASR).
            for atk in ("geotox", "geotox_adaptive"):
                for mal, tau in itertools.product(MALICIOUS_COUNTS, GEOTOX_TAUS):
                    configs.append({**base, "attack": atk, "malicious": mal,
                                    "z": ATTACK_Z[atk], "tau": tau})
            # Optional durability runs (attacker leaves at DURABILITY_UNTIL).
            if DURABILITY_UNTIL:
                for atk in ("geotox", "geotox_adaptive"):
                    for mal, tau in itertools.product(MALICIOUS_COUNTS, DURABILITY_TAUS):
                        configs.append({**base, "attack": atk, "malicious": mal,
                                        "z": ATTACK_Z[atk], "tau": tau,
                                        "attack_until": DURABILITY_UNTIL})
    return configs


def get_result_filename(config: Dict) -> str:
    """Expected result filename, matching utils.logger.FLLogger naming."""
    parts = [config["dataset"], config["aggregation"], config["attack"],
             config["partition"]]
    base = "_".join(parts)
    if config["partition"] == "noniid" and config.get("alpha") is not None:
        base += f"_a{config['alpha']}"
    base += f"_m{config['malicious']}"
    if config.get("attack_until", 0):
        base += f"_u{config['attack_until']}"
    if config["attack"] in ("geotox", "geotox_adaptive") and config.get("tau") is not None:
        base += f"_t{config['tau']}"
    if config.get("seed") is not None:
        base += f"_s{config['seed']}"
    return f"{base}.csv"


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
        "--dataset", config["dataset"],
        "--aggregation", config["aggregation"],
        "--attack", config["attack"],
        "--partition", config["partition"],
        "--malicious", str(config["malicious"]),
        "--z", str(config["z"]),
        "--seed", str(config["seed"]),
        "--rounds", "50",
        "--num_clients", "20",
        "--clients_per_round", "10",
    ]

    if config["partition"] == "noniid" and config.get("alpha") is not None:
        cmd.extend(["--alpha", str(config["alpha"])])
    if config["attack"] in ("geotox", "geotox_adaptive") and config.get("tau") is not None:
        cmd.extend(["--tau", str(config["tau"])])
    if config.get("attack_until", 0):
        cmd.extend(["--attack_until", str(config["attack_until"])])

    config_str = (
        f"{config['dataset']}/{config['aggregation']}/{config['attack']}/"
        f"{config['partition']}"
        + (f"(a={config['alpha']})" if config.get("alpha") is not None else "")
        + f"/m{config['malicious']}/s{config['seed']}"
    )

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
            print(f"  {i:3d}. {get_result_filename(config)}")
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

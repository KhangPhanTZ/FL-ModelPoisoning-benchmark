#!/usr/bin/env python3
"""Experiment scheduler that guarantees completion of all FL configurations."""

import csv
import gc
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

AGGREGATIONS: List[str] = ["mean", "median", "krum"]
ATTACKS: List[str] = ["lie", "minmax", "model_replacement"]
PARTITIONS: List[str] = ["iid", "noniid"]
MALICIOUS_COUNTS: List[int] = [2, 4, 6]

ROUNDS = 20
MAX_RETRIES = 2
DELAY_SECONDS = 2
TIMEOUT_SECONDS = 600
MAX_SCHEDULER_PASSES = 500

SUMMARY_HEADER = ["aggregation", "attack", "partition", "malicious", "status"]
ConfigKey = Tuple[str, str, str, int]


def get_all_configs() -> List[ConfigKey]:
    """Generate all 54 experiment configurations in deterministic order."""
    configs: List[ConfigKey] = []
    for agg in AGGREGATIONS:
        for atk in ATTACKS:
            for part in PARTITIONS:
                for malicious in MALICIOUS_COUNTS:
                    configs.append((agg, atk, part, malicious))
    return configs


def ensure_summary_file(summary_path: Path) -> None:
    """Create summary file and parent directory if missing."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        return
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_HEADER)


def load_latest_status_by_key(summary_path: Path) -> Dict[ConfigKey, str]:
    """
    Load latest status per exact configuration key.

    If a key appears multiple times, later rows overwrite earlier rows.
    """
    statuses: Dict[ConfigKey, str] = {}
    if not summary_path.exists():
        return statuses

    with summary_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key: ConfigKey = (
                    str(row["aggregation"]).strip(),
                    str(row["attack"]).strip(),
                    str(row["partition"]).strip(),
                    int(row["malicious"]),
                )
                status = str(row["status"]).strip().lower()
                if status not in {"success", "fail"}:
                    continue
                statuses[key] = status
            except (KeyError, TypeError, ValueError):
                continue
    return statuses


def load_completed_successes(summary_path: Path) -> Set[ConfigKey]:
    """Load exact keys whose latest status is success."""
    latest_statuses = load_latest_status_by_key(summary_path)
    return {key for key, status in latest_statuses.items() if status == "success"}


def append_summary(summary_path: Path, key: ConfigKey, status: str) -> None:
    """Append a single run result immediately."""
    agg, atk, part, malicious = key
    with summary_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([agg, atk, part, malicious, status])


def run_one_experiment(project_root: Path, key: ConfigKey) -> bool:
    """Execute one experiment with timeout and retries."""
    agg, atk, part, malicious = key
    cmd = [
        sys.executable,
        "main.py",
        "--aggregation",
        agg,
        "--attack",
        atk,
        "--partition",
        part,
        "--malicious",
        str(malicious),
        "--rounds",
        str(ROUNDS),
    ]

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            result = subprocess.run(
                cmd,
                cwd=project_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            print(
                f"  Attempt {attempt}/{MAX_RETRIES + 1} failed "
                f"(timeout after {TIMEOUT_SECONDS}s)."
            )
            gc.collect()
            continue

        if result.returncode == 0:
            del result
            gc.collect()
            return True

        print(
            f"  Attempt {attempt}/{MAX_RETRIES + 1} failed "
            f"(exit={result.returncode})."
        )
        error_tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        if error_tail:
            print("  Error tail:")
            for line in error_tail:
                print(f"    {line}")
        del result
        gc.collect()

    return False


def compute_missing_configs(all_configs: List[ConfigKey], summary_path: Path) -> List[ConfigKey]:
    """Compute missing configs as all_configs - completed_successes."""
    completed_successes = load_completed_successes(summary_path)
    return [cfg for cfg in all_configs if cfg not in completed_successes]


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    summary_path = project_root / "results" / "summary.csv"

    ensure_summary_file(summary_path)
    all_configs = get_all_configs()
    total_required = len(all_configs)

    pass_index = 0
    while True:
        missing_configs = compute_missing_configs(all_configs, summary_path)
        if not missing_configs:
            break

        pass_index += 1
        if pass_index > MAX_SCHEDULER_PASSES:
            print("ERROR: Exceeded scheduler pass limit while missing configs remain.")
            return 1

        print(
            f"Scheduler pass {pass_index}: "
            f"{len(missing_configs)}/{total_required} configs pending."
        )

        for key in missing_configs:
            missing_before = compute_missing_configs(all_configs, summary_path)
            run_num = total_required - len(missing_before) + 1
            agg, atk, part, malicious = key
            print(f"[RUN {run_num}/{total_required}] {agg} + {atk} + {part} + m={malicious}")

            ok = run_one_experiment(project_root, key)
            status = "success" if ok else "fail"
            append_summary(summary_path, key, status)
            print(f"  Status: {status}")

            gc.collect()
            time.sleep(DELAY_SECONDS)

    final_missing = compute_missing_configs(all_configs, summary_path)
    if final_missing:
        print("ERROR: Some configurations are still incomplete:")
        for key in final_missing:
            print(f"  - {key}")
        return 1

    print(f"All {total_required} configurations completed successfully.")
    print(f"Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

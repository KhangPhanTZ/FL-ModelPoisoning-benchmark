#!/usr/bin/env python3
"""
Diagnostic ladder: isolate WHY ASR is ~0 in the quickstart.

It runs a few fast (no-defense `mean`) configs so we can tell whether the
backdoor pipeline works at all, and whether GeoTox's shaping (norm budget /
durability mask) is what suppresses the backdoor.

    1. mean / none            -> clean accuracy (sanity)
    2. mean / model_replacement -> backdoor pipeline sanity (expect HIGH ASR)
    3. mean / geotox tau=0    -> GeoTox with no defense, default mask 0.7
    4. mean / geotox tau=0 mask_ratio=1.0  -> isolate the durability mask
    5. mean / geotox tau=0 mask_ratio=1.0, malicious=8 -> stronger participation

Run:
    python3 run_diagnose.py --dry-run
    python3 run_diagnose.py
    python3 analyze_results.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

DATASET = "mnist"
PARTITION = "noniid"
ALPHA = 0.5
SEED = 42
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 10


def runs(rounds):
    return [
        dict(agg="mean", attack="none", mal=0, tau=None, mask=None),
        dict(agg="mean", attack="model_replacement", mal=4, tau=None, mask=None),
        dict(agg="mean", attack="geotox", mal=4, tau=0.0, mask=0.7),
        dict(agg="mean", attack="geotox", mal=4, tau=0.0, mask=1.0),
        dict(agg="mean", attack="geotox", mal=8, tau=0.0, mask=1.0),
    ]


def build_cmd(cfg, rounds):
    cmd = [sys.executable, "main.py",
           "--dataset", DATASET,
           "--aggregation", cfg["agg"],
           "--attack", cfg["attack"],
           "--partition", PARTITION, "--alpha", str(ALPHA),
           "--malicious", str(cfg["mal"]),
           "--seed", str(SEED),
           "--rounds", str(rounds),
           "--num_clients", str(NUM_CLIENTS),
           "--clients_per_round", str(CLIENTS_PER_ROUND)]
    if cfg["tau"] is not None:
        cmd += ["--tau", str(cfg["tau"])]
    if cfg["mask"] is not None:
        cmd += ["--mask_ratio", str(cfg["mask"])]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfgs = runs(args.rounds)
    print(f"Diagnostic runs: {len(cfgs)} (rounds={args.rounds})")
    for i, cfg in enumerate(cfgs, 1):
        cmd = build_cmd(cfg, args.rounds)
        label = (f"{cfg['agg']}/{cfg['attack']}/m{cfg['mal']}"
                 f"/tau{cfg['tau']}/mask{cfg['mask']}")
        print(f"\n[{i}/{len(cfgs)}] {label}")
        if args.dry_run:
            print("   " + " ".join(cmd))
            continue
        subprocess.run(cmd, cwd=Path(__file__).parent)
    if not args.dry_run:
        print("\nNow run:  python3 analyze_results.py   (look at the ASR column)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

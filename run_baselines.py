#!/usr/bin/env python3
"""
Baseline-attack comparison: run the prior attacks (LIE, Min-Max, Model
Replacement) under the same defenses as GeoTox, so the thesis can present a
head-to-head table (GeoTox vs prior art).

Setting matches run_quickstart (MNIST, non-IID alpha=0.5, 20% malicious, 30
rounds) so the results live in the same results/ folder and analyze_results.py
can compare them directly.

Run:
    python3 run_baselines.py --dry-run
    python3 run_baselines.py --seeds 42,1,7
    python3 analyze_results.py            # see the "Attack comparison" table
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

DATASET = "mnist"
PARTITION = "noniid"
ALPHA = 0.5
MALICIOUS = 4
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 10

DEFENSES = ["mean", "krum", "fltrust", "flame"]
ATTACKS = {"lie": 3.0, "minmax": 15.0, "model_replacement": 1.0}  # attack -> z


def _name(agg, attack, seed):
    return (f"{DATASET}_{agg}_{attack}_{PARTITION}_a{ALPHA}_m{MALICIOUS}"
            f"_s{seed}.csv")


def build_cmd(agg, attack, z, rounds, seed):
    return [sys.executable, "main.py",
            "--dataset", DATASET,
            "--aggregation", agg,
            "--attack", attack,
            "--partition", PARTITION, "--alpha", str(ALPHA),
            "--malicious", str(MALICIOUS),
            "--z", str(z),
            "--seed", str(seed),
            "--rounds", str(rounds),
            "--num_clients", str(NUM_CLIENTS),
            "--clients_per_round", str(CLIENTS_PER_ROUND)]


def main():
    ap = argparse.ArgumentParser(description="Baseline-attack comparison runs")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", default="42,1,7")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    combos = [(agg, atk, z) for agg in DEFENSES for atk, z in ATTACKS.items()]
    total = len(combos) * len(seeds)
    results_dir = Path(args.results_dir)
    print(f"Baseline runs: {len(combos)} x {len(seeds)} seed(s) = {total} "
          f"(rounds={args.rounds})")
    print(f"Rough CPU estimate: ~{total * args.rounds * 6 // 60} min")

    if args.dry_run:
        i = 0
        for seed in seeds:
            for agg, atk, _ in combos:
                i += 1
                print(f"  {i:3d}. {_name(agg, atk, seed)}")
        return 0

    ok = fail = 0
    t0 = time.time()
    i = 0
    for seed in seeds:
        for agg, atk, z in combos:
            i += 1
            fname = _name(agg, atk, seed)
            if args.skip_existing and (results_dir / fname).exists():
                print(f"[{i}/{total}] skip (exists) {fname}")
                continue
            print(f"\n[{i}/{total}] {fname}")
            r = subprocess.run(build_cmd(agg, atk, z, args.rounds, seed),
                               cwd=Path(__file__).parent)
            ok += (r.returncode == 0)
            fail += (r.returncode != 0)

    print(f"\nDone: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    print("Now run:  python3 analyze_results.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

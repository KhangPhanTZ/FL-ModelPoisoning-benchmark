#!/usr/bin/env python3
"""
Quickstart experiment driver: a small, CPU-friendly curated set that is enough
to answer the three research questions, then read with analyze_results.py.

  RQ1  Evasion<->ASR trade-off : GeoTox tau sweep vs mean / Krum / FLTrust / FLAME
  RQ2  non-IID alpha effect    : GeoTox (tau=0.5) vs FLAME across alpha + IID
  RQ3  backdoor durability     : GeoTox / GeoTox-Adaptive, attacker leaves halfway

Run:
    python3 run_quickstart.py --dry-run            # preview + count
    python3 run_quickstart.py --seeds 42,1,7       # 3 seeds (mean +/- std)
    python3 run_quickstart.py --fresh              # archive old results first
    python3 analyze_results.py                     # RQ1/RQ2/RQ3 tables
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DATASET = "mnist"
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 10
MALICIOUS = 4            # 20%

RQ1_DEFENSES = ["mean", "krum", "fltrust", "flame"]
RQ1_TAUS = [0.0, 0.3, 0.6, 0.9]
RQ2_ALPHAS = [("iid", None), ("noniid", 0.1), ("noniid", 0.3),
              ("noniid", 0.5), ("noniid", 1.0)]
RQ2_DEFENSE = "flame"


def _name(agg, attack, partition, alpha, malicious, tau, until, seed):
    base = "_".join([DATASET, agg, attack, partition])
    if partition == "noniid" and alpha is not None:
        base += f"_a{alpha}"
    base += f"_m{malicious}"
    if until:
        base += f"_u{until}"
    if attack in ("geotox", "geotox_adaptive") and tau is not None:
        base += f"_t{tau}"
    if seed is not None:
        base += f"_s{seed}"
    return base + ".csv"


def curated_runs(rounds):
    """Curated config list (without seed), deduplicated by base identity."""
    until = max(1, rounds // 2)
    runs, seen = [], set()

    def add(agg, attack, partition, alpha, tau=None, mal=MALICIOUS, attack_until=0):
        key = (agg, attack, partition, alpha, tau, mal, attack_until)
        if key in seen:
            return
        seen.add(key)
        runs.append(dict(aggregation=agg, attack=attack, partition=partition,
                         alpha=alpha, tau=tau, malicious=mal,
                         attack_until=attack_until))

    # RQ1: trade-off (non-IID alpha=0.5) + clean baseline per defense.
    for agg in RQ1_DEFENSES:
        add(agg, "none", "noniid", 0.5, mal=0)
        for tau in RQ1_TAUS:
            add(agg, "geotox", "noniid", 0.5, tau=tau)
    # RQ2: alpha sensitivity (FLAME, GeoTox tau=0.5).
    for partition, alpha in RQ2_ALPHAS:
        add(RQ2_DEFENSE, "geotox", partition, alpha, tau=0.5)
    # RQ3: durability (FLAME, attacker leaves at `until`).
    for attack in ("geotox", "geotox_adaptive"):
        add("flame", attack, "noniid", 0.5, tau=0.0, attack_until=until)
    return runs


def build_cmd(cfg, rounds, seed):
    cmd = [sys.executable, "main.py",
           "--dataset", DATASET,
           "--aggregation", cfg["aggregation"],
           "--attack", cfg["attack"],
           "--partition", cfg["partition"],
           "--malicious", str(cfg["malicious"]),
           "--seed", str(seed),
           "--rounds", str(rounds),
           "--num_clients", str(NUM_CLIENTS),
           "--clients_per_round", str(CLIENTS_PER_ROUND)]
    if cfg["partition"] == "noniid" and cfg["alpha"] is not None:
        cmd += ["--alpha", str(cfg["alpha"])]
    if cfg["attack"] in ("geotox", "geotox_adaptive") and cfg["tau"] is not None:
        cmd += ["--tau", str(cfg["tau"])]
    if cfg["attack_until"]:
        cmd += ["--attack_until", str(cfg["attack_until"])]
    return cmd


def archive_old(results_dir: Path):
    """Move existing per-run CSV/config files into a timestamped archive."""
    olds = list(results_dir.glob("mnist_*.csv")) + list(results_dir.glob("mnist_*_config.txt"))
    if not olds:
        return
    dest = results_dir.parent / f"results_archive_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    for f in olds:
        shutil.move(str(f), str(dest / f.name))
    print(f"Archived {len(olds)} old result files -> {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Quickstart curated experiments")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seeds", default="42",
                    help="comma-separated seeds, e.g. 42,1,7")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="archive existing results before running")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    runs = curated_runs(args.rounds)
    results_dir = Path(args.results_dir)
    total = len(runs) * len(seeds)
    print(f"Curated runs: {len(runs)} x {len(seeds)} seed(s) = {total}  "
          f"(rounds={args.rounds}, dataset={DATASET}, malicious={MALICIOUS}/{NUM_CLIENTS})")
    print(f"Rough CPU estimate: ~{total * args.rounds * 7 // 60} min "
          f"(geotox_adaptive / FLAME runs are the slow ones).")

    if args.dry_run:
        i = 0
        for seed in seeds:
            for cfg in runs:
                i += 1
                print(f"  {i:3d}. {_name(cfg['aggregation'], cfg['attack'], cfg['partition'], cfg['alpha'], cfg['malicious'], cfg['tau'], cfg['attack_until'], seed)}")
        return 0

    if args.fresh:
        archive_old(results_dir)

    ok = fail = 0
    t0 = time.time()
    i = 0
    for seed in seeds:
        for cfg in runs:
            i += 1
            fname = _name(cfg["aggregation"], cfg["attack"], cfg["partition"],
                          cfg["alpha"], cfg["malicious"], cfg["tau"],
                          cfg["attack_until"], seed)
            if args.skip_existing and (results_dir / fname).exists():
                print(f"[{i}/{total}] skip (exists) {fname}")
                continue
            print(f"\n[{i}/{total}] {fname}")
            r = subprocess.run(build_cmd(cfg, args.rounds, seed),
                               cwd=Path(__file__).parent)
            if r.returncode == 0:
                ok += 1
            else:
                fail += 1
                print(f"  FAILED (exit {r.returncode})")

    print(f"\nDone: {ok} ok, {fail} failed in {time.time()-t0:.0f}s")
    print("Now run:  python3 analyze_results.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

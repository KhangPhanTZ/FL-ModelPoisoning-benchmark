#!/usr/bin/env python3
"""
Quickstart experiment driver: a small, CPU-friendly curated set that is enough
to answer the three research questions, then read with analyze_results.py.

  RQ1  Evasion<->ASR trade-off : GeoTox tau sweep vs Krum / FLTrust / FLAME
                                 (MNIST, non-IID alpha=0.5, 20% malicious)
  RQ2  non-IID alpha effect    : GeoTox (tau=0.5) vs FLAME across alpha + IID
  RQ3  backdoor durability     : GeoTox / GeoTox-Adaptive, attacker leaves
                                 halfway (vs FLAME)

Run:
    python3 run_quickstart.py --dry-run        # preview the runs + count
    python3 run_quickstart.py                  # run them (sequential)
    python3 analyze_results.py                 # RQ1/RQ2/RQ3 tables
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Fixed small FL setting (fast on CPU).
DATASET = "mnist"
NUM_CLIENTS = 20
CLIENTS_PER_ROUND = 10
MALICIOUS = 4            # 20%
SEED = 42

RQ1_DEFENSES = ["mean", "krum", "fltrust", "flame"]
RQ1_TAUS = [0.0, 0.3, 0.6, 0.9]
RQ2_ALPHAS = [(("iid", None)), ("noniid", 0.1), ("noniid", 0.3),
              ("noniid", 0.5), ("noniid", 1.0)]
RQ2_DEFENSE = "flame"


def _name(dataset, agg, attack, partition, alpha, malicious, tau, until, seed):
    base = "_".join([dataset, agg, attack, partition])
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


def curated_runs(rounds: int):
    """Build the curated config list (deduplicated by result filename)."""
    until = max(1, rounds // 2)
    runs, seen = [], set()

    def add(agg, attack, partition, alpha, tau=None, mal=MALICIOUS, attack_until=0):
        cfg = dict(aggregation=agg, attack=attack, partition=partition,
                   alpha=alpha, tau=tau, malicious=mal, attack_until=attack_until)
        fname = _name(DATASET, agg, attack, partition, alpha, mal, tau,
                      attack_until, SEED)
        if fname not in seen:
            seen.add(fname)
            cfg["_file"] = fname
            runs.append(cfg)

    # RQ1: trade-off (non-IID alpha=0.5) + a clean baseline per defense.
    for agg in RQ1_DEFENSES:
        add(agg, "none", "noniid", 0.5, mal=0)
        for tau in RQ1_TAUS:
            add(agg, "geotox", "noniid", 0.5, tau=tau)

    # RQ2: alpha sensitivity (defense = FLAME, GeoTox tau=0.5).
    for partition, alpha in RQ2_ALPHAS:
        add(RQ2_DEFENSE, "geotox", partition, alpha, tau=0.5)

    # RQ3: durability (FLAME, attacker leaves at `until`).
    for attack in ("geotox", "geotox_adaptive"):
        add("flame", attack, "noniid", 0.5, tau=0.0, attack_until=until)

    return runs


def build_cmd(cfg, rounds):
    cmd = [sys.executable, "main.py",
           "--dataset", DATASET,
           "--aggregation", cfg["aggregation"],
           "--attack", cfg["attack"],
           "--partition", cfg["partition"],
           "--malicious", str(cfg["malicious"]),
           "--seed", str(SEED),
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Quickstart curated experiments")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a run if its result CSV already exists")
    args = ap.parse_args()

    runs = curated_runs(args.rounds)
    results_dir = Path(args.results_dir)
    print(f"Curated runs: {len(runs)}  (rounds={args.rounds}, "
          f"dataset={DATASET}, malicious={MALICIOUS}/{NUM_CLIENTS})")

    if args.dry_run:
        for i, cfg in enumerate(runs, 1):
            print(f"  {i:2d}. {cfg['_file']}")
        return 0

    ok = fail = 0
    t0 = time.time()
    for i, cfg in enumerate(runs, 1):
        if args.skip_existing and (results_dir / cfg["_file"]).exists():
            print(f"[{i}/{len(runs)}] skip (exists) {cfg['_file']}")
            continue
        print(f"\n[{i}/{len(runs)}] {cfg['_file']}")
        r = subprocess.run(build_cmd(cfg, args.rounds), cwd=Path(__file__).parent)
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

#!/usr/bin/env python3
"""
Aggregate FL benchmark results into per-run final metrics and the headline
analyses for the thesis:

  - a GeoTox **Evasion <-> ASR trade-off** table (metrics vs tau), and
  - a **non-IID alpha sensitivity** table (ASR/Evasion vs Dirichlet alpha).

It reads each ``results/*.csv`` together with its companion
``*_config.txt`` (written by utils.logger), so it is robust to the
underscore-containing dataset / attack names in the filenames.

Final metrics are averaged over the last ``--last`` rounds for stability.

Usage:
    python3 analyze_results.py                 # writes results/summary_metrics.csv
    python3 analyze_results.py --plot          # also save PNGs (needs matplotlib)
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional


def parse_config(path: Path) -> Dict[str, str]:
    """Parse a ``key: value`` config sidecar into a dict."""
    cfg: Dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line and not line.startswith("="):
            key, _, value = line.partition(":")
            cfg[key.strip()] = value.strip()
    return cfg


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def final_metrics(csv_path: Path, last: int) -> Dict[str, Optional[float]]:
    """Average accuracy / asr / evasion_rate over the last ``last`` rounds."""
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return {"accuracy": None, "asr": None, "evasion": None, "rounds": 0}
    tail = rows[-last:] if last > 0 else rows

    def col(name: str) -> List[float]:
        out = []
        for r in tail:
            v = r.get(name, "")
            if v not in ("", None):
                try:
                    out.append(float(v))
                except ValueError:
                    pass
        return out

    return {
        "accuracy": _mean(col("accuracy")),
        "asr": _mean(col("asr")),
        "evasion": _mean(col("evasion_rate")),
        "rounds": len(rows),
    }


def durability_metrics(csv_path: Path, attack_until: int) -> Dict[str, Optional[float]]:
    """
    Backdoor durability: ASR at the round the attacker leaves vs the final ASR.

    retention = 100 * asr_final / asr_at_stop (how much backdoor survives after
    the attacker stops participating).
    """
    rows = list(csv.DictReader(csv_path.open()))

    def asr_of(row) -> Optional[float]:
        v = row.get("asr", "")
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    asr_at_stop = None
    for r in rows:
        try:
            if int(float(r.get("round", "nan"))) == attack_until:
                asr_at_stop = asr_of(r)
                break
        except (ValueError, TypeError):
            continue

    asr_final = None
    for r in reversed(rows):
        a = asr_of(r)
        if a is not None:
            asr_final = a
            break

    retention = None
    if asr_at_stop is not None and asr_at_stop > 1e-9 and asr_final is not None:
        retention = 100.0 * asr_final / asr_at_stop
    return {"asr_at_stop": asr_at_stop, "asr_final": asr_final, "retention": retention}


def collect(results_dir: Path, last: int) -> List[Dict]:
    """Build one record per (result csv + config) pair."""
    records = []
    for csv_path in sorted(results_dir.glob("*.csv")):
        cfg_path = csv_path.with_name(f"{csv_path.stem}_config.txt")
        if not cfg_path.exists():
            continue  # skip summary / orphan CSVs
        cfg = parse_config(cfg_path)
        m = final_metrics(csv_path, last)
        try:
            attack_until = int(cfg.get("attack_until", "0"))
        except ValueError:
            attack_until = 0
        rec = {
            "dataset": cfg.get("dataset", "mnist"),
            "aggregation": cfg.get("aggregation", ""),
            "attack": cfg.get("attack", ""),
            "partition": cfg.get("partition", ""),
            "alpha": cfg.get("alpha", ""),
            "malicious": cfg.get("malicious", ""),
            "tau": cfg.get("tau", ""),
            "seed": cfg.get("seed", ""),
            "attack_until": attack_until,
            "accuracy": m["accuracy"],
            "asr": m["asr"],
            "evasion": m["evasion"],
            "rounds": m["rounds"],
            "asr_at_stop": None,
            "retention": None,
        }
        if attack_until > 0:
            d = durability_metrics(csv_path, attack_until)
            rec["asr_at_stop"] = d["asr_at_stop"]
            rec["retention"] = d["retention"]
        records.append(rec)
    return records


def write_summary(records: List[Dict], out_path: Path) -> None:
    cols = ["dataset", "aggregation", "attack", "partition", "alpha",
            "malicious", "tau", "seed", "attack_until", "accuracy", "asr",
            "evasion", "asr_at_stop", "retention", "rounds"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)


def _fmt(x) -> str:
    return f"{x:6.2f}" if isinstance(x, (int, float)) else "   -  "


def print_tradeoff(records: List[Dict]) -> None:
    """GeoTox trade-off: ASR / Evasion / Accuracy as tau varies."""
    geo = [r for r in records if r["attack"] in ("geotox", "geotox_adaptive")]
    if not geo:
        print("\n[trade-off] no GeoTox runs found.")
        return
    print("\n=== GeoTox Evasion<->ASR trade-off (mean of last rounds) ===")
    print(f"{'dataset':13} {'agg':9} {'attack':16} {'part':7} {'alpha':5} "
          f"{'m':>2} {'tau':>4} | {'ASR':>6} {'Evas':>6} {'Acc':>6}")
    keyf = lambda r: (r["dataset"], r["aggregation"], r["attack"],
                      r["partition"], r["alpha"], r["malicious"],
                      _taukey(r["tau"]))
    for r in sorted(geo, key=keyf):
        print(f"{r['dataset']:13} {r['aggregation']:9} {r['attack']:16} "
              f"{r['partition']:7} {str(r['alpha']):5} {str(r['malicious']):>2} "
              f"{str(r['tau']):>4} | {_fmt(r['asr'])} {_fmt(r['evasion'])} "
              f"{_fmt(r['accuracy'])}")


def print_alpha_sensitivity(records: List[Dict]) -> None:
    """ASR vs non-IID alpha (RQ2): is non-IID an ally of the attacker?"""
    geo = [r for r in records
           if r["attack"] in ("geotox", "geotox_adaptive") and r["partition"] == "noniid"]
    if not geo:
        print("\n[alpha] no non-IID GeoTox runs found.")
        return
    print("\n=== Non-IID alpha sensitivity (GeoTox, non-IID only) ===")
    print(f"{'dataset':13} {'agg':9} {'attack':16} {'m':>2} {'tau':>4} "
          f"{'alpha':>5} | {'ASR':>6} {'Evas':>6}")
    keyf = lambda r: (r["dataset"], r["aggregation"], r["attack"],
                      r["malicious"], _taukey(r["tau"]), _alphakey(r["alpha"]))
    for r in sorted(geo, key=keyf):
        print(f"{r['dataset']:13} {r['aggregation']:9} {r['attack']:16} "
              f"{str(r['malicious']):>2} {str(r['tau']):>4} {str(r['alpha']):>5} "
              f"| {_fmt(r['asr'])} {_fmt(r['evasion'])}")


def _stats(values):
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None, None, 0
    mean = sum(vals) / len(vals)
    if len(vals) > 1:
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        std = var ** 0.5
    else:
        std = 0.0
    return mean, std, len(vals)


def aggregate_over_seeds(records: List[Dict]) -> List[Dict]:
    """Group runs by config (ignoring seed) and average ASR/Evasion/Acc."""
    groups: Dict[tuple, List[Dict]] = {}
    for r in records:
        key = (r["dataset"], r["aggregation"], r["attack"], r["partition"],
               r["alpha"], r["malicious"], r["tau"], r.get("attack_until", 0))
        groups.setdefault(key, []).append(r)

    out = []
    for key, rs in groups.items():
        asr_m, asr_s, n = _stats([r["asr"] for r in rs])
        ev_m, ev_s, _ = _stats([r["evasion"] for r in rs])
        ac_m, _, _ = _stats([r["accuracy"] for r in rs])
        out.append({
            "dataset": key[0], "aggregation": key[1], "attack": key[2],
            "partition": key[3], "alpha": key[4], "malicious": key[5],
            "tau": key[6], "attack_until": key[7], "n_seeds": n,
            "asr_mean": asr_m, "asr_std": asr_s, "evasion_mean": ev_m,
            "evasion_std": ev_s, "accuracy_mean": ac_m,
        })
    return out


def print_seed_aggregate(records: List[Dict], out_path: Path) -> None:
    """Print and save the per-config mean +/- std across seeds."""
    agg = aggregate_over_seeds(records)
    n_max = max((g["n_seeds"] for g in agg), default=0)
    cols = ["dataset", "aggregation", "attack", "partition", "alpha",
            "malicious", "tau", "attack_until", "n_seeds", "asr_mean",
            "asr_std", "evasion_mean", "evasion_std", "accuracy_mean"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for g in agg:
            w.writerow(g)

    print(f"\n=== Per-config mean +/- std over seeds (n up to {n_max}) -> {out_path.name} ===")
    print(f"{'agg':9} {'attack':16} {'part':7} {'alpha':5} {'tau':>4} "
          f"{'n':>2} | {'ASR(mean+/-std)':>16} {'Evas':>6} {'Acc':>6}")
    keyf = lambda g: (g["aggregation"], g["attack"], g["partition"],
                      _alphakey(g["alpha"]), _taukey(g["tau"]))
    for g in sorted(agg, key=keyf):
        asr = (f"{g['asr_mean']:.1f}+/-{g['asr_std']:.1f}"
               if g["asr_mean"] is not None else "   -  ")
        print(f"{g['aggregation']:9} {g['attack']:16} {g['partition']:7} "
              f"{str(g['alpha']):5} {str(g['tau']):>4} {g['n_seeds']:>2} | "
              f"{asr:>16} {_fmt(g['evasion_mean'])} {_fmt(g['accuracy_mean'])}")


def print_durability(records: List[Dict]) -> None:
    """Backdoor durability (RQ3): ASR retention after the attacker leaves."""
    dur = [r for r in records if r.get("attack_until", 0) and r["attack_until"] > 0]
    if not dur:
        print("\n[durability] no durability runs (attack_until>0) found.")
        return
    print("\n=== Backdoor durability after the attacker leaves (RQ3) ===")
    print(f"{'dataset':13} {'agg':9} {'attack':16} {'m':>2} {'tau':>4} "
          f"{'until':>5} | {'ASR@stop':>8} {'ASR_final':>9} {'retain%':>7}")
    keyf = lambda r: (r["dataset"], r["aggregation"], r["attack"],
                      r["malicious"], _taukey(r["tau"]))
    for r in sorted(dur, key=keyf):
        print(f"{r['dataset']:13} {r['aggregation']:9} {r['attack']:16} "
              f"{str(r['malicious']):>2} {str(r['tau']):>4} {str(r['attack_until']):>5} "
              f"| {_fmt(r['asr_at_stop'])} {_fmt(r['asr']):>9} {_fmt(r['retention'])}")


def _taukey(t):
    try:
        return float(t)
    except (ValueError, TypeError):
        return -1.0


def _alphakey(a):
    try:
        return float(a)
    except (ValueError, TypeError):
        return -1.0


def maybe_plot(records: List[Dict], results_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"\n[plot] matplotlib unavailable ({e}); skipping plots.")
        return

    geo = [r for r in records
           if r["attack"] in ("geotox", "geotox_adaptive")
           and r["asr"] is not None and r["evasion"] is not None]
    if geo:
        plt.figure()
        for atk in sorted(set(r["attack"] for r in geo)):
            pts = [(r["evasion"], r["asr"]) for r in geo if r["attack"] == atk]
            pts.sort()
            plt.scatter([p[0] for p in pts], [p[1] for p in pts], label=atk)
        plt.xlabel("Evasion rate (%)")
        plt.ylabel("ASR (%)")
        plt.title("GeoTox: Evasion vs ASR trade-off")
        plt.legend()
        out = results_dir / "tradeoff_evasion_vs_asr.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"\n[plot] saved {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate FL benchmark results")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--last", type=int, default=5,
                    help="Average metrics over the last N rounds (default: 5)")
    ap.add_argument("--plot", action="store_true", help="Save PNG plots")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    records = collect(results_dir, args.last)
    print(f"Parsed {len(records)} runs from {results_dir}/")
    if not records:
        print("No runs with config sidecars found. Run experiments first.")
        return 0

    out_path = results_dir / "summary_metrics.csv"
    write_summary(records, out_path)
    print(f"Wrote {out_path}")

    print_tradeoff(records)
    print_alpha_sensitivity(records)
    print_durability(records)
    print_seed_aggregate(records, results_dir / "summary_by_config.csv")
    if args.plot:
        maybe_plot(records, results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

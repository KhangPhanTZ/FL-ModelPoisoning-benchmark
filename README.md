# FL Model Poisoning Benchmark

A comprehensive Federated Learning benchmarking framework for evaluating model poisoning attacks and Byzantine-robust aggregation defenses.

## Overview

This repository provides a modular implementation of:
- **Federated Learning** with FedAvg on MNIST using LeNet
- **Model Poisoning Attacks**: LIE, Min-Max, Model Replacement
- **Byzantine-Robust Defenses**: Median, Krum, Multi-Krum, Bulyan, FLTrust
- **Data Partitioning**: IID and Non-IID (Dirichlet distribution)

## Project Structure

```
fl_project/
├── client/          # Federated client implementation
├── server/          # Server with aggregation and attack logic
│   ├── aggregation.py   # Defense mechanisms
│   └── attacks.py       # Attack implementations
├── models/          # Neural network architectures (LeNet)
├── data/            # Data loading and partitioning
├── utils/           # Logging utilities
├── results/         # Experiment CSV results
├── main.py          # Main entry point with CLI
├── run_experiments.py   # Batch experiment runner
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python3 -m venv fl_env
source fl_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single Experiment

```bash
python main.py \
    --aggregation mean \
    --attack lie \
    --partition iid \
    --malicious 4 \
    --z 3.0 \
    --rounds 50
```

### Run All 54 Configurations

```bash
# Dry run to see what would execute
python run_experiments.py --dry-run

# Run all experiments
python run_experiments.py
```

## CLI Arguments

| Argument | Options | Description |
|----------|---------|-------------|
| `--dataset` | `mnist`, `fashion_mnist` | Dataset (default: mnist) |
| `--aggregation` | `mean`, `median`, `krum`, `multi_krum`, `bulyan`, `fltrust` | Aggregation method |
| `--attack` | `none`, `lie`, `minmax`, `model_replacement` | Attack type |
| `--partition` | `iid`, `noniid` | Data distribution |
| `--malicious` | Integer | Number of malicious clients |
| `--z` | Float | Attack strength parameter |
| `--alpha` | Float | Dirichlet alpha for non-IID (default: 0.5) |
| `--root_size` | Integer | Clean root-set size for FLTrust (default: 100) |
| `--rounds` | Integer | Number of FL rounds (default: 50) |
| `--num_clients` | Integer | Total clients (default: 20) |
| `--clients_per_round` | Integer | Clients sampled per round (default: 10) |
| `--seed` | Integer | Random seed (default: 42) |

## Attacks

### LIE (Little Is Enough)
Malicious clients send `mean + z * std` to shift the aggregated model.

### Min-Max
Maximizes distance from benign updates by perturbing in the opposite direction.

### Model Replacement
Scales malicious updates to dominate after FedAvg aggregation.

## Defenses

| Defense | Description |
|---------|-------------|
| **Mean (FedAvg)** | Weighted average of client updates |
| **Median** | Coordinate-wise median |
| **Krum** | Selects update closest to others |
| **Multi-Krum** | Selects k closest updates and averages |
| **Bulyan** | Krum selection + trimmed mean |
| **FLTrust** | Cosine-similarity trust weighting against a clean **server root dataset** (set via `--root_size`); falls back to the coordinate-wise median reference only when no root set is provided |

## Datasets

`mnist` and `fashion_mnist` are supported (both 1x28x28, 10 classes, so the same
LeNet works for either). Select with `--dataset`.

## Results

Experiment results are saved to `results/` as CSV files:
- Format: `{dataset}_{aggregation}_{attack}_{partition}[_a{alpha}]_m{malicious}[_s{seed}].csv`
  (`a{alpha}` only for non-IID; `s{seed}` when a seed is recorded)
- Columns: `round`, `loss`, `accuracy`, `asr`, `evasion_rate`, `timestamp`

## Metrics

- **Accuracy**: Model accuracy on the test set (main-task performance).
- **ASR (Attack Success Rate)**: Backdoor effectiveness. Only meaningful for the
  backdoor attack (`model_replacement`); untargeted attacks (LIE / Min-Max) are
  assessed by the accuracy drop versus the clean `none` baseline.
- **Evasion Rate**: Fraction (%) of malicious updates accepted (not filtered) by
  the defense each round. For coordinate-wise defenses (Median) there is no
  per-client rejection, so this is reported as 100%.

## Implementation notes

Attacks and aggregation operate in **update space** (`u_i = w_local_i - w_global`):
the server forms each client's update, the attack perturbs the malicious
updates, the aggregator returns an aggregated update `Delta`, and the new global
model is reconstructed as `w_global + Delta`. This matches how Byzantine-robust
defenses measure norms and cosine similarities (on gradients, not raw weights).

## Configuration Matrix

`run_experiments.py` sweeps: datasets x aggregations x seeds x partition
settings (IID + a non-IID **alpha sweep**), and for each it runs a clean `none`
baseline plus the attack x malicious grid. The lists (`DATASETS`,
`AGGREGATIONS`, `NONIID_ALPHAS`, `SEEDS`, `MALICIOUS_COUNTS`) are configurable
at the top of the file - use `--dry-run` to preview the count and trim before
launching.

## Testing

```bash
python3 -m pytest tests/ -v        # or: PYTHONPATH=. python3 tests/test_phase0.py
```



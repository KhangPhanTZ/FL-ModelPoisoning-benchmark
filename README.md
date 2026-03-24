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
| `--aggregation` | `mean`, `median`, `krum`, `multi_krum`, `bulyan`, `fltrust` | Aggregation method |
| `--attack` | `none`, `lie`, `minmax`, `model_replacement` | Attack type |
| `--partition` | `iid`, `noniid` | Data distribution |
| `--malicious` | Integer | Number of malicious clients |
| `--z` | Float | Attack strength parameter |
| `--alpha` | Float | Dirichlet alpha for non-IID (default: 0.5) |
| `--rounds` | Integer | Number of FL rounds (default: 50) |
| `--num_clients` | Integer | Total clients (default: 20) |
| `--clients_per_round` | Integer | Clients sampled per round (default: 10) |

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
| **FLTrust** | Cosine similarity-based trust weighting |

## Results

Experiment results are saved to `results/` as CSV files:
- Format: `{aggregation}_{attack}_{partition}_m{malicious}.csv`
- Columns: `round`, `loss`, `accuracy`, `asr`

## Metrics

- **Accuracy**: Model accuracy on test set
- **ASR (Attack Success Rate)**: Measures attack effectiveness

## Configuration Matrix

The benchmark runs 54 configurations:
- 3 aggregations × 3 attacks × 2 partitions × 3 malicious counts



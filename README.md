# Self-Healing Network Testbed

This repository implements the PRD for a software-only, topology-aware self-healing network experiment platform. It is structured around the phases in the PRD and emphasizes a two-tier approach: a fast graph simulator for training and a Mininet-friendly interface for validation and live experiments.

## Architecture overview

- `topology/`: topology generation and converters
- `emulation/`: Mininet-centric network harness and traffic generator wrappers
- `telemetry/`: graph snapshots and feature extraction
- `failures/`: failure injection scenarios and label generation
- `sim/`: fast graph simulator used for RL training and counterfactual evaluation
- `models/`: lightweight topology-aware representation and failure predictor
- `rl/`: Gymnasium-style recovery environment with action masking and reward shaping
- `baselines/`: comparison strategies, including no-op and reactive rerouting
- `eval/`: metrics and experiment runner
- `scripts/`: reproducible training and evaluation entry points

## Quickstart

1. Create a virtual environment and install the project:
   ```bash
   python -m venv .venv
   . .venv/bin/activate
   python -m pip install -U pip
   python -m pip install -e .
   ```
2. Run the smoke test:
   ```bash
   pytest -q
   ```
3. Run a graph simulation example:
   ```bash
   python scripts/collect_data.py
   ```
4. Run the baseline evaluation:
   ```bash
   python scripts/evaluate.py
   ```

## Design decisions

- The simulator intentionally avoids hard dependence on Mininet. Training runs against the graph model and only optionally touches Mininet at validation time.
- Network graphs are built with NetworkX and use simple edge attributes similar to the PRD (bandwidth, delay, loss, and utilization).
- The RL agent uses a minimal, environment-friendly interface that remains compatible with the design intent from the PRD while keeping the code executable in plain Python.
- Real Mininet and Ryu integrations are represented as compatibility wrappers and smoke-test-ready adapters; they do not require privileged Linux execution in the default developer environment.

## Phase coverage

- Phase 0: repo scaffold, topology generation, traffic harness, smoke scripts
- Phase 1: telemetry snapshots and failure scenarios
- Phase 2: fast simulator, metrics, baselines
- Phase 3: GNN-style failure predictor and feature extraction
- Phase 4: RL recovery environment and action masking
- Phase 5: experiment runner and evaluation flow

## Notes

The project intentionally reports honest, reproducible outputs. It does not fabricate experimental results or claims. The repository is designed to be extended with real Mininet, Ryu, PyTorch Geometric, and SB3 integrations.


# Self-Healing Network Testbed

Software testbed for studying topology-aware failure prediction and network recovery. The fast path is an executable NetworkX simulator; optional paths connect the same recovery interface to Mininet, Open vSwitch, Ryu, iperf3, PyTorch, and Stable-Baselines3.

No experiment outcomes are claimed in this repository. Metrics and plots are generated from runs; there are no hard-coded performance claims.

## Architecture

```text
NetworkX topology + flow demands
          |
Failure scheduler / injection -----> timestamped graph snapshots (JSONL)
          |                                      |
Fast capacity-aware graph simulator        failure predictor
          |                                      |
counterfactual baselines <---------- masked Gymnasium recovery environment
                                                 |
                         optional Mininet + Ryu adapter / measured QoS
                                                 |
                                  CSV metrics and comparison plots
```

Core modules live under `src/shnet/`:

- `topology/`: seeded Barabasi-Albert, Erdos-Renyi, Waxman, grid, ring, and fat-tree-style topologies; GML/GraphML loaders.
- `failures/`: seeded hard-link, node, degradation, congestion, and cascading injection, with future-horizon edge labels.
- `sim/`: capacity-aware flow assignment, link/node restoration, weight rebalancing, and non-mutating counterfactual evaluation.
- `telemetry/`: graph feature snapshots, JSONL persistence, periodic collection, and Ryu OpenFlow counter sampling.
- `models/`: trainable edge-risk MLP on message-passing features, plus an optional native-PyTorch edge-aware GNN with a temporal GRU path.
- `rl/`: padded, topology-size-agnostic Gymnasium observation/action spaces, action masks, configurable recovery rewards, and a live Mininet adapter. Padding limits default to 50 nodes, 512 links, and 128 flows.
- `baselines/` and `eval/`: no-op, reactive, load-aware, prediction-informed, and fixed-feature counterfactual strategies; seeded trials, 95% confidence intervals, CSVs, and optional plots.
- `emulation/` and `controller/`: real Mininet topology/traffic handling and Ryu REST clients. These are optional system integrations, not mocks.

## Install and run the simulation

Python 3.10+ is supported for the simulator. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Run the end-to-end software-only workflow:

```bash
bash scripts/run_testbed.sh
```

It collects a labeled snapshot sequence, trains and evaluates the predictor on held-out topology families, demonstrates the Gymnasium recovery environment, and runs the configured seeded baseline matrix. Files are written under the gitignored `results/` directory.

Individual commands:

```bash
python scripts/collect_data.py --config configs/topology.yaml
python scripts/train_gnn.py --config configs/training.yaml
python scripts/train_rl.py                         # environment/policy smoke demo
python scripts/evaluate.py --config configs/evaluation.yaml
```

To use a trained predictor in the prediction-informed baseline:

```bash
python scripts/train_gnn.py --output results/gnn_model.npz
python scripts/evaluate.py --predictor results/gnn_model.npz
```

For comparison charts, install the plotting extra. The evaluator writes trial and aggregate CSV files by default and creates MTTR, reconfiguration-cost, and topology-generalization plots when plotting support is installed:

```bash
python -m pip install -e ".[plot]"
python scripts/evaluate.py
```

### Train and evaluate MaskablePPO

Install the optional RL dependencies (PyTorch, Stable-Baselines3, and sb3-contrib):

```bash
python -m pip install -e ".[rl]"
python scripts/train_rl.py --train --timesteps 100000
python scripts/train_rl.py --train --without-gnn --output results/fixed_feature_ppo
python scripts/evaluate_rl.py --policy results/maskable_ppo.zip
```

The default PPO policy uses an edge-aware message-passing feature extractor and action masks. `--without-gnn` selects the fixed-feature MultiInputPolicy comparison. Training randomizes the failure scenario per episode; the Gymnasium environment pads graph observations and action slots up to the configured limits so one policy can be evaluated on different graph sizes. Increase the capacities in `SelfHealingEnv` when evaluating larger or denser graphs.

## Run the real Mininet/Ryu smoke test

The emulator requires an Ubuntu/Linux host with Mininet, Open vSwitch, `iperf3`, and Ryu installed. It must run with root privileges. Ryu's REST and OpenFlow ports default to 8080 and 6653. The current host used for development does not have `mn`, `ryu-manager`, or `iperf3`, so this integration has not been executed here.

After installing those system dependencies in a supported Python environment:

```bash
sudo env PATH="$PATH" .venv/bin/python scripts/run_mininet.py --recovery-smoke
```

This boots a five-switch path, starts Ryu's topology/ofctl REST apps and the project link-weight endpoint, injects a link failure, selects a reactive restoration, verifies connectivity, generates iperf3 traffic, and tears down the Mininet instance. To start a configurable topology without the recovery scenario:

```bash
sudo env PATH="$PATH" .venv/bin/python scripts/run_mininet.py --topology ring --nodes 10
```

The Mininet harness tears down its own network via `net.stop()`. It does not run global `mn -c`, which could disrupt unrelated Mininet work on the host. If a forced process termination leaves stale OVS state, inspect the host first and run Mininet's cleanup command manually only when safe.

## Configurations and output

- `configs/topology.yaml`: topology, default traffic demand, simulation horizon.
- `configs/training.yaml`: predictor samples/epochs, PPO settings, and configurable reward weights.
- `configs/evaluation.yaml`: topology families, strategies, scenario matrix, seeds, and report directory.
- `results/`: generated snapshots, model checkpoints, reports, and plots; ignored by Git.

`scripts/evaluate.py` accepts `--nodes`, `--horizon`, `--output`, and `--no-plots`. `scripts/train_gnn.py` supports `--backend numpy` (lightweight default) and `--backend torch` (optional PyTorch implementation).

## Design and interpretation notes

- The simulator preserves failed links/nodes as inactive physical components so recovery can restore them. Flow connectivity and delivered throughput are measured against explicit demands rather than inferred from the largest connected component.
- The default NumPy predictor is a compact edge-context model; it is not PyTorch Geometric. `torch_gnn.py` provides a native PyTorch edge-aware message-passing network with temporal GRU support. Neither model's predictive performance is assumed—run training and holdout evaluation.
- The Mininet adapter uses the same Gymnasium action contract, applies link/node actions, verifies connectivity with ping, measures traffic with iperf3, and samples OpenFlow port/flow counters through Ryu. Real measurements and simulation metrics should be reported separately.
- The GNN heuristic and fixed-feature greedy policies are heuristics, not trained RL policies. Compare learned policies using `train_rl.py` and `evaluate_rl.py`.
- Synthetic workloads and simulator results do not replace Mininet validation. The simulation parameters should be calibrated against measured emulation data before drawing research conclusions.

## Remaining host-dependent work

The code paths and scripts are present, but this workspace lacks the required system binaries and privileged Mininet environment. The remaining setup is to install/launch Mininet, Open vSwitch, Ryu, and iperf3 on Ubuntu (or an Ubuntu VM), run the five-switch recovery gate, then calibrate simulator delay/capacity/loss against those measurements. Optional PPO training also requires installing the `.[rl]` extra and running the training/evaluation jobs; no trained policy checkpoint or research results are committed.

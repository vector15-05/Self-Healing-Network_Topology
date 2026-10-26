# PRD: Predictive and Autonomous Self-Healing Network Topologies

**Type:** Research testbed + ML system (Computer Networks course project)
**Authors:** Ayush Mohanty, Vinayak Joshi, Dinesh Karthik
**Audience:** AI coding agent (Antigravity). Build incrementally, phase by phase, and stop at each phase's acceptance gate.

---

## 1. Overview

Build a **software-only self-healing network testbed**. A virtual network (Mininet) is monitored in real time. A Graph Neural Network (GNN) learns topology-aware representations to detect and predict failures. A Deep RL agent then picks the **minimum effective intervention** to restore connectivity and QoS, and an SDN controller (Ryu) applies it.

**Research question:** Can a topology-aware AI system autonomously determine the minimum effective intervention required to restore connectivity and QoS after failure?

**Core idea:** `link failure -> auto-repair`, with as few reconfigurations as possible.

## 2. Goals

1. Reproducible, software-only testbed for controlled failure experiments (no physical hardware).
2. Model the live network as a dynamic graph `G = (V, E)` updated from topology and telemetry.
3. Detect and predict failures, and estimate their impact, *before* full failure.
4. Select recovery actions autonomously via DRL.
5. Minimize recovery time, QoS degradation, and reconfiguration cost.
6. Evaluate against baselines on seen and unseen topologies and failure patterns.

## 3. Non-Goals

- No physical routers, switches, or sensors.
- No production-grade controller HA, security hardening, or multi-tenant features.
- No claim of results yet. The expected outcomes (lower MTTR, fewer reconfigs, better generalization) are **hypotheses to be validated experimentally**. Do not hardcode or fabricate results anywhere.

## 4. Tech Stack

| Layer | Choice |
|---|---|
| Network emulation | Mininet (Open vSwitch, OpenFlow 1.3) |
| SDN controller | **Ryu** (default). ONOS is optional/stretch behind the same interface |
| Orchestration | Python 3.10+ |
| Graph building | NetworkX |
| GNN | PyTorch + PyTorch Geometric |
| DRL | Stable-Baselines3 (+ `sb3-contrib` for MaskablePPO), Gymnasium |
| Traffic / QoS | iperf3, ping, `tc netem` for impairment |
| Telemetry (optional) | Prometheus + Grafana |
| Experiment tracking | CSV/Parquet logs + TensorBoard (MLflow optional) |

**Host:** Linux (Ubuntu recommended), multi-core CPU, 16 GB+ RAM, GPU optional. Mininet requires Linux and root privileges. If the host is macOS/Windows, use an Ubuntu VM and document setup in the README.

## 5. System Architecture

```
Mininet Network (virtual hosts + switches)
        |
Telemetry Layer (latency, loss, throughput, utilization)
        |
Dynamic Graph Representation  G = (V, E)
        |
Failure Detection & Prediction Engine  <-- GNN topology embeddings
        |
Counterfactual Simulator (candidate topology evaluation)
        |
Deep RL Recovery Agent (minimal-intervention action)
        |
SDN Controller (Ryu) -> Network Reconfiguration
        |
QoS Verification -> Feedback / Reward signal -> retrains GNN + RL agent
```

### Key design decision: two-tier environment

Mininet is too slow for the millions of steps RL needs. Therefore:

- **Fast sim env** (`graph_sim`): a pure-Python NetworkX flow/capacity model that mimics the same state, actions, and reward. Used for **training** the GNN and RL agent and as the **counterfactual simulator**.
- **Mininet env** (`mininet_env`): the high-fidelity emulation. Used for **data collection, validation, and final evaluation**.

Both must expose the **same Gymnasium-compatible interface** so a trained policy runs unchanged on either.

## 6. Repository Structure

```
self-healing-net/
  README.md
  pyproject.toml / requirements.txt
  configs/                 # YAML: topologies, failures, training, eval
  src/shnet/
    topology/              # generators + loaders (NetworkX)
    emulation/             # Mininet builder, traffic gen, impairment injector
    controller/            # Ryu app + REST client wrapper
    telemetry/             # collectors, graph snapshot builder
    failures/              # failure injection scenarios
    sim/                   # fast graph simulator (counterfactual + training env)
    models/                # GNN encoder, failure predictor, RL feature extractor
    rl/                    # Gymnasium env, reward, action masking, training
    baselines/             # comparison strategies
    eval/                  # metrics, experiment runner, plots
    utils/                 # logging, seeding, config
  scripts/                 # run_testbed.sh, collect_data.py, train_gnn.py, train_rl.py, evaluate.py
  tests/
  results/                 # gitignored: logs, checkpoints, figures
```

## 7. Functional Requirements

### 7.1 Topology module
- Generate topologies via NetworkX: Barabasi-Albert, Erdos-Renyi (connected), Waxman, grid, ring, fat-tree. Support loading from GML (e.g. Internet Topology Zoo).
- Each link has attributes: `bandwidth_mbps`, `delay_ms`, `loss_pct`. Each node: type (`host`/`switch`), optional capacity.
- Seeded and deterministic. Provide **disjoint train and test topology sets** (varying size 10-50 switches) for generalization experiments.
- Converter: NetworkX graph -> Mininet `Topo` (using `TCLink` for bw/delay/loss).

### 7.2 Emulation and traffic
- Launch Mininet from a topology config with the Ryu controller as remote.
- Traffic generator: iperf3 flows between random host pairs with configurable mix (TCP bulk, UDP constant-rate), rates, and durations. Flow demands must be recorded.
- Provide a clean teardown (`mn -c` equivalent) to avoid stale state between runs.

### 7.3 Telemetry layer
- Poll at a configurable interval (default 1s): per-link **latency, packet loss, throughput, utilization**; per-switch flow counts and port stats (from OpenFlow stats via Ryu).
- Emit a **graph snapshot** per tick: node feature matrix, edge index, edge feature matrix, timestamp, and flow table summary.
- Store snapshots as a time series (Parquet/NPZ) for GNN training. Optional Prometheus exporter.

### 7.4 Failure injection
Scenarios (all seeded and logged with ground-truth labels and timestamps):
- **Hard link failure** (link down).
- **Node/switch failure.**
- **Gradual degradation** (loss/delay ramp via `tc netem`), which serves as the **precursor signal for prediction**.
- **Congestion-induced failure** (traffic surge).
- **Multiple simultaneous** and **cascading** failures (failure of one link shifts load and overloads the next).
- Support a scheduler that mixes scenarios with configurable rates.

### 7.5 Dynamic graph + GNN
- Graph `G=(V,E)` rebuilt/updated each tick from topology and telemetry.
- Encoder: message-passing GNN (GraphSAGE/GAT/GINE, configurable) with edge features, producing node and edge embeddings. Add a temporal component (GRU over a window of K snapshots, or stacked snapshot features).
- Heads:
  1. **Link failure prediction:** probability each link fails within horizon H seconds.
  2. **Impact estimation:** predicted QoS degradation (e.g. affected flows, expected throughput/latency impact).
- Training: supervised on collected labelled snapshots. Handle class imbalance (weighted loss/focal loss). Report PR-AUC, F1, precision/recall, and **lead time** (how early failures are predicted).
- Must be **size-agnostic** (no fixed node count) to support unseen topologies.

### 7.6 Counterfactual simulator
- Given the current graph state and a candidate action, return the predicted post-action state (connectivity, per-flow path, utilization, latency estimate, reconfig cost) **without touching Mininet**.
- Use shortest/k-shortest paths with capacity-aware flow assignment. Must be fast (target: < 10 ms per evaluation on a 30-node graph).
- Doubles as the training environment for the RL agent.

### 7.7 DRL recovery agent
- Gymnasium environment `SelfHealingEnv` (works on `graph_sim` and `mininet_env`).
- **Observation:** graph state (nodes, edges, features, active failures, predicted failure probabilities, current flow routing).
- **Action space (discrete, masked):**
  - `NO_OP` (explicitly valid; this is key to minimal intervention)
  - `REROUTE_FLOW(flow_id, k)` where k indexes the k-th alternative path
  - `ACTIVATE_BACKUP_LINK(link_id)` if available
  - `REBALANCE_WEIGHTS(region)` (shift load across paths)
  - Use **action masking** (MaskablePPO) so invalid actions are never chosen.
- **Reward (weights configurable in YAML):**
  `R = - w1*disconnected_flows - w2*QoS_degradation(loss, latency, throughput shortfall) - w3*reconfig_cost(#flow-rule changes) - w4*time_to_recover`
  plus a terminal bonus when connectivity and QoS are fully restored.
- **Policy:** custom SB3 features extractor wrapping the GNN encoder (PyG) so the policy is topology-size agnostic. Algorithm: MaskablePPO (default); DQN as an optional comparison.
- Curriculum: single failure -> multiple -> cascading; small -> larger topologies.
- The agent can also act **proactively** on predicted failures (before link actually fails).

### 7.8 SDN controller integration
- Ryu app (OpenFlow 1.3) exposing a REST API: get topology, get stats, install/delete flow rules, set path for a flow, set link weights.
- A Python client wrapper in `controller/` used by the Mininet env to apply agent actions.
- Count every flow-rule add/modify/delete as reconfiguration cost (feeds reward and metrics).
- Keep controller interface abstract so ONOS can be swapped in later.

### 7.9 QoS verification and feedback
- After each action, verify via ping/iperf3 (and telemetry) that connectivity and QoS recovered. Compute the reward from measured outcomes.
- Log (state, action, reward, outcome) tuples to support retraining and offline analysis.

### 7.10 Baselines (required for fair comparison)
1. **No-op** (no recovery).
2. **Reactive shortest-path reroute** (OSPF-like recompute after failure detection).
3. **Reactive + load-aware reroute** (ECMP / least-utilized path).
4. **GNN prediction + heuristic action** (no RL).
5. **RL without GNN** (MLP on flattened features; limited to fixed topology).
6. **Full system** (GNN + DRL).

## 8. Evaluation Metrics

| Category | Metrics |
|---|---|
| Recovery | Mean Time to Recovery (MTTR), service restoration time |
| QoS | Packet loss, end-to-end latency, throughput |
| Resilience | Network availability, connectivity restoration rate, cascading failure probability |
| Efficiency | Number of reconfiguration operations, reconfiguration cost, control-plane overhead |
| Generalization | Performance on unseen topologies, performance under unseen failure patterns |
| Prediction (supporting) | PR-AUC, F1, lead time |

Experiment runner requirements:
- Run N seeded trials per (strategy x topology x failure scenario); report mean, std, and 95% CI.
- Auto-generate comparison tables (CSV) and plots (MTTR bar/box, reconfigs vs. MTTR scatter, generalization by topology size).
- Single command reproduces a full experiment from a config file.

## 9. Non-Functional Requirements

- **Reproducibility:** global seeding (python, numpy, torch, Mininet traffic), pinned dependency versions, configs saved with every run.
- **Modularity:** each module independently testable; sim and Mininet share interfaces.
- **Performance:** sim env >= 1,000 steps/sec on a 30-node graph on CPU for RL training to be practical.
- **Robustness:** graceful teardown on crash; no stale Mininet/OVS state.
- **Testing:** pytest unit tests for topology, simulator, reward, action masking, metrics; one integration smoke test that boots a 5-switch Mininet, injects a link failure, and runs a baseline recovery.
- **Documentation:** README with setup, quickstart, and an architecture diagram; docstrings on public APIs.

## 10. Build Phases (with acceptance gates)

**Phase 0: Environment and testbed**
- Repo scaffold, dependencies, Mininet + Ryu running, topology generator, Mininet builder, iperf3 traffic.
- *Gate:* one command starts a 10-switch topology, runs traffic, and tears down cleanly.

**Phase 1: Telemetry and failure injection**
- Telemetry collector -> graph snapshots; all failure scenarios in 7.4; ground-truth labels.
- *Gate:* inject a link failure and see it reflected in the logged snapshot sequence.

**Phase 2: Fast simulator + baselines**
- `graph_sim` with the same interface; baselines 1-3; metrics module; experiment runner.
- *Gate:* baselines run on both sim and Mininet with consistent metric trends; sim speed target met.

**Phase 3: GNN predictor**
- Dataset collection (sim primarily, Mininet for validation), GNN encoder + heads, training/eval scripts.
- *Gate:* beats a non-graph baseline (e.g. logistic regression/MLP on link features) on PR-AUC for failure prediction; evaluated on held-out topologies.

**Phase 4: DRL agent**
- `SelfHealingEnv`, action masking, reward, GNN features extractor, curriculum training.
- *Gate:* agent trains stably and outperforms the reactive baseline on at least one of MTTR or reconfig count in sim.

**Phase 5: Integration and full evaluation**
- Deploy the trained policy on Mininet + Ryu; run the full experiment matrix including unseen topologies and cascading failures; generate figures and tables.
- *Gate:* one-command reproduction of all results; honest reporting even if hypotheses are not supported.

## 11. Optional / Stretch

- ONOS controller backend.
- Prometheus + Grafana dashboards for live telemetry.
- Internet Topology Zoo evaluation set.
- Safe-RL constraint (penalize or block actions that would worsen connectivity).

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Mininet too slow for RL | Two-tier env: train in `graph_sim`, validate in Mininet |
| Sim-to-emulation gap | Calibrate sim against Mininet measurements; report both |
| Too few failure events to predict | Gradual-degradation precursors, oversampling, weighted loss |
| RL instability / sparse reward | Dense shaped reward, curriculum, action masking, `NO_OP` baseline |
| Poor generalization | Size-agnostic GNN, train on diverse topologies, hold out test set |
| Environment setup friction | Ubuntu VM instructions, setup script, smoke test |

## 13. Agent Working Instructions

- Work **one phase at a time**. Finish, run tests, and confirm the gate before moving on.
- Prefer simple, readable code with type hints. Keep configs in YAML, not hardcoded.
- Never fabricate experimental results, metrics, or plots. All reported numbers must come from actual runs.
- When a design choice is ambiguous, pick the simplest option consistent with this PRD and note it in the README under "Design decisions".
- Keep the sim and Mininet environments interface-compatible at all times.
- After each phase, update the README with how to run that phase.

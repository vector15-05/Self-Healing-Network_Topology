#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from shnet.rl.env import SelfHealingEnv
from shnet.sim.reward import RewardWeights
from shnet.topology.generators import generate_topology
from shnet.utils.config import load_config


def build_training_env(
    topology_kind: str,
    nodes: int,
    seed: int,
    reward_weights: RewardWeights | None = None,
) -> SelfHealingEnv:
    graph = generate_topology(topology_kind, nodes, seed=seed)
    return SelfHealingEnv(
        graph,
        horizon=10,
        randomize_failure=True,
        reward_weights=reward_weights,
    )


def train(args: argparse.Namespace) -> None:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "RL training requires Stable-Baselines3, sb3-contrib, and PyTorch. "
            "Install them with `python -m pip install -e '.[rl]'`."
        ) from exc
    env = build_training_env(args.topology, args.nodes, args.seed, args.reward_weights)
    policy_kwargs: dict[str, Any] = {"net_arch": {"pi": [64], "vf": [64]}}
    if not args.without_gnn:
        try:
            from shnet.rl.features_extractor import GraphPolicyFeatureExtractor
        except ImportError as exc:
            raise SystemExit(
                "The graph policy extractor requires PyTorch and Stable-Baselines3. "
                "Install them with `python -m pip install -e '.[rl]'`."
            ) from exc
        policy_kwargs["features_extractor_class"] = GraphPolicyFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {"hidden_dim": 32}
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.learning_rate,
        n_steps=128,
        batch_size=64,
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(
        {
            "checkpoint": str(args.output),
            "timesteps": args.timesteps,
            "features": "fixed-feature-mlp" if args.without_gnn else "graph-message-passing",
        }
    )


def demonstrate(env: SelfHealingEnv, seed: int) -> None:
    observation, info = env.reset(seed=seed)
    del observation
    total_reward = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        mask = info["action_mask"]
        valid = [index for index, allowed in enumerate(mask) if allowed]
        repair_actions = [index for index in valid if index not in (0, env.action_space.n - 1)]
        action = (
            0
            if info["recovery_complete"]
            else (repair_actions[0] if repair_actions else env.action_space.n - 1)
        )
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    print(
        {
            "total_reward": round(total_reward, 3),
            "recovered": bool(info["recovery_complete"]),
            "reconfig_cost": info["reconfig_cost"],
            "steps": env.timestep,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate or train the masked PPO recovery agent.")
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--train", action="store_true", help="Run MaskablePPO training.")
    parser.add_argument(
        "--without-gnn",
        action="store_true",
        help="Use the fixed-feature MultiInputPolicy baseline instead of graph message passing.",
    )
    parser.add_argument("--timesteps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--topology", default="barabasi_albert")
    parser.add_argument("--nodes", type=int, default=12)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("results/maskable_ppo"))
    args = parser.parse_args()
    config = load_config(args.config)
    training = config.get("training", {})
    args.timesteps = (
        args.timesteps
        if args.timesteps is not None
        else int(training.get("rl_timesteps", 100_000))
    )
    args.learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(training.get("rl_learning_rate", 3e-4))
    )
    args.seed = args.seed if args.seed is not None else int(training.get("seed", 7))
    args.reward_weights = RewardWeights.from_mapping(config.get("reward"))
    if args.train:
        if args.timesteps <= 0 or args.learning_rate <= 0:
            parser.error("--timesteps and --learning-rate must be positive")
        train(args)
    else:
        demonstrate(
            build_training_env(args.topology, args.nodes, args.seed, args.reward_weights),
            args.seed,
        )


if __name__ == "__main__":
    main()

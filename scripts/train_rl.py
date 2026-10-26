#!/usr/bin/env python3
from __future__ import annotations

from shnet.rl.env import SelfHealingEnv


def main() -> None:
    env = SelfHealingEnv(horizon=3)
    observation = env.reset()
    total_reward = 0.0
    for action in ["NO_OP", "NO_OP", "NO_OP"]:
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print({"initial_obs": observation["step"], "total_reward": round(total_reward, 2), "final_done": done})


if __name__ == "__main__":
    main()

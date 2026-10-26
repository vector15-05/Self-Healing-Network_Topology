from shnet.rl.env import SelfHealingEnv


def test_self_healing_env_reset_and_step() -> None:
    env = SelfHealingEnv(horizon=2)
    observation = env.reset()

    assert observation["step"] == 0
    assert "action_mask" in observation

    next_observation, reward, done, info = env.step("NO_OP")
    assert isinstance(reward, float)
    assert "meta" in info
    assert not done or next_observation["step"] >= 0

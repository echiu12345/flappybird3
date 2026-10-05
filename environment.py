"""Shared observation settings for training and evaluation."""

import flappy_bird_gymnasium  # Registers FlappyBird-v0.
import gymnasium as gym


def make_env(render=False, state_dim=12, max_steps=10000):
    if state_dim not in (12, 180):
        raise ValueError(f"Unsupported observation size: {state_dim}; expected 12 or 180.")
    return gym.make(
        "FlappyBird-v0",
        render_mode="human" if render else None,
        use_lidar=state_dim == 180,
        normalize_obs=True,
        max_episode_steps=max_steps,
    )

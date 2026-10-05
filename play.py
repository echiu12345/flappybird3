"""Evaluate a saved DQN, with optional visible gameplay."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from environment import make_env


def play(model_path=None, render=True, num_episodes=20,
         seed=10000, max_steps=10000, output_json=None):
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError("Episodes and max_steps must be positive.")
    if model_path is None:
        candidates = list(Path("runs").glob("*/best_flappy_bird_dqn.pt"))
        candidates += list(Path("runs").glob("*/flappy_bird_dqn_final.pt"))
        if candidates:
            latest_run = max(candidates, key=lambda path: path.stat().st_mtime).parent
            model = latest_run / "best_flappy_bird_dqn.pt"
            if not model.exists():
                model = latest_run / "flappy_bird_dqn_final.pt"
        else:
            model = Path("best_flappy_bird_dqn.pt")
            if not model.exists():
                model = Path("flappy_bird_dqn_final.pt")
    else:
        model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Model not found: {model}. Train first or pass --model.")
    torch.set_num_threads(1)
    agent = DQNAgent.from_checkpoint(model)
    env = make_env(render=render, state_dim=agent.state_dim, max_steps=max_steps)
    scores, rewards, lengths, truncations = [], [], [], []
    print(f"Model: {model.resolve()} | State dimensions: {agent.state_dim} | Epsilon: 0")
    try:
        for episode in range(num_episodes):
            state, info = env.reset(seed=seed + episode)
            reward_sum, steps = 0.0, 0
            while True:
                action = agent.select_action(state, epsilon=0.0)
                state, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                steps += 1
                if terminated or truncated:
                    break
            score = int(info.get("score", 0))
            scores.append(score)
            rewards.append(reward_sum)
            lengths.append(steps)
            truncations.append(bool(truncated))
            print(f"Episode {episode + 1:3d} | Score {score:3d} | Reward {reward_sum:.2f} | "
                  f"Steps {steps}" + (" | Time limit" if truncated else ""), flush=True)
    finally:
        env.close()
    result = dict(model=str(model.resolve()), seed=seed, episodes=len(scores),
                  mean_score=float(np.mean(scores)), max_score=max(scores),
                  mean_reward=float(np.mean(rewards)), scores=scores, rewards=rewards,
                  steps=lengths, truncated=truncations)
    print(f"Mean score: {result['mean_score']:.2f} | Max score: {result['max_score']}")
    if output_json:
        path = Path(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Default: best weights from the latest run, then legacy root weights")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()
    play(args.model, render=not args.no_render, num_episodes=args.episodes,
         seed=args.seed, max_steps=args.max_steps, output_json=args.output_json)

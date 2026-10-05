"""Train a DQN policy on normalized Flappy Bird state features."""

import argparse
import csv
from collections import deque
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from agent import DQNAgent
from environment import make_env
from runtime import configure_runtime


def plot_metrics(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    episodes = [row["episode"] for row in rows]
    for axis, metric, average, label in (
        (axes[0], "reward", "avg_reward", "Episode reward"),
        (axes[1], "score", "avg_score", "Pipes passed"),
    ):
        axis.plot(episodes, [row[metric] for row in rows], alpha=0.4, label=label)
        axis.plot(episodes, [row[average] for row in rows], label="100-episode mean")
        axis.set(xlabel="Episode", ylabel=label)
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def train(render=False, num_episodes=10000, output_dir=None, seed=42,
          batch_size=128, hidden_dim=256, learning_rate=1e-4, gamma=0.99,
          memory_size=50000, tau=0.005, epsilon_start=0.1,
          epsilon_end=0.01, epsilon_decay=0.999, learning_starts=1000,
          train_frequency=4, max_steps=10000, log_interval=10, threads=1,
          cpu_max=False, device_name=None):
    """Return the run directory; Ctrl+C saves the policy and completed metrics."""
    threads, device_name = configure_runtime(threads, cpu_max, device_name)
    positive = (num_episodes, batch_size, hidden_dim, memory_size, train_frequency,
                max_steps, log_interval, threads)
    if any(value <= 0 for value in positive):
        raise ValueError("Episode counts, sizes, frequencies and threads must be positive.")
    if memory_size < batch_size or learning_starts < 0:
        raise ValueError("Replay capacity must cover a batch; learning_starts must be nonnegative.")
    if not (0 <= epsilon_end <= epsilon_start <= 1 and 0 < epsilon_decay <= 1):
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1 and 0 < decay <= 1.")
    if not (learning_rate > 0 and 0 <= gamma <= 1 and 0 < tau <= 1):
        raise ValueError("Require learning_rate > 0, 0 <= gamma <= 1 and 0 < tau <= 1.")

    config = dict(num_episodes=num_episodes, seed=seed, batch_size=batch_size,
                  hidden_dim=hidden_dim, learning_rate=learning_rate, gamma=gamma,
                  memory_size=memory_size, tau=tau, epsilon_start=epsilon_start,
                  epsilon_end=epsilon_end, epsilon_decay=epsilon_decay,
                  learning_starts=learning_starts, train_frequency=train_frequency,
                  max_steps=max_steps, threads=threads, cpu_max=cpu_max,
                  logical_cpus=os.cpu_count() or 1, state_dim=12, action_dim=2,
                  use_lidar=False, normalize_obs=True, render=render)
    output = Path(output_dir) if output_dir else Path("runs") / datetime.now().strftime("dqn_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("*.pt")) or (output / "training_metrics.csv").exists():
        raise FileExistsError(f"Choose a new output directory; training artifacts already exist: {output}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    env = make_env(render=render, max_steps=max_steps)
    env.action_space.seed(seed)
    agent = DQNAgent(12, 2, hidden_dim, lr=learning_rate, gamma=gamma,
                     memory_size=memory_size, tau=tau, device_name=device_name)
    config["device"] = str(agent.device)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    rewards_window, scores_window = deque(maxlen=100), deque(maxlen=100)
    epsilon, best_reward, best_score, global_step = epsilon_start, -float("inf"), 0, 0
    started = time.monotonic()
    interrupted = False
    fields = ["episode", "steps", "total_steps", "reward", "score", "avg_reward",
              "avg_score", "epsilon", "loss", "updates", "truncated", "elapsed_seconds"]
    print(f"DQN: 12 -> {hidden_dim} -> {hidden_dim} -> 2 | Device: {agent.device}")
    print(f"PyTorch CPU threads: {threads} | Logical CPUs: {config['logical_cpus']}")
    print(f"Training {num_episodes} episodes | Output: {output.resolve()}", flush=True)
    try:
        with (output / "training_metrics.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for episode in range(1, num_episodes + 1):
                state, info = env.reset(seed=seed if episode == 1 else None)
                reward_sum, loss_sum, updates, steps = 0.0, 0.0, 0, 0
                while True:
                    action = agent.select_action(state, epsilon)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    # Time limits end rollouts but still allow future-value targets.
                    agent.step(state, action, reward, next_state, terminated)
                    global_step += 1
                    steps += 1
                    if global_step >= learning_starts and global_step % train_frequency == 0:
                        loss = agent.optimize(batch_size)
                        if loss is not None:
                            loss_sum += loss
                            updates += 1
                    state = next_state
                    reward_sum += reward
                    if terminated or truncated:
                        break

                score = int(info.get("score", 0))
                rewards_window.append(reward_sum)
                scores_window.append(score)
                avg_reward, avg_score = float(np.mean(rewards_window)), float(np.mean(scores_window))
                row = dict(episode=episode, steps=steps, total_steps=global_step,
                           reward=reward_sum, score=score, avg_reward=avg_reward,
                           avg_score=avg_score, epsilon=epsilon,
                           loss=loss_sum / updates if updates else 0.0, updates=updates,
                           truncated=truncated, elapsed_seconds=time.monotonic() - started)
                rows.append(row)
                writer.writerow(row)
                file.flush()
                if avg_reward > best_reward and episode >= min(50, num_episodes):
                    best_reward = avg_reward
                    agent.save(output / "best_flappy_bird_dqn.pt")
                best_score = max(best_score, score)
                if episode % log_interval == 0 or episode == num_episodes:
                    print(f"Episode {episode:5d} | Reward {reward_sum:7.2f} (avg {avg_reward:7.2f}) | "
                          f"Score {score:3d} (avg {avg_score:5.2f}) | Epsilon {epsilon:.4f} | "
                          f"Loss {row['loss']:.5f} | Best score {best_score}", flush=True)
                epsilon = max(epsilon_end, epsilon * epsilon_decay)
    except KeyboardInterrupt:
        interrupted = True
        print("\nTraining interrupted; saving current weights and completed metrics.")
    finally:
        env.close()
        agent.save(output / "flappy_bird_dqn_final.pt")
        if rows:
            plot_metrics(rows, output / "training_performance.png")
        summary = dict(completed_episodes=len(rows), total_steps=global_step,
                       best_training_score=best_score, interrupted=interrupted,
                       elapsed_seconds=time.monotonic() - started)
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved weights, CSV metrics and plots: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--epsilon-start", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.999)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=10)
    cpu_options = parser.add_mutually_exclusive_group()
    cpu_options.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    cpu_options.add_argument("--cpu-max", action="store_true", help="Force CPU and use all logical CPUs for PyTorch operations")
    parser.add_argument("--device", dest="device_name", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = vars(parser.parse_args())
    args["num_episodes"] = args.pop("episodes")
    if args["device_name"] == "auto":
        args["device_name"] = None
    train(**args)


if __name__ == "__main__":
    main()

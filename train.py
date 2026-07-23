import os
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import DQNAgent

def train(render=False):
    # Training configurations
    num_episodes = 10000
    batch_size = 128
    hidden_dim = 256
    learning_rate = 1e-4
    gamma = 0.99
    memory_size = 50000
    tau = 0.005  # Target network soft update parameter
    
    # Exploration parameters
    # Note: Flappy bird is highly sensitive; random flaps usually lead to instant crashes.
    # Therefore, we start with a small epsilon (0.1) and decay it, letting the network's
    # initial value estimates quickly guide exploration.
    epsilon_start = 0.1
    epsilon_end = 0.001
    epsilon_decay = 0.995  # Decay rate per episode

    # Initialize environment
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]  # Should be 12
    action_dim = env.action_space.n             # Should be 2 (0: idle, 1: flap)
    
    print(f"Initializing DQN Agent...")
    print(f"State space: {state_dim} dimensions")
    print(f"Action space: {action_dim} actions")
    print(f"Device: {DQNAgent.__init__.__globals__['device']}")

    # Initialize DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        lr=learning_rate,
        gamma=gamma,
        memory_size=memory_size,
        tau=tau
    )

    # For tracking metrics
    all_rewards = []
    all_scores = []
    all_losses = []
    scores_window = deque(maxlen=100)  # Last 100 episode scores
    rewards_window = deque(maxlen=100)  # Last 100 episode rewards
    
    epsilon = epsilon_start
    best_running_reward = -float("inf")
    best_score = 0

    print("\n--- Training Started ---")
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_losses = []
        done = False

        while not done:
            # Select action
            action = agent.select_action(state, epsilon)

            # Step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Save experience in replay memory
            agent.step(state, action, reward, next_state, done)

            # Optimize model
            loss = agent.optimize(batch_size)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward

        # Episode completed
        score = info.get("score", 0)
        all_rewards.append(episode_reward)
        all_scores.append(score)
        
        rewards_window.append(episode_reward)
        scores_window.append(score)
        
        # Calculate mean loss
        mean_loss = np.mean(episode_losses) if episode_losses else 0.0
        all_losses.append(mean_loss)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Calculate averages
        avg_reward = np.mean(rewards_window)
        avg_score = np.mean(scores_window)

        # Save the best model based on running average reward
        if avg_reward > best_running_reward and episode >= 50:
            best_running_reward = avg_reward
            agent.save("best_flappy_bird_dqn.pt")
            print(f"--> Saved new BEST model checkpoint at Episode {episode} (Avg Reward: {avg_reward:.2f}, Avg Score: {avg_score:.1f})")
            
        # Track the absolute highest score reached
        if score > best_score:
            best_score = score

        # Print statistics
        if episode % 10 == 0 or score > 15:
            print(
                f"Episode {episode:4d} | "
                f"Reward: {episode_reward:6.1f} (Avg: {avg_reward:6.1f}) | "
                f"Score: {score:3d} (Avg: {avg_score:4.1f}) | "
                f"Epsilon: {epsilon:.4f} | "
                f"Loss: {mean_loss:.4f} | "
                f"Best Score: {best_score}"
            )

    env.close()
    print("--- Training Completed ---")

    # Save final model
    agent.save("flappy_bird_dqn_final.pt")
    print("Saved final model to flappy_bird_dqn_final.pt")

    # Save training performance plots
    print("Generating training plots...")
    plt.figure(figsize=(12, 8))
    
    # Plot Rewards
    plt.subplot(2, 1, 1)
    plt.plot(all_rewards, label='Episode Reward', color='skyblue', alpha=0.6)
    # Calculate simple moving average
    moving_avg_rewards = [np.mean(all_rewards[max(0, i-99):i+1]) for i in range(len(all_rewards))]
    plt.plot(moving_avg_rewards, label='100-Ep Moving Avg', color='blue', linewidth=2)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot Scores
    plt.subplot(2, 1, 2)
    plt.plot(all_scores, label='Episode Score', color='lightgreen', alpha=0.6)
    moving_avg_scores = [np.mean(all_scores[max(0, i-99):i+1]) for i in range(len(all_scores))]
    plt.plot(moving_avg_scores, label='100-Ep Moving Avg', color='green', linewidth=2)
    plt.title('Training Scores (Pipes Passed)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("training_performance.png")
    print("Plot saved as training_performance.png")
    plt.close()

if __name__ == "__main__":
    train()

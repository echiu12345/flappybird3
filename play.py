import os
import time
import gymnasium as gym
import flappy_bird_gymnasium
from agent import DQNAgent

def play(model_path="best_flappy_bird_dqn.pt"):
    # If the requested model doesn't exist, try final model
    if not os.path.exists(model_path):
        fallback = "flappy_bird_dqn_final.pt"
        if os.path.exists(fallback):
            print(f"Model '{model_path}' not found. Falling back to '{fallback}'...")
            model_path = fallback
        else:
            print(f"Error: No model checkpoint found at '{model_path}' or '{fallback}'.")
            print("Please run train.py first to train a model.")
            return

    # Initialize environment in human render mode so we can watch it play
    print(f"Initializing environment in human render mode...")
    env = gym.make("FlappyBird-v0", render_mode="human")
    
    state_dim = env.observation_space.shape[0]  # Should be 12
    action_dim = env.action_space.n             # Should be 2

    # Initialize agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256
    )
    
    # Load model weights
    print(f"Loading trained model from '{model_path}'...")
    agent.load(model_path)
    
    print("\n--- Starting Evaluation ---")
    print("Press Ctrl+C in the console to exit.")
    
    num_episodes = 100
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Under exploitation mode, epsilon = 0.0
            action = agent.select_action(state, epsilon=0.0)
            
            # Step in environment
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Add a slight delay to make the gameplay watchable (optional)
            time.sleep(0.02)
            
        print(f"Episode {episode} | Score (Pipes Passed): {info.get('score', 0):3d} | Total Reward: {episode_reward:.2f}")
        time.sleep(1.0)  # Pause between games
        
    env.close()
    print("--- Evaluation Completed ---")

if __name__ == "__main__":
    play()

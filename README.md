# Flappy Bird Reinforcement Learning using Deep Q-Networks (DQN)

This repository contains a complete reinforcement learning implementation designed to solve the **Flappy Bird** game. Built using PyTorch and Gymnasium, the project implements a **Deep Q-Network (DQN)** agent that learns optimal flying strategies (when to flap and when to glide) based on sensory inputs from the environment.

---

## 🎮 Features

- **Double-Layer MLP DQN**: Utilizes a robust 2-layer neural network ($12 \rightarrow 256 \rightarrow 256 \rightarrow 2$) to estimate actions' Q-values.
- **Stable Target Networks**: Uses a target network updated via **soft updates** ($\theta_{\text{target}} \leftarrow \tau \theta_{\text{policy}} + (1-\tau) \theta_{\text{target}}$ where $\tau = 0.005$) to prevent value estimate oscillations.
- **Experience Replay**: Employs a FIFO replay memory to store past experiences and sample randomized minibatches, breaking temporal correlation.
- **Huber Loss Optimization**: Uses Smooth L1 Loss to ensure stability against gradient explosions.
- **Visual Analytics**: Automatically logs reward history and generates visual training plots (`training_performance.png`).
- **Interactive Menu Launcher**: A simple terminal menu that lets you choose between playing manually, training the DQN agent, or watching the trained agent play.

---

## ⚙️ Installation

To run this project, ensure you have Python installed. It is highly recommended to use a virtual environment or an Anaconda environment.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/flappy-bird-dqn.git
   cd flappy-bird-dqn
   ```

2. **Install required packages**:
   If using `pip` (outside Anaconda), run:
   ```bash
   pip install torch pygame-ce flappy-bird-gymnasium matplotlib gymnasium
   ```
   > 💡 **Note**: On modern Python versions (e.g., Python 3.12+), standard `pygame` might fail to compile. We recommend installing `pygame-ce` (Pygame Community Edition) which is a drop-in replacement.

---

## 🚀 How to Run

You can run the project in two ways: via the command-line launcher, or directly inside a Jupyter Notebook.

### Method 1: Interactive Menu (Terminal)
Run the central entry point file:
```bash
python flappy_bird.py
```
You will be prompted with the following menu:
```text
====================================================
       Flappy Bird Reinforcement Learning           
====================================================
1. Play Manually (Use SPACEBAR to flap)
2. Train the DQN Agent
3. Watch the Trained Agent Play
====================================================
Select an option (1-3): 
```

### Method 2: Jupyter Notebook
If you are working inside **Jupyter Lab** or **Jupyter Notebook**, you can import the scripts directly into code cells:

- **To Train the Agent**:
  ```python
  from train import train
  # Set render=True to watch the bird learn in real-time
  train(render=False)
  ```
- **To Watch the Trained Agent Play**:
  ```python
  from play import play
  play()
  ```
- **To Play Manually**:
  ```python
  from flappy_bird import run_manual_play
  run_manual_play()
  ```

---

## 🧠 Reinforcement Learning Details

### Observation Space (12 Dimensions)
The Gymnasium environment provides a continuous observation vector representing the physical state of the game:
1. Horizontal distance to the next pipe.
2. Vertical distance to the next top pipe.
3. Vertical distance to the next bottom pipe.
4. Horizontal distance to the next-next pipe.
5. Vertical distance to the next-next top pipe.
6. Vertical distance to the next-next bottom pipe.
7. Bird's vertical position.
8. Bird's vertical velocity.
9. Bird's rotation.
10. Horizontal distance to the last pipe.
11. Vertical distance to the last top pipe.
12. Vertical distance to the last bottom pipe.

### Action Space (2 Dimensions)
- `0`: Idle (let the bird fall)
- `1`: Flap (gain height)

## 📈 Results

During training, the agent's performance metrics are saved. You can check the convergence details in `training_performance.png` inside the root folder, which plots:
1. **Cumulative Episode Rewards** (with a 100-episode moving average).
2. **Episode Scores** (number of pipes successfully passed).

The best performing model weights are saved automatically as `best_flappy_bird_dqn.pt`.

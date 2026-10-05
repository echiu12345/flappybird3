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
   git clone https://github.com/echiu12345/flappy_bird.git
   cd flappy_bird
   ```

2. **Install required packages**:
   If using `pip` (outside Anaconda), run:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
   Tested with Python 3.12 on Windows. On macOS/Linux use `.venv/bin/python` instead.

---

## 🚀 How to Run

### 預測下一個動作（不開遊戲視窗）

使用已訓練的 DQN，輸入目前狀態，輸出「不拍翅」或「拍翅」及兩個動作的
Q 值。這個介面不啟動遊戲、不執行動作，也不更新模型。

```powershell
.\.venv\Scripts\python.exe predict.py --state-file example_state.json
```

預設載入本次已訓練的 `runs/dqn_trained/best_flappy_bird_dqn.pt`，也可以用
`--model <模型路徑>` 指定其他權重。`example_state.json` 是種子 42 初始畫面的
12 維正規化狀態範例；對這個固定輸入重複執行會得到相同預測。
實際使用時應傳入每一幀最新的 `state`，維度、順序與正規化方式必須和訓練一致。
若要從截圖輸入，還需要另做影像處理或重新訓練影像輸入模型。

輸出 `action=0` / `action_name=idle` 表示不拍翅；
`action=1` / `action_name=flap` 表示拍翅。選擇 Q 值較大的動作。
Q 值估計的是未來累積回報，不是成功機率或預測信心。

在 Python 程式內呼叫時，先載入一次模型，再重複傳入不同狀態：

```python
from predict import ActionPredictor

predictor = ActionPredictor()
# state 是環境 reset() 或 step() 回傳的正規化 observation
result = predictor.predict(state)
action = result["action"]
q_values = result["q_values"]
```

### DQN 訓練與評估（Windows PowerShell）

DQN 輸入是正規化的 12 維遊戲狀態，輸出為兩個動作的 Q 值：
`0` 不拍翅、`1` 拍翅。模型透過環境回饋學習每個動作的預期累積回報。
這是強化學習策略訓練，不需要事先準備標籤資料。

```powershell
# 訓練；不開遊戲視窗可提高速度
.\.venv\Scripts\python.exe train.py --episodes 10000 --seed 42 --output-dir runs\my_dqn

# 用不同於訓練的種子進行 20 回合評估，不使用隨機探索
.\.venv\Scripts\python.exe play.py --model runs\my_dqn\best_flappy_bird_dqn.pt --episodes 20 --no-render --output-json runs\my_dqn\evaluation.json

# 觀看模型玩遊戲
.\.venv\Scripts\python.exe play.py --model runs\my_dqn\best_flappy_bird_dqn.pt --episodes 5
```

每次訓練保存 `config.json`、逐回合 `training_metrics.csv`、
`training_performance.png`、`summary.json`、最佳與最終 `.pt` 權重。
未指定 `--output-dir` 時，自動建立 `runs/dqn_<timestamp>/`；已有訓練產物的
資料夾會拒絕覆寫。Ctrl+C 可保存當前權重與已完成回合的紀錄。
權重檔不含 optimizer 或 replay buffer，不能精確接續訓練。

預設先收集 1,000 步經驗，每 4 步更新一次；batch size 128、learning rate
`1e-4`、gamma `0.99`、tau `0.005`、replay capacity 50,000。
epsilon 從 `0.1` 開始，每回合乘 `0.999`，最低 `0.01`，以保留較長的探索期。
可用 `python train.py --help` 查看參數。每回合最多 10,000 步；時間限制
造成的截斷仍計算下一狀態的 Q 值，碰撞死亡才取消該值。

最佳模型依訓練時最近最多 100 回合的平均回報選擇；這個數值含探索，
應另外用 `play.py --no-render` 的平均通過水管數評估成果。短測試只能確認
流程可執行，不能證明模型已收斂。
`play.py` 未指定模型時載入最近一次訓練的最佳權重，並支援既有的 180 維
LIDAR 模型。新訓練明確使用 `use_lidar=False`。

### Command line：CPU 執行緒與全部核心模式

`train.py` 與 `predict.py` 支援 `--cpu-max`：強制使用 CPU，並將 PyTorch
計算執行緒數設為機器的邏輯 CPU 數。目前這台機器偵測到 8 個邏輯 CPU。

```powershell
# 動作預測，允許 PyTorch 使用全部邏輯 CPU
.\.venv\Scripts\python.exe predict.py --state-file example_state.json --cpu-max

# 訓練，同樣使用全部邏輯 CPU；預設不開遊戲視窗
.\.venv\Scripts\python.exe train.py --episodes 10000 --cpu-max

# 自行指定 CPU 計算執行緒數，例如 4
.\.venv\Scripts\python.exe train.py --episodes 10000 --device cpu --threads 4
```

`--threads` 與 `--cpu-max` 二選一；`--cpu-max` 可搭配 `--device cpu` 或
預設的 `auto`，不能搭配 GPU 裝置。訓練會顯示並保存執行緒數；預測結果的
`runtime` 會顯示裝置、計算執行緒數與邏輯 CPU 數。

全部核心模式代表允許模型使用所有邏輯 CPU，不保證整台電腦的 CPU 使用率
持續達到 100%。目前網路很小，單筆預測與單一遊戲環境的工作量有限，
增加執行緒可能造成額外開銷。應以每秒訓練步數／預測耗時比較速度。
參考 [PyTorch 計算執行緒設定](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)。

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
1–3. Horizontal position and upper/lower pipe edges of the leftmost pipe.
4–6. Horizontal position and upper/lower edges of the middle pipe.
7–9. Horizontal position and upper/lower edges of the rightmost pipe.
10. Bird's vertical position.
11. Bird's vertical velocity.
12. Bird's rotation.

The environment sorts the three pipe positions horizontally and normalizes these
features. See the [environment implementation](https://github.com/markub3327/flappy-bird-gymnasium/blob/main/flappy_bird_gymnasium/envs/flappy_bird_env.py).

### Action Space (2 Dimensions)
- `0`: Idle (let the bird fall)
- `1`: Flap (gain height)

## 📈 Results

During training, the agent's performance metrics are saved. You can check the convergence details in `training_performance.png` inside the root folder, which plots:
1. **Cumulative Episode Rewards** (with a 100-episode moving average).
2. **Episode Scores** (number of pipes successfully passed).

The best performing model weights are saved as `best_flappy_bird_dqn.pt` inside
the run directory. Existing model files in the repository root are preserved.

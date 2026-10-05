"""Predict the next Flappy Bird action from a normalized observation."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from agent import DQNAgent
from runtime import configure_runtime


DEFAULT_MODEL = Path(__file__).resolve().parent / "runs" / "dqn_trained" / "best_flappy_bird_dqn.pt"


class ActionPredictor:
    """Load once, then predict each new observation without running the game."""

    def __init__(self, model_path=DEFAULT_MODEL, device_name=None):
        self.agent = DQNAgent.from_checkpoint(model_path, device_name=device_name)
        if self.agent.action_dim != 2:
            raise ValueError("Flappy Bird prediction requires two actions: idle and flap.")
        self.state_dim = self.agent.state_dim

    def predict(self, state):
        try:
            observation = np.asarray(state, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("State must be a numeric observation array.") from error
        if observation.shape != (self.state_dim,):
            raise ValueError(
                f"Expected {self.state_dim} normalized state values; got shape {observation.shape}."
            )
        if not np.isfinite(observation).all():
            raise ValueError("State values must be finite numbers (no NaN or infinity).")
        tensor = torch.as_tensor(observation, device=self.agent.device).unsqueeze(0)
        with torch.inference_mode():
            q_values = self.agent.policy_dqn(tensor)[0].cpu()
        action = int(q_values.argmax().item())
        return {
            "action": action,
            "action_name": "flap" if action else "idle",
            "q_values": {
                "idle": float(q_values[0].item()),
                "flap": float(q_values[1].item()),
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    cpu_options = parser.add_mutually_exclusive_group()
    cpu_options.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (default: 1)")
    cpu_options.add_argument("--cpu-max", action="store_true", help="Force CPU and use all logical CPUs for PyTorch operations")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--state", type=float, nargs="+", help="Normalized observation values in environment order")
    inputs.add_argument("--state-file", type=Path, help="JSON file containing a normalized observation array")
    args = parser.parse_args()
    try:
        device_name = None if args.device == "auto" else args.device
        threads, device_name = configure_runtime(args.threads, args.cpu_max, device_name)
        state = json.loads(args.state_file.read_text(encoding="utf-8")) if args.state_file else args.state
        predictor = ActionPredictor(args.model, device_name=device_name)
        result = predictor.predict(state)
        result["runtime"] = {
            "device": str(predictor.agent.device),
            "cpu_threads": threads,
            "logical_cpus": os.cpu_count() or 1,
        }
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

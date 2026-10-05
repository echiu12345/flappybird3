"""Checks for Bellman targets, environment settings, and saved model compatibility."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from agent import DQNAgent
from environment import make_env
from train import train


class CaptureLoss(nn.Module):
    def forward(self, current, target):
        self.targets = target.detach().cpu()
        return nn.functional.mse_loss(current, target)


class DQNTests(unittest.TestCase):
    def test_explicit_cpu_overrides_automatic_gpu_selection(self):
        with patch("agent.device", torch.device("cuda")):
            agent = DQNAgent(12, 2, hidden_dim=8, device_name="cpu")
        self.assertEqual(agent.device.type, "cpu")
        self.assertTrue(all(parameter.device.type == "cpu"
                            for parameter in agent.policy_dqn.parameters()))
        self.assertTrue(all(parameter.device.type == "cpu"
                            for parameter in agent.target_dqn.parameters()))
        self.assertIn(agent.select_action(np.zeros(12)), (0, 1))

    def test_time_limit_keeps_nonterminal_observation(self):
        env = make_env(max_steps=1)
        try:
            state, _ = env.reset(seed=42)
            self.assertEqual(state.shape, (12,))
            self.assertEqual(env.action_space.n, 2)
            _, _, terminated, truncated, _ = env.step(0)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_bellman_targets_bootstrap_only_nonterminal_transitions(self):
        agent = DQNAgent(12, 2, hidden_dim=8, gamma=0.5)
        with torch.no_grad():
            for parameter in agent.policy_dqn.parameters():
                parameter.zero_()
            for parameter in agent.target_dqn.parameters():
                parameter.zero_()
            agent.target_dqn.model[4].bias.fill_(2.0)
        state = np.zeros(12, dtype=np.float32)
        agent.step(state, 0, 1.0, state, True)
        agent.step(state, 1, 1.0, state, False)
        capture = CaptureLoss()
        agent.loss_fn = capture
        loss = agent.optimize(2)
        self.assertEqual(sorted(capture.targets.flatten().tolist()), [1.0, 2.0])
        self.assertAlmostEqual(loss, 2.5)

    def test_replay_owns_observation_values(self):
        agent = DQNAgent(12, 2, hidden_dim=8)
        state = np.zeros(12)
        agent.step(state, 0, 0.1, state, False)
        state[:] = 99
        before, _, _, after, _ = agent.memory.sample(1)[0]
        np.testing.assert_array_equal(before, np.zeros(12))
        np.testing.assert_array_equal(after, np.zeros(12))

    def test_legacy_lidar_checkpoint_dimensions_and_predictions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "model.pt"
            agent = DQNAgent(180, 2, hidden_dim=16)
            agent.save(path)
            loaded = DQNAgent.from_checkpoint(path)
            self.assertEqual((loaded.state_dim, loaded.action_dim), (180, 2))
            state = torch.ones(1, 180, device=loaded.device)
            torch.testing.assert_close(agent.policy_dqn(state), loaded.policy_dqn(state))
            env = make_env(state_dim=loaded.state_dim)
            try:
                observation, _ = env.reset(seed=42)
                self.assertEqual(observation.shape, (180,))
                self.assertIn(loaded.select_action(observation), (0, 1))
            finally:
                env.close()

    def test_training_refuses_to_overwrite_existing_weights(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "existing.pt"
            path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                train(num_episodes=1, output_dir=directory)
            self.assertEqual(path.read_bytes(), b"preserve")


if __name__ == "__main__":
    unittest.main()

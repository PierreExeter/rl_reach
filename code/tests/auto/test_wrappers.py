""" Automated tests for gymnasium wrappers using check_env """

import sys
from pathlib import Path

# Add code/utils directory to path for imports
code_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(code_dir / "utils"))

import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from wrappers import DoneOnSuccessWrapper


class MockEnv(gym.Env):
    """Mock environment for testing wrappers."""

    def __init__(self, success_on_step=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.success_on_step = success_on_step
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        # Use seeded RNG for deterministic observations
        obs = self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
        ).astype(np.float32)
        return obs, {}

    def step(self, action):
        self.current_step += 1
        # Use seeded RNG for deterministic observations
        obs = self.np_random.uniform(
            low=self.observation_space.low,
            high=self.observation_space.high,
        ).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        is_success = (
            self.success_on_step is not None
            and self.current_step >= self.success_on_step
        )
        info = {"is_success": is_success}
        return obs, reward, terminated, truncated, info


class TestDoneOnSuccessWrapper:
    """Tests for DoneOnSuccessWrapper."""

    def test_check_env(self):
        """Verify wrapper passes gymnasium's check_env validation."""
        env = DoneOnSuccessWrapper(MockEnv(), n_successes=1)
        check_env(env, skip_render_check=True)

    def test_terminated_on_success(self):
        """Test that terminated=True when success threshold is reached."""
        env = DoneOnSuccessWrapper(MockEnv(success_on_step=1), n_successes=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        assert terminated is True, f"Expected terminated=True on success, got {terminated}"
        assert truncated is False, f"Expected truncated=False, got {truncated}"

    def test_not_terminated_before_success(self):
        """Test that terminated=False before success threshold."""
        env = DoneOnSuccessWrapper(MockEnv(success_on_step=3), n_successes=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        assert terminated is False, f"Expected terminated=False, got {terminated}"
        assert truncated is False, f"Expected truncated=False, got {truncated}"

    def test_n_successes_threshold(self):
        """Test that n_successes consecutive successes are required."""
        env = DoneOnSuccessWrapper(MockEnv(success_on_step=1), n_successes=3)
        env.reset()

        # First two successes should not terminate
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(np.zeros(2))
            assert terminated is False

        # Third success should terminate
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert terminated is True

    def test_success_counter_resets_on_failure(self):
        """Test that success counter resets when is_success becomes False."""
        class SuccessFailEnv(MockEnv):
            def step(self, action):
                self.current_step += 1
                obs = self.observation_space.sample()
                # Success on steps 1, 2, then fail on 3, then success on 4, 5, 6
                is_success = self.current_step != 3
                return obs, 0.0, False, False, {"is_success": is_success}

        env = DoneOnSuccessWrapper(SuccessFailEnv(), n_successes=3)
        env.reset()

        # Steps 1, 2: success (count=1, 2)
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        assert env.current_successes == 2

        # Step 3: failure (count resets to 0)
        env.step(np.zeros(2))
        assert env.current_successes == 0

        # Steps 4, 5: success (count=1, 2)
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        assert env.current_successes == 2

        # Step 6: success (count=3, should terminate)
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        assert terminated is True

    def test_preserves_truncated_from_underlying_env(self):
        """Test that truncated value from underlying env is preserved."""
        class TruncatingEnv(MockEnv):
            def step(self, action):
                obs, reward, terminated, truncated, info = super().step(action)
                truncated = True  # Simulate time limit truncation
                return obs, reward, terminated, truncated, info

        env = DoneOnSuccessWrapper(TruncatingEnv(), n_successes=1)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        assert truncated is True, f"Expected truncated=True from underlying env, got {truncated}"

    def test_reward_offset(self):
        """Test that reward offset is applied correctly."""
        env = DoneOnSuccessWrapper(MockEnv(), reward_offset=5.0)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))

        assert reward == 5.0, f"Expected reward=5.0 (0.0 + offset), got {reward}"

    def test_reset_clears_success_counter(self):
        """Test that reset clears the success counter."""
        env = DoneOnSuccessWrapper(MockEnv(success_on_step=1), n_successes=3)
        env.reset()

        # Build up success counter
        env.step(np.zeros(2))
        env.step(np.zeros(2))
        assert env.current_successes == 2

        # Reset should clear counter
        env.reset()
        assert env.current_successes == 0

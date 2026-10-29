"""Optional Gymnasium adapter over the real-time session."""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TouhouEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.action_space = spaces.Discrete(9)
        self.observation_space = spaces.Dict({
            "player": spaces.Box(-1, 1, (4,), dtype=np.float32),
            "bullets": spaces.Box(-1, 1, (session.capacity, 7), dtype=np.float32),
        })

    @staticmethod
    def _arrays(obs):
        return {key: np.asarray(value, dtype=np.float32) for key, value in obs.items()}

    def reset(self, seed=None, options=None):
        # This seeds Gymnasium only. Native game RNG control is not available.
        super().reset(seed=seed)
        obs, info = self.session.reset()
        return self._arrays(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.session.step(action)
        return self._arrays(obs), reward, terminated, truncated, info

    def close(self):
        self.session.close()

"""Experimental coherent laser observation adapter, not a training entrypoint."""
import gymnasium as gym
import numpy as np

from agent.laser_policy import LASER_CAPACITY, LASER_FIELDS, LASER_OBSERVATION_CONTRACT, encode_lasers
from .laser_probe import capture, require_layout
from .session import GameSession
from .touhou_env import TouhouEnv, state_space


class LaserSnapshot:
    def __init__(self, sample, lasers):
        self.sample,self.lasers=sample,lasers
    def __getattr__(self,name):
        return getattr(self.sample,name)


class LaserReader:
    """Wrap existing snapshots with the pool captured under the same frame guard."""
    def __init__(self, reader, build):
        require_layout(reader,build)
        self.reader=reader
    def __getattr__(self,name):
        return getattr(self.reader,name)
    def snapshot(self):
        sample,record=capture(self.reader)
        return LaserSnapshot(sample,record['lasers'])


def laser_state_space(capacity):
    spaces=dict(state_space(capacity).spaces)
    spaces['lasers']=gym.spaces.Box(-np.inf,np.inf,(LASER_CAPACITY,LASER_FIELDS),dtype=np.float32)
    return gym.spaces.Dict(spaces)


class LaserSession(GameSession):
    """Include laser rows from the exact snapshot used for each transition."""
    def _with_lasers(self, observation, info):
        try:
            observation['lasers']=encode_lasers(self.sample.lasers)
            info['laser_count']=len(self.sample.lasers)
            info['laser_observation_contract']=LASER_OBSERVATION_CONTRACT
            return observation,info
        except BaseException:
            self.active=False
            self.controls.release_all()
            raise
    def reset(self):
        obs,info=super().reset()
        return self._with_lasers(obs,info)
    def step(self, action):
        obs,reward,done,truncated,info=super().step(action)
        obs,info=self._with_lasers(obs,info)
        return obs,reward,done,truncated,info


class LaserTouhouEnv(TouhouEnv):
    def __init__(self,session):
        super().__init__(session)
        self.observation_space=laser_state_space(session.capacity)

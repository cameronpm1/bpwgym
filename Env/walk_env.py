""" Module implements environment for running task"""
import multiprocessing as mp
import pickle
from collections import OrderedDict
from typing import Dict, Optional

import gym
import time
import numpy as np
from gym import spaces
from gym.utils import seeding
from Sim.robot import Robot

from Env import BPWEnv

class ReorientEnv(BPWEnv):

    """Environment for running task"""

    def __init__(
        self,
        *,
        rew_clip_range=(0, 2),
        **kwargs
    ):
        """
        Args:
            axis: desired axis of rotation
            min_num_contacts: minimum number of contacts to provide reward
                rotation about desired axis
            max_tilt: maximum tilt about desired axis
            rew_clip_range: reward clipping to avoid idiosyncratic behaviors

        """
        super().__init__(**kwargs)
        self._rew_clip_range = rew_clip_range
        self._np_random = dict(axis=np.random.default_rng())
        self._reset_goal_axis()

    def _reward(self):
        vel = self._obs['vel']
        return np.clip(vel, *self._rew_clip_range)

    def step(self, action):
        obs, rew, done, dict = super().step(action)
        if not done:
            done,pos = self.sim.check_state(self._obs)
        return obs, rew, done, {"done": done, "reward": rew, "action": action}
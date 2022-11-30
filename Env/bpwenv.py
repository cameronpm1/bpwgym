"""base class for Bipedal Walker"""
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


class BPWEnv(gym.Env):
    """Environment for Bipedal Walkers"""

    def __init__(
        self,
        sim: Robot,
        step_duration: float = 0.05,
        max_episode_length: int = 5000,
        max_dq: float or np.ndarray = 1.0,
        action_scale: float = 0.2
    ):

        """
        Args:
            sim: Robot simulation object to be wrapped
            step_duration: Simulation duration for each step
            max_episode_length: Episode forced to terminate
        """

        # Simulation
        self.sim = sim
        self.step_duration = step_duration
        self.max_episode_length = max_episode_length
        self.action_scale = action_scale
        self._step = 0
        self._obs = None
        self.action = None

    @property
    def action_dim(self) -> int:
        """Action dimension"""
        return len(self.sim.servos)

    @property
    def action_space(self) -> gym.Space:

        return spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Dict:
        """Observation space

        Return the observation space consisting of hand, object and
        contact information

        """
        obs = self._get_obs()
        space = {}
        for key, val in obs.items():
            space[key] = spaces.Box(low=-np.inf, high=np.inf, shape=val.shape)

        return spaces.Dict(space)

    def reset(self):
        self._step = 0
        self.sim.reset()
        self._obs = self._get_obs()
        self.action = []
        for i,joint in enumerate(self.sim.servos):
            self.action.append(self._obs[joint]['qpos'])
        return self._obs

    def render(self):
        return self.sim.render()

    def seed(self, seed=None):
        # Save the seed so we can re-seed during un-pickling
        self._seed = seed

        # Hash the seed to avoid any correlations
        seed = seeding.hash_seed(seed)

        # Seed environment components with randomness
        seeds = [seed]
        seeds.extend(self.sim.seed(seed))

        return seeds

    
    def step(self, action):
        angles = {}
        for i,joint in enumerate(self.sim.servos):
            angles[joint] = self.action[i] + action[i]*self.action_scale
            self.action[i] += action[i]
        sim_angles = self.sim.rel_to_sim(angles)
        self.sim.step(sim_angles)
        self._obs = self._get_obs()
        self._step += 1
        if self._step > self.max_episode_length:
            done = True
        rew = self._reward()
        return self._obs, rew, done, {"done": done, "reward": rew}

    def _get_obs(self) -> OrderedDict:
        """Return observation

        For attributes which can be noisy ex. hand_joint_position,
        contact_position, contact_normal etc,. both the "accurate" and
        noisy versions are inluded in the observation dictionary. The
        noisy version is the one with suffix "_noise". Helpful towards
        using assymmetric actor-critic architectures.

        """

        data = self.sim.sim_to_rel()

        obs = OrderedDict()

        # Hand
        obs['joint011'] = data['joint011'].copy()
        obs['joint031'] = data['joint031'].copy()
        obs['joint00'] = data['joint00'].copy()
        obs['joint111'] = data['joint111'].copy()
        obs['joint131'] = data['joint131'].copy()
        obs['joint10'] = data['joint10'].copy()
        obs['vel'] = self.sim._state['vel'].copy()
        
        return obs

    def _reward(self):
        return 0.0

def main():
    robot1 = Robot('/home/ubuntu/robot/Sim/robot.xml',0.05)
    env = BPWEnv(robot1,0.5)
    #[-79.937, -148.61, -90.0, -79.937, -148.61, -90.0]
    i=0
    obs = env.reset()
    done = False
    
    while i < 5000:
        if i == 0: time.sleep(1)
        env.render()
        if i > 100 and i < 240:
            obs, rew, done, info = env.step([0,0.15,0,0,0.15,0])
            
        else:
            obs, rew, done, info = env.step([0,0,0,0,0,0])
        #print(env.sim._state['joint032']['qpos'])
        print(env.sim._state['joint012']['qpos'])
        time.sleep(0.0001)
        i += 1

if __name__ == "__main__":
    main()
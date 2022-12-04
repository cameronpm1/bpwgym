import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from Sim.sim import Sim
import matplotlib.pyplot as plt
from biped import matrix_to_RPY, para_period
# import os
# import time
# from stable_baselines3.common.utils import set_random_seed
# from vec_env_utils import make_vec_env
# from robot import Robot
# from arm_dynamics import ArmDynamics
# from stable_baselines3 import PPO


class BipEnv(gym.Env):
    def __init__(self, Bipedal: Sim):
        self.robot = Bipedal
        self.period = 64
        self.render_flag = False
        self.jnames = ['root',
                       'Joint11', 'Joint21', 'Joint31', 'Joint32',
                       'Joint12', 'Joint22', 'Joint33', 'Joint34',
                       'Joint13', 'Joint23', 'Joint35', 'Joint36',
                       'Joint14', 'Joint24', 'Joint37', 'Joint38',
                       'Joint15', 'Joint25', 'Joint39', 'Joint310',
                       'Joint16', 'Joint26', 'Joint311', 'Joint312'
                       ]
        self.servos = ['Joint11', 'Joint12', 'Joint13', 'Joint14', 'Joint15', 'Joint16']
        self.robot.set_state(joints=len(self.jnames), jnames=self.jnames)
        # self.state = None
        self.site_pos = self.robot._data.site_xpos
        self.site_ori = self.robot._data.site_xmat
        self.n_step = 0
        self.epoch_p = 20

    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_obs(self):
        rpy = matrix_to_RPY(self.site_ori)
        return {"pos": self.site_pos, "rpy": rpy}

    def reset(self, *, seed: int = None, options: dict = None):
        self.robot.reset()
        self.n_step = 0
        return self.get_obs()

    def get_reward(self):
        pos = self.get_obs()["pos"][0]
        x, y, z = pos[0], pos[1], pos[2]
        return y

    def step(self, action):
        # Periodicity
        self.n_step += 1
        period_a = para_period(para=action)
        for sid in range(self.period):
            ctrl = period_a[:, sid]
            self.robot.set_ctrl(ctrl)
            self.robot.step()
            if self.render_flag:
                self.robot.render()

        obs = self.get_obs()
        reward = self.get_reward()
        done = False
        rpy = obs["rpy"]
        print(rpy)
        if abs(rpy[0]) > 0.6 or abs(rpy[1]) > 0.6:
            print("Turn over")
            done = True

        if self.n_step == self.epoch_p:
            done = True

        return obs, reward, done, {"done": done, "reward": reward}


if __name__ == "__main__":
    bipedal = Sim(model='./urdf/bluebody-urdf.xml', dt=0.01)
    env = BipEnv(bipedal)
    env.render_flag = True
    done_flag = False
    for _ in range(100):
        a = np.random.rand(3, 6)
        print("--------")
        while not done_flag:
            obs, rew, done_flag, _ = env.step(action=a)
            print(rew)
        env.reset()
        done_flag = False

import numpy as np
from gym.wrappers import FilterObservation, FlattenObservation
from stable_baselines3.common.utils import set_random_seed

from Sim.robot import Robot
from Env.WalkEnv import WalkEnv


def make_env(cfg):

    sim = Robot(
        model=cfg["sim"]["dir"],
        dt=cfg["sim"]["dt"],
    )

    env = WalkEnv(
            sim=sim,
            step_duration=cfg["env"]["step_duration"],
            max_episode_length=cfg["env"]["max_episode_length"],
            action_scale=cfg["env"]["action_scale"]
        )

    #filter_keys=[
    #    "joint011",
    #    "joint031",
    #    "joint00",
    #    "joint111",
    #    "joint131",
    #    "joint10",
    #    "vel",
    #]

    #env = FilterObservation(
    #    env,
    #    filter_keys=filter_keys,
    #)
    
    env = FlattenObservation(env)
    env.seed(cfg["seed"])
    return env

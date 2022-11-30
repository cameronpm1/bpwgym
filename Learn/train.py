import os
import hydra
import numpy as np
import torch
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import SAC

from make_env import make_env
from subproc_vec_env_no_daemon import SubprocVecEnvNoDaemon


@hydra.main(config_path=".", config_name="config", version_base='1.1')
def train(cfg: DictConfig):
    set_random_seed(cfg["seed"])
    logdir = os.getcwd()
    print("current directory:", logdir)

    # make parallel envs
    def env_fn(seed):
        env = make_env(cfg)
        env.seed(seed)
        return env

    class EnvMaker:
        def __init__(self, seed):
            self.seed = seed

        def __call__(self):
            return env_fn(self.seed)

    def make_vec_env(nenvs, seed):
        envs = VecMonitor(
            SubprocVecEnvNoDaemon([EnvMaker(seed + i) for i in range(nenvs)])
        )
        return envs

    env = make_vec_env(cfg["algo"]["nenv"], cfg["seed"])

    # define policy network size
    policy_kwargs = dict(net_arch=dict(pi=cfg["algo"]["pi"], qf=cfg["algo"]["qf"]))
    

    model = SAC(
            "MlpPolicy", 
            env, 
            verbose=2, 
            tensorboard_log=logdir,
            learning_rate=cfg["algo"]["lr"], 
            learning_starts=cfg["algo"]["learning_starts"],
            batch_size=cfg["algo"]["batch_size"],
            tau=cfg["algo"]["tau"],
            gamma=cfg["algo"]["gamma"],
            train_freq=cfg["algo"]["train_freq"], 
            gradient_steps=cfg["algo"]["gradient_steps"],
            action_noise=NormalActionNoise(0, np.array([cfg["algo"]["action_noise_std"]] * 6)),
            ent_coef=cfg["algo"]["ent_coef"],
            target_update_interval=cfg["algo"]["target_update_interval"],
            target_entropy="auto",
            policy_kwargs=policy_kwargs,
            seed=cfg["seed"],
            device=cfg["algo"]["device"],
            )

    model.learn(
            total_timesteps=cfg["algo"]["total_timesteps"]
            )
            
    env.close()
    model.save(os.path.join(logdir, "model"))
    

if __name__ == "__main__":
    torch.set_num_threads(9)
    train()

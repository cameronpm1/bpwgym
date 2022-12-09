import os
import time
from stable_baselines3.common.utils import set_random_seed
from bienv import BipEnv
from stable_baselines3 import PPO
from Sim.sim import Sim


def train():
    save_dir = "log/m4"
    bipedal = Sim(model='./urdf/bluebody-urdf.xml', dt=0.01)
    vec_env = BipEnv(bipedal)
    # vec_env.render_flag = True
    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        batch_size=2048,
        tensorboard_log="log2/",
    )
    model.learn(total_timesteps=1000000)
    model.save(save_dir)


if __name__ == "__main__":
    train()

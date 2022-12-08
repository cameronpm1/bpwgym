import numpy as np
from stable_baselines3 import PPO
from Sim.sim import Sim
from bienv import BipEnv

model_path = "log/m2.zip"
policy = PPO.load(model_path)
bipedal = Sim(model='./urdf/bluebody-urdf.xml', dt=0.01)
vec_env = BipEnv(bipedal)
obs = vec_env.reset()
vec_env.render_flag = True

while True:
    action, _states = policy.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
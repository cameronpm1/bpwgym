import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco.glfw import glfw
import os
import time
from Sim.sim import Sim

def matrix_to_RPY(Rm):
    Rm = Rm.reshape(3, 3)
    yaw = np.arctan2(Rm[1, 0], Rm[0, 0])
    pitch = np.arctan2(-Rm[2, 0], np.sqrt(Rm[2, 1] ** 2 + Rm[2, 2] ** 2))
    roll = np.arctan2(Rm[2, 1], Rm[2, 2])
    return np.array([roll, pitch, yaw])

def para_period(para, gait_points=64, check_plot=False):
    X = [i for i in range(gait_points)]
    Y = np.array([
        [para[0, 0] * np.sin((x / 32. + 2 * para[0, 1]) * np.pi) +
         para[0, 2] * np.sin((x / 16. + 2 * para[0, 3]) * np.pi) +
         para[0, 4] * np.sin((x / 8. + 2 * para[0, 5]) * np.pi)
         for x in X],
        [para[1, 0] * np.sin((x / 32. + 2 * para[1, 1]) * np.pi) +
         para[1, 2] * np.sin((x / 16. + 2 * para[1, 3]) * np.pi) +
         para[1, 4] * np.sin((x / 8. + 2 * para[1, 5]) * np.pi)
         for x in X],
        [para[2, 0] * np.sin((x / 32. + 2 * para[2, 1]) * np.pi) +
         para[2, 2] * np.sin((x / 16. + 2 * para[2, 3]) * np.pi) +
         para[2, 4] * np.sin((x / 8. + 2 * para[2, 5]) * np.pi)
         for x in X],
    ]) / 2.

    if check_plot:
        plt.figure()
        plt.plot(Y[0])
        plt.plot(Y[1])
        plt.plot(Y[2])
        plt.show()

    gait_f = np.concatenate((Y, np.flip(Y, 1)))
    return gait_f

class Biped(Sim):
    def __init__(self, model: str, dt: float):
        super().__init__(model, dt)
        self.loop = 64
        self.jnames = ['root',
                       'Joint11', 'Joint21', 'Joint31', 'Joint32',
                       'Joint12', 'Joint22', 'Joint33', 'Joint34',
                       'Joint13', 'Joint23', 'Joint35', 'Joint36',
                       'Joint14', 'Joint24', 'Joint37', 'Joint38',
                       'Joint15', 'Joint25', 'Joint39', 'Joint310',
                       'Joint16', 'Joint26', 'Joint311', 'Joint312'
                       ]
        self.servos = ['Joint11', 'Joint12', 'Joint13', 'Joint14', 'Joint15', 'Joint16']
        self.set_state(joints=len(self.jnames), jnames=self.jnames)
        self.site_pos = self._data.site_xpos
        self.site_ori = self._data.site_xmat

    def run_robot(self, gait_function, steps, render_flag=False):
        for s in range(steps):
            for i in range(self.loop):
                ctrl = gait_function[:, i]
                self.set_ctrl(ctrl)
                self.step()
                if render_flag:
                    self.render()

                # print(self.get_state())
            print(self.site_pos)
            print(matrix_to_RPY(self.site_ori))


def random_generate_sin_gait(gait_points=64, check_plot=False):
    para = np.random.rand(3, 6)
    X = [i for i in range(gait_points)]

    Y = np.array([
        [para[0, 0] * np.sin((x / 32. + 2 * para[0, 1]) * np.pi) +
         para[0, 2] * np.sin((x / 16. + 2 * para[0, 3]) * np.pi) +
         para[0, 4] * np.sin((x / 8. + 2 * para[0, 5]) * np.pi)
         for x in X],
        [para[1, 0] * np.sin((x / 32. + 2 * para[1, 1]) * np.pi) +
         para[1, 2] * np.sin((x / 16. + 2 * para[1, 3]) * np.pi) +
         para[1, 4] * np.sin((x / 8. + 2 * para[1, 5]) * np.pi)
         for x in X],
        [para[2, 0] * np.sin((x / 32. + 2 * para[2, 1]) * np.pi) +
         para[2, 2] * np.sin((x / 16. + 2 * para[2, 3]) * np.pi) +
         para[2, 4] * np.sin((x / 8. + 2 * para[2, 5]) * np.pi)
         for x in X],
    ]) / 2.

    if check_plot:
        plt.figure()
        plt.plot(Y[0])
        plt.plot(Y[1])
        plt.plot(Y[2])
        plt.show()

    return Y


if __name__ == "__main__":
    gait_f = random_generate_sin_gait(check_plot=False)
    # gait_f = np.loadtxt("IK_angles/circle02.csv").reshape(3,64)
    print(gait_f.shape)
    gait_f = np.concatenate((gait_f, np.flip(gait_f, 1)))
    print(gait_f.shape)

    bipedal = Biped(model='./urdf/bluebody-urdf.xml', dt=0.01)
    bipedal.run_robot(gait_function=gait_f, steps=10, render_flag=True)

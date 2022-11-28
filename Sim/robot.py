from util import *
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Union, Dict, List, Optional

from sim import Sim

class robot(Sim):
    def __init__(
        self,
        model: str,
        dt: float,
    ) -> None:
        super().__init__(model=model, dt=dt)
        self.servo_min = [-105, -180, -100, -105, -180, -100]
        self.servo_max = [-15, -50, 0, -15, -20, 0]
        self.front_leg_servos = [0, 3]
        self.back_leg_servos = [1, 4]
        self.ftl = 63  # front top leg
        self.ble = 30  # back leg extension
        self.bl = 90  # back leg
        self.fbl = 68 # front bottom leg
        self.fbls = 50  # front bottom leg segment
        self.servo_back_pos = [-53.5,-6.45]
        self.init_pos = np.array([-40,-107])
        self.jnames = ['root','joint00','joint011','joint012','joint014','joint013','joint021','joint022','joint031','joint032',
                        'joint10','joint111','joint112','joint114','joint113','joint121','joint122','joint131','joint132']
        self.servos = ['joint011','joint031','joint00','joint111','joint131','joint10']

    def set_state(self) -> dict:
        state = super().set_state(joints=len(self.jnames), jnames=self.jnames)
        self._state['vel'] = np.sqrt(state['root']['qvel'][0]**2+state['root']['qvel'][1]**2)
        return self._state

    
    def get_state(self) -> dict:
        state = super().get_state()
        self._state['vel'] = np.sqrt(state['root']['qvel'][0]**2+state['root']['qvel'][1]**2)
        return self._state

    def sim_to_rel(self) -> dict:
        """
        Take the qpos of actuated joints (radians) and return angles realtive to robot (degrees)
        """
        angles = {}
        angles['joint00'] = -90 - np.degrees(self._state['joint00']['qpos'][0])
        angles['joint10'] = -90 + np.degrees(self._state['joint10']['qpos'][0])
        angles['joint011'] = -79.937 - np.degrees(self._state['joint011']['qpos'][0])
        angles['joint031'] = -148.610 - np.degrees(self._state['joint011']['qpos'][0])
        angles['joint111'] = -79.937 - np.degrees(self._state['joint011']['qpos'][0])
        angles['joint131'] = -148.610 - np.degrees(self._state['joint011']['qpos'][0])

        return angles

    def rel_to_sim(self,
        angles: dict,
    ) -> bool:
        angles_sim = {}
        angles_sim['joint00'] = np.radians(-90 - angles['joint00'])
        angles_sim['joint10'] = np.radians(angles['joint10'] + 90)
        angles_sim['joint011'] = np.radians(-79.937 - angles['joint011'])
        angles_sim['joint031'] = np.radians(-148.610 - angles['joint031'])
        angles_sim['joint111'] = np.radians(-79.937 - angles['joint111'])
        angles_sim['joint131'] = np.radians(-148.610 - angles['joint131'])

        return angles_sim

    def actuate(self,
        angles: dict,
    ):
        ctrl = []
        ctrl.append(angles['joint011'])
        ctrl.append(angles['joint031'])
        ctrl.append(angles['joint00'])
        ctrl.append(angles['joint111'])
        ctrl.append(angles['joint131'])
        ctrl.append(angles['joint10'])
        ctrl.append(angles['joint011'])
        ctrl.append(angles['joint111'])
        self.set_ctrl(ctrl)


    def check_angles(self,
        angles: dict,
    ) -> bool:
        for i,joint in enumerate(self.servos):
            if angles[joint] < self.servo_min[i] or angles[joint] > self.servo_max[i]:
                return False
        return True

    def check_kinematics(self,
        angles: dict,
    ) -> bool:
        foot_pos_left = self.forward_kinematics(angles['joint011'],angles['joint031'])
        foot_pos_right = self.forward_kinematics(angles['joint111'],angles['joint131'])
        if foot_pos_left is None or foot_pos_right is None:
            return False
        else:
            return True

    def check_fall(self) -> bool:
        if self._state['root']['qpos'][2] < 0.1:
            return True
        else:
            return False

    def front_servo_pos(self, angle):
        return (self.ftl*np.cos(math.radians(angle)),self.ftl*np.sin(math.radians(angle)))

    def back_servo_pos(self, angle):
        return (self.ble*np.cos(math.radians(angle)) + self.servo_back_pos[0], self.ble*np.sin(math.radians(angle)) + self.servo_back_pos[1])

    def calc_joint_pos(self,back_pos,front_pos):
        return calcIntersection(back_pos[0],back_pos[1],front_pos[0],front_pos[1],self.bl,self.fbls)

    def inverse_kinematics(self,pos):
        #angle1 - front servo angle
        #angle2 - back servo angle
        #RETURNS DEGREES
        x = []
        y = []
        intersections = calcIntersection(0, 0, pos[0], pos[1], self.ftl, self.fbl)
        if intersections[0] > intersections[2]:
            intersection = (intersections[0],intersections[1])
        else:
            intersection = (intersections[2], intersections[3])
        angle1 = -np.arctan(abs(intersection[1]) / abs(intersection[0]))
        knee_pos = intersection
        if intersection[0] < 0:
            angle1 = -np.pi - angle1
        foot_angle = np.arctan(abs(intersection[1]-pos[1])/abs(intersection[0]-pos[0]))
        joint_pos = [(self.fbl-self.fbls)*np.cos(foot_angle)+pos[0],(self.fbl-self.fbls)*np.sin(foot_angle)+pos[1]]
        intersections = calcIntersection(joint_pos[0],joint_pos[1],self.servo_back_pos[0],self.servo_back_pos[1],self.bl,self.ble)
        if intersections[1] < intersections[3]:
            intersection = (intersections[0],intersections[1])
        else:
            intersection = (intersections[2], intersections[3])
        angle2 = -np.arctan(abs(intersection[1]-self.servo_back_pos[1])/abs(intersection[0]-self.servo_back_pos[0]))
        if intersection[1] > 0:
            angle2 -= np.pi
        elif intersection[0] < self.servo_back_pos[0]:
            angle2 = -np.pi - angle2
        return (np.degrees(angle1),np.degrees(angle2)),{'joint pos':joint_pos, 'foot pos':pos, 'knee pos':knee_pos}

    def forward_kinematics(self, angle1,angle2):
        #INPUT AS DEGREES
        front_link_pos = self.front_servo_pos(angle1)
        back_link_pos = self.back_servo_pos(angle2)
        joints = self.calc_joint_pos(back_link_pos,front_link_pos)
        if joints is None:
            return None
        if joints[1] < joints[3]:
            joint = [joints[0],joints[1]]
        else:
            joint = [joints[2],joints[3]]
        vec_to_foot = np.array(joint) - np.array([front_link_pos[0],front_link_pos[1]])
        vec_to_foot /= self.fbls#np.linalg.norm(vec_to_foot)
        vec_to_foot *= self.fbl-self.fbls
        foot_pos = joint + vec_to_foot
        return foot_pos

        

def main():
    robot1 = robot('/home/ubuntu/robot/robot.xml',0.05)
    #print(robot1.inverse_kinematics(robot1.init_pos))
    #print(robot1.set_state())
    # i=0
    # robot1.set_up_screen()
    # robot1.set_state()
    # while i < 500:
    #     robot1.step()
    #     robot1.render()
    #     robot1.get_state()
    #     angles = robot1.sim_to_rel()
    # robot1.terminate()

if __name__ == "__main__":
    main()
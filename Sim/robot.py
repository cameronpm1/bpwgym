from Sim.util import *
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Callable, Union, Dict, List, Optional

from Sim.sim import Sim

class Dummy():
    def __init__(self):
        pass

class Robot(Sim):
    def __init__(
        self,
        model: str,
        dt: float,
    ) -> None:
        super().__init__(model=model, dt=dt)
        self.servo_min = [-105, -180, -100, -105, -180, -100]
        self.servo_max = [-15, -50, 0, -15, -50, 0]
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
        self.time_step = dt
        self.knee_off = 1.0238628616194603
        self.support_off = 1.8096838437102432
        self.foot_off = 0.6190878380815508
        
        self.reset()

    def set_state(self) -> dict:
        state = super().set_state(joints=len(self.jnames), jnames=self.jnames)
        self._state['vel'] = np.sqrt(state['root']['qvel'][0]**2+state['root']['qvel'][1]**2)
        return self._state

    
    def get_state(self) -> dict:
        state = super().get_state()
        self._state['vel'] = np.sqrt(state['root']['qvel'][0]**2+state['root']['qvel'][1]**2)
        return self._state

    def reset(self):
        super().reset()
        if self._state is None:
            return self.set_state()
        else:
            return self.get_state()

    def sim_to_rel(self) -> dict:
        """
        Take the qpos of actuated joints (radians) and return angles realtive to robot (degrees)
        """
        angles = {}
        angles['joint00'] = {'qpos' : -90 - np.degrees(self._state['joint00']['qpos'][0]), 'qvel' : self._state['joint00']['qvel'][0]}
        angles['joint10'] = {'qpos' : -90 + np.degrees(self._state['joint10']['qpos'][0]), 'qvel' : self._state['joint10']['qvel'][0]}
        angles['joint011'] = {'qpos' : -79.937 - np.degrees(self._state['joint011']['qpos'][0]), 'qvel' : self._state['joint011']['qvel'][0]}
        angles['joint031'] = {'qpos' : -148.610 - np.degrees(self._state['joint031']['qpos'][0]), 'qvel' : self._state['joint031']['qvel'][0]}
        angles['joint111'] = {'qpos' : -79.937 - np.degrees(self._state['joint111']['qpos'][0]), 'qvel' : self._state['joint111']['qvel'][0]}
        angles['joint131'] = {'qpos' : -148.610 - np.degrees(self._state['joint131']['qpos'][0]), 'qvel' : self._state['joint131']['qvel'][0]}
        return angles

    def rel_to_sim(self,
        angles: dict,
    ) -> bool:
        angles_sim = {}
        angles_sim['joint00'] = {'qpos' : np.radians(-90 - angles['joint00'])}
        angles_sim['joint10'] = {'qpos' : np.radians(angles['joint10'] + 90)}
        angles_sim['joint011'] = {'qpos' : np.radians(-79.937 - angles['joint011'])}
        angles_sim['joint031'] = {'qpos' : np.radians(-148.610 - angles['joint031'])}
        angles_sim['joint111'] = {'qpos' : np.radians(-79.937 - angles['joint111'])}
        angles_sim['joint131'] = {'qpos' : np.radians(-148.610 - angles['joint131'])}
        angles_sim['joint012'] = {'qpos' : self.front_bottom_pos(angles['joint011'],angles['joint031']) - self.knee_off} 
        angles_sim['joint112'] = {'qpos' : self.front_bottom_pos(angles['joint111'],angles['joint131']) - self.knee_off} 
        angles_sim['joint032'] = {'qpos' : -self.back_support_pos(angles['joint011'],angles['joint031']) + self.support_off} 
        angles_sim['joint132'] = {'qpos' : -self.back_support_pos(angles['joint111'],angles['joint131']) + self.support_off} 
        angles_sim['joint014'] = {'qpos' : self.foot_pos(angles['joint011'],angles['joint031']) - self.foot_off} 
        angles_sim['joint114'] = {'qpos' : self.foot_pos(angles['joint111'],angles['joint131']) - self.foot_off} 
        return angles_sim

    def step(self,
        angles: dict,
    ):
        ctrl = []
        ctrl.append(angles['joint011']['qpos'])
        ctrl.append(angles['joint031']['qpos'])
        ctrl.append(angles['joint00']['qpos'])
        ctrl.append(angles['joint111']['qpos'])
        ctrl.append(angles['joint131']['qpos'])
        ctrl.append(angles['joint10']['qpos'])
        ctrl.append(angles['joint011']['qpos'])
        ctrl.append(angles['joint111']['qpos'])
        ctrl.append(angles['joint012']['qpos'])
        ctrl.append(angles['joint112']['qpos'])
        ctrl.append(angles['joint032']['qpos'])
        ctrl.append(angles['joint132']['qpos'])
        ctrl.append(angles['joint012']['qpos'])
        ctrl.append(angles['joint112']['qpos'])
        ctrl.append(angles['joint014']['qpos'])
        ctrl.append(angles['joint114']['qpos'])
        self.set_ctrl(ctrl)
        super().step()
        self.get_state()

    def check_state(self,
        angles: dict,
    ):
        angles = angles.copy()
        for key in angles.keys():
            if 'joint' in key:
                temp = angles[key]
                angles[key] = {'qpos' : temp[0], 'qvel' : temp[1]}
        if self.check_angles(angles):
            return True,None
        if self.check_fall():
            return True,None
        valid,pos = self.check_kinematics(angles)
        if valid:
            return True,None
        else:
            return False,pos
            

                
                

    def check_angles(self,
        angles: dict,
    ) -> bool:
        """
        returns false if angles are within range
        """
        for i,joint in enumerate(self.servos):
            if angles[joint]['qpos'] < self.servo_min[i] or angles[joint]['qpos'] > self.servo_max[i]:
                return True
        return False

    def check_kinematics(self,
        angles: dict,
    ):
        """
        returns false if anlges are feasible
        """
        foot_pos_left = self.forward_kinematics(angles['joint011']['qpos'],angles['joint031']['qpos'])
        foot_pos_right = self.forward_kinematics(angles['joint111']['qpos'],angles['joint131']['qpos'])
        if foot_pos_left is None or foot_pos_right is None:
            return True,(foot_pos_left,foot_pos_right)
        else:
            return False,(foot_pos_left,foot_pos_right)

    def check_fall(self) -> bool:
        """
        returns flase if robot has not fallen
        """
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
        return {'joint pos':joint, 'foot pos':foot_pos, 'knee pos':front_link_pos, 'back pos': back_link_pos}

    def front_bottom_pos(self, angle1, angle2):
        pos = self.forward_kinematics(angle1,angle2)
        vec1 = np.array(pos['knee pos'])
        vec2 = np.array(pos['foot pos']) - np.array(pos['knee pos'])
        angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return angle

    def back_support_pos(self, angle1, angle2):
        pos = self.forward_kinematics(angle1,angle2)
        vec1 = np.array(pos['back pos']) - np.array(self.servo_back_pos)
        vec2 = np.array(pos['joint pos']) - np.array(pos['back pos'])
        angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return angle

    def foot_pos(self, angle1, angle2):
        pos = self.forward_kinematics(angle1,angle2)
        vec1 = np.array([1, 0])
        vec2 = np.array(pos['knee pos']) - np.array(pos['foot pos'])
        angle = np.arccos(np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
        return angle
        


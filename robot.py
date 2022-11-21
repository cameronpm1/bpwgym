from util import *
from pylx16a.lx16a import *
import math
import numpy as np
import time
import matplotlib.pyplot as plt

class robot():
    def __init__(self,com,debug=True):
        self.servo_min = [-105, -180, -100, -105, -180, -100]
        self.servo_max = [-15, -50, 0, -15, -20, 0]
        self.servo_off = [4, -22.8, 18, 15, -9, 24]
        self.servo_abs = [-1, -1, -1, -1, -1, -1]
        self.servo_rel = [-1, -1, -1, -1, -1, -1]
        self.servo_inverse = [False,False,False,True,True,False]
        self.init_angles = [-90,-135,-90,-90,-135,-90]
        self.front_leg_servos = [0, 3]
        self.back_leg_servos = [1, 4]
        self.ftl = 63  # front top leg
        self.ble = 30  # back leg extension
        self.bl = 90  # back leg
        self.fbl = 68 # front bottom leg
        self.fbls = 50  # front bottom leg segment
        self.servo_back_pos = [-53.5,-6.45]
        self.init_pos = np.array([-40,-107])
        self.gait1 = None
        self.gait2 = None
        self.gait1a = None
        self.gait2a = None
        if not debug:
            self.servos = self.init_servos(com)

    def init_servos(self,com):
        LX16A.initialize(com)
        try:
            servo1 = LX16A(1)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        try:
            servo2 = LX16A(2)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        try:
            servo3 = LX16A(3)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        try:
            servo11 = LX16A(11)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        try:
            servo12 = LX16A(12)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        try:
            servo13 = LX16A(13)  # top leg servo

        except ServoTimeoutError as e:
            print(f"Servo {e.id_} is not responding. Exiting...")
            quit()
        print('all servos responding')
        return [servo1,servo2,servo3,servo11,servo12,servo13]

    def auto_init(self,maitenance=False):
        for i, servo in enumerate(self.servos):
            if maitenance:
                self.move_servo(i, -90)
            else:
                self.move_servo(i,self.init_angles[i])

    def move_servo(self,indx,angle):
        if angle < self.servo_min[indx] or angle > self.servo_max[indx]:
            print('Error: invalid angles being given to servo, please stop process')
            quit()
        else:
            self.servos[indx].move(self.rel_to_abs(indx, angle))

    def update_servo_angles(self):
        for i, servo in enumerate(self.servos):
            self.servo_abs[i] = servo.get_physical_angle()
            if self.servo_abs[i] > 240: self.servo_abs[i] = 240
            self.servo_rel[i] = self.abs_to_rel(i,self.servo_abs[i])

    def abs_to_rel(self,indx,angle):
        if self.servo_inverse[indx]:
            rel = -1*(180-(angle+self.servo_off[indx]))
        else:
            rel = -1*(angle+self.servo_off[indx])
        return rel

    def rel_to_abs(self,indx,angle):
        if self.servo_inverse[indx]:
            abs = angle+180-self.servo_off[indx]
        else:
            abs = -1*angle-self.servo_off[indx]
        if abs < 0: abs = 0
        if abs > 240: abs = 240
        return abs

    def servo_test_mini(self,indx):
        self.move_servo_test(indx, self.servo_rel[indx], self.servo_min[indx], -1)
        self.move_servo_test(indx, self.servo_rel[indx], self.servo_max[indx], 1)
        print('servo full ranges successfully tested')

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
        return (np.degrees(angle1),np.degrees(angle2))

    def forward_kinematics(self, angle1,angle2):
        #INPUT AS DEGREES
        front_link_pos = self.front_servo_pos(angle1)
        back_link_pos = self.back_servo_pos(angle2)
        joints = self.calc_joint_pos(back_link_pos,front_link_pos)
        if joints is None:
            print('Not feasible servo positions')
            quit()
        if joints[1] < joints[3]:
            joint = [joints[0],joints[1]]
        else:
            joint = [joints[2],joints[3]]
        vec_to_foot = np.array(joint) - np.array([front_link_pos[0],front_link_pos[1]])
        vec_to_foot /= self.fbls#np.linalg.norm(vec_to_foot)
        vec_to_foot *= self.fbl-self.fbls
        foot_pos = joint + vec_to_foot
        return foot_pos

    def init_legs(self,leg1_pos,leg2_pos):
        indx1 = self.front_leg_servos[0]
        indx2 = self.back_leg_servos[0]
        indx3 = self.front_leg_servos[1]
        indx4 = self.back_leg_servos[1]
        self.update_servo_angles()
        x1 = leg1_pos[0]
        y1 = leg1_pos[1]
        x2 = leg2_pos[0]
        y2 = leg2_pos[1]
        foot_pos1 = self.forward_kinematics(self.servo_rel[indx1], self.servo_rel[indx2])
        foot_pos2 = self.forward_kinematics(self.servo_rel[indx3], self.servo_rel[indx4])
        if (abs(foot_pos1[0] - x1[0]) > 0.1 and abs(foot_pos1[1] - x1[1]) > 0.1) or (abs(foot_pos2[0] - x2[0]) > 0.1 and abs(foot_pos2[1] - x2[1]) > 0.1):
            temp_x1 = np.linspace(foot_pos1[0], x1[0], num=30)
            temp_y1 = np.linspace(foot_pos1[1], y1[0], num=30)
            temp_x2 = np.linspace(foot_pos2[0], x2[0], num=30)
            temp_y2 = np.linspace(foot_pos2[1], y2[0], num=30)
            for i in range(len(temp_x1)):
                angles1 = self.inverse_kinematics([temp_x1[i], temp_y1[i]])
                angles2 = self.inverse_kinematics([temp_x2[i], temp_y2[i]])
                self.move_servo(indx1, angles1[0])
                self.move_servo(indx2, angles1[1])
                self.move_servo(indx3, angles2[0])
                self.move_servo(indx4, angles2[1])
                time.sleep(0.1)
        return None

    def gait(self,loop=5):
        wait=0.05
        if self.gait1 is None or self.gait2 is None:
            self.gait1 = self.load_traj1()
            self.gait2 = self.load_traj2()
            gait1a = [[],[]]
            gait2a = [[],[]]
            for i in range(len(self.gait1[0][0])):
                angles1_1 = self.inverse_kinematics([self.gait1[0][0][i], self.gait1[0][1][i]])
                angles1_2 = self.inverse_kinematics([self.gait1[1][0][i], self.gait1[1][1][i]])
                angles2_1 = self.inverse_kinematics([self.gait2[0][0][i], self.gait2[0][1][i]])
                angles2_2 = self.inverse_kinematics([self.gait2[1][0][i], self.gait2[1][1][i]])
                gait1a[0].append(angles1_1)
                gait1a[1].append(angles1_2)
                gait2a[0].append(angles2_1)
                gait2a[1].append(angles2_2)
            self.gait1a=gait1a
            self.gait2a=gait2a
        #self.init_legs(self.init_pos,self.init_pos)
        indx1 = self.front_leg_servos[0]
        indx2 = self.back_leg_servos[0]
        indx3 = self.front_leg_servos[1]
        indx4 = self.back_leg_servos[1]
        x = []
        a1 = []
        a2 = []
        for i in range(len(self.gait1a[0])):
            x.append(i)
            a1.append(gait1a[0][i][0])
            a2.append(gait1a[0][i][1])
            """
            self.servo_pos[indx1] = gait1a[0][i][0]
            self.servo_pos[indx2] = gait1a[0][i][1]
            self.servo_pos[indx3] = gait1a[1][i][0]
            self.servo_pos[indx4] = gait1a[1][i][1]
            self.set_servos()
            #self.move_servo(indx1, gait1a[0][i][0])
            #self.move_servo(indx2, gait1a[0][i][1])
            #self.move_servo(indx3, gait1a[1][i][0])
            #self.move_servo(indx4, gait1a[1][i][1])
            time.sleep(wait)
            """
        for l in range(loop):
            for i in range(len(self.gait2a[0])):
                if l%2 == 0:
                    x.append(x[-1]+1)
                    a1.append(gait2a[1][i][0])
                    a2.append(gait2a[1][i][1])
                    """
                    self.servo_pos[indx1] = gait2a[1][i][0]
                    self.servo_pos[indx2] = gait2a[1][i][1]
                    self.servo_pos[indx3] = gait2a[0][i][0]
                    self.servo_pos[indx4] = gait2a[0][i][1]
                    self.set_servos()
                    #self.move_servo(indx1, gait2a[1][i][0])
                    #self.move_servo(indx2, gait2a[1][i][1])
                    #self.move_servo(indx3, gait2a[0][i][0])
                    #self.move_servo(indx4, gait2a[0][i][1])
                    time.sleep(wait)
                    """
                else:
                    x.append(x[-1] + 1)
                    a1.append(gait2a[0][i][0])
                    a2.append(gait2a[0][i][1])
                    """
                    self.servo_pos[indx1] = gait2a[0][i][0]
                    self.servo_pos[indx2] = gait2a[0][i][1]
                    self.servo_pos[indx3] = gait2a[1][i][0]
                    self.servo_pos[indx4] = gait2a[1][i][1]
                    self.set_servos()
                    #self.move_servo(indx1, gait2a[0][i][0])
                    #self.move_servo(indx2, gait2a[0][i][1])
                    #self.move_servo(indx3, gait2a[1][i][0])
                    #self.move_servo(indx4, gait2a[1][i][1])
                    time.sleep(wait)
                    """
        #self.init_legs(self.init_pos, self.init_pos)
        plt.plot(x,a1)
        plt.plot(x,a2)
        plt.xlabel('step')
        plt.ylabel('servo angle')
        plt.legend(['Servo1', 'Servo2'])
        plt.show()
        return None

    def move_leg(self,foot,pos):
        indx1 = self.front_leg_servos[foot]
        indx2 = self.back_leg_servos[foot]
        self.update_servo_angles()
        x = pos[0]
        y = pos[1]
        foot_pos = self.forward_kinematics(self.servo_rel[indx1],self.servo_rel[indx2])
        if abs(foot_pos[0]-x[0]) > 0.1 and abs(foot_pos[1]-y[0]) > 0.1:
            temp_x = np.linspace(foot_pos[0],x[0],num=30)
            temp_y = np.linspace(foot_pos[1],y[0],num=30)
            for i in range(len(temp_x)):
                angles = self.inverse_kinematics([temp_x[i],temp_y[i]])
                self.move_servo(indx1,angles[0])
                self.move_servo(indx2,angles[1])
                time.sleep(0.1)
        for i in range(len(x)):
            angles = self.inverse_kinematics([x[i], y[i]])
            self.move_servo(indx1, angles[0])
            self.move_servo(indx2, angles[1])
            time.sleep(0.1)

    def load_traj1(self,num=15):
        init = self.init_pos
        angles = np.linspace(170,10,num=num)
        forward = 10
        center_yoff = (forward/2)/np.tan(np.radians(80))
        center = np.array([init[0]+forward/2,init[1]-center_yoff])
        radius = np.sqrt(forward**2/4+center_yoff**2)
        x = []
        y = []
        for angle in angles:
            x.append(np.cos(np.radians(angle))*radius + center[0])
            y.append(np.sin(np.radians(angle))*radius + center[1])
        foot1 = (x,y)
        x_s = np.linspace(init[0],init[0]-forward,num=num)
        y_s = np.ones(num)*init[1]
        #plt.plot(x,y)
        #plt.plot(x_s,y_s)
        #plt.axis('equal')
        #plt.show()
        foot2 = (x_s,y_s)
        return foot1,foot2

    def load_traj2(self,num=15):
        forward = 20
        back_foot = np.array([self.init_pos[0]-forward/2,self.init_pos[1]])
        angles = np.linspace(170,10,num=num)
        center_yoff = (forward/2)*np.tan(np.radians(10))
        center = np.array([back_foot[0]+forward/2,back_foot[1]-center_yoff])
        radius = np.sqrt(forward**2/4+center_yoff**2)
        x = []
        y = []
        for angle in angles:
            x.append(np.cos(np.radians(angle))*radius + center[0])
            y.append(np.sin(np.radians(angle))*radius + center[1])
        foot1 = (x,y)
        x_s = np.linspace(back_foot[0]+forward, back_foot[0], num=num)
        y_s = np.ones(num) * back_foot[1]
        #plt.plot(x,y)
        #plt.plot(x_s,y_s)
        #plt.axis('equal')
        #plt.show()
        foot2 = (x_s,y_s)
        return foot1,foot2
        

def main():
    robot1 = robot("COM4",debug=True)
    print(robot1.inverse_kinematics(robot1.init_pos))
    #x = np.linspace(-50,0,num=50)
    #y = np.ones(50)*-97
    #robot1.inverse_kinematics_leg(0,[x,y])

if __name__ == "__main__":
    main()
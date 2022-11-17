#!/usr/bin/env python3
"""
Example of how bodies interact with each other. For a body to be able to
move it needs to have joints. In this example, the "robot" is a red ball
with X and Y slide joints (and a Z slide joint that isn't controlled).
On the floor, there's a cylinder with X and Y slide joints, so it can
be pushed around with the robot. There's also a box without joints. Since
the box doesn't have joints, it's fixed and can't be pushed around.
"""
from mujoco_py import load_model_from_xml, MjSim, MjViewer
import math
import os

MODEL_XML = """
<?xml version="1.0" ?>
<mujoco>
    <compiler angle="radian" balanceinertia="true" meshdir="/home/ubuntu/robot/left leg"/>
 
    <!-- import our stl files -->
    <asset>
        <mesh file="foot.STL" />
        <mesh file="bottom_leg.STL" />
        <mesh file="bottom_leg_back_linkage.STL" />
        <mesh file="upper_leg_back_linkage.STL" />
        <mesh file="right_leg_servos.STL" />
        <mesh file="bottom_leg_linkage_extension.STL" />
        <mesh file="front_leg_assem.STL" />
        <mesh file="bottom_leg_linkage.STL" />
        <mesh file="center_joint_linkage.STL" />
    </asset>

    <option timestep="0.0001" />
    
    <worldbody>
        <!-- set up a light pointing down on the robot -->
        <light diffuse="0.5 0.5 0.5" pos="0 0 3" dir="0 0 -1" />
        <!-- add a floor so we don't stare off into the abyss -->
        <geom name="floor" pos="0 0 0" size="1000 1000 100" type="plane" rgba="0 0.9 0 1.0"/>

        <body name="right_leg_servos" pos="0 0 152">
            <freejoint name="root"/>
            <geom name="right_leg_servos" type="mesh" mesh="right_leg_servos" pos="0 0 0"/>
            <inertial pos="0 0 0" mass="0.1" fullinertia="0.00002416 0.00008991 0.00007368 0.00000440 -0.0000292 -0.00000112"/>
            <body name="bottom_leg_linkage_extension" pos="0 0 0">
                <geom name="bottom_leg_linkage_extension" type="mesh" mesh="bottom_leg_linkage_extension" pos="0 0 0" />
                <inertial pos="0 0 0" mass="0.0052" fullinertia="0.00000436 0.00000252 0.00000205 0.0 0.0 0.0"/>
                <joint axis="0 1 0" name="joint0" pos="0 0 0" type="hinge"/>
                <body name="bottom_leg_linkage" pos="18.9 0 -23.3">
                     <geom name="bottom_leg_linkage" type="mesh" mesh="bottom_leg_linkage" pos="0 0 0" conaffinity="0"/>
                     <inertial pos="0 0 0" mass="0.00866" fullinertia="0.000016 0.000014 0.000001 0.0 0.0 0.0"/>
                     <joint axis="0 1 0" name="joint5" pos="0 0 0" type="hinge"/>
                </body>
            </body>
            <body name="front_leg_assem" pos="-53.5 0 6.45">
                <geom name="front_leg_assem" type="mesh" mesh="front_leg_assem" pos="0 0 0" />
                <inertial pos="0 0 0" mass="0.0082" fullinertia="0.00000786 0.00000608 0.00000204 0.0 0.0 0.0"/>
                <joint axis="0 1 0" name="joint1" pos="0 0 0" type="hinge"/>
                <body name="bottom_leg" pos="0 0 -63">
                    <geom name="bottom_leg" type="mesh" mesh="bottom_leg" pos="0 0 0" />
                    <inertial pos="0 0 0" mass="0.00866" fullinertia="0.00001137 0.00001129 0.00000038 0.0 0.00000014 0.0"/>
                    <joint axis="0 1 0" name="joint2" pos="0 0 0" type="hinge"/>
                    <body name="foot" pos="0 0 -68">
                        <geom name="foot" type="mesh" mesh="foot" pos="0 0 0" />
                        <inertial pos="0 0 0" mass="0.0083" fullinertia="0.00000325 0.00000406 0.00000250 0.0 -0.00000095 0.0"/>
                        <joint axis="0 1 0" name="joint6" pos="0 0 0" type="hinge"/>
                    </body>
                </body>
            </body>
            <body name="upper_leg_back_linkage" pos="-37.4 0 -56.55">
                <geom name="upper_leg_back_linkage" type="mesh" mesh="upper_leg_back_linkage" pos="0 0 0" conaffinity="0"/>
                <inertial pos="0 0 0" mass="0.0026" fullinertia="0.00000429 0.00000398 0.00000063 0.0 0.00000064 0.0"/>
                <joint axis="0 1 0" name="joint3" pos="0 0 63" type="hinge"/>
                <body name="bottom_leg_back_linkage" pos="0 0 0">
                     <geom name="bottom_leg_back_linkage" type="mesh" mesh="bottom_leg_back_linkage" pos="0 0 0" conaffinity="0" />
                     <inertial pos="0 0 0" mass="0.0017" fullinertia="0.000003 0.000003 0.000001 0.0 0.0 0.0"/>
                     <joint axis="0 1 0" name="joint4" pos="0 0 0" type="hinge"/>
                </body>
            </body>
        </body>

    </worldbody>

    <contact>
        <exclude body1="bottom_leg_back_linkage" body2="foot"/>
        <exclude body1="bottom_leg_linkage" body2="bottom_leg"/>
        <exclude body1="bottom_leg_linkage" body2="bottom_leg_back_linkage"/>
        <exclude body1="bottom_leg_linkage" body2="upper_leg_back_linkage"/>
    </contact>

    <equality>
		<connect anchor="-37.4 0 27.45" body1="bottom_leg_back_linkage" body2="foot" name="equality_constraint1" />
        <connect anchor="-53.5 0 75.45" body1="bottom_leg_linkage" body2="bottom_leg_linkage_extension" name="equality_constraint2" solref="100 1"/>
	</equality>

    <actuator>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint0" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint1" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint2" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint3" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint4" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint5" kp="1"/>
        <position ctrllimited="false" ctrlrange="0 6.283185308" gear="1" joint="joint6" kp="1"/>
    </actuator>

</mujoco>
"""

#<body pos="-37.4 0 95.45">
#            <joint type="free"/>
#            <geom type="sphere" size="5 5 5" rgba="0 .9 0 1"/>
#        </body>
#<body name="bottom_leg_linkage" pos="0 0 -30">
#                    <geom name="bottom_leg_linkage" type="mesh" mesh="bottom_leg_linkage" pos="0 0 0" />
#                    <inertial pos="0 0 0" mass="0.00866" fullinertia="0.000016 0.000014 0.000001 0.0 0.0 0.0"/>
#                    <joint axis="0 1 0" damping="1" name="joint5" pos="0 0 0" type="hinge"/>
#                    <joint axis="0 1 0" damping="1" name="joint9" pos="-90 0 0" type="hinge"/>
#                </body>
#<body name="center_joint_linkage" pos="0 0 -63">
#                    <geom name="center_joint_linkage" type="mesh" mesh="center_joint_linkage" pos="0 0 0" />
#                    <inertial pos="0 0 0" mass="0.0003" fullinertia="0.000000005 0.00000002 0.00000002 0.0 0.0 0.0"/>
#                    <joint axis="0 1 0" damping="1" name="joint7" pos="0 0 0" type="hinge"/>
#                </body>
#<body name="upper_leg_back_linkage" pos="-37.4 -1 -56.55">
#                <geom name="upper_leg_back_linkage" type="mesh" mesh="upper_leg_back_linkage" pos="0 0 0" />
#                <inertial pos="0 0 0" mass="0.0026" fullinertia="0.00000429 0.00000398 0.00000063 0.0 0.00000064 0.0"/>
#                <joint axis="0 1 0" damping="1" name="joint3" pos="0 0 63" type="hinge"/>
#                <body name="bottom_leg_back_linkage" pos="0 0 0">
#                    <geom name="bottom_leg_back_linkage" type="mesh" mesh="bottom_leg_back_linkage" pos="0 0 0" />
#                    <inertial pos="0 0 0" mass="0.0017" fullinertia="0.000003 0.000003 0.000001 0.0 0.0 0.0"/>
#                    <joint axis="0 1 0" damping="1" name="joint4" pos="0 0 0" type="hinge"/>
#                </body>
#            </body>
#<joint active="true" joint1="joint9" joint2="joint10" solref="2 100"></joint>
#<joint active="true" joint1="joint8" joint2="joint3" solref="2 100"></joint>
model = load_model_from_xml(MODEL_XML)
sim = MjSim(model)
#sim.data.qpos[8] = 1
#sim.data.qpos[7] = -0.68
#sim.data.qpos[8] = 0.88
viewer = MjViewer(sim)
t = 0
viewer.render()
print(sim.data.ctrl,sim.data.qpos)
while True:
    viewer.render()
    sim.data.qpos[9] = sim.data.qpos[12]
    sim.data.qpos[10] = sim.data.qpos[13]
    t += 1
    sim.step()
    viewer.render()
    if t > 100 and os.getenv('TESTING') is not None:
        break
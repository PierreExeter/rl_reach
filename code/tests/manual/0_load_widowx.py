"""
Load WidowX manipulator and obstacle in Pybullet
"""
import time
import random
import pybullet as p
import numpy as np
import pybullet_data

# start pybullet simulation
p.connect(p.GUI)

# reset the simulation to its original state
p.resetSimulation()

# load urdf file path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# import plane
plane = p.loadURDF("plane.urdf")

# import Kuka urdf and fix it to the ground
robot = p.loadURDF("gym_envs/gym_envs/widowx_env/URDFs/widowx/widowx.urdf", [0, 0, 0], useFixedBase=1)
obstacle = p.loadURDF("gym_envs/gym_envs/widowx_env/URDFs/circular_window_small.urdf", [0, 0, 0], useFixedBase=1)

# move obstacle
p.resetBasePositionAndOrientation(obstacle, [0.1, .0, 0.26], p.getQuaternionFromEuler([0, np.pi/2, 0]))

print("robot ID: ", robot)

# request the position and orientation of the robot
position, orientation = p.getBasePositionAndOrientation(robot)
print("The robot position is {}".format(position))
print("The robot orientation (x, y, z, w) is {}".format(orientation))
print("--------------------------")

# print the number of joints of the robot
nb_joints = p.getNumJoints(robot)
print("The robot is made of {} joints.".format(nb_joints))
print("--------------------------")

# print joint information
joints_index_list = range(nb_joints)

for joint_id in joints_index_list:
    joint_info = p.getJointInfo(robot, joint_id)
    print("Joint index: {}".format(joint_info[0]))
    print("Joint name: {}".format(joint_info[1]))
    print("Joint type: {}".format(joint_info[2]))
    print("First position index: {}".format(joint_info[3]))
    print("First velocity index: {}".format(joint_info[4]))
    print("flags: {}".format(joint_info[5]))
    print("Joint damping value: {}".format(joint_info[6]))
    print("Joint friction value: {}".format(joint_info[7]))
    print("Joint positional lower limit: {}".format(joint_info[8]))
    print("Joint positional upper limit: {}".format(joint_info[9]))
    print("Joint max force: {}".format(joint_info[10]))
    print("Joint max velocity {}".format(joint_info[11]))
    print("Name of link: {}".format(joint_info[12]))
    print("Joint axis in local frame: {}".format(joint_info[13]))
    print("Joint position in parent frame: {}".format(joint_info[14]))
    print("Joint orientation in parent frame: {}".format(joint_info[15]))
    print("Parent link index: {}".format(joint_info[16]))
    print("--------------------------")

# # print state of all joints
joints_state_list = p.getJointStates(robot, joints_index_list)

for joint_id in joints_index_list:
    print("Joint position: {}".format(joints_state_list[joint_id][0]))
    print("Joint velocity: {}".format(joints_state_list[joint_id][1]))
    print("Joint reaction forces (Fx, Fy, Fz, Mx, My, Mz): {}".format(joints_state_list[joint_id][2]))
    print("Torque applied to joint: {}".format(joints_state_list[joint_id][3]))
    print("--------------------------")

# # print state of all links
# link_state_list = p.getLinkState(robot, 2)

for link_id in joints_index_list:
    link_state_list = p.getLinkState(robot, link_id)
    print("Link position (center of mass): {}".format(link_state_list[0]))
    print("Link orientation (center of mass): {}".format(link_state_list[1]))
    print("Local position offset of inertial frame: {}".format(link_state_list[2]))
    print("Local orientation offset of inertial frame: {}".format(link_state_list[3]))
    print("Link frame position: {}".format(link_state_list[4]))
    print("Link frame orientation: {}".format(link_state_list[5]))
    print("--------------------------")


def create_random_pos():
    """ create position vector within lower and upper joint limit"""
    pos = []
    for joint_id in joints_index_list:
        joint_info = p.getJointInfo(robot, joint_id)

        pos.append(random.uniform(joint_info[8], joint_info[9]))
    return pos


# Define gravity in x, y and z
p.setGravity(0, 0, -9.81)

# define a target angle position for each joint (note, you can also control by velocity or torque)
# pos = create_random_pos()
# print(pos)
# # p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL, targetPositions=[-1, 0, -0.5, 1, 1, 0, 0, 0, 0])  

# step through the simulation
for t in range(1000):
    print('Time step: ', t)
    p.stepSimulation()
    time.sleep(1./30.)  # slow down the simulation

    if t % 100 == 0:
        pos = create_random_pos()
        p.setJointMotorControlArray(robot, joints_index_list, p.POSITION_CONTROL, targetPositions=pos)  

p.disconnect()

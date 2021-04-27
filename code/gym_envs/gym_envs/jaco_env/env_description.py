"""
Classes with observation shapes, action shapes
and reward functions
"""

import numpy as np
import pybullet as p
# import time


class ObservationShapes:
    """ Implements observations shapes 1 to 7 """

    def __init__(
        self,
        endeffector_pos,
        endeffector_orient,
        torso_pos,
        torso_orient,
        goal_pos,
        goal_orient,
        joint_positions):

        self.endeffector_pos = endeffector_pos
        self.endeffector_orient = endeffector_orient
        self.torso_pos = torso_pos
        self.torso_orient = torso_orient
        self.goal_pos = goal_pos
        self.goal_orient = goal_orient
        self.joint_positions = joint_positions

        self.end_torso_pos = self.endeffector_pos - self.torso_pos
        self.end_goal_pos = self.endeffector_pos - self.goal_pos
        self.end_torso_orient = self.endeffector_orient - self.torso_orient
        self.end_goal_orient = self.endeffector_orient - self.goal_orient


    def get_obs1(self):
        """ Returns observation #1 """
        robot_obs = np.concatenate(
            [self.endeffector_pos, self.joint_positions]).ravel()

        return robot_obs

    def get_obs2(self):
        """ Returns observation #2 """
        robot_obs = np.concatenate(
            [self.goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def get_obs3(self):
        """ Returns observation #3 """
        robot_obs = np.concatenate(
            [self.end_torso_pos, self.end_goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def get_obs4(self):
        """ Returns observation #4 """
        robot_obs = np.concatenate(
            [self.end_goal_pos, self.joint_positions]).ravel()

        return robot_obs

    def get_obs5(self):
        """ Returns observation #5 """
        robot_obs = np.concatenate(
            [self.end_torso_pos, self.end_goal_pos, self.goal_pos, self.joint_positions]).ravel()
        return robot_obs

    def get_obs6(self):
        """ Returns observation #6 """
        robot_obs = np.concatenate(
            [
                self.end_torso_pos,
                self.end_goal_pos,
                self.end_torso_orient,
                self.end_goal_orient,
                self.goal_pos,
                self.goal_orient,
                self.endeffector_pos,
                self.endeffector_orient,
                self.joint_positions
                ]).ravel()

        return robot_obs

    def get_obs7(self):
        """ Returns observation #7 """
        robot_obs = np.concatenate(
            [
                self.end_torso_pos,
                self.end_goal_pos,
                self.goal_pos,
                self.endeffector_pos,
                self.joint_positions
                ]).ravel()

        return robot_obs


class ActionShapes:
    """ Implement actions 1 and 2 """

    def __init__(
        self,
        pybullet_action,
        joint_positions,
        joint_min,
        joint_max,
        arm,
        physics_client,
        frame_skip):

        self.pybullet_action = pybullet_action
        self.joint_positions = joint_positions
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.arm = arm
        self.physics_client = physics_client
        self.frame_skip = frame_skip

        # Update the new joint position with the action
        self.new_joint_positions = self.joint_positions + self.pybullet_action

        # Clip the joint position to fit the joint's allowed boundaries
        self.new_joint_positions = np.clip(
            np.array(self.new_joint_positions),
            self.joint_min,
            self.joint_max)

    def take_action1(self):
        """ select action #1 (increments from previous joint position) """

        # Instantaneously reset the joint position (no torque applied)
        self.force_joint_positions(self.new_joint_positions)

    def take_action2(self):
        """ select action #2: position control """

        # Position control
        self.set_joint_positions(self.new_joint_positions)

        for _ in range(self.frame_skip):
            p.stepSimulation(physicsClientId=self.physics_client)


        # p.stepSimulation(physicsClientId=self.physics_client)
        # time.sleep(1/240)

        # p.setRealTimeSimulation(1)

    def set_joint_positions(self, joint_positions):
        """ Position control (not reset) """
        # In Pybullet, gripper halves are controlled separately
        joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [0, 1, 2, 3, 4, 7, 8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )

    def force_joint_positions(self, joint_positions):
        """ Instantaneous reset of the joint angles (not position control) """
        for i in range(5):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[i]
            )
        # In Pybullet, gripper halves are controlled separately
        for i in range(7, 9):
            p.resetJointState(
                self.arm,
                i,
                joint_positions[-1]
            )


class RewardFunctions:


    def __init__(
        self,
        dist,
        alpha_reward,
        action,
        delta_dist,
        delta_pos,
        orient,
        collision):

        self.dist = dist
        self.alpha_reward = alpha_reward
        self.action = action
        self.delta_dist = delta_dist
        self.delta_pos = delta_pos
        self.orient = orient
        self.collision = collision

        self.term1 = 0
        self.term2 = 0

    def get_reward1(self):
        """ Compute reward function 1 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward2(self):
        """ Compute reward function 2 (dense) """
        self.term1 = - self.dist
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward3(self):
        """ Compute reward function 3 (dense) """
        self.term1 = - self.dist ** 3
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward4(self):
        """ Compute reward function 4 (dense) """
        self.term1 = - self.dist ** 4
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward5(self):
        """ Compute reward function 5 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * np.linalg.norm(self.action)
        rew = self.term1 + self.term2
        return rew

    def get_reward6(self):
        """ Compute reward function 6 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * np.linalg.norm(self.action) / self.dist ** 2
        rew = self.term1 + self.term2
        return rew

    def get_reward7(self):
        """ Compute reward function 7 (dense) """
        self.term1 = self.delta_dist
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward8(self):
        """ Compute reward function 8 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = self.alpha_reward * abs(self.delta_dist / self.dist)
        rew = self.term1 + self.term2
        return rew

    def get_reward9(self):
        """ Compute reward function 9 (dense) """
        self.term1 = self.delta_pos
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward10(self):
        """ Compute reward function 10 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * self.delta_pos / self.dist
        rew = self.term1 + self.term2
        return rew

    def get_reward11(self):
        """ Compute reward function 11 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -1
        else:
            self.term1 = 0
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward12(self):
        """ Compute reward function 12 (sparse) """
        if self.dist >= 0.001:
            self.term1 = 0
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward13(self):
        """ Compute reward function 13 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -0.02
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward14(self):
        """ Compute reward function 14 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -0.001
        else:
            self.term1 = 10
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward15(self):
        """ Compute reward function 15 (sparse + dense) """
        if self.dist >= 0.001:
            self.term1 = - self.dist
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward16(self):
        """ Compute reward function 16 (sparse + dense) """
        if self.dist >= 0.001:
            self.term1 = self.delta_dist
        else:
            self.term1 = self.delta_dist + 10
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward17(self):
        """ Compute reward function 17 (dense) """
        self.term1 = - self.orient ** 2
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward18(self):
        """ Compute reward function 18 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha_reward * self.orient ** 2
        rew = self.term1 + self.term2
        return rew

    def get_reward19(self):
        """ Compute reward function 19 (sparse + dense) """
        if ((self.dist >= 0.001) or (self.orient >= 0.01)):
            self.term1 = - self.dist **2 - self.alpha_reward * self.orient ** 2
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def get_reward20(self):
        """ Compute reward function 20 (sparse + dense) + penalty for collision """
        if self.dist >= 0.001:
            self.term1 = - self.dist
        else:
            self.term1 = 1

        if self.collision:
            self.term2 = -1
        else:
            self.term2 = 0

        rew = self.term1 + self.term2

        return rew

    def get_reward21(self):
        """ Compute reward function 21 (dense): maximise action 1 """
        self.term1 = self.action[0]
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

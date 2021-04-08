"""
Classes with observation shapes, action shapes
and reward functions
"""

import numpy as np
import pybullet as p


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
        physics_client):

        self.pybullet_action = pybullet_action
        self.joint_positions = joint_positions
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.arm = arm
        self.physics_client = physics_client
        self.frame_skip = 10
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

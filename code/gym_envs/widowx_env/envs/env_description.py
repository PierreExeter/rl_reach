"""
Classes with observation shapes, action shapes
and reward functions
"""

import numpy as np


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

"""
Implements the Gym training environment in Pybullet
WidowX MK-II robot manipulator reaching a target position
"""

import os
import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces


# Initial joint angles
RESET_VALUES = [
    0.015339807878856412,
    -1.2931458041875956,
    1.0109710760673565,
    -1.3537670644267164,
    -0.07158577010132992,
    .027]

MIN_GOAL_COORDS = np.array([-.14, -.13, 0.26])
MAX_GOAL_COORDS = np.array([.14, .13, .39])
MIN_END_EFF_COORDS = np.array([-.16, -.15, 0.14])
MAX_END_EFF_COORDS = np.array([.16, .15, .41])
FIXED_GOAL_COORDS  = np.array([.14, .0, 0.26])


class WidowxEnv(gym.Env):
    """ WidowX reacher Gym environment """

    def __init__(
        self,
        random_goal,
        goal_oriented,
        obs_type,
        reward_type,
        action_type,
        joint_limits,
        action_coeff,
        normalize_action,
        alpha):
        """
        Initialise the environment
        """

        self.random_goal = random_goal
        self.goal_oriented = goal_oriented
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.action_type = action_type
        self.joint_limits = joint_limits
        self.action_coeff = action_coeff
        self.normalize_action = normalize_action
        self.alpha = alpha

        self.old_endeffector_pos = None
        self.endeffector_pos = None
        self.torso_pos = None
        self.end_torso = None
        self.end_goal = None
        self.joint_positions = None
        self.reward = None
        self.obs = None
        self.action = np.zeros(6)
        self.normalized_action = np.zeros(6)
        self.new_joint_positions = None
        self.dist = 0
        self.old_dist = 0
        self.term1 = 0
        self.term2 = 0
        self.delta_pos = 0
        self.delta_dist = 0

        # Define action space
        self.normalized_action_min = np.float32(np.array([-1, -1, -1, -1, -1, -1]))
        self.normalized_action_max = np.float32(np.array([1, 1, 1, 1, 1, 1]))
        self.action_min = np.float32(np.array([-0.5, -0.25, -0.25, -0.25, -0.5, -0.005]) / self.action_coeff)
        self.action_max = np.float32(np.array([0.5, 0.25, 0.25, 0.25, 0.5, 0.005]) / self.action_coeff)

        if self.normalize_action:
            self.action_space = spaces.Box(
                    low=self.normalized_action_min,
                    high=self.normalized_action_max,
                    dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                    low=self.action_min,
                    high=self.action_max,
                    dtype=np.float32)

        # Define observation space
        if self.joint_limits == "small":
            self.joint_min = np.array([-3.1, -1.6, -1.6, -1.8, -3.1, 0.0])
            self.joint_max = np.array([3.1, 1.6, 1.6, 1.8, 3.1, 0.0])
        elif self.joint_limits == "large":
            self.joint_min = np.array([-3.2, -3.2, -3.2, -3.2, -3.2, -3.2])
            self.joint_max = np.array([3.2, 3.2, 3.2, 3.2, 3.2, 3.2])

        if self.obs_type == 1:
            self.obs_space_low = np.float32(
                np.concatenate((MIN_END_EFF_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((MAX_END_EFF_COORDS, self.joint_max), axis=0))

        elif self.obs_type == 2:
            self.obs_space_low = np.float32(
                np.concatenate((MIN_GOAL_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((MAX_GOAL_COORDS, self.joint_max), axis=0))

        elif self.obs_type == 3:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*6, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*6, self.joint_max), axis=0))

        elif self.obs_type == 4:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*3, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*3, self.joint_max), axis=0))

        elif self.obs_type == 5:
            self.obs_space_low = np.float32(
                np.concatenate(([-1.0]*6, MIN_GOAL_COORDS, self.joint_min), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate(([1.0]*6, MAX_GOAL_COORDS, self.joint_max), axis=0))

        self.observation_space = spaces.Box(
                    low=self.obs_space_low,
                    high=self.obs_space_high,
                    dtype=np.float32)

        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(
                    low=np.float32(MIN_GOAL_COORDS),
                    high=np.float32(MAX_GOAL_COORDS),
                    dtype=np.float32),
                achieved_goal=spaces.Box(
                    low=np.float32(MIN_END_EFF_COORDS),
                    high=np.float32(MAX_END_EFF_COORDS),
                    dtype=np.float32),
                observation=self.observation_space))

        # Initialise goal position
        if self.random_goal:
            self.goal = self.sample_random_goal()
        else:
            self.goal = FIXED_GOAL_COORDS

        # Connect to physics client. By default, do not render
        self.physics_client = p.connect(p.DIRECT)

        # Load URDFs
        self.create_world()

        # reset environment
        self.reset()

    def sample_random_goal(self):
        """ Sample random goal """
        return np.random.uniform(low=MIN_GOAL_COORDS, high=MAX_GOAL_COORDS)

    def create_world(self):
        """ Setup camera and load URDFs"""

        # Initialise camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=0.6,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0.2, 0, 0.1],
            physicsClientId=self.physics_client)

        # Load robot, sphere and plane urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        path = os.path.abspath(os.path.dirname(__file__))
        self.arm = p.loadURDF(
            os.path.join(
                path,
                "URDFs/widowx/widowx.urdf"),
            useFixedBase=True)
        self.sphere = p.loadURDF(
            os.path.join(
                path,
                "URDFs/sphere.urdf"),
            useFixedBase=True)
        self.plane = p.loadURDF('plane.urdf')

    def reset(self):
        """
        Reset robot and goal at the beginning of an episode.
        Returns observation
        """

        if self.random_goal:
            self.goal = self.sample_random_goal()

        # Reset robot at the origin and move sphere to the goal position
        p.resetBasePositionAndOrientation(
            self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(
            self.sphere, self.goal, p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))

        # Reset joint at initial angles
        self._force_joint_positions(RESET_VALUES)

        # Get observation
        if self.obs_type == 1:
            self.obs = self._get_obs1()
        elif self.obs_type == 2:
            self.obs = self._get_obs2()
        elif self.obs_type == 3:
            self.obs = self._get_obs3()
        elif self.obs_type == 4:
            self.obs = self._get_obs4()
        elif self.obs_type == 5:
            self.obs = self._get_obs5()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        return self.obs

    def _get_general_obs(self):
        """ Get information for generating observation array """
        self.endeffector_pos = self._get_end_effector_position()
        self.torso_pos = self._get_torso_position()
        self.end_torso = self.endeffector_pos - self.torso_pos
        self.end_goal = self.endeffector_pos - self.goal
        self.joint_positions = self._get_joint_positions()

    def _get_obs1(self):
        """ Returns observation #1 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.endeffector_pos, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs2(self):
        """ Returns observation #2 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.goal, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs3(self):
        """ Returns observation #3 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_torso, self.end_goal, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs4(self):
        """ Returns observation #4 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_goal, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs5(self):
        """ Returns observation #5 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.end_torso, self.end_goal, self.goal, self.joint_positions]).ravel()

        return robot_obs

    def _get_joint_positions(self):
        """ Return current joint position """
        return np.array([x[0] for x in p.getJointStates(self.arm, range(6))])

    def _get_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(p.getLinkState(
                self.arm,
                5,
                computeForwardKinematics=True)
            [0])

    def _get_torso_position(self):
        """ Get torso coordinates """
        return np.array(p.getLinkState(
                self.arm,
                0,
                computeForwardKinematics=True)
            [0])

    def _get_goal_oriented_obs(self):
        """ return goal_oriented observation """
        obs = {}
        obs['observation'] = self.obs
        obs['desired_goal'] = self.goal
        obs['achieved_goal'] = self.endeffector_pos
        return obs

    def step(self, action):
        """
        Execute the action and return obs, reward, episode_over, info (tuple)

        Parameters
        ----------
        action (array)

        Returns
        -------
        obs, reward, episode_over, info : tuple
            obs (array)
            reward (float)
            episode_over (bool)
            info (dict)
        """

        # get distance and end effector position before taking the action
        self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal)
        self.old_endeffector_pos = self.endeffector_pos

        # take action
        self.normalized_action = np.array(action, dtype=np.float32)

        if self.normalize_action:
            for i in range(6):
                self.action[i] = self._normalize_scalar(
                    self.normalized_action[i],
                    self.normalized_action_min[i],
                    self.normalized_action_max[i],
                    self.action_min[i],
                    self.action_max[i])

        if self.action_type == 1:
            self._take_action1()

        # get observation
        if self.obs_type == 1:
            self.obs = self._get_obs1()
        elif self.obs_type == 2:
            self.obs = self._get_obs2()
        elif self.obs_type == 3:
            self.obs = self._get_obs3()
        elif self.obs_type == 4:
            self.obs = self._get_obs4()
        elif self.obs_type == 5:
            self.obs = self._get_obs5()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        # get new distance
        self.dist = np.linalg.norm(self.endeffector_pos - self.goal)

        # get reward
        if self.reward_type == 1:
            self.reward = self._get_reward1()
        elif self.reward_type == 2:
            self.reward = self._get_reward2()
        elif self.reward_type == 3:
            self.reward = self._get_reward3()
        elif self.reward_type == 4:
            self.reward = self._get_reward4()
        elif self.reward_type == 5:
            self.reward = self._get_reward5()
        elif self.reward_type == 6:
            self.reward = self._get_reward6()
        elif self.reward_type == 7:
            self.reward = self._get_reward7()
        elif self.reward_type == 8:
            self.reward = self._get_reward8()
        elif self.reward_type == 9:
            self.reward = self._get_reward9()
        elif self.reward_type == 10:
            self.reward = self._get_reward10()
        elif self.reward_type == 11:
            self.reward = self._get_reward11()
        elif self.reward_type == 12:
            self.reward = self._get_reward12()

        # Create info
        self.delta_dist = self.old_dist - self.dist
        self.delta_pos = np.linalg.norm(self.old_endeffector_pos - self.endeffector_pos)

        info = {}
        info['old_distance'] = self.old_dist
        info['distance'] = self.dist
        info['goal_pos'] = self.goal
        info['old_endeffector_pos'] = self.old_endeffector_pos
        info['endeffector_pos'] = self.endeffector_pos
        info['joint_pos'] = self.joint_positions
        info['joint_min'] = self.joint_min
        info['joint_max'] = self.joint_max
        info['term1'] = self.term1
        info['term2'] = self.term2
        info['normalized_action'] = self.normalized_action
        info['action'] = self.action
        info['normalized_action_min'] = self.normalized_action_min
        info['normalized_action_max'] = self.normalized_action_max
        info['action_min'] = self.action_min
        info['action_max'] = self.action_max
        # According to the Pybullet documentation, 1 timestep = 240 Hz
        info['vel_dist'] = self.delta_dist * 240
        info['vel_pos'] = self.delta_pos * 240

        # Create "episode_over": never end episode prematurily
        episode_over = False
        # if self.new_distance < 0.0005:
        #     episode_over = True

        return self.obs, self.reward, episode_over, info

    def _take_action1(self):
        """ select action #1 (increments from previous joint position """
        # Update the new joint position with the action
        self.new_joint_positions = self.joint_positions + self.action

        # Clip the joint position to fit the joint's allowed boundaries
        self.new_joint_positions = np.clip(
            np.array(self.new_joint_positions),
            self.joint_min,
            self.joint_max)

        # Instantaneously reset the joint position (no torque applied)
        self._force_joint_positions(self.new_joint_positions)

    def _normalize_scalar(self, var, old_min, old_max, new_min, new_max):
        """ normalize scalar var from one range to another """
        return ((new_max - new_min) * (var - old_min) / (old_max - old_min)) + new_min

    def _get_reward1(self):
        """ Compute reward function 1 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward2(self):
        """ Compute reward function 2 (dense) """
        self.term1 = 1 / abs(self.normalized_action[0])
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward3(self):
        """ Compute reward function 3 (dense) """
        self.term1 = 1
        self.term2 = - abs(self.normalized_action[0])
        rew = self.term1 + self.term2
        return rew

    def _get_reward4(self):
        """ Compute reward function 4 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha * np.linalg.norm(self.normalized_action)
        rew = self.term1 + self.term2
        return rew

    def _get_reward5(self):
        """ Compute reward function 5 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha * np.linalg.norm(self.normalized_action) / self.dist ** 2
        rew = self.term1 + self.term2
        return rew

    def _get_reward6(self):
        """ Compute reward function 6 (dense) """
        self.term1 = self.delta_dist
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward7(self):
        """ Compute reward function 7 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = self.alpha * self.delta_dist / self.dist
        rew = self.term1 + self.term2
        return rew

    def _get_reward8(self):
        """ Compute reward function 8 (dense) """
        self.term1 = self.delta_pos
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward9(self):
        """ Compute reward function 9 (dense) """
        self.term1 = - self.dist ** 2
        self.term2 = - self.alpha * self.delta_pos / self.dist
        rew = self.term1 + self.term2
        return rew

    def _get_reward10(self):
        """ Compute reward function 10 (sparse) """
        if self.dist >= 0.001:
            self.term1 = -1
        else:
            self.term1 = 0
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward11(self):
        """ Compute reward function 11 (sparse) """
        if self.dist >= 0.001:
            self.term1 = 0
        else:
            self.term1 = 1
        self.term2 = 0
        rew = self.term1 + self.term2
        return rew

    def _get_reward12(self):
        """ Compute reward function 12 (dense) """
        self.term1 = np.linalg.norm([1, 1, 1, 1, 1, 1])
        self.term2 = - np.linalg.norm(self.normalized_action)
        rew = self.term1 + self.term2
        return rew

    def render(self, mode='human'):
        """ Render Pybullet simulation """
        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI)
        self.create_world()

    def compute_reward(self, achieved_goal, goal):
        """ Function necessary for goal Env"""
        return - (np.linalg.norm(achieved_goal - goal)**2)

    def _set_joint_positions(self, joint_positions):
        """ Position control (not reset) """
        # In Pybullet, gripper halves are controlled separately
        joint_positions = list(joint_positions) + [joint_positions[-1]]
        p.setJointMotorControlArray(
            self.arm,
            [0, 1, 2, 3, 4, 7, 8],
            controlMode=p.POSITION_CONTROL,
            targetPositions=joint_positions
        )

    def _force_joint_positions(self, joint_positions):
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

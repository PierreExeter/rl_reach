import gym
import pybullet as p
import pybullet_data
import os
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

# Joint boundaries
JOINT_MIN = np.array([
    -3.1,
    -1.6,
    -1.6,
    -1.8,
    -3.1,
    0.0
])

JOINT_MAX = np.array([
    3.1,
    1.6,
    1.6,
    1.8,
    3.1,
    0.0
])


MIN_GOAL_COORDS = np.array([-.14, -.13, 0.26])
MAX_GOAL_COORDS = np.array([.14, .13, .39])
FIXED_GOAL_COORDS  = np.array([.14, .0, 0.26])
MIN_END_EFF_COORDS = np.array([-.16, -.15, 0.14])
MAX_END_EFF_COORDS = np.array([.16, .15, .41])


class WidowxEnv(gym.Env):

    def __init__(self):
        """
        Initialise the environment
        """

        self.random_goal = False
        self.goal_oriented = False
        self.obs_type = 3
        self.reward_type = 1
        self.action_type = 1

        # Define action space
        self.action_space = spaces.Box(
            low=np.float32(np.array([-0.5, -0.25, -0.25, -0.25, -0.5, -0.005]) / 30),
            high=np.float32(np.array([0.5, 0.25, 0.25, 0.25, 0.5, 0.005]) / 30),
            dtype=np.float32)

        # Define observation space
        if self.obs_type == 1:
            self.obs_space_low = np.float32(np.concatenate((MIN_END_EFF_COORDS, JOINT_MIN), axis=0))
            self.obs_space_high = np.float32(np.concatenate((MAX_END_EFF_COORDS, JOINT_MAX), axis=0))
            
        elif self.obs_type == 2:
            self.obs_space_low = np.float32(np.concatenate((MIN_GOAL_COORDS, JOINT_MIN), axis=0))
            self.obs_space_high = np.float32(np.concatenate((MAX_GOAL_COORDS, JOINT_MAX), axis=0))
            
        elif self.obs_type == 3:
            self.obs_space_low = np.float32(np.concatenate(([-1.0]*6, JOINT_MIN), axis=0))
            self.obs_space_high = np.float32(np.concatenate(([1.0]*6, JOINT_MAX), axis=0))
            
        elif self.obs_type == 4:
            self.obs_space_low = np.float32(np.concatenate(([-1.0]*3, JOINT_MIN), axis=0))
            self.obs_space_high = np.float32(np.concatenate(([1.0]*3, JOINT_MAX), axis=0))
            
        elif self.obs_type == 5:
            self.obs_space_low = np.float32(np.concatenate(([-1.0]*6, MIN_GOAL_COORDS, JOINT_MIN), axis=0))
            self.obs_space_high = np.float32(np.concatenate(([1.0]*6, MAX_GOAL_COORDS, JOINT_MAX), axis=0))

        self.observation_space = spaces.Box(
                    low=self.obs_space_low,
                    high=self.obs_space_high,
                    dtype=np.float32)

        if self.goal_oriented:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(low=np.float32(MIN_GOAL_COORDS), high=np.float32(MAX_GOAL_COORDS), dtype=np.float32),
                achieved_goal=spaces.Box(low=np.float32(MIN_END_EFF_COORDS), high=np.float32(MAX_END_EFF_COORDS), dtype=np.float32),
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
        """ sample random goal """
        return np.random.uniform(low=MIN_GOAL_COORDS, high=MAX_GOAL_COORDS)

    def create_world(self):

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
        self.endeffector_pos = self._get_end_effector_position()
        self.torso_pos = self._get_torso_position()
        self.ET = self.endeffector_pos - self.torso_pos 
        self.EG = self.endeffector_pos - self.goal
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
            [self.ET, self.EG, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs4(self):
        """ Returns observation #4 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.EG, self.joint_positions]).ravel()

        return robot_obs

    def _get_obs5(self):
        """ Returns observation #5 """
        self._get_general_obs()

        robot_obs = np.concatenate(
            [self.ET, self.EG, self.goal, self.joint_positions]).ravel()

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

        # take action
        if self.action_type == 1:
            self._take_action1(action)

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
        elif self.obs_type == 2:
            self.obs = self._get_obs2()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        # get reward
        if self.reward_type == 1:
            self.reward = self._get_reward1()

        # Create info
        info = {}
        info['distance'] = np.linalg.norm(self.endeffector_pos - self.goal)
        info['goal_pos'] = self.goal
        info['endeffector_pos'] = self.endeffector_pos
        info['joint_pos'] = self.joint_positions

        # Create "episode_over": never end episode prematurily
        episode_over = False
        # if self.new_distance < 0.0005:
        #     episode_over = True

        return self.obs, self.reward, episode_over, info

    def _take_action1(self, action):
        """ select action #1 (increments from previous joint position """
        self.action = np.array(action, dtype=np.float32)

        # Update the new joint position with the action
        self.new_joint_positions = self.joint_positions + self.action

        # Clip the joint position to fit the joint's allowed boundaries
        self.new_joint_positions = np.clip(
            np.array(self.new_joint_positions),
            JOINT_MIN,
            JOINT_MAX)

        # Instantaneously reset the joint position (no torque applied)
        self._force_joint_positions(self.new_joint_positions)

    def _get_reward1(self):
        """ Calculate the reward as - distance **2 """
        rew = - (np.linalg.norm(self.endeffector_pos - self.goal) ** 2)
        return rew

    def render(self, mode='human'):
        """ Render Pybullet simulation """
        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI)
        self.create_world()

    def compute_reward(self, achieved_goal, goal, info):
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

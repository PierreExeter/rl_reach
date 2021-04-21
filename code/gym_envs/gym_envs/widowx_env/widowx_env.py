"""
Implements the Gym training environment in Pybullet
WidowX MK-II robot manipulator reaching a target position
"""

import os
import copy
import gym
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
from .env_description import ObservationShapes, ActionShapes, RewardFunctions


# Initial joint angles
RESET_VALUES = [
    0.015339807878856412,
    -1.2931458041875956,
    1.0109710760673565,
    -1.3537670644267164,
    -0.07158577010132992,
    0]


# MIN_GOAL_COORDS = np.array([-.14, -.13, 0.26])  # changed by Pierre: adding obstacle
# MAX_GOAL_COORDS = np.array([.14, .13, .39]) # changed by Pierre: adding obstacle
MIN_GOAL_COORDS = np.array([-.2, -.13, 0.26])
MAX_GOAL_COORDS = np.array([.2, .13, .39])
# MIN_END_EFF_COORDS = np.array([-.16, -.15, 0.14]) # changed by Pierre: adding obstacle
# MAX_END_EFF_COORDS = np.array([.16, .15, .41]) # changed by Pierre: adding obstacle
MIN_END_EFF_COORDS = np.array([-.21, -.15, 0.14])
MAX_END_EFF_COORDS = np.array([.21, .15, .41])
FIXED_GOAL_COORDS_SPHERE = np.array([.14, .0, 0.26])
FIXED_GOAL_COORDS_SPHERE2 = np.array([.2, .0, 0.26])
FIXED_GOAL_COORDS_ARROW = np.array([.0, .0, 0.26])
FIXED_GOAL_COORDS_MOVING = np.array([.14, -.125, 0.26])
MIN_GOAL_ORIENTATION = np.array([-np.pi, 0, 0])
MAX_GOAL_ORIENTATION = np.array([np.pi, 0, 0])
FIXED_GOAL_ORIENTATION  = np.array([-np.pi/4, 0, -np.pi/2])
ARROW_OBJECT_ORIENTATION_CORRECTION = np.array([np.pi/2 , 0, 0])
FIXED_OBSTACLE_ORIENTATION  = np.array([0, np.pi/2, 0])
FIXED_OBSTACLE_POS = np.array([0.1, .0, 0.26])
TARGET_SPEED = 0.0025


class WidowxEnv(gym.Env):
    """ WidowX reacher Gym environment """

    def __init__(
        self,
        random_position,
        random_orientation,
        moving_target,
        target_type,
        goal_oriented,
        obstacle,
        obs_type,
        reward_type,
        action_type,
        joint_limits,
        action_min,
        action_max,
        alpha_reward,
        reward_coeff,
        lidar,
        camera_sensor,
        frame_skip,
        pybullet_action_coeff,
        widowx_type):
        """
        Initialise the environment
        """

        self.random_position = random_position
        self.random_orientation = random_orientation
        self.moving_target = moving_target
        self.target_type = target_type
        self.goal_oriented = goal_oriented
        self.obstacle = obstacle
        self.obs_type = obs_type
        self.reward_type = reward_type
        self.action_type = action_type
        self.joint_limits = joint_limits
        self.action_min = np.array(action_min)
        self.action_max = np.array(action_max)
        self.alpha_reward = alpha_reward
        self.reward_coeff = reward_coeff
        self.lidar = lidar
        self.camera_sensor = camera_sensor
        self.frame_skip = frame_skip
        self.pybullet_action_coeff = pybullet_action_coeff
        self.widowx_type = widowx_type

        self.endeffector_pos = np.zeros(3)
        self.old_endeffector_pos = np.zeros(3)
        self.endeffector_orient = np.zeros(3)
        self.old_endeffector_orient = np.zeros(3)
        self.torso_pos = np.zeros(3)
        self.torso_orient = np.zeros(3)
        self.end_torso_pos = np.zeros(3)
        self.end_goal_pos = np.zeros(3)
        self.end_torso_orient = np.zeros(3)
        self.end_goal_orient = np.zeros(3)
        self.joint_positions = np.zeros(6)
        self.new_joint_positions = np.zeros(6)
        self.delta_orient = np.zeros(3)
        self.delta_endeff_orient = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.goal_orient = np.zeros(3)
        self.target_object_orient = np.zeros(3)
        self.reward = None
        self.obs = None
        self.action = np.zeros(6)
        self.pybullet_action = np.zeros(6)
        # self.pybullet_action_min = np.array([-0.05, -0.025, -0.05, -0.05, -0.05, 0]) * self.pybullet_action_coeff
        # self.pybullet_action_max = np.array([0.05, 0.025, 0.05, 0.05, 0.05, 0.025]) * self.pybullet_action_coeff
        self.pybullet_action_min = np.array([-0.05, -0.025, -0.025, -0.025, -0.05, 0]) * self.pybullet_action_coeff # old values
        self.pybullet_action_max = np.array([0.05, 0.025, 0.025, 0.025, 0.05, 0.025]) * self.pybullet_action_coeff # old values
        # self.pybullet_action_min = np.array([-3, -0.3, -0.3, -0.5, -0.5, 0])  # with action2
        # self.pybullet_action_max = np.array([3, 0.3, 0.3, 0.5, 0.5, 0.025])   # with action2
        self.dist = 0
        self.old_dist = 0
        self.orient = 0
        self.old_orient = 0
        self.term1 = 0
        self.term2 = 0
        self.delta_pos = 0
        self.delta_dist = 0

        # render settings
        self.renderer = p.ER_TINY_RENDERER  # p.ER_BULLET_HARDWARE_OPENGL
        self._width = 224
        self._height = 224
        self._cam_dist = 0.8
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._cam_roll = 0
        self.camera_target_pos = [0.2, 0, 0.1]
        self._screen_width = 3840 #1920
        self._screen_height = 2160 #1080

        # Define action space
        self.action_space = spaces.Box(
                low=np.float32(self.action_min),
                high=np.float32(self.action_max),
                dtype=np.float32)

        # Define observation space
        if self.joint_limits == "small":
            # original values
            # self.joint_min = np.array([-3.1, -1.6, -1.6, -1.8, -3.1, 0.0])
            # self.joint_max = np.array([3.1, 1.6, 1.6, 1.8, 3.1, 0.0])

            # measured joint amplitude
            self.joint_min = np.array([-2.6, -1.5, -1.5, -1.7, -2.6, 0.0])
            self.joint_max = np.array([2.6, 0.8, 1.5, 1.7, 2.6, 0.0])
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

        elif self.obs_type == 6:
            self.obs_space_low = np.float32(
                np.concatenate((
                    [-1.0]*6,
                    [-2*np.pi]*6,
                    MIN_GOAL_COORDS,
                    MIN_GOAL_ORIENTATION,
                    MIN_END_EFF_COORDS,
                    [-np.pi]*3,
                    self.joint_min
                    ), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((
                    [1.0]*6,
                    [2*np.pi]*6,
                    MAX_GOAL_COORDS,
                    MAX_GOAL_ORIENTATION,
                    MAX_END_EFF_COORDS,
                    [np.pi]*3,
                    self.joint_max
                    ), axis=0))

        elif self.obs_type == 7:
            self.obs_space_low = np.float32(
                np.concatenate((
                    [-1.0]*6,
                    MIN_GOAL_COORDS,
                    MIN_END_EFF_COORDS,
                    self.joint_min
                    ), axis=0))
            self.obs_space_high = np.float32(
                np.concatenate((
                    [1.0]*6,
                    MAX_GOAL_COORDS,
                    MAX_END_EFF_COORDS,
                    self.joint_max
                    ), axis=0))

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

        # Connect to physics client
        self.physics_client = p.connect(p.DIRECT)
        
        # Load URDFs
        self.create_world()

        self.action_shape = ActionShapes(
            self.pybullet_action,
            self.joint_positions,
            self.joint_min,
            self.joint_max,
            self.arm,
            self.physics_client,
            self.frame_skip)

        self.obs_shape = ObservationShapes(
            self.endeffector_pos,
            self.endeffector_orient,
            self.torso_pos,
            self.torso_orient,
            self.goal_pos,
            self.goal_orient,
            self.joint_positions
        )

        self.reward_function = RewardFunctions(
            self.dist,
            self.alpha_reward,
            self.action,
            self.delta_dist,
            self.delta_pos,
            self.orient,
            self._detect_collision()
            )

    def sample_random_position(self):
        """ Sample random target position """
        return np.random.uniform(low=MIN_GOAL_COORDS, high=MAX_GOAL_COORDS)

    def sample_random_orientation(self):
        """ Sample random target orientation """
        return np.random.uniform(low=MIN_GOAL_ORIENTATION, high=MAX_GOAL_ORIENTATION)

    def create_world(self):
        """ Setup camera and load URDFs"""

        # # Set gravity
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        # # p.setGravity(0, 0, 0, physicsClientId=self.physics_client)

        # Load robot, target object and plane urdf
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        path = os.path.abspath(os.path.dirname(__file__))

        if self.widowx_type == "normal":
            self.arm = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/widowx/widowx.urdf"),
                useFixedBase=True)
        elif self.widowx_type == "light":
            self.arm = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/widowx/widowx_light.urdf"),
                useFixedBase=True)

        if self.target_type == "arrow":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/arrow.urdf"),
                useFixedBase=True)
        elif self.target_type == "sphere":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/sphere.urdf"),
                useFixedBase=True)

        if self.obstacle == "circular_window":
            self.obstacle_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/circular_window.urdf"),
                useFixedBase=True)
        elif self.obstacle == "circular_window_small":
            self.obstacle_object = p.loadURDF(
                os.path.join(
                    path,
                    "URDFs/circular_window_small_vhacd.urdf"),
                useFixedBase=True)

        self.plane = p.loadURDF('plane.urdf')

        # Define lidar rays
        if self.lidar:
            self.rayFrom = []
            self.rayTo = []
            self.rayIds = []

            self.lidar_height = 0.26
            self.numRays = 96  # 1024
            self.rayLen = 1
            self.rayHitColor = [1, 0, 0]  # green
            self.rayMissColor = [0, 1, 0] # red
            self.lidarStartPositon = [0, 0, self.lidar_height]

            for i in range(self.numRays):
                self.rayFrom.append(self.lidarStartPositon)
                self.rayTo.append([
                self.rayLen * np.sin(2. * np.pi * float(i) / self.numRays),
                self.rayLen * np.cos(2. * np.pi * float(i) / self.numRays), self.lidar_height])

                # replace lines
                self.rayIds.append(p.addUserDebugLine(self.rayFrom[i], self.rayTo[i], self.rayMissColor))

    def reset(self):
        """
        Reset robot and goal at the beginning of an episode.
        Returns observation
        """

        # Initialise goal position
        if self.random_position:
            self.goal_pos = self.sample_random_position()
        else:
            if self.moving_target:
                # deepcopy is necessary to avoid changing the value of FIXED_GOAL_COORDS_MOVING
                self.goal_pos = copy.deepcopy(FIXED_GOAL_COORDS_MOVING)
            else:
                if self.obstacle == "circular_window" or self.obstacle == "circular_window_small":
                    if self.target_type == "arrow":
                        self.goal_pos = FIXED_GOAL_COORDS_ARROW
                    elif self.target_type == "sphere":
                        self.goal_pos = FIXED_GOAL_COORDS_SPHERE2
                else:
                    if self.target_type == "arrow":
                        self.goal_pos = FIXED_GOAL_COORDS_ARROW
                    elif self.target_type == "sphere":
                        self.goal_pos = FIXED_GOAL_COORDS_SPHERE

        # Initialise goal orientation
        if self.random_orientation:
            self.goal_orient = self.sample_random_orientation()
        else:
            self.goal_orient = FIXED_GOAL_ORIENTATION

        # Initialise obstacle position and orientation
        self.obstacle_pos = FIXED_OBSTACLE_POS
        self.obstacle_orient = FIXED_OBSTACLE_ORIENTATION

        # Correct the orientation of the target object for consistency with rendering
        # in Pybullet (This is due to the arrow's STL being oriented along a different axis)
        self.target_object_orient = self.goal_orient + ARROW_OBJECT_ORIENTATION_CORRECTION

        # Reset robot at the origin and move the target object to the goal position and orientation
        p.resetBasePositionAndOrientation(
            self.arm, [0, 0, 0], p.getQuaternionFromEuler([np.pi, np.pi, np.pi]))
        p.resetBasePositionAndOrientation(
            self.target_object, self.goal_pos, p.getQuaternionFromEuler(self.target_object_orient))

        if self.obstacle is not None:
            p.resetBasePositionAndOrientation(
                self.obstacle_object, self.obstacle_pos, p.getQuaternionFromEuler(self.obstacle_orient))

        # Reset joint at initial angles
        self.action_shape.force_joint_positions(RESET_VALUES)

        # Get observation
        self._get_general_obs()
        self.obs_shape = ObservationShapes(
            self.endeffector_pos,
            self.endeffector_orient,
            self.torso_pos,
            self.torso_orient,
            self.goal_pos,
            self.goal_orient,
            self.joint_positions
        )

        if self.obs_type == 1:
            self.obs = self.obs_shape.get_obs1()
        elif self.obs_type == 2:
            self.obs = self.obs_shape.get_obs2()
        elif self.obs_type == 3:
            self.obs = self.obs_shape.get_obs3()
        elif self.obs_type == 4:
            self.obs = self.obs_shape.get_obs4()
        elif self.obs_type == 5:
            self.obs = self.obs_shape.get_obs5()
        elif self.obs_type == 6:
            self.obs = self.obs_shape.get_obs6()
        elif self.obs_type == 7:
            self.obs = self.obs_shape.get_obs7()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        return self.obs

    def _get_general_obs(self):
        """ Get information for generating observation array """
        self.endeffector_pos = self._get_end_effector_position()
        self.endeffector_orient = self._get_end_effector_orientation()
        self.torso_pos = self._get_torso_position()
        self.torso_orient = self._get_torso_orientation()
        self.joint_positions, self.joint_vel, self.joint_rf, self.joint_torques = self._get_joint_info()

    def _get_joint_info(self):
        """ Return current joint positions, velocities, reaction forces and torques """
        # np.array([x[0] for x in p.getJointStates(self.arm, range(6))])

        info = p.getJointStates(self.arm, range(6))
        pos, vel, rf, t = [], [], [], []

        for joint_info in info:
            pos.append(joint_info[0])
            vel.append(joint_info[1])
            rf.append(joint_info[2])
            t.append(joint_info[3])

        return np.array(pos), np.array(vel), np.array(rf), np.array(t)

    def _get_end_effector_position(self):
        """ Get end effector coordinates """
        return np.array(p.getLinkState(
                self.arm,
                5,
                computeForwardKinematics=True)
            [0])

    def _get_end_effector_orientation(self):
        """ Get end effector orientation """
        orient_quat = p.getLinkState(self.arm, 5, computeForwardKinematics=True)[1]
        orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_euler)

    def _get_torso_position(self):
        """ Get torso coordinates """
        return np.array(p.getLinkState(
                self.arm,
                0,
                computeForwardKinematics=True)
            [0])

    def _get_torso_orientation(self):
        """ Get torso orientation """
        orient_quat = p.getLinkState(self.arm, 0, computeForwardKinematics=True)[1]
        orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_euler)

    def _get_goal_oriented_obs(self):
        """ return goal_oriented observation """
        obs = {}
        obs['observation'] = self.obs
        obs['desired_goal'] = self.goal_pos
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
        self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.old_endeffector_pos = self.endeffector_pos
        self.old_orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)
        self.old_endeffector_orient = self.endeffector_orient

        # Update target position and move the target object
        if self.moving_target:
            self.goal_pos[1] += TARGET_SPEED
            p.resetBasePositionAndOrientation(
                self.target_object,
                self.goal_pos,
                p.getQuaternionFromEuler(self.target_object_orient))

        # take action
        self.action = np.array(action, dtype=np.float32)

        # Scale action to pybullet range
        self._scale_action_pybullet()

        self.action_shape = ActionShapes(
            self.pybullet_action,
            self.joint_positions,
            self.joint_min,
            self.joint_max,
            self.arm,
            self.physics_client,
            self.frame_skip)

        if self.action_type == 1:
            self.action_shape.take_action1()
        elif self.action_type == 2:
            self.action_shape.take_action2()


        # print("desired joint position : ", self.action_shape.new_joint_positions)
        # print("actual joint position: ", self._get_joint_positions())

        # get observation
        self._get_general_obs()
        self.obs_shape = ObservationShapes(
            self.endeffector_pos,
            self.endeffector_orient,
            self.torso_pos,
            self.torso_orient,
            self.goal_pos,
            self.goal_orient,
            self.joint_positions
        )

        if self.obs_type == 1:
            self.obs = self.obs_shape.get_obs1()
        elif self.obs_type == 2:
            self.obs = self.obs_shape.get_obs2()
        elif self.obs_type == 3:
            self.obs = self.obs_shape.get_obs3()
        elif self.obs_type == 4:
            self.obs = self.obs_shape.get_obs4()
        elif self.obs_type == 5:
            self.obs = self.obs_shape.get_obs5()
        elif self.obs_type == 6:
            self.obs = self.obs_shape.get_obs6()
        elif self.obs_type == 7:
            self.obs = self.obs_shape.get_obs7()

        # update observation if goal oriented environment
        if self.goal_oriented:
            self.obs = self._get_goal_oriented_obs()

        # get new distance
        self.dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)

        # get reward
        self.reward_function = RewardFunctions(
            self.dist,
            self.alpha_reward,
            self.action,
            self.delta_dist,
            self.delta_pos,
            self.orient,
            self._detect_collision()
            )

        if self.reward_type == 1:
            self.reward = self.reward_function.get_reward1()
        elif self.reward_type == 2:
            self.reward = self.reward_function.get_reward2()
        elif self.reward_type == 3:
            self.reward = self.reward_function.get_reward3()
        elif self.reward_type == 4:
            self.reward = self.reward_function.get_reward4()
        elif self.reward_type == 5:
            self.reward = self.reward_function.get_reward5()
        elif self.reward_type == 6:
            self.reward = self.reward_function.get_reward6()
        elif self.reward_type == 7:
            self.reward = self.reward_function.get_reward7()
        elif self.reward_type == 8:
            self.reward = self.reward_function.get_reward8()
        elif self.reward_type == 9:
            self.reward = self.reward_function.get_reward9()
        elif self.reward_type == 10:
            self.reward = self.reward_function.get_reward10()
        elif self.reward_type == 11:
            self.reward = self.reward_function.get_reward11()
        elif self.reward_type == 12:
            self.reward = self.reward_function.get_reward12()
        elif self.reward_type == 13:
            self.reward = self.reward_function.get_reward13()
        elif self.reward_type == 14:
            self.reward = self.reward_function.get_reward14()
        elif self.reward_type == 15:
            self.reward = self.reward_function.get_reward15()
        elif self.reward_type == 16:
            self.reward = self.reward_function.get_reward16()
        elif self.reward_type == 17:
            self.reward = self.reward_function.get_reward17()
        elif self.reward_type == 18:
            self.reward = self.reward_function.get_reward18()
        elif self.reward_type == 19:
            self.reward = self.reward_function.get_reward19()
        elif self.reward_type == 20:
            self.reward = self.reward_function.get_reward20()
        elif self.reward_type == 21:
            self.reward = self.reward_function.get_reward21()

        # Apply reward coefficient
        self.reward *= self.reward_coeff

        # Create info
        self.delta_dist = self.old_dist - self.dist
        self.delta_pos = np.linalg.norm(self.old_endeffector_pos - self.endeffector_pos)
        self.delta_orient = self.old_orient - self.orient
        self.delta_endeff_orient = np.linalg.norm(self.old_endeffector_orient - self.endeffector_orient)

        info = {}
        info['distance'] = self.dist
        info['goal_pos'] = self.goal_pos
        info['endeffector_pos'] = self.endeffector_pos
        info['orientation'] = self.orient
        info['goal_orient'] = self.goal_orient
        info['endeffector_orient'] = self.endeffector_orient
        info['joint_pos'] = self.joint_positions
        info['joint_vel'] = self.joint_vel
        info['joint_tor'] = self.joint_torques
        info['desired_joint_pos'] =  self.action_shape.new_joint_positions
        info['joint_min'] = self.joint_min
        info['joint_max'] = self.joint_max
        info['term1'] = self.term1
        info['term2'] = self.term2
        info['action'] = self.action
        info['action_min'] = self.action_min
        info['action_max'] = self.action_max
        info['pybullet_action'] = self.pybullet_action
        info['pybullet_action_min'] = self.pybullet_action_min
        info['pybullet_action_max'] = self.pybullet_action_max
        # According to the Pybullet documentation, 1 timestep = 240 Hz
        info['vel_dist'] = self.delta_dist * 240
        info['vel_pos'] = self.delta_pos * 240
        info['collision'] = self._detect_collision()

        # Create "episode_over": never end episode prematurily
        episode_over = False
        # if self.new_distance < 0.0005:
        #     episode_over = True

        if self.lidar:
            self.results_lidar = p.rayTestBatch(self.rayFrom, self.rayTo)

            for i in range(self.numRays):
                # print(self.results_lidar[i])

                self.hitObjectUid = self.results_lidar[i][0]

                # if no object detected
                if self.hitObjectUid < 0:
                    self.hitPosition = [0, 0, 0]
                    p.addUserDebugLine(
                        self.rayFrom[i],
                        self.rayTo[i],
                        self.rayMissColor,
                        replaceItemUniqueId=self.rayIds[i])
                else:
                    self.hitPosition = self.results_lidar[i][3]
                    p.addUserDebugLine(
                        self.rayFrom[i],
                        self.hitPosition,
                        self.rayHitColor,
                        replaceItemUniqueId=self.rayIds[i])

        if self.camera_sensor:

            viewMatrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.camera_target_pos,
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=self._cam_roll,
                upAxisIndex=2)

            projectionMatrix = p.computeProjectionMatrixFOV(
                fov=60,
                aspect=self._width / self._height,
                nearVal=0.1,
                farVal=100.0)

            frame = p.getCameraImage(
                width=self._width,
                height=self._height,
                viewMatrix=viewMatrix,
                projectionMatrix=projectionMatrix,
                renderer=self.renderer)

            # frame = width, height, rgbImg, depthImg, segImg
            # rgb_array = np.array(frame[2])
            # depth_array = np.array(frame[3])
            # segmentation_array = np.array(frame[4])

        return self.obs, self.reward, episode_over, info

    def _detect_collision(self):
        """ Detect any collision with the arm (require physics enabled) """
        if len(p.getContactPoints(self.arm)) > 0:
            return True
        else:
            return False

    def _normalize_scalar(self, var, old_min, old_max, new_min, new_max):
        """ Normalize scalar var from one range to another """
        return ((new_max - new_min) * (var - old_min) / (old_max - old_min)) + new_min

    def _scale_action_pybullet(self):
        """ Scale action to Pybullet action range """
        for i in range(6):
            self.pybullet_action[i] = self._normalize_scalar(
                self.action[i],
                self.action_min[i],
                self.action_max[i],
                self.pybullet_action_min[i],
                self.pybullet_action_max[i])

    def render(self, mode='human'):
        """ Render Pybullet simulation """

        p.disconnect(self.physics_client)
        self.physics_client = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self._screen_width, self._screen_height))
        self.create_world()

        # Initialise debug camera angle
        p.resetDebugVisualizerCamera(
            cameraDistance=self._cam_dist,
            cameraYaw=self._cam_yaw,
            cameraPitch=self._cam_pitch,
            cameraTargetPosition=self.camera_target_pos,
            physicsClientId=self.physics_client)

    def compute_reward(self, achieved_goal, goal, info):
        """ Function necessary for goal Env"""
        return - (np.linalg.norm(achieved_goal - goal)**2)

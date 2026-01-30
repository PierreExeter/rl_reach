import os
import copy
from gymnasium import spaces
import numpy as np
import pybullet as p
from .env import RobotEnv
from .env_description import ObservationShapes, ActionShapes, RewardFunctions


class ReachingEnv(RobotEnv):

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
        alpha_reward,
        action_amplitude,
        observation_amplitude,
        robot_gains):

        super(ReachingEnv, self).__init__(
            action_amp=action_amplitude,
            obs_amp=observation_amplitude,
            frame_skip=5,
            time_step=0.02,
            action_robot_len=7,
            obs_robot_len=17)

        self.random_position = random_position
        self.random_orientation = random_orientation
        self.moving_target = moving_target
        self.target_type = target_type
        self.goal_oriented = goal_oriented  # Not implemented yet
        self.obstacle = obstacle
        self.obs_type = obs_type  # Not implemented yet
        self.reward_type = reward_type
        self.action_type = action_type  # Not implemented yet
        self.alpha_reward = alpha_reward
        self.robot_gains = robot_gains

        self.robot_forces = 1.0
        self.task_success_threshold = 0.03
        self.fixed_goal_coord = np.array([0.7, 0.0, 1.0])
        self.obstacle_pos = np.array([0.6, 0.0, 1.0])
        self.obstacle_orient = np.array([0, np.pi/2, 0])
        self.dist = 0
        self.old_dist = 0
        self.orient = 0
        self.old_orient = 0
        self.term1 = 0
        self.term2 = 0
        self.delta_pos = 0
        self.delta_dist = 0
        self.target_speed = 0.01

        self.endeffector_pos = np.zeros(3)
        self.old_endeffector_pos = np.zeros(3)
        self.endeffector_orient = np.zeros(4)   # this is a quaternion!
        self.old_endeffector_orient = np.zeros(4)  # this is a quaternion!
        self.torso_pos = np.zeros(3)
        self.torso_orient = np.zeros(4)
        self.end_torso_pos = np.zeros(3)
        self.end_goal_pos = np.zeros(3)
        self.end_torso_orient = np.zeros(4)
        self.end_goal_orient = np.zeros(4)
        self.delta_orient = np.zeros(3)
        self.delta_endeff_orient = np.zeros(3)
        self.goal_pos = np.zeros(3)
        self.goal_orient = np.zeros(3)
        self.target_object_orient = np.zeros(3)
        self.joint_positions = np.zeros(7)
        self.new_joint_positions = np.zeros(7)

        self.pybullet_action_min = - np.array([self.robot_gains]*self.action_robot_len)
        self.pybullet_action_max = np.array([self.robot_gains]*self.action_robot_len)

    def step(self, action):

        # get distance and end effector position before taking the action
        self.old_dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.old_endeffector_pos = self.endeffector_pos
        self.old_orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)
        self.old_endeffector_orient = self.endeffector_orient

        # Update target position and move the target object
        if self.moving_target:
            self.goal_pos[1] += self.target_speed
            p.resetBasePositionAndOrientation(
                self.target_object,
                self.goal_pos,
                p.getQuaternionFromEuler(self.target_object_orient))

        # Execute action
        self.take_step(
            action=action,
            gains=self.robot_gains,
            forces=self.robot_forces,
            indices=self.robot_arm_joint_indices,
            upper_limit=self.robot_upper_limits,
            lower_limit=self.robot_lower_limits,
            robot=self.robot)

        # Get observations
        self._get_general_obs()
        obs = self.get_obs1()

        # get distance and orientation
        self.dist = np.linalg.norm(self.endeffector_pos - self.goal_pos)
        self.orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)

        # get reward
        # reward = - self.dist

        self.reward_function = RewardFunctions(
            self.dist,
            self.alpha_reward,
            action,
            self.delta_dist,
            self.delta_pos,
            self.orient,
            self._detect_collision()
            )

        if self.reward_type == 1:
            reward = self.reward_function.get_reward1()
        elif self.reward_type == 2:
            reward = self.reward_function.get_reward2()
        elif self.reward_type == 3:
            reward = self.reward_function.get_reward3()
        elif self.reward_type == 4:
            reward = self.reward_function.get_reward4()
        elif self.reward_type == 5:
            reward = self.reward_function.get_reward5()
        elif self.reward_type == 6:
            reward = self.reward_function.get_reward6()
        elif self.reward_type == 7:
            reward = self.reward_function.get_reward7()
        elif self.reward_type == 8:
            reward = self.reward_function.get_reward8()
        elif self.reward_type == 9:
            reward = self.reward_function.get_reward9()
        elif self.reward_type == 10:
            reward = self.reward_function.get_reward10()
        elif self.reward_type == 11:
            reward = self.reward_function.get_reward11()
        elif self.reward_type == 12:
            reward = self.reward_function.get_reward12()
        elif self.reward_type == 13:
            reward = self.reward_function.get_reward13()
        elif self.reward_type == 14:
            reward = self.reward_function.get_reward14()
        elif self.reward_type == 15:
            reward = self.reward_function.get_reward15()
        elif self.reward_type == 16:
            reward = self.reward_function.get_reward16()
        elif self.reward_type == 17:
            reward = self.reward_function.get_reward17()
        elif self.reward_type == 18:
            reward = self.reward_function.get_reward18()
        elif self.reward_type == 19:
            reward = self.reward_function.get_reward19()
        elif self.reward_type == 20:
            reward = self.reward_function.get_reward20()
        elif self.reward_type == 21:
            reward = self.reward_function.get_reward21()
        elif self.reward_type == 22:
            reward = self.reward_function.get_reward22()
        elif self.reward_type == 23:
            reward = self.reward_function.get_reward23()
        elif self.reward_type == 24:
            reward = self.reward_function.get_reward24()
        elif self.reward_type == 25:
            reward = self.reward_function.get_reward25()
        elif self.reward_type == 26:
            reward = self.reward_function.get_reward26()
        elif self.reward_type == 27:
            reward = self.reward_function.get_reward27()

        # get info
        self.delta_dist = self.old_dist - self.dist
        self.delta_pos = np.linalg.norm(self.old_endeffector_pos - self.endeffector_pos)
        self.delta_orient = self.old_orient - self.orient
        self.delta_endeff_orient = np.linalg.norm(self.old_endeffector_orient - self.endeffector_orient)

        info = {}
        info['task_success'] = int(self.dist <= self.task_success_threshold)
        info['action_robot_len'] = self.action_robot_len
        info['obs_robot_len'] = self.obs_robot_len
        info['distance'] = self.dist
        info['goal_pos'] = self.goal_pos
        info['endeffector_pos'] = self.endeffector_pos
        info['orientation'] = self.orient
        info['goal_orient'] = self.goal_orient
        info['endeffector_orient'] = self.endeffector_orient
        info['joint_pos'] = self.joint_positions
        info['joint_vel'] = self.joint_vel
        info['joint_tor'] = self.joint_torques
        info['desired_joint_pos'] =  self.robot_joint_positions
        info['joint_min'] = self.robot_lower_limits
        info['joint_max'] = self.robot_upper_limits
        info['term1'] = self.term1
        info['term2'] = self.term2
        info['action'] = action
        info['action_min'] = self.action_space.low
        info['action_max'] = self.action_space.high
        info['pybullet_action'] = self.action_robot
        info['pybullet_action_min'] = self.pybullet_action_min
        info['pybullet_action_max'] = self.pybullet_action_max
        # # According to the Pybullet documentation, 1 timestep = 240 Hz
        info['vel_dist'] = self.delta_dist / self.time_step
        info['vel_pos'] = self.delta_pos / self.time_step
        info['collision'] = self._detect_collision()

        # get done
        done = False

        return np.array(obs, dtype=np.float32), reward, done, False, info

    def _detect_collision(self):
        """ Detect any collision with the arm (require physics enabled) """
        if len(p.getContactPoints(self.robot)) > 0:
            return True
        else:
            return False

    def _get_general_obs(self):
        """ Get information for generating observation array """
        self.endeffector_pos = self._get_end_effector_position()
        self.endeffector_orient = self._get_end_effector_orientation()
        self.torso_pos = self._get_torso_position()
        self.torso_orient = self._get_torso_orientation()
        self.joint_positions, self.joint_vel, self.joint_rf, self.joint_torques = self._get_joint_info()

        self.end_torso_pos = self.endeffector_pos - self.torso_pos
        self.end_goal_pos = self.endeffector_pos - self.goal_pos
        self.end_torso_orient = self.endeffector_orient - self.torso_orient
        self.end_goal_orient = self.endeffector_orient - self.goal_orient

    def _get_joint_info(self):
        """ Return current joint positions, velocities, reaction forces and torques """
        info = p.getJointStates(
            self.robot,
            jointIndices=self.robot_arm_joint_indices,
            physicsClientId=self.id)

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
                self.robot,
                linkIndex=8,
                computeForwardKinematics=True,
                physicsClientId=self.id)
            [0])

    def _get_end_effector_orientation(self):
        """ Get end effector orientation """
        orient_quat = p.getLinkState(
            self.robot,
            linkIndex=8,
            computeForwardKinematics=True,
            physicsClientId=self.id)[1]
        # orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_quat)

    def _get_torso_position(self):
        """ Get torso coordinates """
        return np.array(p.getLinkState(
                self.robot,
                linkIndex=0,
                computeForwardKinematics=True,
                physicsClientId=self.id)
            [0])

    def _get_torso_orientation(self):
        """ Get torso orientation """
        orient_quat = p.getLinkState(
            self.robot,
            linkIndex=0,
            computeForwardKinematics=True,
            physicsClientId=self.id)[1]
        # orient_euler = p.getEulerFromQuaternion(orient_quat)
        return np.array(orient_quat)

    def get_obs1(self):
        robot_obs = np.concatenate([self.end_torso_pos, self.end_goal_pos, self.joint_positions, self.endeffector_orient]).ravel()
        return robot_obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.setup_timing()
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.robot, self.robot_lower_limits, self.robot_upper_limits, self.robot_arm_joint_indices = self.world_creation.create_new_world(print_joints=False)

        self.robot_lower_limits = self.robot_lower_limits[self.robot_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_arm_joint_indices]
        self.reset_robot_joints(robot=self.robot)

        # Disable gravity
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        # p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)

        # Initialise goal position
        if self.random_position:
            self.goal_pos = self.sample_random_position()
        else:
            # deepcopy is necessary to avoid changing the value of fixed_goal_coord
            self.goal_pos = copy.deepcopy(self.fixed_goal_coord)

        # Initialise goal orientation
        if self.random_orientation:
            self.goal_orient = self.sample_random_orientation()
        else:
            self.goal_orient = p.getQuaternionFromEuler(
                np.array([0, np.pi/2.0, 0]),
                physicsClientId=self.id)

        # Spawn target object
        path = os.path.abspath(os.path.dirname(__file__))

        if self.target_type == "arrow":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "assets/URDFs/arrow.urdf"),
                useFixedBase=True)
        elif self.target_type == "sphere":
            self.target_object = p.loadURDF(
                os.path.join(
                    path,
                    "assets/URDFs/sphere.urdf"),
                useFixedBase=True)

        p.resetBasePositionAndOrientation(
            self.target_object,
            self.goal_pos,
            p.getQuaternionFromEuler(self.target_object_orient))

        # Spawn obstacle
        if self.obstacle == "circular_window":
            self.obstacle_object = p.loadURDF(
                os.path.join(
                    path,
                    "assets/URDFs/circular_window.urdf"),
                useFixedBase=True)
        elif self.obstacle == "circular_window_small":
            self.obstacle_object = p.loadURDF(
                os.path.join(
                    path,
                    "assets/URDFs/circular_window_small_vhacd.urdf"),
                useFixedBase=True)

        if self.obstacle is not None:
            p.resetBasePositionAndOrientation(
                self.obstacle_object,
                self.obstacle_pos,
                p.getQuaternionFromEuler(self.obstacle_orient))

        # OLD IMPLEMENTATION
        # sphere_collision = -1
        # sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
        # self.target = p.createMultiBody(
        #     baseMass=0.0,
        #     baseCollisionShapeIndex=sphere_collision,
        #     baseVisualShapeIndex=sphere_visual,
        #     basePosition=self.goal_pos,
        #     useMaximalCoordinates=False,
        #     physicsClientId=self.id)

        # Jaco
        _, _ , _ = self.position_robot_toc(
            robot=self.robot,
            joints=8,
            joint_indices=self.robot_arm_joint_indices,
            lower_limits=self.robot_lower_limits,
            upper_limits=self.robot_upper_limits,
            pos_offset=np.array([0, 0, 0.6]))

        self.world_creation.set_gripper_open_position(self.robot, position=1.1, set_instantly=True)

        # load tool
        self.tool = self.world_creation.init_tool(
            self.robot,
            pos_offset=[-0.01, 0, 0.03],
            orient_offset=p.getQuaternionFromEuler([0, -np.pi/2.0, 0],
            physicsClientId=self.id))

        # Load a nightstand in the environment for the jaco arm
        self.nightstand_scale = 0.275

        visual_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')
        collision_filename = os.path.join(self.world_creation.directory, 'nightstand', 'nightstand.obj')

        nightstand_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=visual_filename,
            meshScale=[self.nightstand_scale]*3,
            rgbaColor=[0.5, 0.5, 0.5, 1.0],
            physicsClientId=self.id)

        nightstand_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=collision_filename,
            meshScale=[self.nightstand_scale]*3,
            physicsClientId=self.id)

        nightstand_pos = np.array([0, 0, 0])
        nightstand_orient = p.getQuaternionFromEuler(
            np.array([np.pi/2.0, 0, 0]),
            physicsClientId=self.id)

        self.nightstand = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=nightstand_collision,
            baseVisualShapeIndex=nightstand_visual,
            basePosition=nightstand_pos,
            baseOrientation=nightstand_orient,
            baseInertialFramePosition=[0, 0, 0],
            useMaximalCoordinates=False,
            physicsClientId=self.id)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return np.array(self.get_obs1(), dtype=np.float32), {}

    def sample_random_position(self):

        # Assuming that the max reach of the Jaco is 0.7,
        # the target is generated randomly inside the inscribed cube of the reaching sphere
        goal_pos = np.array([0, 0, 0.95]) + np.array([
            self.np_random.uniform(-0.7*np.cos(np.pi/4), 0.7*np.cos(np.pi/4)),
            self.np_random.uniform(-0.7*np.cos(np.pi/4), 0.7*np.cos(np.pi/4)),
            self.np_random.uniform(0, 0.7*np.cos(np.pi/4))])

        return goal_pos

    def sample_random_orientation(self):
        """ Sample random target orientation """

        MIN_GOAL_ORIENTATION = np.array([-np.pi, np.pi/2.0, 0.0])
        MAX_GOAL_ORIENTATION = np.array([np.pi, np.pi/2.0, 0.0])

        euler_orient = np.random.uniform(low=MIN_GOAL_ORIENTATION, high=MAX_GOAL_ORIENTATION)

        return p.getQuaternionFromEuler(euler_orient, physicsClientId=self.id)

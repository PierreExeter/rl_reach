import os
from gym import spaces
import numpy as np
import pybullet as p
from .env import RobotEnv


class ReachingEnv(RobotEnv):

    def __init__(self):
        super(ReachingEnv, self).__init__(frame_skip=5, time_step=0.02, action_robot_len=7, obs_robot_len=17)
        self.robot_forces = 1.0
        self.robot_gains = 0.05
        self.distance_weight = 1.0
        self.task_success_threshold = 0.03

    def step(self, action):

        # Execute action
        self.take_step(action, gains=self.robot_gains, forces=self.robot_forces)

        # Get the Euclidean distance between the robot's end effector and the target
        gripper_pos = np.array(p.getLinkState(self.robot, 8, computeForwardKinematics=True, physicsClientId=self.id)[0])
        reward_distance_target = -np.linalg.norm(gripper_pos - self.target_pos)

        # self.orient = np.linalg.norm(self.endeffector_orient - self.goal_orient)
        
        # Get end effector velocity
        # end_effector_velocity = np.linalg.norm(p.getLinkState(self.robot, 8, computeForwardKinematics=True, computeLinkVelocity=True, physicsClientId=self.id)[6])

        # Get observations and reward
        obs = self._get_obs()
        reward = self.distance_weight * reward_distance_target

        info = {}
        info['task_success'] = int(-reward_distance_target <= self.task_success_threshold)
        info['action_robot_len'] = self.action_robot_len
        info['obs_robot_len'] = self.obs_robot_len

        info['distance'] = -reward_distance_target
        # info['goal_pos'] = self.target_pos
        # info['endeffector_pos'] = gripper_pos
        info['orientation'] = 1 # self.orient TO CHANGE LATER
        # info['goal_orient'] = self.goal_orient
        # info['endeffector_orient'] = self.endeffector_orient
        # info['joint_pos'] = self.joint_positions
        # info['joint_vel'] = self.joint_vel
        # info['joint_tor'] = self.joint_torques
        # info['desired_joint_pos'] =  self.action_shape.new_joint_positions
        # info['joint_min'] = self.joint_min
        # info['joint_max'] = self.joint_max
        # info['term1'] = self.term1
        # info['term2'] = self.term2
        # info['action'] = self.action
        # info['action_min'] = self.action_min
        # info['action_max'] = self.action_max
        # info['pybullet_action'] = self.pybullet_action
        # info['pybullet_action_min'] = self.pybullet_action_min
        # info['pybullet_action_max'] = self.pybullet_action_max
        # # According to the Pybullet documentation, 1 timestep = 240 Hz
        # info['vel_dist'] = self.delta_dist * 240
        # info['vel_pos'] = self.delta_pos * 240
        # info['collision'] = self._detect_collision()

        done = False

        return obs, reward, done, info

    def _get_obs(self):
        torso_pos = np.array(p.getLinkState(self.robot, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
        robot_joint_positions = np.array([x[0] for x in p.getJointStates(self.robot, jointIndices=self.robot_arm_joint_indices, physicsClientId=self.id)])
        gripper_pos, gripper_orient = p.getLinkState(self.robot, 8, computeForwardKinematics=True, physicsClientId=self.id)[:2]

        robot_obs = np.concatenate([gripper_pos-torso_pos, gripper_pos-self.target_pos, robot_joint_positions, gripper_orient]).ravel()

        return robot_obs

    def reset(self):
        self.setup_timing()
        self.task_success = 0
        self.contact_points_on_arm = {}
        self.robot, self.robot_lower_limits, self.robot_upper_limits, self.robot_arm_joint_indices = self.world_creation.create_new_world(print_joints=False)
        
        self.robot_lower_limits = self.robot_lower_limits[self.robot_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_arm_joint_indices]
        self.reset_robot_joints()

        # Disable gravity
        # p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        # p.setGravity(0, 0, 0, body=self.robot, physicsClientId=self.id)

        # assuming that the max reach of the Jaco is 0.7, the target is generated randomly inside the inscribed cube of the reaching sphere
        self.target_pos = np.array([0, 0, 0.95]) + np.array([self.np_random.uniform(-0.7*np.cos(np.pi/4), 0.7*np.cos(np.pi/4)), self.np_random.uniform(-0.7*np.cos(np.pi/4), 0.7*np.cos(np.pi/4)), self.np_random.uniform(0, 0.7*np.cos(np.pi/4))])
        self.target_orient = p.getQuaternionFromEuler(np.array([0, np.pi/2.0, 0]), physicsClientId=self.id)
        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 1, 0, 1], physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision, baseVisualShapeIndex=sphere_visual, basePosition=self.target_pos, useMaximalCoordinates=False, physicsClientId=self.id)

        # Jaco
        _, _ , _ = self.position_robot_toc(self.robot, 8, self.robot_arm_joint_indices, self.robot_lower_limits, self.robot_upper_limits, pos_offset=np.array([0, 0, 0.6]))
        
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
        nightstand_visual = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=visual_filename, meshScale=[self.nightstand_scale]*3, rgbaColor=[0.5, 0.5, 0.5, 1.0], physicsClientId=self.id)
        nightstand_collision = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=collision_filename, meshScale=[self.nightstand_scale]*3, physicsClientId=self.id)
        nightstand_pos = np.array([0, 0, 0])
        nightstand_orient = p.getQuaternionFromEuler(np.array([np.pi/2.0, 0, 0]), physicsClientId=self.id)
        self.nightstand = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=nightstand_collision, baseVisualShapeIndex=nightstand_visual, basePosition=nightstand_pos, baseOrientation=nightstand_orient, baseInertialFramePosition=[0, 0, 0], useMaximalCoordinates=False, physicsClientId=self.id)
        
        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        return self._get_obs()

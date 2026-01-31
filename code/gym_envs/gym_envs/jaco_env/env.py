import os, time, datetime
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
import pybullet as p
import cv2
from screeninfo import get_monitors
from .world_creation import WorldCreation


class RobotEnv(gym.Env):
    def __init__(
        self,
        action_amp,
        obs_amp,
        frame_skip,
        time_step,
        action_robot_len,
        obs_robot_len):

        # Start the bullet physics server
        self.id = p.connect(p.DIRECT)

        self.gui = False
        self.action_robot_len = action_robot_len
        self.obs_robot_len = obs_robot_len

        # Define action space
        self.action_space = spaces.Box(
            low=np.float32(np.array([-action_amp]*(self.action_robot_len))),
            high=np.float32(np.array([action_amp]*(self.action_robot_len))),
            dtype=np.float32)

        # Define observation space
        self.observation_space = spaces.Box(
            low=np.float32(np.array([-obs_amp]*(self.obs_robot_len))),
            high=np.float32(np.array([obs_amp]*(self.obs_robot_len))),
            dtype=np.float32)

        # Execute actions at 10 Hz by default. A new action every 0.1 seconds
        self.frame_skip = frame_skip
        self.time_step = time_step

        self.setup_timing()
        self.np_random, _ = seeding.np_random(1001)

        self.world_creation = WorldCreation(
            self.id,
            time_step=self.time_step,
            np_random=self.np_random)

        # self.util = Util(self.id, self.np_random)

        # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4") # added by Pierre

        self.record_video = False #True
        self.video_writer = None  #cv2.VideoWriter('Hola.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (10, 10)) # (don't work)
        try:
            self.width = get_monitors()[0].width
            self.height = get_monitors()[0].height
        except Exception as e:
            self.width = 1920
            self.height = 1080
            # self.width = 3840
            # self.height = 2160

    def step(self, action):
        raise NotImplementedError('Implement observations')

    def _get_obs(self, forces):
        raise NotImplementedError('Implement observations')

    def reset(self, seed=None, options=None):
        raise NotImplementedError('Implement reset')

    def take_step(
        self,
        action,
        gains,
        forces,
        indices,
        upper_limit,
        lower_limit,
        robot,
        step_sim=True):

        # clip action to fit the action space
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        # print('cameraYaw=%.2f, cameraPitch=%.2f, distance=%.2f' % p.getDebugVisualizerCamera(physicsClientId=self.id)[-4:-1])

        # print('Total time:', self.total_time)
        # self.total_time += 0.1
        self.iteration += 1
        if self.last_sim_time is None:
            self.last_sim_time = time.time()

        action *= 0.05
        self.action_robot = action

        robot_joint_states = p.getJointStates(
            bodyUniqueId=robot,
            jointIndices=indices,
            physicsClientId=self.id)
        self.robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        for _ in range(self.frame_skip):
            self.action_robot[self.robot_joint_positions + self.action_robot < lower_limit] = 0
            self.action_robot[self.robot_joint_positions + self.action_robot > upper_limit] = 0
            self.robot_joint_positions += self.action_robot

        p.setJointMotorControlArray(
            bodyUniqueId=robot,
            jointIndices=indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=self.robot_joint_positions,
            positionGains=np.array([gains]*self.action_robot_len),
            forces=[forces]*self.action_robot_len,
            physicsClientId=self.id)

        # self.setup_record_video()  # pierre

        if step_sim:
            # Update robot position
            for _ in range(self.frame_skip):
                p.stepSimulation(physicsClientId=self.id)

                if self.gui:
                    # Slow down time so that the simulation matches real time
                    self.slow_time()
            # self.record_video_frame()  # commented

    def reset_robot_joints(self, robot):
        # Reset all robot joints
        for rj in range(p.getNumJoints(robot, physicsClientId=self.id)):
            p.resetJointState(
                robot,
                jointIndex=rj,
                targetValue=0,
                targetVelocity=0,
                physicsClientId=self.id)

    def joint_limited_weighting(self, q, lower_limits, upper_limits):
        phi = 0.5
        lam = 0.05
        weights = []
        for qi, l, u in zip(q, lower_limits, upper_limits):
            qr = 0.5*(u - l)
            weights.append(1.0 - np.power(phi, (qr - np.abs(qr - qi + l)) / (lam*qr) + 1))
            if weights[-1] < 0.001:
                weights[-1] = 0.001
        # Joint-limited-weighting
        joint_limit_weight = np.diag(weights)
        return joint_limit_weight

    def get_motor_joint_states(self, robot):
        num_joints = p.getNumJoints(robot, physicsClientId=self.id)
        joint_states = p.getJointStates(robot, range(num_joints), physicsClientId=self.id)
        joint_infos = [p.getJointInfo(robot, i, physicsClientId=self.id) for i in range(num_joints)]
        joint_states = [j for j, i in zip(joint_states, joint_infos) if i[2] != p.JOINT_FIXED]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return joint_positions, joint_velocities, joint_torques

    def position_robot_toc(
        self,
        robot,
        joints,
        joint_indices,
        lower_limits,
        upper_limits,
        pos_offset=np.zeros(3)):

        if type(joints) == int:
            joints = [joints]
            joint_indices = [joint_indices]
            lower_limits = [lower_limits]
            upper_limits = [upper_limits]

        # Reset all robot joints
        self.reset_robot_joints(robot)

        # best_position = np.array([-0.03778406, -0.07913912,  0.        ])
        best_position = np.array([0, 0, 0])
        best_orientation = np.array([0.0, 0.0, -0.029410315999288762, 0.9995674230950217])
        best_start_joint_poses = [[ 0.79499653,  1.66096791, -1.29904527,  2.52316092, -0.14608365, 1.67627263, -1.80315766]]

        # spawn robot
        p.resetBasePositionAndOrientation(
            robot,
            pos_offset + best_position,
            best_orientation,
            physicsClientId=self.id)

        for i, joint in enumerate(joints):
            self.world_creation.setup_robot_joints(
                robot,
                joint_indices[i],
                lower_limits[i],
                upper_limits[i],
                randomize_joint_positions=False,
                default_positions=np.array(best_start_joint_poses[i]),
                tool=None)

        return best_position, best_orientation, best_start_joint_poses

    def slow_time(self):
        # Slow down time so that the simulation matches real time
        t = time.time() - self.last_sim_time
        if t < self.time_step:
            time.sleep(self.time_step - t)
        self.last_sim_time = time.time()

    def setup_timing(self):
        self.total_time = 0
        self.last_sim_time = None
        self.iteration = 0

    def setup_record_video(self):
        if self.record_video and self.gui:
            if self.video_writer is not None:
                self.video_writer.release()
            now = datetime.datetime.now()
            date = now.strftime('%Y-%m-%d_%H-%M-%S')
            self.video_writer = cv2.VideoWriter('%s.avi' % (date), cv2.VideoWriter_fourcc(*'MJPG'), 10, (self.width, self.height))

    def record_video_frame(self):
        if self.record_video and self.gui:
            frame = np.reshape(
                p.getCameraImage(
                    width=self.width,
                    height=self.height,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    physicsClientId=self.id)[2],
                    (self.height, self.width, 4)
                    )[:, :, :3]
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.video_writer.write(frame)

    def render(self, mode='human'):
        if not self.gui:
            self.gui = True
            p.disconnect(self.id)
            self.id = p.connect(p.GUI, options='--background_color_red=0.8 --background_color_green=0.9 --background_color_blue=1.0 --width=%d --height=%d' % (self.width, self.height))

            self.world_creation = WorldCreation(
                self.id,
                time_step=self.time_step,
                np_random=self.np_random)
            # self.util = Util(self.id, self.np_random)
            # print('Physics server ID:', self.id)

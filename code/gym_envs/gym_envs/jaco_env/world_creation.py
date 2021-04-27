""" Implement WorldCreation class"""
import os
import numpy as np
import pybullet as p
from screeninfo import get_monitors


class WorldCreation:
    """ Initialise Pybullet world for the Jaco arm """

    def __init__(self, pid, time_step=0.02, np_random=None):
        self.id = pid
        self.time_step = time_step
        self.np_random = np_random
        self.directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets')

    def create_new_world(self, print_joints=False):
        """ Create world with Jaco arm on a stand + tool + goal """

        p.resetSimulation(physicsClientId=self.id)

        # Configure camera position
        p.resetDebugVisualizerCamera(
            cameraDistance=1.75,
            cameraYaw=-25,
            cameraPitch=-45,
            cameraTargetPosition=[-0.2, 0, 0.4],
            physicsClientId=self.id)

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.id)

        # Load all models off screen and then move them into place
        p.loadURDF(os.path.join(self.directory, 'plane', 'plane.urdf'), physicsClientId=self.id)

        # Disable rendering during creation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=self.id)

        # Set simulation time step
        p.setTimeStep(self.time_step, physicsClientId=self.id)

        # Disable real time simulation so that the simulation only advances when we call stepSimulation
        p.setRealTimeSimulation(0, physicsClientId=self.id)

        # Jaco
        robot, robot_lower_limits, robot_upper_limits, robot_arm_joint_indices = self.init_jaco(print_joints)

        return robot, robot_lower_limits, robot_upper_limits, robot_arm_joint_indices

    def enforce_joint_limits(self, body):
        """ Enforce joint limits """
        joint_states = p.getJointStates(
            body,
            jointIndices=list(range(p.getNumJoints(body, physicsClientId=self.id))),
            physicsClientId=self.id)
        joint_positions = np.array([x[0] for x in joint_states])
        lower_limits = []
        upper_limits = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            joint_info = p.getJointInfo(body, j, physicsClientId=self.id)
            joint_name = joint_info[1]
            joint_pos = joint_positions[j]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            if lower_limit == 0 and upper_limit == -1:
                lower_limit = -1e10
                upper_limit = 1e10
            lower_limits.append(lower_limit)
            upper_limits.append(upper_limit)
            # print(joint_name, joint_pos, lower_limit, upper_limit)
            if joint_pos < lower_limit:
                p.resetJointState(
                    body,
                    jointIndex=j,
                    targetValue=lower_limit,
                    targetVelocity=0,
                    physicsClientId=self.id)
            elif joint_pos > upper_limit:
                p.resetJointState(
                    body,
                    jointIndex=j,
                    targetValue=upper_limit,
                    targetVelocity=0,
                    physicsClientId=self.id)
        lower_limits = np.array(lower_limits)
        upper_limits = np.array(upper_limits)
        return lower_limits, upper_limits

    def init_jaco(self, print_joints=False):
        """ Load Jaco arm """
        robot = p.loadURDF(
            os.path.join(self.directory, 'jaco', 'j2s7s300_gym.urdf'),
            useFixedBase=True,
            basePosition=[0, 0, 0],
            flags=p.URDF_USE_SELF_COLLISION,
            physicsClientId=self.id)

        robot_arm_joint_indices = [1, 2, 3, 4, 5, 6, 7]

        if print_joints:
            self.print_joint_info(robot, show_fixed=True)

        # Initialize and position
        p.resetBasePositionAndOrientation(
            robot,
            [-2, -2, 0.975],
            [0, 0, 0, 1],
            physicsClientId=self.id)

        # Grab and enforce robot arm joint limits
        lower_limits, upper_limits = self.enforce_joint_limits(robot)

        return robot, lower_limits, upper_limits, robot_arm_joint_indices

    def set_gripper_open_position(self, robot, position=0, set_instantly=False, indices=None):
        """ Set gripper open position """
        indices_new = [9, 11, 13]
        positions = [position, position, position]
        if indices is None:
            indices = indices_new

        if set_instantly:
            for i, j in enumerate(indices):
                p.resetJointState(
                    robot,
                    jointIndex=j,
                    targetValue=positions[i],
                    targetVelocity=0,
                    physicsClientId=self.id)

        p.setJointMotorControlArray(
            robot,
            jointIndices=indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=positions,
            positionGains=np.array([0.05]*len(indices)),
            forces=[500]*len(indices),
            physicsClientId=self.id)

    def init_tool(self, robot, pos_offset=[0]*3, orient_offset=[0, 0, 0, 1]):
        """ Initialise tool """
        gripper_pos, gripper_orient = p.getLinkState(
            robot,
            8,
            computeForwardKinematics=True,
            physicsClientId=self.id
            )[:2]

        transform_pos, transform_orient = p.multiplyTransforms(
            positionA=gripper_pos,
            orientationA=gripper_orient,
            positionB=pos_offset,
            orientationB=orient_offset,
            physicsClientId=self.id)

        tool = p.loadURDF(
            os.path.join(self.directory, 'tools', 'tool_scratch.urdf'),
            basePosition=transform_pos,
            baseOrientation=transform_orient,
            physicsClientId=self.id)

        # Disable collisions between the tool and robot
        for j in [7, 8, 9, 10, 11, 12, 13, 14]:
            for tj in list(range(p.getNumJoints(tool, physicsClientId=self.id))) + [-1]:
                p.setCollisionFilterPair(robot, tool, j, tj, False, physicsClientId=self.id)

        # Create constraint that keeps the tool in the gripper
        constraint = p.createConstraint(
            robot,
            8,
            tool,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            parentFramePosition=pos_offset,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=orient_offset,
            physicsClientId=self.id)

        p.changeConstraint(constraint, maxForce=500, physicsClientId=self.id)
        return tool

    def setup_robot_joints(
        self,
        robot,
        robot_joint_indices,
        lower_limits,
        upper_limits,
        randomize_joint_positions=False,
        default_positions=[1, 1, 0, -1.75, 0, -1.1, -0.5],
        tool=None):
        """ Setup initial robot joints """

        if randomize_joint_positions:
            # Randomize arm joint positions
            # Keep trying random joint positions until the end effector is not colliding with anything
            retry = True
            while retry:
                for j, lower_limit, upper_limit in zip(robot_joint_indices, lower_limits, upper_limits):
                    if lower_limit == -1e10:
                        lower_limit = -np.pi
                        upper_limit = np.pi
                    joint_range = upper_limit - lower_limit
                    p.resetJointState(
                        robot,
                        jointIndex=j,
                        targetValue=self.np_random.uniform(lower_limit + joint_range/6.0, upper_limit - joint_range/6.0),
                        targetVelocity=0,
                        physicsClientId=self.id)
                p.stepSimulation(physicsClientId=self.id)
                retry = len(p.getContactPoints(bodyA=robot, physicsClientId=self.id)) > 0
                if tool is not None:
                    retry = retry or (len(p.getContactPoints(bodyA=tool, physicsClientId=self.id)) > 0)
        else:
            default_positions[default_positions < lower_limits] = lower_limits[default_positions < lower_limits]
            default_positions[default_positions > upper_limits] = upper_limits[default_positions > upper_limits]
            for i, j in enumerate(robot_joint_indices):
                p.resetJointState(
                    robot,
                    jointIndex=j,
                    targetValue=default_positions[i],
                    targetVelocity=0,
                    physicsClientId=self.id)

    def print_joint_info(self, body, show_fixed=True):
        joint_names = []
        for j in range(p.getNumJoints(body, physicsClientId=self.id)):
            if show_fixed or p.getJointInfo(body, j, physicsClientId=self.id)[2] != p.JOINT_FIXED:
                print(p.getJointInfo(body, j, physicsClientId=self.id))
                joint_names.append((j, p.getJointInfo(body, j, physicsClientId=self.id)[1]))
        print(joint_names)

import pybullet as p
import time
import math
import pybullet_data
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.loadURDF("r2d2.urdf", [3, 3, 1])

# obstacle_object = p.loadURDF("gym_envs/widowx_env/envs/URDFs/circular_window_small.urdf") #, useFixedBase=True)
obstacle_object = p.loadURDF("gym_envs/widowx_env/envs/URDFs/circular_window_small_vhacd.urdf") #, useFixedBase=True)

p.resetBasePositionAndOrientation(
    obstacle_object,
    [0.1, .0, 1], 
    p.getQuaternionFromEuler([0, np.pi/2, 0]))


rayFrom = []
rayTo = []
rayIds = []

numRays = 96# 1024
rayLen = 13
rayHitColor = [1, 0, 0]  # green
rayMissColor = [0, 1, 0] # red
lidarPositon = [0, 0, 1]

for i in range(numRays):
    rayFrom.append(lidarPositon)
    rayTo.append([
      rayLen * math.sin(2. * math.pi * float(i) / numRays),
      rayLen * math.cos(2. * math.pi * float(i) / numRays), 1])

    # replace lines
    rayIds.append(p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor))

# print(rayFrom)
# print(rayTo)
# print(rayIds)

numSteps = 327680

for i in range(numSteps):
    p.stepSimulation()
    # for j in range(8):
    #     results = p.rayTestBatch(rayFrom, rayTo, j + 1)
    results = p.rayTestBatch(rayFrom, rayTo)

    for i in range(numRays):
        print(results[i])

        hitObjectUid = results[i][0]

        if hitObjectUid < 0:
            hitPosition = [0, 0, 0]
            p.addUserDebugLine(rayFrom[i], rayTo[i], rayMissColor, replaceItemUniqueId=rayIds[i])
        else:
            hitPosition = results[i][3]
            p.addUserDebugLine(rayFrom[i], hitPosition, rayHitColor, replaceItemUniqueId=rayIds[i])

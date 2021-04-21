" create pybullet simulation with r2d2 robot from urdf file"
import pybullet as p
import time
import pybullet_data

# Start pybullet simulation
p.connect(p.GUI)    
# p.connect(p.DIRECT) # don't render

# load urdf file path
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 

# load urdf and set gravity
# p.setGravity(0,0,-10)
# planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0,0,1]
cubeStartOrientation = p.getQuaternionFromEuler([0,0,0])
# boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)

boxId = p.loadURDF("arrow.urdf", cubeStartPos, cubeStartOrientation)

# step through the simluation
for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()

""" Create convex decomposition of an OBJ file """

import pybullet as p

p.connect(p.DIRECT)
name_in = "circular_window_small.obj"
name_out = "circular_window_small_vhacd2.obj"
name_log = "log.txt"
p.vhacd(
    name_in,
    name_out,
    name_log,
    concavity=0.0001,
    alpha=0.05,
    beta=0.05,
    gamma=0.00125,
    minVolumePerCH=0.0001,
    maxNumVerticesPerCH=312,
    depth=20,
    planeDownsampling=4,
    convexhullDownsampling=4,
    pca=0,
    mode=0,
    convexhullApproximation=1,
    resolution=50000)
# p.vhacd(name_in, name_out, name_log)



# import pybullet_data as pd
# import os
# name_in = os.path.join(pd.getDataPath(), "duck.obj")
# name_out = "duck_vhacd2.obj"
# name_log = "log.txt"
# p.vhacd(name_in, name_out, name_log, alpha=0.01,resolution=50000 )

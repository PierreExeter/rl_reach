""" Create convex decomposition of an OBJ file """

import pybullet as p

p.connect(p.DIRECT)
name_in = "circular_window_small.obj"
name_out = "circular_window_small_vhacd.obj"
name_log = "log.txt"
# p.vhacd(name_in, name_out, name_log, alpha=0.04,resolution=50000)
p.vhacd(name_in, name_out, name_log)

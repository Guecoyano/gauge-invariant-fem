"""script to use watershed from a landscape computed on a mesh

the landscape is interpolated on a grid via P1 elements, and the grid should
be chosen so as to be finer than the mesh (2x finer for instance)"""

import numpy as np
import os
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_merge_utils as wsm
from scipy.interpolate import griddata

x = np.linspace(-0.5, 0.5, 2000)
grid = np.meshgrid(x, x)
nameu = os.path.realpath(os.path.join(data_path, "landscapes", "Na400NV10NB10.npz"))
umesh = np.load(nameu, "u")
points = np.load(nameu, "q")
ugrid = griddata(points, umesh, grid)

M = wsm.find_minima(ugrid)
wsm.store_watershed_transform(M, ugrid)

threshold = 0.1
# This is rhe threshold beyond which 2 points are considered
# not likely to be negihbors on a noramlized domain of size 1.
# This may need to be adjusted based on the potential.
# If no value if provided, it is 0.5 by default, but this will take long to execute.
wsm.find_neighbors(threshold=threshold)
wsm.construct_network(ugrid)
independent_minima = wsm.merge_algorithm(ugrid)

"""script to use watershed from a landscape computed on a mesh

the landscape is interpolated on a grid via P1 elements, and the grid should be chosen so as to be finer than the mesh (2x finer for instance)"""

import numpy as np
import os
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm
import pickle

h = 0.001

print("load mesh")
with open(
    os.path.realpath(
        os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")
    ),
    "rb",
) as f:
    Th = pickle.load(f)

print("load u")
nameu = os.path.realpath(
    os.path.join(data_path, "landscapes", "Na400x15sig22v0NV100NB50.npz")
)
u = np.load(nameu, allow_pickle=True)["u"]
if len(u) != Th.nq:
    print("u does not have th.nq values")
    exit()
# we make sure we can invert u by adding epsilon
epsilon = 10**-20
u = u + epsilon

print("vertex structuration")
vertex_w = ws.verticesdata_frommesh(Th, values=1 / u)

print("find local minima")
M = ws.label_local_minima(vertex_w, mode="irr")

print("watershed")
O = ws.watershed(vertex_w, M, mode="irr")

print(np.max(M))

"""threshold = 0.1  #This is the threshold beyond which 2 points are considered
#not likely to be negihbors on a noramlized domain of size 1.
#This may need to be adjusted based on the potential.
#If no value if provided, it is 0.5 by default, but this will take long to
# execute.
wsm.find_neighbors(threshold = threshold)
wsm.construct_network(u)
independent_minima =  wsm.merge_algorithm(u)"""

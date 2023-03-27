"""script to use watershed from a landscape computed on a mesh

the landscape is interpolated on a grid via P1 elements, and the grid should be chosen so as to be finer than the mesh (2x finer for instance)"""

import numpy as np
import os
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm
import pickle
from fem_base.graphics import PlotVal, PlotIsolines

h = 0.005
NB = 1
NV = 100
coeff = 1.1
E = NB**2 / 2
print("load mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)

print("load u")
nameu = os.path.realpath(
    os.path.join(
        data_path,
        "landscapes",
        "h200" + "Na400x15sig22v0NV" + str(NV) + "NB" + str(NB) + ".npz",
    )
)
u = np.load(nameu, allow_pickle=True)["u"]
if len(u) != Th.nq:
    print("u does not have th.nq values")
    exit()
# we make sure we can invert u by adding epsilon
epsilon = 10**-20
u = np.real(u + epsilon)

print("vertex structuration")
vertex_w = ws.vertices_data_from_mesh(Th, values=(1 / u))

print("find local minima")
M = ws.label_local_minima(vertex_w, mode="irr")

print("initial watershed")
W = ws.watershed(vertex_w, M, mode="irr")

print("initial numberof regions:", np.max(M))

print("Structuring data")
regions = wsm.init_regions(vertex_w, M, W)


print("merging regions")


def cond(min, barr):
    return barr > coeff * min


def shifted_cond(min, barr):
    return barr - E > coeff * (min - E)


merged = wsm.merge_algorithm(regions, shifted_cond)

final_min = 0
for r in merged.regions:
    if not r.removed:
        final_min += 1

print("final number of regions:", final_min)

print("Plotting boundaries over effective potential")
x = []
y = []
for n in range(Th.nq):
    if len(regions.global_boundary[n]) >= 2:
        x.append(Th.q[n, 0])
        y.append(Th.q[n, 1])
plt.figure()
plt.clf()
vmin, vmax = colorscale(1 / u, "hypercube")
PlotVal(Th, 1 / u, vmin=vmin, vmax=vmax)
plt.scatter(x, y, c="k", s=2)
plt.show()
plt.clf()
plt.close()

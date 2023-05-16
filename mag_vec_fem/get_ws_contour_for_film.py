"""
This script aims at computing the grad(log(modulus psi)) to check wether there is an exponential decay when crossing regions of a landscape. 
"""
# coding=utf-8
import numpy as np
from fem_base.exploit_fun import *
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from fem_base.graphics import *
from scipy.interpolate import griddata
import pickle
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm

# from watershed_landscape import *
load_path = os.path.realpath(os.path.join(data_path,"20230404film-01"))
save_path = os.path.realpath(os.path.join(data_path,"20230404film-01"))
h=0.001
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", f"h{int(1/h)}.pkl")), "rb",
) as f:
    Th = pickle.load(f)
tri = matplotlib.tri.Triangulation(Th.q[:, 0], Th.q[:, 1], triangles=Th.me)
print("Mesh done")
namepot="N_a400x15sig22v0"
nframes = 151
NBmax, NBmin = 30, 0
NV=100
nbs = np.linspace(NBmin, NBmax, nframes)#np.sqrt(np.linspace(NBmin**2, NBmax**2, nframes))
for i,NB in enumerate(nbs):
    nameu=f"u_h{int(1 / h)}{namepot}NV{NV}NB{int(NB)}"
    print("load u")
    u = np.load(nameu, allow_pickle=True)["u"]
    if len(u) != Th.nq:
        print("u does not have th.nq values")
        exit()
    # we make sure we can invert u by adding epsilon
    epsilon = 10**-20
    u = np.abs(u) + epsilon

    print("vertex structuration")
    vertex_w = ws.vertices_data_from_mesh(Th, values=(1 / u))

    print("find local minima")
    M = ws.label_local_minima(vertex_w, mode="irr")

    print("initial watershed")
    W = ws.watershed(vertex_w, M, mode="irr")

    print("initial number of regions:", np.max(M))

    print("Structuring data")
    regions = wsm.init_regions(vertex_w, M, W)


    print("merging regions")


    def cond(min, barr):
        return barr > coeff * min

    E=NB**2/2
    def shifted_cond(min, barr):
        return barr - E > coeff * (min - E)


    merged = wsm.merge_algorithm(regions, shifted_cond)
    final_min = 0
    for r in merged.regions:
        if not r.removed:
            final_min += 1

    print("final number of regions:", final_min)

    x = []
    y = []
    z = []
    for n in range(Th.nq):
        if len(regions.global_boundary[n]) >= 2:
            x.append(Th.q[n, 0])
            y.append(Th.q[n, 1])
            z.append(u[n])
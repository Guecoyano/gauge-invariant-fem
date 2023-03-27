# coding=utf-8
import fem_base.gaugeInvariantFEM as gi

# from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
import os

# from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np
import pickle


res_path = data_path
path = os.path.realpath(os.path.join(res_path, "film_poles"))
h = 0.01
gauge = "Sym"
N_eig = 1
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")
N_a = 200
x = 0.15
sigma = 2.2
v = 0
nframes = 15
NBmax, NBmin = 10, 0
"""nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))
namepot = (
            "Na"
            + str(N_a)
            + "x"
            + str(int(100 * x))
            + "sig"
            + str(int(10 * sigma))
            + "v"
            + str(v)
        )
V1, Th = vth_data(h, namepot, Th=Th,N_a=200)"""
dtype = complex
A0 = gi.A_LandauX
G = gi.FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)

d = Th.d
ndfe = d + 1
mu = np.zeros((Th.nme, ndfe, ndfe), dtype)
Kg_A = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d):
    mu += gi.KgP1_OptV3_gdudv(Th, 1, G, i, i, dtype)
phi_A = gi.phi(A0, Th)
for i in range(d + 1):
    for j in range(i):
        Kg_A[:, i, i] = Kg_A[:, i, i] + mu[:, i, j]
        Kg_A[:, j, j] = Kg_A[:, j, j] + mu[:, i, j]
        Kg_A[:, i, j] = Kg_A[:, i, j] - np.multiply(mu[:, i, j], phi_A[:, i, j])
        Kg_A[:, j, i] = Kg_A[:, j, i] - np.multiply(mu[:, i, j], phi_A[:, j, i])


Kg_fp = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d + 1):
    for j in range(i):
        Kg_fp[:, i, i] = Kg_fp[:, i, i] + mu[:, i, j]
        Kg_fp[:, j, j] = Kg_fp[:, j, j] + mu[:, i, j]

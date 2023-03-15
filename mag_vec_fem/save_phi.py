"""This computation of circulation of the magnetic vector potential is really not efficient considering that for the simple linear gauges one may compute everything in a vectorized matricial way instead of creating a lambda function to be integrated at each edge. Therefore this file saves the computation for a given mesh once and for all.
"""
 # coding=utf-8
from fem_base.exploit_fun import *
from fem_base.gaugeInvariantFEM import *
import fem_base.FEMtools as FEMTools
from fem_base.mesh import HyperCube
import os
import numpy as np
import pickle

for h in (0.01, 0.005, 0.001):
    with open(
        os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
        "rb",
    ) as f:
        Th = pickle.load(f)
    
    print("Circulation of symmetric gauge")
    d1 = Th.d + 1
    pA = np.ones((Th.nme, d1, d1), dtype=complex)
    for i in range(d1):
        for j in range(i, d1):
            for k in range(Th.nme):
                qi, qj = Th.q[Th.me[k, i], :], Th.q[Th.me[k, j], :]
                x = lambda t: (1 - t) * qi + t * qj
                A0t = lambda t: np.dot(A_Sym(x(t)[0], x(t)[1]), qj - qi)
                pA[k,i,j]= integrate.quad(A0t, 0, 1)[0]
                pA[k,j,i] = -pA[k,i,j]
                
    with open(
        os.path.realpath(os.path.join(data_path, "logPhi", "Symh" + str(int(1 / h)) + ".pkl")),
        "wb",
    ) as f:
        pickle.dump(pA, f)

    print("circulation of Landau X gauge")
    d1 = Th.d + 1
    pA = np.ones((Th.nme, d1, d1), dtype=complex)
    for i in range(d1):
        for j in range(i, d1):
            for k in range(Th.nme):
                qi, qj = Th.q[Th.me[k, i], :], Th.q[Th.me[k, j], :]
                x = lambda t: (1 - t) * qi + t * qj
                A0t = lambda t: np.dot(A_LandauX(x(t)[0], x(t)[1]), qj - qi)
                pA[k,i,j]= integrate.quad(A0t, 0, 1)[0]
                pA[k,j,i] = -pA[k,i,j]
    with open(
        os.path.realpath(os.path.join(data_path, "logPhi", "LandauXh" + str(int(1 / h)) + ".pkl")),
        "wb",
    ) as f:
        pickle.dump(pA, f)

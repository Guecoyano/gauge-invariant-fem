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
    # mass lumped matrix m^0
    print("assemble $m^0$")
    Kg = Kg_guv_ml(Th, 1, complex)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    NN = Th.nme * (Th.d + 1) ** 2
    M = sparse.csc_matrix(
        (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
        shape=(Th.nq, Th.nq),
    )
    M.eliminate_zeros()
    with open(
        os.path.realpath(
            os.path.join(data_path, "Th", "M0h" + str(int(1 / h)) + ".pkl")
        ),
        "wb",
    ) as f:
        pickle.dump(M, f)

    dtype = complex

    # mass lumped matrix m^1
    print("assemble mu")
    d = Th.d
    ndfe = d + 1
    G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
    mu = np.zeros((Th.nme, ndfe, ndfe), dtype)
    for i in range(d):
        mu += Kg_gdudv(Th, 1, G, i, i, dtype)
    with open(
        os.path.realpath(
            os.path.join(data_path, "Th", "muh" + str(int(1 / h)) + ".pkl")
        ),
        "wb",
    ) as f:
        pickle.dump(mu, f)

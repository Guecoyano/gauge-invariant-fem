# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *
import pickle

pot_version = 0
lnm = 500
h = 0.001
gauge = "Sym"
N_eig = 100
res_pathldhsbc = data_path
"""for pot_version in range(1):
    for lnm in (50, 100, 200, 500, 1000):
        nameV = os.path.realpath(
            os.path.join(
                res_path,
                "pre_interp_pot",
                "pre_interp_potv" + str(pot_version) + "l" + str(lnm) + "E1eVx15.npy",
            )
        )
        V_unscaled = np.load(nameV)
        V_scaled = 1 / np.max(V_unscaled) * V_unscaled
        np.save(nameV, V_scaled)"""
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
for N_a in (200, 50, 100, 300, 400):
    # for N_a in (400,):
    for v in range(5):
        for x in (0.15, 0.05, 0.30, 0.50):
            # for x in (0.15,):
            for sigma in (2.2,):
                namepot = f"Na{N_a}x{int(100 * x)}sig{int(10 * sigma)}v{v}"
                name = os.path.realpath(
                    os.path.join(
                        res_pathldhsbc,
                        f"pre_interp_pot/{namepot}.npy",
                    )
                )
                name_mean = os.path.realpath(
                    os.path.join(
                        res_pathldhsbc,
                        f"pre_interp_pot/{namepot}mean1.npy",
                    )
                )
                V_unscaled = np.load(name)
                V1, Th = vth_data(h, namepot, Th=Th, N_a=N_a)
                V_scaled = 1 / np.mean(V1) * V_unscaled
                print(np.max(V_unscaled), np.max(V_scaled), np.mean(V1))
                np.save(name_mean, V_scaled)

# coding=utf-8
import fem_base.gaugeInvariantFEM as gi
from fem_base.mesh import HyperCube
from fem_base.graphics import *
from fem_base.potentials import interpolate_pot
from fem_base.exploit_fun import *

pot_version = 0
lnm = 500
h = 0.005
gauge = "Sym"
N_eig = 100
res_path = data_path
for pot_version in range(9):
    for lnm in (50, 100, 200, 500, 1000):
        nameV = os.path.realpath(
            os.path.join(
                res_path,
                "pre_interp_pot",
                "pre_interp_potv"
                + str(pot_version)
                + "l"
                + str(lnm)
                + "E1eVx15.npy",
            )
        )
        V_unscaled = np.load(nameV)
        V_scaled = 1 / np.max(V_unscaled) * V_unscaled
        np.save(nameV, V_scaled)

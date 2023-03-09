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
for N_a in (200,50,100,300):
    for v in range(1):    
        for x in (0.15,0.05,0.30,0.50):
            for sigma in (2.2,):
                name=os.path.realpath(os.path.join(res_path,'pre_interp_pot/Na'+str(N_a)+'x'+str(int(100*x))+'sig'+str(int(10*sigma))+'v'+str(v)+'.npy'))
                V_unscaled = np.load(name)
                V_scaled = 1 / np.max(V_unscaled) * V_unscaled
                np.save(name, V_scaled)
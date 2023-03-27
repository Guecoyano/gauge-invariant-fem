# coding=utf-8
import numpy as np
from fem_base.potentials import gauss_pot
from fem_base.potentials import pot_from_funct
from math import exp
from fem_base.exploit_fun import data_path
import os

res_path = data_path
"""for l in [50,100,200,500,1000]:
    for v in range(5):
        pot,q_pot=gauss_pot(eV,2.2,int(l/0.56),l*10**-9,0.15)
        name=res_path+'pre_interp_pot/pre_interp_potv'+str(v)+'l'+str(l)+'E1eVx15.npy'
        np.save(name,pot)
    np.save(res_path+'/Vq/Vql'+str(l),q_pot)

v = 10
for f in (
    lambda x, y: 2 * exp(-0.1 * ((x - 8) ** 2 + (y - 8) ** 2))
    - 3 * exp(-0.3 * ((x - 5.5) ** 2 + (y - 1) ** 2))
    + 4 / ((x - 2) ** 2 + (y - 5) ** 2 + 1),
    lambda x, y: (1 - 1 / ((x - 5) ** 2 / 2 + (y - 4) ** 2 / 4 + 1)),
    lambda x, y: 2 * exp(-0.1 * ((x - 8) ** 2 + (y - 8) ** 2))
    + 3 * exp(-0.3 * ((x - 5.5) ** 2 + (y - 1) ** 2))
    + 4 / ((x - 2) ** 2 + (y - 5) ** 2 + 1),
    lambda x, y: (1 / ((x - 5) ** 2 / 2 + (y - 4) ** 2 / 4 + 1)),
):
    v += 1
    N = 400
    pot = pot_from_funct(f, N)
    name = os.path.realpath(
        os.path.join(data_path, "pre_interp_pot", "v" + str(v) + ".npy")
    )
    np.save(name, pot)


"""
l = 1
for N_a in (200, 50, 100, 300):
    for v in range(1):
        for x in (0.15, 0.05, 0.30, 0.50):
            for sigma in (2.2,):
                pot, q_pot = gauss_pot(1, sigma, N_a, l, x)
                name = os.path.realpath(
                    os.path.join(
                        res_path,
                        "pre_interp_pot/Na"
                        + str(N_a)
                        + "x"
                        + str(int(100 * x))
                        + "sig"
                        + str(int(10 * sigma))
                        + "v"
                        + str(v)
                        + ".npy",
                    )
                )
                np.save(name, pot / np.max(pot))
    np.save(res_path + "/Vq/VqNa" + str(N_a), q_pot)

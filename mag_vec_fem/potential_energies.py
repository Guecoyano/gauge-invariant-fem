import numpy as np
from scipy.stats import linregress
import os
import sys
from fem_base.exploit_fun import *
from fem_base.gaugeInvariantFEM import *
from scipy import sparse

params = [
    ("load_file_x", True, None),
    ("load_file_u", True, None),
    ("dir_to_save", True, None),
    ("name_save", True, None),
    ("namepot", True, None),    
    ("eta", False, None),    
]
load_file_x=eta=namepot=load_file_u=dir_to_save=name_save=None
h=0.001
L=200
N_a=400

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default

x_sol=np.load(load_file_x,allow_pickle=True)["eig_vec"]
w=np.load(load_file_x,allow_pickle=True)["eig_val"]
eff_pot=1/(10**-20+np.real(np.load(load_file_u,allow_pickle=True)["u"]))

with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q = L * Th.q
    Th.vols = (L**2) * Th.vols

V_unscaled, Th = vth_data(h, namepot, L=L, Th=Th, N_a=N_a)
V = (eta / np.mean(V_unscaled)) * V_unscaled
ones = np.ones(Th.nq)


ndfe=Th.d+1
Kg = Kg_guv_ml(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (ndfe) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()
m0=np.real(np.sum(M,0))

pot_energies,eff_pot_energies=[],[]
for i,energy in enumerate(w):
    vec2=np.real(x_sol[:,i]*np.conj(x_sol[:,i]))
    pot_en=np.sum(np.dot(m0,V*vec2))
    eff_pot_en=np.sum(np.dot(m0,eff_pot*vec2))
    pot_energies.append(pot_en)
    eff_pot_energies.append(eff_pot_en)
    
np.savez_compressed(os.path.join(dir_to_save,name_save), potential_energies=np.array(pot_energies), effective_potential_energies=np.array(eff_pot_energies), eig_val=w)
# coding=utf-8
"""This script is meant to compute efficiently the IPRs of eigenvectors of the first landau level for given mesh size, potential, disorder and magnetic parameters."""
import sys
import os
import pickle
import time
from fem_base.exploit_fun import variable_value,data_path,vth_data
from fem_base.gaugeInvariantFEM import *
from scipy import sparse
from scipy.sparse.linalg import eigsh


t0=time.time()

# parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("h", False, 0.001),
    ("gauge", True, "Sym"),
    ("N_eig", False, 12),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("L", False, 200),
    ("beta", False, 0.1),
    ("eta", False, 0.1),
    ("namepot", True, None),
    ("target_energy", False, None),
    ("serial_solve", False, False),
    ("dir_to_save", True, None),
    ("name_pr", True, None),
    ("which", True, "LM"),
]
(
    h
) = (
    gauge
) = (
    N_eig
) = (
    N_a
) = (
    x
) = (
    sigma
) = (
    v
) = serial_solve = L = which = name_pr = eta = beta = namepot = target_energy = dir_to_save = None

print(sys.argv)

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default
if namepot is None:
    namepot = f"Na{N_a}x{int(100 * x)}sig{int(10 * sigma)}v{v}"
if target_energy is None:
    target_energy = beta + eta * 0
if dir_to_save is None:
    dir_to_save = data_path
if N_eig is None:
    N_eig = int(L**2*beta/(2*np.pi))# trying to get the whole first Landau level.
if name_pr is None:
    name_pr="pr{N_eig}eta1e-1beta2e-1"
if serial_solve:
    pr=np.fromfile(f"{dir_to_save}/{name_pr}")
    pr=pr.reshape((-1,2))
    target_energy=pr[-1,1]#*1.0000001
    which="LA"
else:
    pr=np.array([[]])

print("Creating mesh, time is:", time.time()-t0)
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q = L * Th.q
    Th.vols = (L**2) * Th.vols

print("Mesh done, time is:", time.time()-t0)
# Th=None
V_unscaled, Th = vth_data(h, namepot, L=L, Th=Th, N_a=N_a)
V1 = (1 / np.mean(V_unscaled)) * V_unscaled
print(np.mean(V1))
time1=time.time()
ones = np.ones(Th.nq)

fichier=f"{data_path}/20230522_charon/Na400x15sig22v0eta1e-1beta1e-1Symtarget0h1000Neig10.npz"
fichier_u=f"{data_path}/20230522_charon/u_Na400x15sig22v0L200eta1e-1beta1e-1h1000.npz"
file= np.load(fichier,allow_pickle=True)
x_sol,w=file["eig_vec"],file["eig_val"]
unsuru=np.array(V1)

potpoints=60
areapoints=60
list_pot_val=np.linspace(0,2.5*eta,potpoints)
min_unsuru=0.01*int(100*np.min(unsuru))
list_unsuru=np.linspace(min_unsuru,2.5*eta+beta,potpoints)
list_exp=np.linspace(0,20,areapoints)
list_proba_val=[10**-i for i in list_exp]

# mass lumped matrix m^0
print("assemble $m^0$, time is:", time.time()-t0)
Kg = Kg_guv_ml(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (Th.d+1) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()
m0=np.real(np.sum(M,0))

print("compute pr's, time is:", time.time()-t0)
#compute0 and store PR's


histograms_u,histograms_V,histograms_proba=[],[],[]
for i,energy in enumerate(w):
    vec2=(x_sol[:,i]*np.conj(x_sol[:,i])).real
    psi2=np.sum(np.dot(m0,vec2))
    proba_density=vec2/psi2
    histogram_u,histogram_V,histogram_proba=[],[],[]

    for potential_value in list_pot_val:
        hist_V=np.where(eta*V1<potential_value,proba_density,0)
        point_V=np.array((potential_value,np.sum(np.dot(m0,hist_V))))
        histogram_V.append(point_V)
    histograms_V.append(np.array[histogram_V])
    
    for unsuru_value in list_unsuru:
        hist_u=np.where(unsuru<unsuru_value,proba_density,0)
        point_u=np.array((unsuru_value,np.sum(np.dot(m0,hist_u))))
        histogram_u.append(point_u)
    histograms_u.append(np.array(histogram_u))

    for proba_value in list_proba_val:
        hist_proba=np.where(proba_density>proba_value,1,0)
        point_proba=np.array((proba_value,np.sum(np.dot(m0,hist_proba))))
        histogram_proba.append(point_proba)
    histograms_proba.append(histogram_proba)
    
pr.tofile(f"{dir_to_save}/{name_pr}_readable",sep=" ")
pr.tofile(f"{dir_to_save}/{name_pr}")
print("ending at ", time.time()-t0)
print(pr)

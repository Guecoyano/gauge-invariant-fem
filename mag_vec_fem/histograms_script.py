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
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("L", False, 200),
    ("beta", False, 0.1),
    ("eta", False, 0.1),
    ("namepot", True, None),
    ("dir_to_save", True, None),
    ("name_file_psi", True, None),
    ("name_file_u", True, None),
    ("name_hist", True, None),
]
(
    h
) = (
    gauge
) = (
    N_a
) = (
    x
) = (
    sigma
) = (
    v
) = name_file_psi = name_file_u = name_hist = L = eta = beta = namepot = dir_to_save = None

print(sys.argv)

for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default
if namepot is None:
    namepot = f"Na{N_a}x{int(100 * x)}sig{int(10 * sigma)}v{v}"
if dir_to_save is None:
    dir_to_save = data_path
if name_hist is None:
    print("no name to save the file")


print("Creating mesh, time is:", time.time()-t0)
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q = L * Th.q
    Th.vols = (L**2) * Th.vols
print("Mesh done, time is:", time.time()-t0)


print("Creating potential, time is:", time.time()-t0)
V_unscaled, Th = vth_data(h, namepot, L=L, Th=Th, N_a=N_a)
V = (eta / np.mean(V_unscaled)) * V_unscaled
print("Potential done, time is:", time.time()-t0)


fichier=f"{data_path}/{name_file_psi}"
fichier_u=f"{data_path}/{name_file_u}"
file= np.load(fichier,allow_pickle=True)
x_sol,w=file["eig_vec"],file["eig_val"]
unsuru=1/(10**-20+np.real(np.load(fichier_u,allow_pickle=True)["u"]))

potpoints=200
areapoints=100
list_pot_val=np.linspace(0,2.5*eta,potpoints)
vmax=np.max(V)
min_unsuru=0.01*int(100*np.min(unsuru))
list_v=np.linspace(0,vmax,potpoints)
list_unsuru=np.linspace(min_unsuru,vmax+beta,potpoints)
list_exp=np.linspace(0,30,areapoints)
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


histograms_u,histograms_V,histograms_proba=len(w)*[0],len(w)*[0],len(w)*[0]
for i,energy in enumerate(w):
    vec2=(x_sol[:,i]*np.conj(x_sol[:,i])).real
    psi2=np.sum(np.dot(m0,vec2)) 
    proba_density=vec2/psi2
    histogram_u,histogram_V,histogram_proba=[],[],[]

    for potential_value in list_pot_val:
        hist_V=np.where(eta*V<potential_value,proba_density,0)
        point_V=np.array((potential_value,np.sum(np.dot(m0,hist_V))))
        histogram_V.append(point_V)
    histograms_V[i]=np.array(histogram_V)
    
    for unsuru_value in list_unsuru:
        hist_u=np.where(unsuru<unsuru_value,proba_density,0)
        point_u=np.array((unsuru_value,np.sum(np.dot(m0,hist_u))))
        histogram_u.append(point_u)
    histograms_u[i]=np.array(histogram_u)

    for proba_value in list_proba_val:
        hist_proba=np.where(proba_density>proba_value,1,0)
        point_proba=np.array((proba_value,np.sum(np.dot(m0,hist_proba))))
        histogram_proba.append(point_proba)
    histograms_proba[i]=np.array(histogram_proba)

np.savez_compressed(
        os.path.join(dir_to_save, name_hist), eig_val=w, histograms_proba=histograms_proba, histograms_u=histograms_u,histograms_V=histograms_V
    )
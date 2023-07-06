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
    ("h", False, 0.01),
    ("gauge", True, "Sym"),
    ("N_eig", False, 12),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("L", False, 200),
    ("beta", False, 0.25),
    ("eta", False, 0.1),
    ("namepot", True, None),
    ("target_energy", False, None),
    ("dir_to_save", True, None),
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
) = L = eta = beta = namepot = target_energy = dir_to_save = None

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


# getsaveeig


meshfile = None
N = 1
magpde = init_magpde(h, beta, L, gauge, V1, Th)
ndfe=Th.d+1
Num = 1
AssemblyVersion = "OptV3"
SolveOptions = None
verbose = False
Tcpu = np.zeros((4,))
tstart = time.time()

# mass lumped matrix m^0
print("assemble $m^0$, time is:", time.time()-t0)
Kg = Kg_guv_ml(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (ndfe) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()

dtype = complex

# compute normalized V term
print("assemble normalized V term, time is:", time.time()-t0)
Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
Kg_V = Kg_guv_ml(Th,eta * V1, dtype)

# magnetic kinetic term
print("assemble kinetic magnetic term, time is:", time.time()-t0)
G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
Kg_A = Kg_kinetic_mag(Th, magpde.op, G, complex)

Kg=Kg_V-Kg_A

# boundaries
print("assemble boundaries, time is:", time.time()-t0)
Tcpu[0] = time.time() - tstart
ndof = Th.nq
tstart = time.time()
# bN=NeumannBC(pde,AssemblyVersion,Num);
# [AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
[ID, IDc, gD] = DirichletBC(magpde, Num)

print("computing eigenproblem, time is:", time.time()-t0)
print(
    "h=",
    h,
    "namepot=",
    namepot,
    f"beta={beta}",
    f"eta={eta}",
    "target energy=",
    target_energy,
    "Neig=",
    N_eig,
)


A = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
A.eliminate_zeros()

# A = A + AR
Tcpu[2] = time.time() - tstart
x_sol = np.zeros((ndof, N_eig), dtype=magpde.dtype)
w = np.zeros(N_eig, dtype=complex)
tstart = time.time()
xx = np.repeat(gD[ID], N_eig, axis=0)
x_sol[ID, :] = np.reshape(xx, (len(ID), -1))
print("solving..., time is:", time.time()-t0)
w, x_sol[IDc, :] = eigsh(
    (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N_eig, sigma=target_energy, which="LM"
)
print(f"assemble and solve time for h={h}:",time.time()-time1)


print("Ordering eigendata, time is:", time.time()-t0)
# ordering eigenstates:
wtype = [("energy", float), ("rank", int)]
w_ = [(w[i], i) for i in range(N_eig)]
w_disordered = np.array(w_, dtype=wtype)
w_ord = np.sort(w_disordered, axis=0, order="energy")
I = [i[1] for i in w_ord]
w = np.array([i[0] for i in w_ord])
x_ = np.copy(x_sol)
for i in range(N_eig):
    x_sol[:, i] = x_[:, I[i]]

"""print("compute pr's, time is:", time.time()-t0)
#compute and store PR's
pr=[]
for i,energy in enumerate(w):
    m0=np.sum(M,0)
    vec2=x_sol[:,i]*np.conj(x_sol[:,i])
    psi4=np.sum(np.dot(m0,vec2**2))
    psi2=np.sum(np.dot(m0,vec2))
    pr_elem = psi2**2/psi4
    pr.append((pr_elem,energy))
pr.tofile(f"{dir_to_save}/pr{N_eig}eta1e-1beta2e-1_readable",sep=" ")
pr.tofile(f"{dir_to_save}/pr{N_eig}eta1e-1beta2e-1")
print("ending at ", time.time()-t0)"""
print(w)
# coding=utf-8
"""This script is meant to solve efficiently a lot of eigenvalue problems with on the same mesh (and gauge) but for different values of magnetic strength and/or disorder. """

from fem_base.exploit_fun import *
import pickle
from fem_base.gaugeInvariantFEM import *

res_pathldhsbc = data_path
path = os.path.realpath(os.path.join(res_pathldhsbc, "film_poles"))

#parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("eig",False,True),
    ("u",False, False),
    ("h", False, 0.001),
    ("gauge", True, "Sym"),
    ("N_eig", False, 10),
    ("N_a", False, 400),
    ("x", False, 0.15),
    ("sigma", False, 2.2),
    ("v", False, 0),
    ("NB", False, 0),
    ("NV", False, 0),
    ("namepot", True, None),
    ("target_energy", False, None),
    ("dir_to_save", True, None),
    ("name_eig", True, None),
    ("name_u", True, None),
]
eig=u=h=gauge=N_eig=N_a=x=sigma=v=NB=NV=namepot=target_energy=dir_to_save=name_eig=name_u=None
print(sys.argv)
for param, is_string, default in params:
    prescribed = variable_value(param, is_string, sys.argv[1:])
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default
if namepot is None:
    namepot=(
            "Na"
            + str(N_a)
            + "x"
            + str(int(100 * x))
            + "sig"
            + str(int(10 * sigma))
            + "v"
            + str(v)
    )
if target_energy is None:
    target_energy=(NB**2)/2
if dir_to_save is None:
    dir_to_save = os.path.join(
            res_pathldhsbc,
            "film_poles"
    )
if name_eig is None:
    name_eig=(
        namepot
        + "NV"
        + str(NV)
        + "NB"
        + str(int(NB))
        + gauge
        + "h"
        + str(int(1 / h))
        + "Neig"
        + str(N_eig)
    )
if name_u is None:
    name_u= (
        "u_h"
        + str(int(1 / h))
        + namepot
        + "NV"
        + str(NV)
        + "NB"
        + str(int(NB))
    )
B=NB**2

print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")

V1, Th = vth_data(h, namepot, Th=Th, N_a=N_a)
ones = np.ones(Th.nq)


# getsaveeig


meshfile = None
N = 1
l = 1
V = np.copy(V1)
magpde = init_magpde(h * l, 1, l, gauge, V, Th)
Num = 1
AssemblyVersion = "OptV3"
SolveOptions = None
verbose = False
Tcpu = np.zeros((4,))
tstart = time.time()

# mass lumped matrix m^0
print("assemble $m^0$")
Kg = KgP1_OptV3_ml(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (Th.d + 1) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()

dtype = complex

# mass lumped matrix m^1
print("assemble m^1")
d = Th.d
ndfe = d + 1
G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
mu = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d):
    mu += KgP1_OptV3_gdudv(Th, 1, G, i, i, dtype)

# compute normalized V term
print("assemble normalized V term")
Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
Kg_V = KgP1_OptV3_ml(Th, (magpde.op).V, dtype)

# circulationn of the normalized vector poltential
print("normalized circulation of A")
# phi_A = phi((magpde.op).A0, Th)
with open(
    os.path.realpath(
        os.path.join(data_path, "logPhi", gauge+"h" + str(int(1 / h)) + ".pkl")
    ),
    "rb",
) as f:
    logPhi = pickle.load(f)


# prepare grad_A term
Kg_A = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d + 1):
    for j in range(i):
        Kg_A[:, i, i] = Kg_A[:, i, i] + mu[:, i, j]
        Kg_A[:, j, j] = Kg_A[:, j, j] + mu[:, i, j]

# boundaries
print("assemble boundaries")
Tcpu[0] = time.time() - tstart
ndof = Th.nq
tstart = time.time()
Tcpu[1] = time.time() - tstart
tstart = time.time()
# bN=NeumannBC(pde,AssemblyVersion,Num);
# [AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
[ID, IDc, gD] = DirichletBC(magpde, Num)


# operators for landscape
Kg_delta = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d):
    Kg_delta = Kg_delta + KgP1_OptV3_gdudv(Th, ones, G, i, i, dtype)
Kg_uV = KgP1_OptV3_guv(Th, V, dtype)
Kg_u1 = KgP1_OptV3_guv(Th, ones, dtype)

# RHS for landscapes
b = RHS(magpde.Th, magpde.f, Num, dtype=magpde.dtype, version=AssemblyVersion)

if eig:
    print("computing eigenproblem")
    print("h=", h,"namepot=",namepot, "NB=",NB,"NV=",NV, "target energy=",target_energy,"Neig=",N_eig)
    phi_A = np.exp(B * logPhi * 1j)
    for i in range(d + 1):
        for j in range(i):
            Kg_A[:, i, j] = Kg_A[:, i, j] - np.multiply(mu[:, i, j], phi_A[:, i, j])
            Kg_A[:, j, i] = Kg_A[:, j, i] - np.multiply(mu[:, i, j], phi_A[:, j, i])
    Kg = -Kg_A + NV**2 * Kg_V

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
    print("solving...")
    w, x_sol[IDc, :] = eigsh(
        (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N_eig, sigma=target_energy, which="LM"
    )
    Tcpu[3] = time.time() - tstart

    print("times:", Tcpu)

    print("Ordering eigendata")
    tstart = time.time()

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
    t_order = time.time() - tstart
    print("ordering time:", t_order)
    tstart = time.time()

    print("Post-processing")
    # save in one compressed numpy file: V in nq array, th.q , w ordered in N_eig array, x ordered in nq*N_eig array
    
    
    
    np.savez_compressed(os.path.join(dir_to_save,name_eig), q=Th.q, V=V, eig_val=w, eig_vec=x_sol)
    t_postpro = time.time() - tstart
    print("saving time:", t_postpro)

if u:
    Kg_u = Kg_delta + NV**2 * Kg_uV + (NB**2)/2 * Kg_u1

    M_u = sparse.csc_matrix(
        (np.reshape(Kg_u, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
        shape=(Th.nq, Th.nq),
    )
    M_u.eliminate_zeros()
    print("computing landscape")
    x_sol, flag = classicSolve(M_u, b, ndof, gD, ID, IDc, complex, SolveOptions)
    np.savez_compressed(
        os.path.join(dir_to_save, name_u),
        q=Th.q,
        u=x_sol,
    )

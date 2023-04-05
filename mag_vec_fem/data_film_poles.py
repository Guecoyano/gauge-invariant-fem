# coding=utf-8
"""This script is meant to solve efficiently a lot of eigenvalue problems with on the same mesh (and gauge) but for different values of magnetic strength and/or disorder. """

from fem_base.exploit_fun import *
import pickle
from fem_base.gaugeInvariantFEM import *

res_pathldhsbc = data_path
path = os.path.realpath(os.path.join(res_pathldhsbc, "film_poles"))
params = [
    ("h", 0.001),
    ("gauge", "Sym"),
    ("N_eig", 10),
    ("N_a", 400),
    ("x", 0.15),
    ("sigma", 2.2),
    ("v", 0),
    ("nframes", 2),
    ("NBmax", 10),
    ("NBmin", 0),
]
for param, default in params:
    prescribed = variable_value(param, sys.argv)
    if prescribed is not None:
        globals()[param] = prescribed
    else:
        globals()[param] = default

nbs = np.sqrt(np.linspace(NBmin**2, NBmax**2, nframes))


print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")


"""namepot = (
            "Na"
            + str(N_a)
            + "x"
            + str(int(100 * x))
            + "sig"
            + str(int(10 * sigma))
            + "v"
            + str(v)
        )"""
namepot = "v6"
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
        os.path.join(data_path, "logPhi", "Symh" + str(int(1 / h)) + ".pkl")
    ),
    "rb",
) as f:
    logPhi = pickle.load(f)


# prepare grad_A term
Kg_A0 = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d + 1):
    for j in range(i):
        Kg_A0[:, i, i] = Kg_A0[:, i, i] + mu[:, i, j]
        Kg_A0[:, j, j] = Kg_A0[:, j, j] + mu[:, i, j]

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

# loop on parameters.
for frame in range(len(nbs)):
    NB = nbs[frame]
    for NV in (100,):
        B = NB**2
        E_0 = B / 2
        print("h=", h, "E_0=", E_0)
        Kg_A = np.copy(Kg_A0)
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
        x = np.zeros((ndof, N_eig), dtype=magpde.dtype)
        w = np.zeros(N_eig, dtype=complex)
        tstart = time.time()
        xx = np.repeat(gD[ID], N_eig, axis=0)
        x[ID, :] = np.reshape(xx, (len(ID), -1))
        print("solving...")
        w, x[IDc, :] = eigsh(
            (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N_eig, sigma=E_0, which="LM"
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
        x_ = np.copy(x)
        for i in range(N_eig):
            x[:, i] = x_[:, I[i]]
        t_order = time.time() - tstart
        print("ordering time:", t_order)
        tstart = time.time()

        print("Post-processing")
        # save in one compressed numpy file: V in nq array, th.q , w ordered in N_eig array, x ordered in nq*N_eig array
        dir_to_save = os.path.join(
            path,
            namepot
            + "NV"
            + str(NV)
            + "NBmin"
            + str(int(NBmin))
            + "NBmax"
            + str(int(NBmax))
            + gauge
            + "h"
            + str(int(1 / h))
            + "Neig"
            + str(N_eig)
            + "frame"
            + str(frame),
        )
        np.savez_compressed(dir_to_save, q=Th.q, V=V, eig_val=w, eig_vec=x)
        t_postpro = time.time() - tstart
        print("saving time:", t_postpro)

        Kg_u = Kg_delta + NV**2 * Kg_uV + E_0 * Kg_u1

        M_u = sparse.csc_matrix(
            (np.reshape(Kg_u, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
            shape=(Th.nq, Th.nq),
        )
        M_u.eliminate_zeros()
        print("computing landscape")
        x, flag = classicSolve(M_u, b, ndof, gD, ID, IDc, complex, SolveOptions)
        namedata = (
            "u_h"
            + str(int(1 / h))
            + namepot
            + "NV"
            + str(NV)
            + "NBmin"
            + str(int(NBmin))
            + "NBmax"
            + str(int(NBmax))
            + "frame"
            + str(frame)
        )
        np.savez_compressed(
            os.path.join(path, namedata),
            q=Th.q,
            u=x,
        )

# coding=utf-8
"""This script is meant to compute the invariant parts of the matrix of the eigen pb for a given potential, mesh, gauge, boundary condition, and store them in a file """

from fem_base.exploit_fun import *
import pickle
from fem_base.gaugeInvariantFEM import *

h, gauge, N_eig, N_a, x, sigma, v, nframes, NBmax, NBmin = 10 * [None]
print(sys.argv)
get_args()
if h is None:
    h = 0.001
if gauge is None:
    gauge = "Sym"
if N_a is None:
    N_a = 400
if x is None:
    x = 0.15
if sigma is None:
    sigma = 2.2
if v is None:
    v = 0

print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")


namepot = (
            "Na"
            + str(N_a)
            + "x"
            + str(int(100 * x))
            + "sig"
            + str(int(10 * sigma))
            + "v"
            + str(v)
        )

V, Th = vth_data(h, namepot, Th=Th, N_a=N_a)
ones = np.ones(Th.nq)


# getsaveeig

l = 1
magpde = init_magpde(h * l, 1, l, gauge, V, Th)
dtype = complex
numéro = 1
AssemblyVersion = "OptV3"
SolveOptions = None
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


# prepare grad_A term
Kg_A0 = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d + 1):
    for j in range(i):
        Kg_A0[:, i, i] = Kg_A0[:, i, i] + mu[:, i, j]
        Kg_A0[:, j, j] = Kg_A0[:, j, j] + mu[:, i, j]

# boundaries
print("assemble boundaries")
Tcpu[0] = time.time() - tstart
tstart = time.time()
Tcpu[1] = time.time() - tstart
tstart = time.time()



# operators for landscape
Kg_delta = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d):
    Kg_delta = Kg_delta + KgP1_OptV3_gdudv(Th, ones, G, i, i, dtype)
Kg_uV = KgP1_OptV3_guv(Th, V, dtype)
Kg_u1 = KgP1_OptV3_guv(Th, ones, dtype)

# RHS for landscapes
b = RHS(magpde.Th, magpde.f, numéro, dtype=magpde.dtype, version=AssemblyVersion)


namedata = ("h"
            + str(int(1 / h))
            + namepot
        )
np.savez_compressed(
    os.path.join(data_path,'files', namedata),
    M=M,
    Kg_A0=Kg_A0,
    Kg_V=Kg_V,
    Kg_delta=Kg_delta,
    Kg_uV = Kg_uV,
    Kg_u1 =Kg_u1,
    Ig=Ig,
    Jg=Jg,
    mu=mu,
    b=b
)
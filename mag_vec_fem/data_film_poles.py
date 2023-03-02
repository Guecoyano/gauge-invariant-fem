# coding=utf-8
from fem_base.exploit_fun import *
import pickle
from fem_base.gaugeInvariantFEM import *

res_path = data_path
h = 0.001
gauge = "Sym"
N_eig = 5
print("Creating mesh")
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
print("Mesh done")
N_a = 400
x = 0.15
sigma = 2.2
v = 0
nframes=51
NBmax,NBmin=0,15
nbs = np.sqrt(np.linspace(NBmin**2,NBmax**2,nframes))

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
V1, Th = vth_data(h, namepot, Th=Th)


#getsaveeig
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

Kg = KgP1_OptV3_guv(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (Th.d + 1) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()

'''A_0 = magAssemblyP1(
    magpde.Th,
    D:magpde.op,
    dtype=magpde.dtype,
    version=AssemblyVersion,
)'''

dtype = complex
#Kg = KgP1_OptV3_A(Th, magpde.op, G, dtype=dtype)
d = Th.d
ndfe = d + 1
Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
Kg_V=KgP1_OptV3_guv(Th, (magpde.op).V, dtype)
G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)

# boundaries
Tcpu[0] = time.time() - tstart
ndof = Th.nq
tstart = time.time()
Tcpu[1] = time.time() - tstart
tstart = time.time()
# bN=NeumannBC(pde,AssemblyVersion,Num);
[AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
[ID, IDc, gD] = DirichletBC(magpde, Num)



Kg_A=KgP1_OptV3_A_A(Th, magpde.op, G, dtype)
Kg = Kg - Kg_A+ NV**2*Kg_V

A = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
A.eliminate_zeros()
#    return A
A_0=np.copy(A)

A = A_0 + AR
Tcpu[2] = time.time() - tstart
x = np.zeros((ndof, N), dtype=magpde.dtype)
w = np.zeros(N, dtype=complex)
tstart = time.time()
xx = np.repeat(gD[ID], N, axis=0)
x[ID, :] = np.reshape(xx, (len(ID), -1))
E_0 = B / 2
t = kwargs.get("target_energy", E_0)
w, x[IDc, :] = eigsh(
    (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N, sigma=t, which="LM"
)
Tcpu[3] = time.time() - tstart

print("h=", h, "E_0=", E_0)
print("times:", Tcpu)

#    return w, x
for NB in nbs:
    for NV in (100,):
        path=os.path.realpath(os.path.join(res_path,"film_poles"))
        
    

        E_s = (NB**2) / 2
        V = (NV**2) * V1 + E_s
        x = gi.getSol(Th=Th, B=0.0, V=V)
        namedata = "u_h" + str(int(1 / h)) + namepot + "NV" + str(NV) + "frame"+ str(int(np.round(np.sqrt(NB**2*(nframes-1)/(NBmax-NBmin-1)),0)))
        np.savez_compressed(
            os.path.join(path, namedata),
            q=Th.q,
            u=x,
        )
# coding=utf-8
"""This script is meant to solve efficiently a lot of eigenvalue problems with on the same mesh (and gauge) but for different values of magnetic strength and/or disorder. """

from fem_base.exploit_fun import *
import pickle
from fem_base.gaugeInvariantFEM import *
import sys

res_path = data_path
path = os.path.realpath(os.path.join(res_path, "film_poles"))
t0=time.time()
# parameters of the file in the form of tuples: (name of parameter, is it a string, default value)
params = [
    ("eig", False, True),
    ("u", False, False),
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
    ("name_eig", True, None),
    ("name_u", True, None),
]
eig = (
    u
) = (
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
) = L = eta = beta = namepot = target_energy = dir_to_save = name_eig = name_u = None

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
    dir_to_save = res_path
if name_eig is None:
    name_eig = f"{namepot}L{int(L)}eta{int(100*eta)}beta{int(100*beta)}{gauge}h{int(1 / h)}Neig{N_eig}"
if name_u is None:
    name_u = f"u_h{int(1 / h)}{namepot}L{int(L)}eta{int(100*eta)}beta{int(100*beta)}"

print("Creating mesh",time.time()-t0)
with open(
    os.path.realpath(os.path.join(data_path, "Th", "h" + str(int(1 / h)) + ".pkl")),
    "rb",
) as f:
    Th = pickle.load(f)
    Th.q = L * Th.q
    Th.vols = (L**2) * Th.vols

print("Mesh done",time.time()-t0)
# Th=None
V_unscaled, Th = vth_data(h, namepot, L=L, Th=Th, N_a=N_a)
V1 = (1 / np.mean(V_unscaled)) * V_unscaled
print(np.mean(V1))


# getsaveeig

ttest=time.time()
meshfile = None
N = 1
magpde = init_magpde(h, 1, L, gauge, V1, Th)
Num = 1
AssemblyVersion = "OptV3"
SolveOptions = None
verbose = False
Tcpu = np.zeros((4,))
tstart = time.time()

# mass lumped matrix m^0
print("assemble $m^0$",time.time()-t0)
Kg = Kg_guv_ml(Th, 1, complex)
Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
NN = Th.nme * (Th.d + 1) ** 2
M = sparse.csc_matrix(
    (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
    shape=(Th.nq, Th.nq),
)
M.eliminate_zeros()

dtype = complex

# mass lumped matrix m^1
print("assemble m^1",time.time()-t0)
d = Th.d
ndfe = d + 1
G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
mu = np.zeros((Th.nme, ndfe, ndfe), dtype)
for i in range(d):
    mu += Kg_gdudv(Th, 1, G, i, i, dtype)
    
# compute normalized V term
print("assemble normalized V term",time.time()-t0)
Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
Kg_V = Kg_guv_ml(Th, (magpde.op).V, dtype)


# boundaries
print("assemble boundaries",time.time()-t0)
ndof = Th.nq
tstart = time.time()
# bN=NeumannBC(pde,AssemblyVersion,Num);
# [AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
[ID, IDc, gD] = DirichletBC(magpde, Num)

print("Boundaries done",time.time()-t0)
# operators for landscape
ones=np.ones(Th.nq)
if u:
    Kg_delta = np.zeros((Th.nme, ndfe, ndfe), dtype)
    for i in range(d):
        Kg_delta = Kg_delta + Kg_gdudv(Th, ones, G, i, i, dtype)
    Kg_uV = KgP1_OptV3(Th, V1, dtype)
    Kg_u1 = KgP1_OptV3(Th, ones, dtype)

    """# operators for landscape
    D=magpde.op
    D.A0=None
    Kg_delta = Kg_kinetic_mag(Th,D,G,float)"""

    # RHS for landscapes
    b = RHS(Th, ones, 1, dtype=dtype, version="OptV3")

if eig:
    # circulationn of the normalized vector poltential
    print("normalized circulation of A",time.time()-t0)
    # phi_A = phi((magpde.op).A0, Th)
    with open(
        os.path.realpath(
            os.path.join(data_path, "logPhi", gauge + "h" + str(int(1 / h)) + ".pkl")
        ),
        "rb",
    ) as f:
        logPhi = (L**2) * pickle.load(f)

    # prepare grad_A term
    Kg_A = np.zeros((Th.nme, ndfe, ndfe), dtype)
    for i in range(d + 1):
        for j in range(i):
            Kg_A[:, i, i] = Kg_A[:, i, i] + mu[:, i, j]
            Kg_A[:, j, j] = Kg_A[:, j, j] + mu[:, i, j]

    print("computing eigenproblem",time.time()-t0)
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
    phi_A = np.exp(beta * logPhi * 1j)
    for i in range(d + 1):
        for j in range(i):
            Kg_A[:, i, j] = Kg_A[:, i, j] - np.multiply(mu[:, i, j], phi_A[:, i, j])
            Kg_A[:, j, i] = Kg_A[:, j, i] - np.multiply(mu[:, i, j], phi_A[:, j, i])
    Kg = -Kg_A + eta * Kg_V

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
    print("solving...",time.time()-t0)
    w, x_sol[IDc, :] = eigsh(
        (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N_eig, sigma=target_energy, which="LM"
    )
    Tcpu[3] = time.time() - tstart

    #print("times:", Tcpu)

    print("Ordering eigendata",time.time()-t0)
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
    print("ordering done:",time.time()-t0)

'''
    print("Post-processing")
    # save in one compressed numpy file: V in nq array, th.q , w ordered in N_eig array, x ordered in nq*N_eig array

    np.savez_compressed(
        os.path.join(dir_to_save, name_eig), q=Th.q, V=V1, eig_val=w, eig_vec=x_sol
    )
    t_postpro = time.time() - tstart
    print("saving time:", t_postpro)
    print(w)
    """
    for n in range(N_eig):
        E_proxy = "{:.2e}".format(w[n - 1])
        plt.figure(n)
        plt.clf()
        # PlotBounds(Th,legend=False,color='k')
        plt.axis("off")
        print(n-1)
        PlotIsolines(Th, np.abs(x_sol[:, n - 1]), fill=True, colorbar=True, color="turbo")
        ti, tle = (f"{namepot}: eigenfunction {n} ",
            f" $E_{n}={E_proxy}$)",
        )
        plt.title(f"{ti}(modulus){tle}")
    plt.show()
    plt.close()
    plt.clf()"""
if u:
    Kg_u = Kg_delta + eta * Kg_uV + beta * Kg_u1

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
'''
print(w)
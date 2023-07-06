"""This file aims at defining a gauge invariant discrete operator to
approximate (P+eA)^2 in Landau hamiltonian, 
according to the procedure given by Christiansen and Halvorsen, using the FEM
python structure provided by F. Cuvelier et al. pyVECFEMP1

The computations are made not in physical units:
-1/2m(-i hbar nabla - q A)^2(psi) + V psi = E psi in a box of size lnm
is re tractable from 
-(nabla - i A)^2 + V psi = E psi in a box of size 1
Author: Alioune SEYE
"""


from fem_base.FEMOptV3 import *
from fem_base.pde import *
from fem_base import operators
from fem_base import mesh
import numpy as np
import types
import scipy as sp
from scipy import integrate
from scipy.sparse.linalg import eigsh
from scipy.interpolate import griddata
from numbers import Integral


class L_Aoperator:
    """Operator class similar to L_Operator but includes a magnetic field via a
    specified magnetic vector potential
    """

    def __init__(cls, **kwargs):
        cls.d = kwargs.get("d", 2)
        cls.m = 1
        cls.A0 = kwargs.get("A0", None)
        cls.name = kwargs.get("name", "No name")
        cls.order = 0
        cls.V = kwargs.get("V", 0)
        cls.order = 2
        # for i in range(cls.d):
        #   if isinstance(cls.A0[i],types.FunctionType):
        #    cls.A0[i]=np.vectorize(cls.A0[i])
        if isinstance(cls.A0, types.FunctionType):
            cls.A0 = np.vectorize(cls.A0)
        if isinstance(cls.V, types.FunctionType):
            cls.V = np.vectorize(cls.V)

    def __repr__(cls):
        return "L_Aoperator %s" % (cls.name)

    def __str__(cls, *args):
        return "member of Test"


def Kg_total(Th, D, G, **kwargs):
    d = Th.d
    ndfe = d + 1
    dtype = kwargs.get("dtype", complex)
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
    Kg = Kg + Kg_guv_ml(Th, D.V, dtype)
    Kg = Kg - Kg_kinetic_mag(Th, D, G, dtype)
    return Kg


def Kg_guv_ml(Th, g, dtype):
    gh = FEMtools.setFdata(g, Th, dtype=dtype)
    d = Th.d
    ndfe = d + 1
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype=dtype)
    gme = gh[Th.me]
    for il in range(ndfe):
        Kg[:, il, il] = gme[:, il] * Th.vols / 3
    return Kg


def KgP1_OptV3_A_ml(Th, D, G, **kwargs):
    d = Th.d
    ndfe = d + 1
    dtype = kwargs.get("dtype", complex)
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
    Kg = Kg + Kg_guv_ml(Th, D.V, dtype)
    G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
    Kg = Kg - Kg_kinetic_mag(Th, D, G, dtype)
    return Kg


def Kg_kinetic_mag(Th, D, G, dtype):
    d = Th.d
    ndfe = d + 1
    mu = np.zeros((Th.nme, ndfe, ndfe), dtype)
    Kg_A = np.zeros((Th.nme, ndfe, ndfe), dtype)
    for i in range(d):
        mu += Kg_gdudv(Th, 1, G, i, i, dtype)
    if D.A0 is None:
        return mu
    logPhi_A = circultion_A(D.A0, Th)
    phi_A = np.exp(logPhi_A * 1j)
    for i in range(d + 1):
        for j in range(i):
            Kg_A[:, i, i] = Kg_A[:, i, i] + mu[:, i, j]
            Kg_A[:, j, j] = Kg_A[:, j, j] + mu[:, i, j]
            Kg_A[:, i, j] = Kg_A[:, i, j] - np.multiply(mu[:, i, j], phi_A[:, i, j])
            Kg_A[:, j, i] = Kg_A[:, j, i] - np.multiply(mu[:, i, j], phi_A[:, j, i])
    return Kg_A


def phi(A0, Th):
    d1 = Th.d + 1
    pA = np.ones((Th.nme, d1, d1), dtype=complex)
    for i in range(d1):
        for j in range(i, d1):
            for k in range(Th.nme):
                qi, qj = Th.q[Th.me[k, i], :], Th.q[Th.me[k, j], :]
                x = lambda t: (1 - t) * qi + t * qj
                A0t = lambda t: np.dot(A0(x(t)[0], x(t)[1]), qj - qi)
                Akij = integrate.quad(A0t, 0, 1)[0]
                pA[k, i, j] = np.exp(complex(0, Akij))
            pA[:, j, i] = np.conjugate(pA[:, i, j])
    return pA

def circultion_A(A,Th): #takes Ax and Ay as arrays of size Th.nq and returns an array of circulation of A along edges of Th of size (Th.nme,Th.d+1,Th.d+1)
    A_x,A_y=A[0],A[1]
    A_xme=A_x[Th.me]
    A_yme=A_y[Th.me]
    qme=Th.q[Th.me]
    computed_circ=np.zeros((Th.nme,Th.d+1,Th.d+1))
    for i in range(Th.d+1):
        for j in range(i):
            computed_circ[:,i,j]=(qme[:,j,0]-qme[:,i,0])*(A_xme[:,i]+A_xme[:,j])/2+(qme[:,j,1]-qme[:,i,1])*(A_yme[:,i]+A_yme[:,j])/2
            computed_circ[:,j,i]=-computed_circ[:,i,j]
    return computed_circ

def AxAySym(L,N):#returns A_xsym and A_ysym for hypercube mesh of size LxL and NxN points (N=1/h)
    x=np.linspace(-L/2,L/2,N)
    y=np.linspace(-L/2,L/2,N)
    #in Th.q x changes fastest so to cast Ay to Th.nq shape, we have to reshape.
    A_x=np.repeat(-y/2,N)
    A_y=np.ravel(np.reshape(np.repeat(x/2,N),(N,N)),'F')
    return np.array([A_x,A_y])

def AxAyLandauX(L,N):#returns A_xsym and A_ysym for hypercube mesh of size LxL and NxN points (N=1/h)
    y=np.linspace(-L/2,L/2,N)
    #in Th.q x changes fastest so to cast Ay to Th.nq shape, we have to reshape.
    A_x=np.repeat(-y,N)
    A_y=np.zeros(N*N)
    return (A_x,A_y)

def AxAyLandauY(L,N):#returns A_xsym and A_ysym for hypercube mesh of size LxL and NxN points (N=1/h)
    x=np.linspace(-L/2,L/2,N)
    #in Th.q x changes fastest so to cast Ay to Th.nq shape, we have to reshape.
    A_x=np.zeros(N*N)
    A_y=np.ravel(np.reshape(np.repeat(x/2,N),(N,N)),'F')
    return (A_x,A_y)

def A_LandauX(x, y):
    A = np.array([-y, 0])
    return A


def A_LandauY(x, y):
    A = np.array([0, x])
    return A


def A_Sym(x, y):
    A = np.array([-y / 2, x / 2])
    return A


class magPDE:
    def __init__(cls, Op, Th, **kwargs):
        assert isinstance(Op, L_Aoperator) and isinstance(Th, mesh.Mesh)
        assert Op.d == Th.d
        cls.dtype = kwargs.get("dtype", complex)
        cls.m = 1
        cls.d = Th.d
        cls.Th = Th
        cls.op = Op
        cls.f = kwargs.get("f", None)
        cls.Bh = mesh.BuildBoundaryMeshes(Th)
        cls.labels = np.unique(Th.bel)
        cls.nlab = cls.labels.shape[0]
        cls.bclD = operators.NoneMatrix(Op.m, cls.nlab)
        cls.bclR = operators.NoneMatrix(Op.m, cls.nlab)
        cls.uex = None


def magAssemblyP1(Th, D, G=None, **kwargs):
    dtype = kwargs.get("dtype", complex)
    Kg = Kg_total(Th, D, G, dtype=dtype)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    N = Th.nme * (Th.d + 1) ** 2
    A = sparse.csc_matrix(
        (np.reshape(Kg, N), (np.reshape(Ig, N), np.reshape(Jg, N))),
        shape=(Th.nq, Th.nq),
    )
    A.eliminate_zeros()
    return A


def massLumpAssemblyP1(Th, D, G=None, **kwargs):
    dtype = kwargs.get("dtype", complex)
    Kg = KgP1_OptV3_A_ml(Th, D, G, dtype=dtype)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    N = Th.nme * (Th.d + 1) ** 2
    A = sparse.csc_matrix(
        (np.reshape(Kg, N), (np.reshape(Ig, N), np.reshape(Jg, N))),
        shape=(Th.nq, Th.nq),
    )
    A.eliminate_zeros()
    return A


def buildMagPDEsystem(magpde, Num=1):
    M = magAssemblyP1(magpde.Th, magpde.op, dtype=magpde.dtype)
    b = RHS(magpde.Th, magpde.f, 1, dtype=magpde.dtype)
    [AR, bR] = RobinBC(magpde, "OptV3", 1)
    A = M + AR
    b = b + bR
    [ID, IDc, gD] = DirichletBC(magpde, Num)
    return A, b, ID, IDc, gD


def solveMagPDE(pde, **kwargs):
    Num = 1
    AssemblyVersion = "OptV3"
    SolveVersion = kwargs.get("SolveVersion", "classic")
    SolveOptions = kwargs.get("SolveOptions", None)
    verbose = kwargs.get("verbose", False)
    split = kwargs.get("split", False)
    Tcpu = np.zeros((4,))
    tstart = time.time()
    M = magAssemblyP1(pde.Th, pde.op, Num=Num, dtype=pde.dtype, version=AssemblyVersion)
    Tcpu[0] = time.time() - tstart
    ndof = M.get_shape()[0]
    tstart = time.time()
    b = RHS(pde.Th, pde.f, Num, dtype=pde.dtype, version=AssemblyVersion)
    Tcpu[1] = time.time() - tstart
    tstart = time.time()
    # bN=NeumannBC(pde,AssemblyVersion,Num);
    [AR, bR] = RobinBC(pde, AssemblyVersion, Num)
    b = b + bR  # +bN
    [ID, IDc, gD] = DirichletBC(pde, Num)
    A = M + AR
    # b[IDc]=b[IDc]-A[IDc,::]*gD
    Tcpu[2] = time.time() - tstart
    x = np.zeros((ndof,), dtype=pde.dtype)
    tstart = time.time()
    X, flag = globals()[SolveVersion + "Solve"](
        A, b, ndof, gD, ID, IDc, pde.dtype, SolveOptions
    )
    Tcpu[3] = time.time() - tstart
    if split:
        x = splitPDEsol(pde, X, Num)
    else:
        x = X
    if verbose:
        bb = b[IDc] - A[IDc, ::] * gD
        residu = np.linalg.norm(A[IDc][::, IDc] * X[IDc] - bb)
        Nh = np.linalg.norm(b[IDc])
        if Nh > 1e-9:
            residu /= Nh
        return x, SolveInfo(
            AssemblyVersion,
            SolveVersion,
            SolveOptions,
            Tcpu,
            flag,
            residu,
            ndof,
        )
    else:
        return x


def init_magpde(h, B, l, gauge, V, Th=None):

    if Th is None:
        T = mesh.HyperCube(2, int(l / h), l=l)
    else:
        T = Th
    L=-2*Th.q[0,0]
    N=int(np.sqrt(Th.nq))
    #A0 = lambda x, y: B * (globals()["A_" + gauge](x, y))
    A0 = B*globals()[f"AxAy{gauge}"](L,N)
    Op = L_Aoperator(A0=A0, V=V)
    magpde = magPDE(Op, T, f=1)
    magpde = setBC_PDE(magpde, 1, 0, "Dirichlet", 0, None)
    magpde = setBC_PDE(magpde, 2, 0, "Dirichlet", 0, None)
    magpde = setBC_PDE(magpde, 3, 0, "Dirichlet", 0, None)
    magpde = setBC_PDE(magpde, 4, 0, "Dirichlet", 0, None)
    return magpde



def getSol(**kwargs):
    meshfile = kwargs.get("meshfile", None)
    Th = kwargs.get("Th", None)
    if meshfile != None:
        Th = mesh.readGMSH(meshfile)
    h = kwargs.get("h", 0.1)  # relative size (for a 1x1 square)
    B = kwargs.get("B", 1.0)
    l = kwargs.get("l", 1.0)
    gauge = kwargs.get("gauge", "LandauX")
    V = kwargs.get("V", 0)
    magpde = init_magpde(h * l, B, l, gauge, V, Th)
    return solveMagPDE(magpde)


def get_eigvv(**kwargs):
    meshfile = kwargs.get("meshfile", None)
    Th = kwargs.get("Th", None)
    if meshfile != None:
        Th = mesh.readGMSH(meshfile)
    h = kwargs.get("h", 0.1)  # relative size (for a 1x1 square)
    B = kwargs.get("B", 1)
    N = kwargs.get("N", 1)
    l = kwargs.get("l", 1)
    gauge = kwargs.get("gauge", "LandauX")
    V = kwargs.get("V", 0)
    ml = kwargs.get("mass_lumping", False)
    magpde = init_magpde(h * l, B, l, gauge, V, Th)
    Num = 1
    AssemblyVersion = "OptV3"
    SolveOptions = kwargs.get("SolveOptions", None)
    verbose = kwargs.get("verbose", False)
    Tcpu = np.zeros((4,))
    tstart = time.time()

    print("h=", h, "E_0=", B / 2)
    if ml:
        A_0 = massLumpAssemblyP1(
            magpde.Th,
            magpde.op,
            Num=Num,
            dtype=magpde.dtype,
            version=AssemblyVersion,
        )
        Kg = Kg_guv_ml(Th, 1, complex)
    else:
        A_0 = magAssemblyP1(
            magpde.Th,
            magpde.op,
            Num=Num,
            dtype=magpde.dtype,
            version=AssemblyVersion,
        )
        Kg = KgP1_OptV3(Th, 1, complex)

    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    NN = Th.nme * (Th.d + 1) ** 2
    M = sparse.csc_matrix(
        (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
        shape=(Th.nq, Th.nq),
    )
    M.eliminate_zeros()

    Tcpu[0] = time.time() - tstart
    ndof = A_0.get_shape()[0]
    tstart = time.time()
    Tcpu[1] = time.time() - tstart
    tstart = time.time()
    # bN=NeumannBC(pde,AssemblyVersion,Num);
    [AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
    [ID, IDc, gD] = DirichletBC(magpde, Num)
    A = A_0 + AR
    Tcpu[2] = time.time() - tstart
    x = np.zeros((ndof, N), dtype=magpde.dtype)
    w = np.zeros(N, dtype=complex)
    tstart = time.time()
    xx = np.repeat(gD[ID], N, axis=0)
    x[ID, :] = np.reshape(xx, (len(ID), -1))
    E_0 = B / 2
    t = kwargs.get("target_energy", E_0)
    print("solving...")
    w, x[IDc, :] = eigsh(
        (A[IDc])[::, IDc], M=(M[IDc])[::, IDc], k=N, sigma=t, which="LM"
    )
    Tcpu[3] = time.time() - tstart

    print("times:", Tcpu)

    return w, x


def get_eigvv_ml(**kwargs):
    meshfile = kwargs.get("meshfile", None)
    Th = kwargs.get("Th", None)
    if meshfile != None:
        Th = mesh.readGMSH(meshfile)
    h = kwargs.get("h", 0.1)  # relative size (for a 1x1 square)
    B = kwargs.get("B", 1)
    N = kwargs.get("N", 1)
    l = kwargs.get("l", 1)
    gauge = kwargs.get("gauge", "LandauX")
    V = kwargs.get("V", 0)
    magpde = init_magpde(h * l, B, l, gauge, V, Th)
    Num = 1
    AssemblyVersion = "OptV3"
    Tcpu = np.zeros((4,))
    tstart = time.time()
    A_0 = massLumpAssemblyP1(
        magpde.Th,
        magpde.op,
        Num=Num,
        dtype=magpde.dtype,
        version=AssemblyVersion,
    )

    Kg = Kg_guv_ml(Th, 1, complex)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    NN = Th.nme * (Th.d + 1) ** 2
    M = sparse.csc_matrix(
        (np.reshape(Kg, NN), (np.reshape(Ig, NN), np.reshape(Jg, NN))),
        shape=(Th.nq, Th.nq),
    )
    M.eliminate_zeros()

    Tcpu[0] = time.time() - tstart
    ndof = A_0.get_shape()[0]
    tstart = time.time()
    Tcpu[1] = time.time() - tstart
    tstart = time.time()
    # bN=NeumannBC(pde,AssemblyVersion,Num);
    [AR, bR] = RobinBC(magpde, AssemblyVersion, Num)
    [ID, IDc, gD] = DirichletBC(magpde, Num)
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

    return w, x

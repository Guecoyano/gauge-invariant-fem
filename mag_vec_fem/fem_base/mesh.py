import numpy as np
from scipy.spatial import Delaunay
from scipy import linalg
from math import factorial
from scipy.special import comb
import itertools
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fem_base.gmsh import gmshMesh


class Mesh:
    def __init__(
        self,
        d=0,
        nq=0,
        q=[],
        ql=[],
        nme=0,
        me=[],
        mel=[],
        vols=[],
        nbe=0,
        be=[],
        bel=[],
    ):
        self.d = d
        self.nq = nq
        self.q = q
        self.ql = ql
        self.nme = nme
        self.me = me
        self.mel = mel
        self.nbe = nbe
        self.be = be
        self.bel = bel
        self.vols = vols
        self.h = GetMaxLengthEdges(q, me)


# A ameliorer!
# Th=HyperCube(2,[100,10],trans=lambda q: np.c_[20*q[:,0],2*q[:,1]-1+np.cos(2*pi*q[:,0])])
def HyperCube(d, N=10, **kwargs):
    trans = kwargs.get("trans", None)
    l = kwargs.get("l", 1)
    a = -l / 2
    b = l / 2
    if isinstance(N, int):
        n_bins = (N) * np.ones(d)
    else:
        n_bins = np.flipud(N)
    bounds = np.repeat([(a, b)], d, axis=0)

    A = np.mgrid[
        [slice(row[0], row[1], n * 1j) for row, n in zip(bounds, n_bins)]
    ]
    q = np.array([A[i].ravel() for i in range(d)]).T
    q = q[:, range(d - 1, -1, -1)]  # Matlab like
    me = Delaunay(q).simplices
    nme = me.shape[0]
    volumes = ComputeSignVolVec(q, me)
    Ix = np.where(abs(volumes) < l * l * 1e-10)[0]
    if Ix.shape[0] > 0:
        Ii = np.setdiff1d(range(0, nme), Ix)
        me = me[Ii]
        nme = me.shape[0]
        volumes = volumes[Ii]
    Ix = np.where(volumes < 0)[0]
    if Ix.shape[0] > 0:  # On permute pour etre dans le sens "direct"
        # print(me.shape)
        tmp = me[Ix, d]
        me[Ix, d] = me[Ix, d - 1]
        me[Ix, d - 1] = tmp
    # Boundary
    V = np.array([x for x in itertools.combinations(range(d + 1), d)])
    BE = np.array([me[::, V[i]] for i in range(d + 1)]).reshape(
        nme * (d + 1), d
    )
    BE.sort()
    be = np.array(
        [np.array(x) for x in set(tuple(x) for x in BE)]
    )  # equivalent to unique(...,'row') of Matlab
    nbe = be.shape[0]
    bel = np.zeros(nbe)
    Qb = np.array([q[be[::, i], ::] for i in range(d)])
    # print(Qb.shape)
    tol = 1e-12 * l
    label = 1
    for i in range(d):
        I = np.arange(nbe)
        for j in range(d):
            II = (abs(Qb[j, ::, i] - a) < tol).nonzero()[0]
            I = np.intersect1d(I, II)
        bel[I] = label
        label += 1
        I = np.arange(nbe)
        for j in range(d):
            II = (abs(Qb[j, ::, i] - b) < tol).nonzero()[0]
            I = np.intersect1d(I, II)
        bel[I] = label
        label += 1

    I = (bel == 0).nonzero()[0]
    J = np.setdiff1d(np.arange(nbe), I)
    bb = bel[J]
    ii = bb.argsort()
    be = be[J[ii]]
    bel = bel[J[ii]]
    nbe = bel.shape[0]
    if trans != None:
        q = trans(q)
        volumes = ComputeSignVolVec(q, me)
    return Mesh(d, q.shape[0], q, [], nme, me, [], abs(volumes), nbe, be, bel)


def HyperCubeKuhn(d, N):
    q, me = hypercube.GlobalKuhn(N * np.ones((d,)))
    me = me.T
    a = 0.0
    b = 1.0
    nme = me.shape[0]
    volumes = np.ones((nme,)) / nme
    V = np.array([x for x in itertools.combinations(range(d + 1), d)])
    BE = np.array([me[::, V[i]] for i in range(d + 1)]).reshape(
        nme * (d + 1), d
    )
    BE.sort()
    be = np.array(
        [np.array(x) for x in set(tuple(x) for x in BE)]
    )  # equivalent to unique(...,'row') of Matlab
    nbe = be.shape[0]
    bel = np.zeros(nbe)
    Qb = np.array([q[be[::, i], ::] for i in range(d)])
    tol = 1e-12
    label = 1
    for i in range(d):
        I = np.arange(nbe)
        for j in range(d):
            II = (abs(Qb[j, ::, i] - a) < tol).nonzero()[0]
            I = np.intersect1d(I, II)
        bel[I] = label
        label += 1
        I = np.arange(nbe)
        for j in range(d):
            II = (abs(Qb[j, ::, i] - b) < tol).nonzero()[0]
            I = np.intersect1d(I, II)
        bel[I] = label
        label += 1

    I = (bel == 0).nonzero()[0]
    J = np.setdiff1d(np.arange(nbe), I)
    bb = bel[J]
    ii = bb.argsort()
    be = be[J[ii]]
    bel = bel[J[ii]]
    nbe = bel.shape[0]
    return Mesh(d, q.shape[0], q, [], nme, me, [], abs(volumes), nbe, be, bel)


def ComputeVolVecOld(d, q, me):
    nme = me.shape[0]
    D = np.zeros((d, d, nme))
    for i in range(d):
        for j in range(1, d + 1):
            D[i, j - 1] = q[me[::, j], i] - q[me[::, 0], i]

    C = factorial(d)
    vol = np.array(
        [abs(linalg.det(D[::, ::, k]) / C) for k in range(D.shape[2])]
    )
    return vol


def ComputeVolVec(d, q, me):
    n = q.shape[1]
    nme = me.shape[0]
    X = np.zeros((d, nme, n))
    for i in range(d):
        X[i] = q[me[::, i + 1]] - q[me[::, 0]]
    V = np.zeros((d, d, nme))
    for i in range(d):
        V[i, i] = (X[i] * X[i]).sum(axis=1)
        for j in range(i + 1, d):
            V[i, j] = V[j, i] = (X[i] * X[j]).sum(axis=1)

    vol = np.array(
        [
            np.sqrt(abs(linalg.det(V[::, ::, k]))) / factorial(d)
            for k in range(nme)
        ]
    )
    return vol


def ComputeSignVolVec(q, me):
    n = q.shape[1]
    d = me.shape[1] - 1
    assert d == n
    nme = me.shape[0]
    X = np.zeros((n, n, nme))
    for i in range(n):
        for j in range(n):
            X[i, j] = q[me[::, i + 1], j] - q[me[::, 0], j]
        # X[i]=(q[me[::,i+1]]-q[me[::,0]]).T
    vol = np.array(
        [linalg.det(X[::, ::, k]) / factorial(n) for k in range(nme)]
    )
    return vol  # ,X


def readFreeFEM(meshfile):
    fp = open(meshfile, "rt")
    nq, nme, nbe = np.fromfile(fp, sep=" ", dtype=np.int32, count=3)
    data = np.fromfile(fp, sep=" ", dtype=np.float64, count=3 * nq)
    data.shape = (nq, 3)
    q = data[:, [0, 1]]
    ql = np.int32(data[:, 2])
    data = np.fromfile(fp, sep=" ", dtype=np.int32, count=4 * nme)
    data.shape = (nme, 4)
    me = data[:, [0, 1, 2]] - 1
    mel = data[:, 3]
    data = np.fromfile(fp, sep=" ", dtype=np.int32, count=3 * nbe)
    data.shape = (nbe, 3)
    be = data[:, [0, 1]] - 1
    bel = data[:, 2]
    volumes = ComputeVolVec(2, q, me)
    return Mesh(2, nq, q, ql, nme, me, mel, volumes, nbe, be, bel)


def readFreeFEM3D(filename):
    fp = open(filename, "rt")
    line = ""
    while line.find("Vertices") == -1:
        line = fp.readline()
    nq = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
    data = np.fromfile(fp, sep=" ", dtype=np.float64, count=4 * nq)
    data.shape = (nq, 4)
    q = data[:, [0, 1, 2]]
    ql = np.int32(data[:, 3])
    fp.seek(0)
    line = ""
    while line.find("Triangles") == -1:
        line = fp.readline()
    nbe = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
    data = np.fromfile(fp, sep=" ", dtype=np.int32, count=4 * nbe)
    data.shape = (nbe, 4)
    be = data[:, [0, 1, 2]] - 1
    bel = data[:, 3]
    fp.seek(0)
    line = ""
    while line.find("Tetrahedra") == -1:
        line = fp.readline()
    nme = np.fromfile(fp, sep=" ", dtype=np.int32, count=1)[0]
    data = np.fromfile(fp, sep=" ", dtype=np.int32, count=5 * nme)
    data.shape = (nme, 5)
    me = data[:, [0, 1, 2, 3]] - 1
    mel = data[:, 4]
    fp.close()
    volumes = ComputeVolVec(3, q, me)
    return Mesh(3, nq, q, ql, nme, me, mel, volumes, nbe, be, bel)


def readGMSH(meshfile):
    M = gmshMesh()
    M.read_msh(meshfile)
    ElmtsDims = list(M.Elmts.keys())
    d = 0
    if 1 and 2 in ElmtsDims:
        d = 2
        pMe = 2
        pBe = 1
    if 2 and 4 in ElmtsDims:  # 4 for tetrahedrons
        d = 3
        pMe = 4
        pBe = 2
    assert d != 0
    me = M.Elmts[pMe][1]
    mel = M.Elmts[pMe][0]
    be = M.Elmts[pBe][1]
    bel = M.Elmts[pBe][0]
    q = M.Verts[:, 0:d]

    volumes = ComputeVolVec(d, q, me)
    nq = q.shape[0]
    nme = me.shape[0]
    nbe = be.shape[0]
    return Mesh(d, nq, q, [], nme, me, mel, volumes, nbe, be, bel)


def GetMaxLengthEdges(q, me):
    ne = me.shape[1]
    h = 0.0
    for i in range(ne):
        for j in range(i + 1, ne):
            h = max(
                h, np.sum((q[me[::, i]] - q[me[::, j]]) ** 2, axis=1).max()
            )
    return np.sqrt(h)


# 3) Boundary Meshes
class bdMesh:
    def __init__(
        self,
        d=0,
        nq=0,
        q=[],
        nme=0,
        me=[],
        mel=[],
        vols=[],
        toGlobal=[],
        nqGlobal=[],
        label=[],
    ):
        self.d = d
        self.nq = nq
        self.q = q
        self.nme = nme
        self.me = me
        self.mel = mel
        self.toGlobal = toGlobal
        self.nqGlobal = nqGlobal
        self.label = label
        self.vols = vols


def BuildBoundaryMesh(Th, Label):
    I = (Th.bel == Label).nonzero()[0]
    BE = Th.be[I]
    indQ = np.unique(BE)
    Q = Th.q[indQ]
    lQ = np.arange(indQ.shape[0])
    J = np.zeros((Th.nq,), dtype=np.int64)
    J[indQ] = lQ
    ME = J[BE]
    return bdMesh(
        Th.d - 1,
        Q.shape[0],
        Q,
        ME.shape[0],
        ME,
        [],
        ComputeVolVec(Th.d - 1, Q, ME),
        indQ,
        Th.nq,
        Label,
    )


def BuildBoundaryMeshes(Th):
    labels = np.unique(Th.bel)
    nlab = labels.shape[0]
    Bh = []
    for l in range(nlab):
        Bh.append(BuildBoundaryMesh(Th, labels[l]))
    return Bh

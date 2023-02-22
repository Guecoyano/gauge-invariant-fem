from . import operators
from . import mesh
from .FEMtools import *

import numpy as np
from scipy import linalg
from scipy import sparse
from math import *


# def DAssemblyP1_base(Th,D,*args,**kwargs):
def DAssemblyP1_base(Th, Lop, G=None, **kwargs):
    assert operators.isoperatorL(Lop)
    spformat = kwargs.get("spformat", "csc")
    Lh = Lhoperator(Lop, Th)
    M = sparse.lil_matrix((Th.nq, Th.nq))
    ndfe = Th.d + 1
    for k in range(Th.nme):
        E = DElemP1(Th, Lh, k)
        for il in range(ndfe):
            i = Th.me[k, il]
            for jl in range(ndfe):
                j = Th.me[k, jl]
                M[i, j] = M[i, j] + E[il, jl]
    if spformat == "lil":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def DAssemblyP1_basea(Th, Lop, G=None, **kwargs):
    assert operators.isoperatorL(Lop)
    spformat = kwargs.get("spformat", "csc")
    Lh = Lhoperator(Lop, Th)
    M = sparse.lil_matrix((Th.nq, Th.nq))
    ndfe = Th.d + 1
    for k in range(Th.nme):
        E = DElemP1(Th, Lh, k)
        for il in range(ndfe):
            i = Th.me[k, il]
            for jl in range(ndfe):
                j = Th.me[k, jl]
                M[i, j] = M[i, j] + E[il, jl]
    if spformat == "lil":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def HAssemblyP1_base(Th, Hop, Num, **kwargs):
    spformat = kwargs.get("spformat", "csc")
    assert operators.isoperatorH(Hop)
    assert isinstance(Th, mesh.Mesh)
    m = Hop.m
    nq = Th.nq
    ndof = m * nq
    M = sparse.csc_matrix((ndof, ndof))
    ndfe = Th.d + 1
    VFNum = getVFindices(Num, m, nq)
    I = np.arange(Th.nq)
    for il in range(m):
        Il = VFNum(I, il)
        for jl in range(m):
            Jl = VFNum(I, jl)
            D = DAssemblyP1_base(Th, Hop.H[il][jl]).tocoo()
            Dr = sparse.csc_matrix(
                (D.data, (VFNum(D.row, il), VFNum(D.col, jl))),
                shape=(ndof, ndof),
            )
            M = M + Dr
    if spformat == "csc":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def HAssemblyP1_baseb(Th, Hop, Num, **kwargs):
    spformat = kwargs.get("spformat", "csc")
    assert operators.isoperatorH(Hop)
    assert isinstance(Th, mesh.Mesh)
    m = Hop.m
    nq = Th.nq
    ndof = m * nq
    M = sparse.lil_matrix((ndof, ndof))
    ndfe = Th.d + 1
    VFNum = getVFindices(Num, m, nq)
    I = np.arange(Th.nq)
    for il in range(m):
        Il = VFNum(I, il)
        for jl in range(m):
            Jl = VFNum(I, jl)
            M[np.ix_(Il, Jl)] = DAssemblyP1_base(
                Th, Hop.H[il][jl], spformat="lil"
            )
    if spformat == "lil":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def HAssemblyP1_basea(Th, Hop, Num, **kwargs):
    spformat = kwargs.get("spformat", "csc")
    assert operators.isoperatorH(Hop)
    assert isinstance(Th, mesh.Mesh)
    m = Hop.m
    nq = Th.nq
    ndof = m * nq
    M = sparse.lil_matrix((ndof, ndof))
    ndfe = Th.d + 1
    VFNum = getVFindices(Num, m, nq)
    I = np.arange(Th.nq)
    for jl in range(m):
        Jl = VFNum(I, jl)
        for il in range(m):
            Il = VFNum(I, il)
            M[np.ix_(Il, Jl)] = DAssemblyP1_base(
                Th, Hop.H[il][jl], spformat="lil"
            )
    if spformat == "lil":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def DElemP1(Th, Dh, k):
    d = Th.d
    vol = Th.vols[k]
    mek = Th.me[k]
    # E=np.zeros((d+1,d+1))
    if Dh.a0 != None:
        E = DElemP1_guv(d, vol, getLocalfData(Dh.a0, Th, k))
    else:
        E = np.zeros((d + 1, d + 1))
    if Dh.order == 0:
        return E
    G = GradientLocal(Th.q[Th.me[k]], Th.vols[k])

    if Dh.A != None:
        for i in range(d):
            for j in range(d):
                if Dh.A[i][j] != None:
                    E = DElemP1_gdudv(
                        d, vol, getLocalfData(Dh.A[i][j], Th, k), j, i, G, E
                    )

    if Dh.b != None:
        for i in range(d):
            if Dh.b[i] != None:
                E = DElemP1_gudv(
                    d, vol, getLocalfData(Dh.b[i], Th, k), i, G, E
                )
    if Dh.c != None:
        for i in range(d):
            if Dh.c[i] != None:
                E = DElemP1_gduv(
                    d, vol, getLocalfData(Dh.c[i], Th, k), i, G, E
                )
    return E


def DElemP1_guv(d, vol, gl):
    MatElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 3)))
        * (1 + (il == jl))
        * vol
        * (gl.sum() + gl[il] + gl[jl])
    )
    De = np.zeros((d + 1, d + 1))
    for il in range(d + 1):
        for jl in range(d + 1):
            De[il, jl] = MatElem(il, jl)
    return De


def DElemP1_gdudv(d, vol, gl, i, j, G, De):
    MatElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 1)))
        * vol
        * gl.sum()
        * G[i, jl]
        * G[j, il]
    )
    for il in range(d + 1):
        for jl in range(d + 1):
            De[il, jl] += MatElem(il, jl)
    return De


def DElemP1_gduv(d, vol, gl, i, G, De):
    MatElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 2)))
        * vol
        * (gl.sum() + gl[il])
        * G[i, jl]
    )
    for il in range(d + 1):
        for jl in range(d + 1):
            De[il, jl] += MatElem(il, jl)
    return De


def DElemP1_gudv(d, vol, gl, i, G, De):
    MatElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 2)))
        * vol
        * (gl.sum() + gl[jl])
        * G[i, il]
    )
    for il in range(d + 1):
        for jl in range(d + 1):
            De[il, jl] -= MatElem(il, jl)
    return De


def getLocalfData(fh, Th, k):
    return fh[Th.me[k]]


def GradientLocal(ql, vol):
    d = ql.shape[1]
    G = np.zeros((d, d + 1))
    if d == 1:
        G[0, 0] = -1 / vol
        G[0, 1] = 1 / vol
        return G
    if d == 2:
        D23 = ql[1] - ql[2]
        D13 = ql[0] - ql[2]
        D12 = ql[0] - ql[1]
        C = 1 / (2.0 * vol)
        G[0, 0] = D23[1] * C
        G[1, 0] = -D23[0] * C
        G[0, 1] = -D13[1] * C
        G[1, 1] = D13[0] * C
        G[0, 2] = D12[1] * C
        G[1, 2] = -D12[0] * C
        return G
    if d == 3:
        D12 = ql[0] - ql[1]
        D13 = ql[0] - ql[2]
        D14 = ql[0] - ql[3]
        D23 = ql[1] - ql[2]
        D24 = ql[1] - ql[3]
        D34 = ql[2] - ql[3]
        C = 1 / (6 * vol)
        G[0, 0] = (-D23[1] * D24[2] + D23[2] * D24[1]) * C
        G[1, 0] = (D23[0] * D24[2] - D23[2] * D24[0]) * C
        G[2, 0] = (-D23[0] * D24[1] + D23[1] * D24[0]) * C
        G[0, 1] = (D13[1] * D14[2] - D13[2] * D14[1]) * C
        G[1, 1] = (-D13[0] * D14[2] + D13[2] * D14[0]) * C
        G[2, 1] = (D13[0] * D14[1] - D13[1] * D14[0]) * C
        G[0, 2] = (-D12[1] * D14[2] + D12[2] * D14[1]) * C
        G[1, 2] = (D12[0] * D14[2] - D12[2] * D14[0]) * C
        G[2, 2] = (-D12[0] * D14[1] + D12[1] * D14[0]) * C
        G[0, 3] = (D12[1] * D13[2] - D12[2] * D13[1]) * C
        G[1, 3] = (-D12[0] * D13[2] + D12[2] * D13[0]) * C
        G[2, 3] = (D12[0] * D13[1] - D12[1] * D13[0]) * C
        return G
    return ComputeGradientdD(ql, vol)


def ComputeGradientdD(ql, vol):
    d = ql.shape[1]
    Grad = np.zeros((d + 1, d))
    Grad[1 : d + 1] = np.eye(d)
    Grad[0] = -1
    X = np.zeros((d, d))
    for i in range(d):
        X[:, i] = ql[i + 1] - ql[0]
    G1 = np.dot(Grad, linalg.inv(X))
    return G1.transpose()


# Operator L discretized on mesh Th
class Lhoperator:
    def __init__(self, Lop, Th):
        assert operators.isoperatorL(Lop)
        assert isinstance(Th, mesh.Mesh) or isinstance(Th, mesh.bdMesh)
        self.d = Lop.d
        self.m = 1
        self.order = Lop.order
        if Lop.a0 == None:
            self.a0 = None
        else:
            self.a0 = setFdata(Lop.a0, Th)

        if Lop.A == None:
            self.A = None
        else:
            self.A = operators.NoneMatrix(self.d, self.d)
            for i in range(self.d):
                for j in range(self.d):
                    self.A[i][j] = setFdata(Lop.A[i][j], Th)

        if Lop.b == None:
            self.b = None
        else:
            self.b = operators.NoneVector(self.d)
            for i in range(self.d):
                self.b[i] = setFdata(Lop.b[i], Th)
        if Lop.c == None:
            self.c = None
        else:
            self.c = operators.NoneVector(self.d)
            for i in range(self.d):
                self.c[i] = setFdata(Lop.c[i], Th)


# Operator Hop discretized on mesh Th
class Hhoperator:
    def __init__(self, Hop, Th):
        assert operators.isoperatorH(Hop)
        assert isinstance(Th, mesh.Mesh)
        self.d = Hop.d
        self.m = Hop.m
        self.H = operators.NoneMatrix(self.m, self.m)
        for i in range(self.m):
            for j in range(self.m):
                self.H[i][j] = Lhoperator(Hop.H[i][j], Th)

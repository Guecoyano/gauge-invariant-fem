from math import factorial
from scipy import sparse
import numpy as np

from fem_base import FEMtools
from fem_base import operators


def DAssemblyP1_OptV3(Th, D, G=None, **kwargs):
    dtype = kwargs.get("dtype", float)
    Kg = KgP1_OptV3(Th, D, G, dtype=dtype)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    N = Th.nme * (Th.d + 1) ** 2
    A = sparse.csc_matrix(
        (np.reshape(Kg, N), (np.reshape(Ig, N), np.reshape(Jg, N))),
        shape=(Th.nq, Th.nq),
    )
    A.eliminate_zeros()
    return A


def HAssemblyP1_OptV3(Th, Hop, Num, **kwargs):
    dtype = kwargs.get("dtype", float)
    spformat = kwargs.get("spformat", "csc")
    m = Hop.m
    nq = Th.nq
    ndof = m * nq
    VFNum = FEMtools.getVFindices(Num, m, nq)
    M = sparse.csc_matrix((ndof, ndof), dtype=dtype)
    G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
    Ig, Jg = IgJgP1_OptV3(Th.d, Th.nme, Th.me)
    N = Th.nme * (Th.d + 1) ** 2
    Ig = np.reshape(Ig, N)
    Jg = np.reshape(Jg, N)

    for i in range(m):
        for j in range(m):
            if Hop.H[i][j] != None:
                Kg = np.reshape(KgP1_OptV3(Th, Hop.H[i][j], G, dtype=dtype), N)
                M = M + sparse.csc_matrix(
                    (Kg, (VFNum(Ig, i), VFNum(Jg, j))),
                    shape=(ndof, ndof),
                    dtype=dtype,
                )
    if spformat == "csc":
        return M
    else:
        return eval("M.to%s()" % spformat, globals(), locals())


def IgJgP1_OptV3(d, nme, me):
    ndfe = d + 1
    Ig = np.zeros((nme, ndfe, ndfe), dtype=np.int32)
    Jg = np.zeros((nme, ndfe, ndfe), dtype=np.int32)
    for il in range(ndfe):
        mel = me[::, il]
        for jl in range(ndfe):
            Ig[:, il, jl] = Jg[:, jl, il] = mel
    return Ig, Jg


def KgP1_OptV3(Th, D, G, **kwargs):
    d = Th.d
    ndfe = d + 1
    dtype = kwargs.get("dtype", float)
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype)
    Kg = Kg + KgP1_OptV3_guv(Th, D.a0, dtype)
    if D.order == 0:
        return Kg
    G = FEMtools.ComputeGradientVec(Th.q, Th.me, Th.vols)
    if D.A != None:
        for i in range(d):
            for j in range(d):
                Kg = Kg + KgP1_OptV3_gdudv(Th, D.A[i][j], G, j, i, dtype)
    if D.b != None:
        for i in range(d):
            Kg = Kg - KgP1_OptV3_gudv(Th, D.b[i], G, i, dtype)
    if D.c != None:
        for i in range(d):
            Kg = Kg + KgP1_OptV3_gduv(Th, D.c[i], G, i, dtype)
    return Kg


def KgP1_OptV3_guv(Th, g, dtype):
    if not (isinstance(g, np.ndarray) and (g.shape[0] == Th.nq)) and g == None:
        return 0
    gh = FEMtools.setFdata(g, Th, dtype=dtype)
    d = Th.d
    ndfe = d + 1
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype=dtype)
    gme = gh[Th.me]
    gs = (gme.sum(axis=1)).reshape(Th.nme)
    KgElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 3)))
        * (1 + (il == jl))
        * Th.vols
        * (gs + gme[:, il] + gme[:, jl])
    )
    for il in range(ndfe):
        for jl in range(ndfe):
            Kg[:, il, jl] = KgElem(il, jl)
    return Kg


def KgP1_OptV3_gdudv(Th, g, G, i, j, dtype):
    if g == None:
        return 0
    gh = FEMtools.setFdata(g, Th, dtype=dtype)
    d = Th.d
    ndfe = d + 1
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype=dtype)
    gme = gh[Th.me]
    gs = (gme.sum(axis=1)).reshape(Th.nme)
    KgElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 1)))
        * Th.vols
        * gs
        * G[jl, i]
        * G[il, j]
    )
    for il in range(ndfe):
        for jl in range(ndfe):
            Kg[:, il, jl] = KgElem(il, jl)
    return Kg


def KgP1_OptV3_gudv(Th, g, G, i, dtype):
    if g == None:
        return 0
    gh = FEMtools.setFdata(g, Th, dtype=dtype)
    d = Th.d
    ndfe = d + 1
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype=dtype)
    gme = gh[Th.me]
    gs = (gme.sum(axis=1)).reshape(Th.nme)
    KgElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 2)))
        * Th.vols
        * (gs + gme[:, jl])
        * G[il, i]
    )
    for il in range(ndfe):
        for jl in range(ndfe):
            Kg[:, il, jl] = KgElem(il, jl)
    return Kg


def KgP1_OptV3_gduv(Th, g, G, i, dtype):
    if g == None:
        return 0
    gh = FEMtools.setFdata(g, Th, dtype=dtype)
    d = Th.d
    ndfe = d + 1
    Kg = np.zeros((Th.nme, ndfe, ndfe), dtype=dtype)
    gme = gh[Th.me]
    gs = (gme.sum(axis=1)).reshape(Th.nme)
    KgElem = (
        lambda il, jl: (factorial(d) / float(factorial(d + 2)))
        * Th.vols
        * (gs + gme[:, il])
        * G[jl, i]
    )
    for il in range(ndfe):
        for jl in range(ndfe):
            Kg[:, il, jl] = KgElem(il, jl)
    return Kg

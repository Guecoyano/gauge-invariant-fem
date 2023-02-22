import numpy as np
import itertools
from math import factorial


def perms(L):
    # perms(range(3))
    return np.array([x for x in itertools.permutations(np.flipud(L), len(L))])


def Vertices(N, isIndice):
    # q=hypercube.Vertices([3,4,5],False)
    n_bins = np.flipud(N)
    d = len(N)
    if isIndice:
        bounds = np.zeros((d, 2))
        for i in range(d):
            bounds[i, 1] = n_bins[i] - 1
    else:
        bounds = np.repeat([(0, 1)], d, axis=0)
    A = np.mgrid[
        [slice(row[0], row[1], n * 1j) for row, n in zip(bounds, n_bins)]
    ]
    q = np.array([A[i].ravel() for i in range(d)]).T
    q = q[:, np.arange(d - 1, -1, -1)]  # Matlab like
    return q


def LocalKuhn(d):
    qBase = Vertices(2 * np.ones((d,)), False)
    BaseSimplex = np.vstack([np.zeros((1, d)), np.tril(np.ones((d, d)))]).T
    P = np.array([x for x in itertools.permutations(range(d - 1, -1, -1), d)])
    nmeBase = factorial(d)
    meBase = np.zeros((d + 1, nmeBase), dtype=int)
    A = np.matrix(2 ** np.arange(d))
    for i in range(nmeBase):
        meBase[:, i] = A * BaseSimplex[P[i]]
        ql = qBase[meBase[:, i]]
        s = orientation(ql)
        if s == -1:
            tmp = meBase[1, i]
            meBase[1, i] = meBase[2, i]
            meBase[2, i] = tmp

    return qBase, meBase


def orientation(ql):
    d = ql.shape[1]
    D = np.c_[np.ones((d + 1, 1)), ql]
    return np.sign(np.linalg.det(D))


def GlobalKuhn(N):
    d = len(N)
    qBase, meBase = LocalKuhn(d)
    q = Vertices(N, False)
    NN = np.array(N)
    Nhypercube = np.prod(NN - 1)
    C = np.ones((d,))
    C[1:] = np.cumprod(NN[0:-1])
    CK = np.ones((d,))
    CK[1:] = np.cumprod(NN[0:-1]) - 1
    J = np.sum(C * qBase, axis=1)
    nme = Nhypercube * meBase.shape[1]
    nmeBase = meBase.shape[1]
    q = Vertices(N, False)
    NN = np.array(N)
    Nhypercube = np.prod(NN - 1)
    C = np.ones((d,))
    C[1:] = np.cumprod(NN[0:-1])
    CK = np.ones((d,))
    CK[1:] = np.cumprod(NN[0:-1]) - 1
    J = np.sum(C * qBase, axis=1)
    nme = Nhypercube * meBase.shape[1]

    qInd = Vertices(NN - 1, True)
    ki = np.sum(C * qInd, axis=1)
    nJ = len(J)
    Ind = np.zeros((nJ, Nhypercube), dtype=int)
    for l in range(nJ):
        Ind[l, :] = ki + J[l]

    K = np.arange(0, nme, nmeBase, dtype=int)
    me = np.zeros((d + 1, nme), dtype=int)

    for l in range(nmeBase):
        me[:, K] = Ind[meBase[:, l], :]
        K = K + 1
    return q, me

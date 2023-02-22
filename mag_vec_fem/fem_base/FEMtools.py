import numpy as np
from scipy import linalg
from math import *


def genericFunc(d, sFunc):
    S = "np.vectorize(lambda x1"
    for i in range(2, d + 1):
        S = S + ",x" + str(i)
    S = S + ":" + sFunc + ")"
    # print(S)
    return eval(S)


def getVFindices(Num, m, nq):
    if Num == 1:
        return lambda I, i: I + i * nq
    else:
        return lambda I, i: m * I + i


def setFdata(f, Th, **kwargs):
    # if f==None:
    #  return None
    dtype = kwargs.get("dtype", float)
    Num = kwargs.get("Num", 1)
    if np.isscalar(f):
        return f * np.ones((Th.nq,), dtype=dtype)
    if isinstance(f, type(lambda: None)):
        f = np.vectorize(f)
    if isinstance(f, np.lib.function_base.vectorize):
        d = Th.q.shape[1]
        S = "f(Th.q[:,0]"
        for i in range(1, d):
            S = S + ",Th.q[:," + str(i) + "]"
        S = S + ")"
        V = eval(S)
        if np.isscalar(V):
            return V * np.ones((Th.nq,), dtype=dtype)
        return V
    if isinstance(f, np.ndarray) and (f.shape[0] == Th.nq):
        return f
    if isinstance(f, list):  # Vector Field case
        n = len(f)
        Fh = np.zeros((n * Th.nq,))
        I = np.arange(Th.nq)
        VFInd = getVFindices(Num, n, Th.nq)
        for i in range(n):
            Fh[VFInd(I, i)] = setFdata(f[i], Th)
        # Fh=np.zeros((n,Th.nq))
        # for i in range(n):
        #  Fh[i]=setFdata(f[i],Th,dtype=dtype)
        return Fh
    else:
        assert 0 == 1


def ComputeGradientVec(q, me, vols):
    d = q.shape[1]
    if d == 1:
        return ComputeGradientVec1D(q, me, vols)
    elif d == 2:
        return ComputeGradientVec2D(q, me, vols)
    elif d == 3:
        return ComputeGradientVec3D(q, me, vols)
    else:
        return ComputeGradientVecdD(q, me, vols)


def ComputeGradientVec1D(q, me, vols):
    nme = me.shape[0]
    G = np.ndarray(shape=(2, 1, nme))
    G[0, 1] = 1 / vols
    G[0, 0] = -G[0, 1]


def ComputeGradientVec2D(q, me, vols):
    nme = me.shape[0]
    C = 1 / (2 * vols)

    G = np.ndarray(shape=(3, 2, nme))
    u = (q[me[:, 1]] - q[me[:, 2]]).T
    G[0, 0] = u[1] * C
    G[0, 1] = -u[0] * C
    u = (q[me[:, 2]] - q[me[:, 0]]).T
    G[1, 0] = u[1] * C
    G[1, 1] = -u[0] * C
    u = (q[me[:, 0]] - q[me[:, 1]]).T
    G[2, 0] = u[1] * C
    G[2, 1] = -u[0] * C
    return G


def ComputeGradientVec3D(q, me, vols):
    nme = me.shape[0]
    q1 = q[me[:, 0]].T
    q2 = q[me[:, 1]].T
    q3 = q[me[:, 2]].T
    q4 = q[me[:, 3]].T
    C = 1 / (6 * vols)
    D12 = q1 - q2
    D13 = q1 - q3
    D14 = q1 - q4
    D23 = q2 - q3
    D24 = q2 - q4

    G = np.ndarray(shape=(4, 3, nme))
    G[0, 0] = (D23[2] * D24[1] - D23[1] * D24[2]) * C
    G[0, 1] = (D23[0] * D24[2] - D23[2] * D24[0]) * C
    G[0, 2] = (D23[1] * D24[0] - D23[0] * D24[1]) * C
    G[1, 0] = (D13[1] * D14[2] - D13[2] * D14[1]) * C
    G[1, 1] = (D13[2] * D14[0] - D13[0] * D14[2]) * C
    G[1, 2] = (D13[0] * D14[1] - D13[1] * D14[0]) * C
    G[2, 0] = (D12[2] * D14[1] - D12[1] * D14[2]) * C
    G[2, 1] = (D12[0] * D14[2] - D12[2] * D14[0]) * C
    G[2, 2] = (D12[1] * D14[0] - D12[0] * D14[1]) * C
    G[3, 0] = (D12[1] * D13[2] - D12[2] * D13[1]) * C
    G[3, 1] = (D12[2] * D13[0] - D12[0] * D13[2]) * C
    G[3, 2] = (D12[0] * D13[1] - D12[1] * D13[0]) * C
    return G


def ComputeGradientVecdD(q, me, vols):
    d = q.shape[1]
    nme = me.shape[0]
    Grad = np.zeros((d + 1, d))
    Grad[1 : d + 1] = np.eye(d)
    Grad[0] = -1

    X = np.zeros((nme, d, d))
    for i in range(d):
        X[:, :, i] = q[me[:, i + 1]] - q[me[:, 0]]

    # G=np.ndarray(shape=(d+1,d,nme))
    G1 = np.array([np.dot(Grad, linalg.inv(X[k])) for k in range(nme)])
    # for k in range(nme):
    # G[::,::,k]=np.dot(Grad,linalg.inv(X[k]))
    return G1.swapaxes(0, 1).swapaxes(1, 2)

import numpy as np

rnd = np.random.random
binom = np.random.binomial
from math import *


def nrandgaussi(N, deltax, deltay, force):
    for k in range(N):
        ampl, r2, r3 = rnd() * force, rnd(), rnd()
        x = deltax * (1 / 2 - r2)
        y = deltay * (1 / 2 - r3)
        print("+")
        print(str(ampl) + "*gaussXY[" + str(x) + "," + str(y) + "]")


def ngaussi(N, deltax, deltay):
    for k in range(N):
        r2, r3 = rnd(), rnd()
        x = deltax * (1 / 2 - r2)
        y = deltay * (1 / 2 - r3)
        print("+")
        print(str(1) + "*gaussXY[" + str(x) + "," + str(y) + "]")


def alloytypestring(N, x):
    for k in range(N):
        for j in range(N):
            d = binom(21, x)
            print("+")
            print(
                str(d)
                + "*gaussXY["
                + str(-1 / 2 + k / N)
                + ","
                + str(-1 / 2 + j / N)
                + "]"
            )


def alloymatrix(N, V, x):
    M = np.zeros((N, N))
    for k in range(N):
        for j in range(N):
            M[k, j] = binom(V, x) / V
    return M


def boolmatrix(N, x):
    M = np.zeros((N, N))
    for k in range(N):
        for j in range(N):
            M[k, j] = binom(1, x)
    return M


def condbandfrombool(N, x, ampl, sigma):
    M0 = boolmatrix(N, x)
    M = np.zeros((N, N))
    t = 0.0
    s = int(sigma // 1)
    for l in range(-s, s):
        l1 = int(sqrt(sigma**2 - l**2) // 1)
        for m in range(-l1, l1):
            t += 1
            for k in range(N):
                for j in range(N):
                    if k + l in range(N) and j + m in range(N):
                        M[k, j] += M0[k + l, j + m]
                    else:
                        M[k, j] += x
    return M * (ampl / t)


def potential(l, sigma, ampl, x, name):
    """
    creates a potential from alloy disorder in a .dat file
    :param l: size of the box in nm
    :param a: crystal mesh parameter in nm
    :param sigma: smearing length
    :param ampl: energy difference in conduction band between the two atomic crystals in meV
    :param x: alloy proportion in percent
    :name: name of the file
    """
    a = 0.56
    N = int(l // a)
    M = condbandfrombool(N, x / 100, ampl, sigma)
    nname = (
        name
        + "l"
        + str(l)
        + "sig"
        + str(int(10 * sigma))
        + "E"
        + str(ampl)
        + "x"
        + str(x)
        + ".dat"
    )
    with open(nname, "wb") as f:
        f.write(
            b"      "
            + str(N).encode("ascii")
            + b"    "
            + str(N).encode("ascii")
            + b"\n"
        )
        for i in range(N):
            f.write(str(i / N - 1 / 2).encode("ascii") + b"    ")
        f.write(b"\n")
        for i in range(N):
            f.write(str(i / N - 1 / 2).encode("ascii") + b"    ")
        f.write(b"\n")
        for i in range(N):
            for j in range(N):
                f.write(b"" + str(M[i, j]).encode("ascii") + b"     ")
            f.write(b"\n")


potential(10, 2.2, 50, 15, "v4")

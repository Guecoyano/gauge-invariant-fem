import os, socket
import numpy

from .common_dev import *


def GetFreeFEMcmd():
    Hostname = socket.gethostname()
    FreeFEMcmd = "FreeFem++ "
    Initcmd = " "
    if Hostname == "gpuschwarz":
        FreeFEMcmd = "/usr/local/freefem/freefem++-3.22/bin/FreeFem++ "
        Initcmd = "export LD_LIBRARY_PATH=/usr/local/freefem/freefem++-3.22/lib/ff++/3.22/lib; "
    if Hostname == "hercule":
        FreeFEMcmd = "/usr/local/FreeFEM++/3.31/bin/FreeFem++ "
        Initcmd = (
            "export LD_LIBRARY_PATH=/usr/local/FreeFEM++/3.31/lib/ff++/3.32-1/lib; "
        )
    if Hostname == "gpucreos1":
        FreeFEMcmd = "/usr/local/FREEFEM/freefem++-3.31-3/bin/FreeFem++ "
        Initcmd = "export LD_LIBRARY_PATH=/usr/local/FREEFEM/freefem++-3.31-3/lib/ff++/3.31-3/lib; "
    return FreeFEMcmd, Initcmd


def GetFreeFEMBenchFEM():
    Hostname = getComputerName()
    Username = getUserName()
    if Hostname == "hercule":
        return "/home/cuvelier/Travail/Recherch/FreeFEM/benchsFEM"
    if Hostname == "gpucreos1":
        if Username == "cuvelier":
            return "/home/cuvelier/FreeFEM/benchsFEM"
        if Username == "scarella":
            return "/home/scarella/MoveFEM/benchsFEM"
    print(
        "Function FreeFEM::GetFreeFEMBenchFEM : not yet configured for user '%s' on computer '%s'"
        % (Username, Hostname)
    )
    assert 0 == 1  #


def RunFreeFEM(Name, d, N):
    FreeFEMcmd, Initcmd = GetFreeFEMcmd()
    DirWork = os.path.dirname(Name)
    File = os.path.basename(Name)
    os.system(
        Initcmd
        + "cd "
        + DirWork
        + ";echo "
        + str(N)
        + " | "
        + FreeFEMcmd
        + File
        + ".edp"
    )
    if d == 2:
        FFmesh = Name + "-" + str(N) + ".msh"
    if d == 3:
        FFmesh = Name + "-" + str(N) + ".mesh"
    FFsol = Name + "-" + str(N) + ".txt"
    return FFmesh, FFsol


def RunFreeFEMV2(Name, DirWork, d, N):
    FreeFEMcmd, Initcmd = GetFreeFEMcmd()
    # DirWork=os.path.dirname(Name)
    # print(Initcmd+ "cd "+DirWork+";echo "+str(N)+" | " + FreeFEMcmd +Name+".edp")
    os.system(
        Initcmd
        + "cd "
        + DirWork
        + ";echo "
        + str(N)
        + " | "
        + FreeFEMcmd
        + Name
        + ".edp"
    )
    if d == 2:
        FFmesh = Name + "-" + str(N) + ".msh"
    if d == 3:
        FFmesh = Name + "-" + str(N) + ".mesh"
    FFsol = Name + "-" + str(N) + ".txt"
    return FFmesh, FFsol


def RunFreeFEMV3(FFScriptFile, N):
    FreeFEMcmd, Initcmd = GetFreeFEMcmd()
    # DirWork=os.path.dirname(Name)
    # print(Initcmd+ "cd "+DirWork+";echo "+str(N)+" | " + FreeFEMcmd +Name+".edp")
    command = Initcmd + " echo " + str(N) + " | " + FreeFEMcmd + " -cd " + FFScriptFile

    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    p_status = proc.wait()
    if p_status:
        print(out.decode("utf-8"))


def LoadFreeFEMsol(FileName, m, d):
    fp = open(FileName, "rt")
    mm = int(fp.readline())
    assert mm == m
    n = int(fp.readline())
    u = numpy.zeros((m, n))
    u[0] = numpy.fromfile(fp, sep=" ", dtype=numpy.float64, count=n)
    for i in range(1, m):
        nn = int(fp.readline())
        assert nn == n
        u[i] = numpy.fromfile(fp, sep=" ", dtype=numpy.float64, count=n)
    T = float(fp.readline())
    if m == 1:
        u = u[0]
    return u, T


def LoadFreeFEMSolV2(FileName, m):
    fp = open(FileName, "rt")
    mm = int(fp.readline())
    assert mm == m
    n = int(fp.readline())
    U = numpy.zeros((n, 1))
    U = numpy.fromfile(fp, sep=" ", dtype=numpy.float64, count=n)
    ndof = n / m
    u = numpy.zeros((m, ndof))
    I = numpy.arange(0, n, m)
    for i in range(0, m):
        u[i, :] = U[I]
        I = I + 1
    T = float(fp.readline())
    if m == 1:
        u = u[0]
    return u, T

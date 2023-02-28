import numpy as np


def vtkWriteValues(FileName, Th, U, names):
    fid = open(FileName, "w")
    vtkWriteMesh(Th, fid)
    if U != None:
        fid.write("POINT_DATA %d\n" % Th.nq)
        assert isinstance(U, list) & isinstance(names, list)
        assert len(U) == len(names)
        n = len(U)
        for i in range(n):
            assert U[i].shape[0] == Th.nq
            vtkWrite(fid, U[i], names[i])
    fid.close()


def vtkWriteMesh(Th, fid):
    fid.write("# vtk DataFile Version 2.0\n")
    fid.write("Generated Volume Mesh\n")
    fid.write("ASCII\n")
    fid.write("DATASET UNSTRUCTURED_GRID\n")
    fid.write("POINTS %d float\n" % Th.nq)
    if Th.d == 3:
        for i in range(Th.nq):
            fid.write("%g %g %g\n" % (Th.q[i, 0], Th.q[i, 1], Th.q[i, 2]))
        fid.write("CELLS %d %d\n" % (Th.nme, 5 * Th.nme))
        for k in range(Th.nme):
            fid.write(
                "4 %d %d %d %d\n" % (Th.me[k, 0], Th.me[k, 1], Th.me[k, 2], Th.me[k, 3])
            )
        fid.write("CELL_TYPES %d\n" % Th.nme)
        for k in range(Th.nme):
            fid.write("10\n")  # 10 for tetrahedra
    else:
        if Th.q.shape[1] == 2:
            for i in range(Th.nq):
                fid.write("%g %g 0\n" % (Th.q[i, 0], Th.q[i, 1]))
        else:
            for i in range(Th.nq):
                fid.write("%g %g %g\n" % (Th.q[i, 0], Th.q[i, 1], Th.q[i, 2]))
        fid.write("CELLS %d %d\n" % (Th.nme, 4 * Th.nme))
        for k in range(Th.nme):
            fid.write("3 %d %d %d \n" % (Th.me[k, 0], Th.me[k, 1], Th.me[k, 2]))
        fid.write("CELL_TYPES %d\n" % Th.nme)
        for k in range(Th.nme):
            fid.write("5\n")  # 5 for triangles
    fid.write("\n")


def vtkWrite(fid, U, name):
    if not isinstance(U, np.ndarray):
        print("vtkWrite: unable to write %s object in file.\n" % type(U))
        print("          Must be a  <class " "numpy.ndarray" "> object\n")
        return
    if len(U.shape) == 1:
        d = 1
    else:
        d = U.shape[1]
    nq = U.shape[0]
    assert d <= 3
    if d == 1:
        fid.write("SCALARS %s float 1\n" % name)
        fid.write("LOOKUP_TABLE table_%s\n" % name)
        for i in range(nq):
            fid.write("%g\n" % U[i])
    else:
        X = np.zeros((nq, 3))
        for i in range(d):
            X[:, i] = U[:, i]
        fid.write("VECTORS vectors float\n")
        for i in range(nq):
            fid.write("%g %g %g\n" % (X[i, 0], X[i, 1], X[i, 2]))
    fid.write("\n")

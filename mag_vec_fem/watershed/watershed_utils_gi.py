"""Watershed_utils.py.

Douglas N. Arnold, 2015-10-17
Modified by Perceval Desforges 
Modified by Alioune Seye, 2023-02

This module defines a watershed(I, M), a marker-based watershed transform
and label_local_minima(I) to label the local minima of an array in order
of minimum.  As utilities it also defines a function for returning
the index pairs neighboring a given index pair, and a class
FIFOPriorityQueue.

Class defined:

    FIFOPriorityQueue:
        a queue in which items are returned in order of priority (low to high)
        with FIFO used to break ties

Functions defined:

    neighbors:
        find the index pairs neighboring a given pair of indices, optionally
        taking into account periodicity

    label_local_minima:
        given a real 2D array, return another of the same size everywhere set
        equal to 0., except for the strict lower minima set equal to 1.,
        2., ... in increasing order of their value

    watershed:
        from image and markers create image of labeled watershed basins and
        watershed basin boundaries, optionally taking into account periodicity

Modifications consisted in allowing to perform watershed on mesh structured
data. Thus Vertex class andconnectivity, vertices_from_data functions were
added to be used in the 'irr' mode.
"""
import numpy as np
from queue import PriorityQueue
from collections import Counter
import scipy.ndimage.morphology as morphology
import scipy.ndimage.filters as filters


class FIFOPriorityQueue(PriorityQueue):
    """
    FIFOPriorityQueue: this is a FIFO priority queue, meaning that items are
    added to the queue with a real number priority, and when items are gotten
    from the queue, the lower priority items go first, with FIFO used to break 
    ties.

        main calls:
        pq.put(item, prio)      # put item on queue with given priority
        pq.get()                # get item from queue
        pq.getall()             # returns priority, counter, and item
        pq.empty()              # returns True if empty
    """

    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0

    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1

    def getall(self, *args, **kwargs):
        prio, ctr, item = PriorityQueue.get(self, *args, **kwargs)
        return prio, ctr, item

    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, *args, **kwargs)
        return item


class Vertex:
    """The vertex class is for mesh structured data. 
    Parameters are:
    n: vertex index
    a,b: x,y coordinates of the vertex
    c: value of the data at the vertex
    vneighbors is the list of indices of neighbouring vertices.
    """
    def __init__(self, n, a, b, c):
        self.number = n
        self.x = a
        self.y = b
        self.z = c
        self.vneighbors = []

    def add_neighbor(self, n):
        (self.vneighbors).append(n)


def connectivity(Th):
    """Build connectivity array from mesh Th.

    Th.q is the list of vertices coordinates, Th.me is the list of mesh
    elements.
    Th.me[b,k] is the index of vertex b in the k-th mesh element.
    """
    conn = Th.nq * [[]]
    for me in Th.me:
        if not me[0] in conn[me[1]]:
            conn[me[0]] = conn[me[0]] + [me[1]]
            conn[me[1]] = conn[me[1]] + [me[0]]
        if not me[0] in conn[me[2]]:
            conn[me[0]] = conn[me[0]] + [me[2]]
            conn[me[2]] = conn[me[2]] + [me[0]]
        if not me[2] in conn[me[1]]:
            conn[me[2]] = conn[me[2]] + [me[1]]
            conn[me[1]] = conn[me[1]] + [me[2]]
    return conn


def vertices_data_from_mesh(Th, values=None):
    """This function returns mesh structured data compatible with watershed
    'irr' mode.

    It takes a mesh with mesh structure of gauge-invariant-fem, and
    optionnally a values array os shape (Th.nq) and returns a list of vertex
    class objects of length Th.nq containing the mesh structure and the data.
    """
    if values is None:
        val = np.zeros(Th.nq)
    else:
        val = values
    Vertices = []
    q = Th.q
    for i in range(Th.nq):
        v = Vertex(i, q[i, 0], q[i, 1], val[i])
        Vertices.append(v)
    for me in Th.me:
        if not me[0] in Vertices[me[1]].vneighbors:
            (Vertices[me[0]].vneighbors).append(me[1])
            (Vertices[me[1]].vneighbors).append(me[0])
        if not me[0] in Vertices[me[2]].vneighbors:
            (Vertices[me[0]].vneighbors).append(me[2])
            (Vertices[me[2]].vneighbors).append(me[0])
        if not me[2] in Vertices[me[1]].vneighbors:
            (Vertices[me[2]].vneighbors).append(me[1])
            (Vertices[me[1]].vneighbors).append(me[2])
    return Vertices


def label_local_minima(I, mode=None):
    """
    Takes a real 2D array and returns another of the same size which is equal
    to zero except at the strict local minima, which are set to 1., 2., ...
    with 1. labeling the global minimum, 2. labeling the next lowest minimum,
    etc.  For this purpose a strict local minimum is a point with a strictly
    lower value than its neighbors, so if two contiguous points take on
    exactly the same minimum value, that won't be detected as a minimum.
    This may not be what is wanted.

    The parameter mode can be set to 'wrap' to handle periodic functions.  In
    that case we interpret I[i, j] = I[i + k n1, j + l n2] where (n1, n2) = I
    shape and k, l are arbitrary integers.
    
    Mode irr:

    take an array of vertex class objects and returns an index list with value
    zero if the vertex is not a local minimum and the ordered index of their
    local minimum elseways.
    """

    if mode == "irr":
        nv = len(I)
        J = np.zeros_like(I)

        lnmin = []
        lzmin = []

        for nn in range(nv):
            if I[nn].z < np.min([I[nnn].z for nnn in I[nn].vneighbors]):
                J[nn] = 1
                lnmin.append(int(nn))
                lzmin.append(I[nn].z)

        order = np.argsort(lzmin)

        lnmin_copy = list(lnmin)
        lzmin_copy = list(lzmin)

        for nn in range(len(lnmin_copy)):
            lnmin[nn] = lnmin_copy[order[nn]]
            lzmin[nn] = lzmin_copy[order[nn]]

        for nn in range(len(lnmin)):
            J[lnmin[nn]] = nn + 1

    else:
        # create an array with value 0. except for value 1. where I has a strict local minima
        J = np.zeros_like(I)
        n1, n2 = I.shape
        for i in range(n1):
            for j in range(n2):
                if I[i, j] < np.min(
                    [
                        I[ii, jj]
                        for (ii, jj) in neighbors(i, j, n1, n2, mode=mode)
                    ]
                ):
                    J[i, j] = 1.0
        mini, minj = np.where(J)
        minvals = I[mini, minj]
        order = np.argsort(minvals)
        mini = mini[order]
        minj = minj[order]
        for k in range(len(mini)):
            J[mini[k], minj[k]] = k + 1

    return J


def neighbors(i, j, n1, n2, mode=None):
    """
    Given the pair i, j of indices, with 0 <= i < n1, 0 <= j < n2, returns a
    numpy array with 2 columns and one row for each neighbor, the contents
    being the indices of the neighbors.

    There are 8 neighbors except for pixels on the boundary (i = 0 or n1-1 or
    j = 0 or n2-1).

    If mode is set to wrap, periodic wrapping is used and i, j need not belong
    to [0, n1) and [0, n2), but the eight neighbors computed will all have
    indices belonging to these intervals.
    """
    if mode == "wrap":
        i = i % n1
        im1 = (i - 1) % n1
        ip1 = (i + 1) % n1
        j = j % n2
        jm1 = (j - 1) % n2
        jp1 = (j + 1) % n2
        l = np.array(
            [
                [im1, jm1],
                [im1, j],
                [im1, jp1],
                [i, jm1],
                [i, jp1],
                [ip1, jm1],
                [ip1, j],
                [ip1, jp1],
            ]
        )
    else:
        l = np.array(
            [
                [i - 1, j - 1],
                [i - 1, j],
                [i - 1, j + 1],
                [i, j - 1],
                [i, j + 1],
                [i + 1, j - 1],
                [i + 1, j],
                [i + 1, j + 1],
            ]
        )
    return l[
        np.all(
            (l[:, 0] > -1, l[:, 0] < n1, l[:, 1] > -1, l[:, 1] < n2), axis=0
        ),
        :,
    ]


def watershed(I, M, mode=None):
    """
    This function defines a watershed transform based on markers and flooding.

    This should duplicate the Matlab imaging toolbox watershed as long as the
    markers are the local minima. It is also a translation of the C++ code
    provided at

    http://www.insight-journal.org/browse/publication/92

    and described in

    Richard Beare and Gaetan Lehmann: The watershed transform in ITK -
    discussion and new developments, The Insight Journal, 2006

    The C++ code for the MEX file from Matlab R-2009b can be found at
    https://lost-contact.mit.edu/afs/cs.stanford.edu/package/matlab-r2009b/
    matlab/r2009b/toolbox/images/images/private/mexsrc/watershed_meyer/
    watershed_meyer.cpp

    Given a 2D real array I of values on the vertices and another M containing
    markers, returns a third such array which shows the watersheds.  Each
    element on the output array is either 0 for points on the watershed
    boundaries, or 1, 2, 3, ... for points in the watershed of marker 1, 2,
    3, ...

    I have added the mode option.  Set mode='wrap' for periodic wrapping.

    Mode 'irr' allows for mesh structured data.
    """
    if mode == "irr":
        S = np.zeros_like(I, dtype=bool) # array: have we started with vertex
        O = M.copy()       #watershed matrix
        pq = FIFOPriorityQueue()   #waiting list for vertices to be assigned

        # Make list of local minima
        minn = []
        for nn in range(len(M)):
            if M[nn] != 0.0:
                minn.append(nn)

        # Initialize priority queue with first neighbors of minima, priority is
        # given by the value at the point.
        for nn in range(len(minn)):
            n = minn[nn]
            S[n] = True
            nbrs = I[n].vneighbors
            for l in range(len(nbrs)):
                nnn = nbrs[l]
                if (not S[nnn]) and (M[nnn] == 0.0):
                    S[nnn] = True
                    pq.put(nnn, I[nnn].z)

        # Examine each vertex in the priority queue to mark it
        # Add its neighbors to the priority queue if they aren't in it 
        while not pq.empty():
            prio, _, n = pq.getall()
            label = 0 # Up-to date Potential marker for n
            watershed = False # Will tell when we consider n marked
            nbrs = I[n].vneighbors
            for l in range(len(nbrs)):
                nn = nbrs[l]
                if (O[nn] != 0) and (not watershed):
                    if (label != 0) and (O[nn] != label):
                        watershed = True # n is marked with zero
                    else:
                        label = O[nn]
            if not watershed:
                O[n] = label # n is marked as the only neighboring region label
                # Neighbors are added to pq
                for l in range(len(nbrs)):
                    nn = nbrs[l]
                    if not S[nn]:
                        S[nn] = True
                        pq.put(nn, max(I[nn].z, prio)) 

    else:
        n1, n2 = M.shape
        S = np.zeros_like(I, dtype=bool)
        O = M.copy()
        pq = FIFOPriorityQueue()

        mini, minj = np.where(M != 0.0)
        for k in range(len(mini)):
            i = mini[k]
            j = minj[k]
            S[i, j] = True
            nbrs = neighbors(i, j, n1, n2, mode=mode)
            for l in range(len(nbrs)):
                ii, jj = nbrs[l]
                if (not S[ii, jj]) and (M[ii, jj] == 0.0):
                    S[ii, jj] = True
                    pq.put((ii, jj), I[ii, jj])

        while not pq.empty():
            prio, _, (i, j) = pq.getall()
            label = 0
            watershed = False
            nbrs = neighbors(i, j, n1, n2, mode=mode)
            for l in range(len(nbrs)):
                ii, jj = nbrs[l]
                if (O[ii, jj] != 0) and (not watershed):
                    if (label != 0) and (O[ii, jj] != label):
                        watershed = True
                    else:
                        label = O[ii, jj]
            if not watershed:
                O[i, j] = label
                for l in range(len(nbrs)):
                    ii, jj = nbrs[l]
                    if not S[ii, jj]:
                        S[ii, jj] = True
                        pq.put((ii, jj), max(I[ii, jj], prio))

    return O
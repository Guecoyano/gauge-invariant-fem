"""watershed_utils.py
Douglas N. Arnold, 2015-10-17

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


def label_local_minima(I, mode=None):
    """
    Takes a real 2D array and returns another of the same size which is equal
    to zero except at the strict local minima, which are set to 1., 2., ...
    with 1. labeling the global minimum, 2. labeling the next lowest minimum,
    etc.  For this purpose a strict local minimum is a point with a strictly
    lower value than its neighbors, so if two contiguous points take on exactly
    the same minimum value, that won't be detected as a minimum.
    This may not be what is wanted.

    The parameter mode can be set to 'wrap' to handle periodic functions.  In
    that case we interpret I[i, j] = I[i + k n1, j + l n2]
    where (n1, n2) = I.shape and k, l are arbitrary integers.
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
        # create an array with value 0. except for value 1. where I has a
        # strict local minima
        J = np.zeros_like(I)
        n1, n2 = I.shape
        for i in range(n1):
            for j in range(n2):
                if I[i, j] < np.min(
                    [I[ii, jj] for (ii, jj) in neighbors(i, j, n1, n2, mode=mode)]
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


def list_local_minima(J, mode=None):
    """
    Takes a real 2D array generated from the label_local_minima function and
    returns a list of the coordinates of the local minima
    """

    l = []

    if mode == "irr":
        nv = len(J)
        for n in range(nv):
            if J[n] != 0:
                l.append(n)

    else:
        n1, n2 = J.shape
        for i in range(n1):
            for j in range(n2):
                if J[i, j] != 0:
                    l.append([i, j])

    return l


def index(i, L):
    ii = (i + L) % L
    return ii


def boundary(I, i, j, mode=None):
    """
    Given a pont i,j and a binary array I with 0 as the values of the contour,
    finds the boundary points of the point.
    The parameter mode can be set to 'wrap' to handle periodic functions.
    """

    # l is the list of contour points

    l = []

    # Find the first points

    l_points = []
    l_used = []

    Li, Lj = I.shape

    if mode == "wrap":
        for x in range(-1, 2):
            for y in range(-1, 2):
                if I[index(i + x, Li), index(j + y, Lj)] == 0:
                    l.append([index(i + x, Li), index(j + y, Lj)])
                    l_used.append(index(i + x, Li) + Li * index(j + y, Lj))
                elif l_used.count(index(i + x, Li) + Li * index(j + y, Lj)) == 0:
                    l_points.append([index(i + x, Li), index(j + y, Lj)])
                    l_used.append(index(i + x, Li) + Li * index(j + y, Lj))

        while len(l_points) > 0:
            l_new_points = []

            for n in range(len(l_points)):
                ii, jj = l_points[n]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if I[index(ii + x, Li), index(jj + y, Lj)] == 0:
                            l.append([index(ii + x, Li), index(jj + y, Lj)])
                            l_used.append(index(ii + x, Li) + Li * index(jj + y, Lj))
                        elif (
                            l_used.count(index(ii + x, Li) + Li * index(jj + y, Lj))
                            == 0
                        ):
                            l_new_points.append([index(ii + x, Li), index(jj + y, Lj)])
                            l_used.append(index(ii + x, Li) + Li * index(jj + y, Lj))

            l_points = l_new_points

    else:
        for x in range(-1, 2):
            for y in range(-1, 2):
                ii = i + x
                jj = j + y
                if ii >= 0 and jj >= 0:
                    if I[ii, jj] == 0:
                        l.append([ii, jj])
                        l_used.append(ii + Li * jj)
                    elif l_used.count(ii + Li * jj) == 0:
                        l_points.append([ii, jj])
                        l_used.append(ii + Li * jj)

        while len(l_points) > 0:
            l_new_points = []

            for n in range(len(l_points)):
                ii, jj = l_points[n]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        iii = ii + x
                        jjj = jj + y
                        if iii >= 0 and jjj >= 0:
                            if I[iii, jjj] == 0:
                                l.append([iii, jjj])
                                l_used.append(iii + Li * jjj)
                            elif l_used.count(iii + Li * jjj) == 0:
                                l_new_points.append([iii, jjj])
                                l_used.append(iii + Li * jjj)

        l_points = l_new_points

    return l


def points_in_common(l1, l2):
    # l1 and l2 are the list of points to be compared. l is the list of points in common

    l = []

    for n in range(len(l1)):
        x1 = (l1[n])[0]
        y1 = (l1[n])[1]

        for nn in range(len(l2)):
            x2 = (l2[nn])[0]
            y2 = (l2[nn])[1]

            if x1 == x2 and y1 == y2:
                l.append([x1, y1])

    return l


def region(I, i, j, mode=None):
    """
    Given a pont i,j and a binary array I with 0 as the values of the contour
    finds the region points of the point.
    Mode is always wrap here, so periodic wraping is always used.
    """

    # l is the list of contour points

    l = []

    # l_regions is the list of region points

    l_points = []
    l_used = []
    l_regions = []

    Li, Lj = I.shape

    if mode == "wrap":
        for x in range(-1, 2):
            for y in range(-1, 2):
                if I[index(i + x, Li), index(j + y, Lj)] == 0:
                    l.append([index(i + x, Li), index(j + y, Lj)])
                    l_regions.append([index(i + x, Li), index(j + y, Lj)])
                    l_used.append(index(i + x, Li) + Li * index(j + y, Lj))
                elif l_used.count(index(i + x, Li) + Li * index(j + y, Lj)) == 0:
                    l_points.append([index(i + x, Li), index(j + y, Lj)])
                    l_used.append(index(i + x, Li) + Li * index(j + y, Lj))
                    l_regions.append([index(i + x, Li), index(j + y, Lj)])

        while len(l_points) > 0:
            l_new_points = []

            for n in range(len(l_points)):
                ii, jj = l_points[n]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if I[index(ii + x, Li), index(jj + y, Lj)] == 0:
                            l.append([index(ii + x, Li), index(jj + y, Lj)])
                            l_used.append(index(ii + x, Li) + Li * index(jj + y, Lj))
                            l_regions.append([index(ii + x, Li), index(jj + y, Lj)])
                        elif (
                            l_used.count(index(ii + x, Li) + Li * index(jj + y, Lj))
                            == 0
                        ):
                            l_new_points.append([index(ii + x, Li), index(jj + y, Lj)])
                            l_used.append(index(ii + x, Li) + Li * index(jj + y, Lj))
                            l_regions.append([index(ii + x, Li), index(jj + y, Lj)])

            l_points = l_new_points

    else:
        for x in range(-1, 2):
            for y in range(-1, 2):
                ii = i + x
                jj = j + y
                if ii >= 0 and jj >= 0:
                    if I[ii, jj] == 0:
                        l.append([ii, jj])
                        l_regions.append([ii, jj])
                        l_used.append(ii + Li * jj)
                    elif l_used.count(ii + Li * jj) == 0:
                        l_points.append([ii, jj])
                        l_used.append(ii + Li * jj)
                        l_regions.append([ii, jj])

        while len(l_points) > 0:
            l_new_points = []

            for n in range(len(l_points)):
                ii, jj = l_points[n]
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        iii = i + x
                        jjj = j + y
                        if iii >= 0 and jjj >= 0:
                            if I[iii, jjj] == 0:
                                l.append([iii, jjj])
                                l_used.append(iii + Li * jjj)
                                l_regions.append([iii, jjj])
                            elif l_used.count(iii + Li * jjj) == 0:
                                l_new_points.append([iii, jjj])
                                l_used.append(iii + Li * jjj)
                                l_regions.append([iii, jjj])

            l_points = l_new_points

    return l_regions


def neighbors(i, j, n1, n2, mode=None):
    """
    Given the pair i, j of indices, with 0 <= i < n1, 0 <= j < n2, returns a
    numpy array with 2 columns and one row for each neighbor, the contents
    being the indices of the neighbors.  There are 8 neighbors except for
    pixels on the boundary (i = 0 or n1-1 or j = 0 or n2-1).

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
        np.all((l[:, 0] > -1, l[:, 0] < n1, l[:, 1] > -1, l[:, 1] < n2), axis=0),
        :,
    ]


def watershed(I, M, mode=None):
    """
    This function defines a watershed transform based on markers and flooding.
    This should duplicate the Matlab imaging toolbox watershed as long as the
    markers are the local minima.
    It is also a translation of the C++ code provided at

    http://www.insight-journal.org/browse/publication/92

    and described in

    Richard Beare and Gaetan Lehmann: The watershed transform in ITK -
    discussion and new developments, The Insight Journal, 2006

    The C++ code for the MEX file from Matlab R-2009b can be found at
    https://lost-contact.mit.edu/afs/cs.stanford.edu/package/matlab-r2009b/
    matlab/r2009b/toolbox/images/images/private/mexsrc/watershed_meyer/
    watershed_meyer.cpp

    Given a 2D real array I and another M where M contains markers, returns a
    third such array which shows the watersheds.  Each element on the output
    array is either 0 for points on the watershed boundaries, or 1, 2, 3, ...
    for points in the watershed of marker 1, 2, 3, ...

    I have added the mode option.  Set mode='wrap' for periodic wrapping.
    """
    if mode == "irr":
        nv = len(M)
        S = np.zeros_like(I, dtype=bool)
        O = M.copy()
        pq = FIFOPriorityQueue()

        minn = []

        for nn in range(len(M)):
            if M[nn] != 0.0:
                minn.append(nn)

        for nn in range(len(minn)):
            n = minn[nn]
            S[n] = True
            nbrs = I[n].vneighbors
            for l in range(len(nbrs)):
                nnn = nbrs[l]
                if (not S[nnn]) and (M[nnn] == 0.0):
                    S[nnn] = True
                    pq.put(nnn, I[nnn].z)

        while not pq.empty():
            prio, _, n = pq.getall()
            label = 0
            watershed = False
            nbrs = I[n].vneighbors
            for l in range(len(nbrs)):
                nn = nbrs[l]
                if (O[nn] != 0) and (not watershed):
                    if (label != 0) and (O[nn] != label):
                        watershed = True
                    else:
                        label = O[nn]
            if not watershed:
                O[n] = label
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


class Vertex:
    def __init__(self, n, a, b, c):
        self.number = n
        self.x = a
        self.y = b
        self.z = c
        self.vneighbors = []

    def add_neighbor(self, n):
        (self.vneighbors).append(n)

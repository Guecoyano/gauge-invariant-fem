"""
Watershed_merge_utils_gi.py.

by Alioune Seye, 2023-02 (inspired by Perceval Desforges' watershed_merge_utils.py)

This module allows to select relevant regions from a watershed(I, M), based on height of barriers between the regions of the watershed.

This module was designed to be used with data produced with watershed_utils.py with 'irr' mode (mesh structured data).
"""
import numpy as np
from matplotlib.pyplot import *


class Region:
    """
    Region class object is a region of the watershed, to be potentially merged with another one.

    Data contained:

        name (index of the min of the region)
        index of local minima of the region (from the original watershed)
        set of indices of interior vertices
        set of indices of boundary vertices

    Methods:

        add_interior:
            Extends the set of interior indices with another set

        add_boundary:
            Extends the boundary indices with another set

        boundary_to_interior:
            Transfers indices from the boundary set to the interior set.

        removed:
            indicates that region was removed by merging algo (and where to)
    """

    def __init__(self, min_index, min_value):
        self.minima = {min_index}
        self.interior = set()
        self.boundary = set()
        self.removed = False
        self.mergedwith = None
        self.min_value = min_value

    def add_interior(self, set):
        (self.interior).update(set)

    def add_boundary(self, set):
        (self.boundary).update(set)

    def boundary_to_interior(self, set):
        self.boundary = self.boundary - set
        (self.interior).update(set)

    def add_minima(self, set):
        (self.minima).update(set)

    def remove(self, mergedwith=None):
        self.removed = True
        if mergedwith is not None:
            self.mergedwith = mergedwith


class NeighborRegions:
    """
    Data for two neighboring regions

    Attributes are:

        #(name1,name2)

        indices: set of boundary indices

        min_index: index of the min of the boundary between the regions

        min_value: value at min_index

    """

    def __init__(self, name1, name2, set_of_indices, min_index, min_value):
        if name1 <= name2:
            self.regions = (name1, name2)
        else:
            self.regions = (name2, name1)
        self.indices = set_of_indices
        self.min_index = min_index
        self.min_value = min_value

    def add_index(
        self, index, value
    ):  # to be used for initialization from original watershed only
        (self.indices).add(index)
        if value < self.min_value:
            self.min_index = index
            self.min_value = value

    def merge(self, nbreg):
        """add neighbouring data of neighbouring regions nbreg ex: self is neighbouring regions (2,5), nbreg is (2,8)"""
        (self.indices).update(nbreg.indices)
        if nbreg.min_value < self.min_value:
            self.min_index = np.copy(nbreg.min_index)
            self.min_value = np.copy(nbreg.min_value)


class Regions:
    """
    Regions objects will contain all relevant information for the watershed merging algorithm. Region info will mainly contain the indices, and should be used together with the data info.

    Regions have attributes:

        regions: the list of regions which are region class objects.

        neighbors_couples: list of couples (name1,name2) of regions

        neighbors: a list of NeighborRegions objects

        irrelevant_min: list of minima discarded from Watershed

        boundary: a list giving a set for each vertex for neighbouring regions
                  it is the same info as in each region but indexed by vertices
    """

    def __init__(self, min_indices=[], min_values=[]):
        self.regions = []
        for m in range(len(min_indices)):
            min_index, min_value = min_indices[m], min_values[m]
            (self.regions).append(Region(min_index, min_value))
        self.neighbors_couples = []
        self.neighbors = []
        self.irrelevant_min = []
        self.global_boundary = []

    def add_neighbors(self, NR):
        (self.neighbors).append(NR)

    def add_irrelevant_min(self, list):
        (self.irrelevant_min).extend(list)

    def remove(self, name):
        (self.regions[name]).remove()
        self.irrelevant_min.append(name)

    def merge(self, mergedto, removed):  # removed and mergedto are the regions'number
        print("merging region", removed, "into", mergedto)
        # transfer interior
        (self.regions)[removed].remove(mergedto)
        interior = self.regions[removed].interior
        for n in interior:
            self.global_boundary[n] = {mergedto}
        (self.regions)[mergedto].add_interior(interior)

        # transfer removed boundary completely to mergedto
        boundary_removed = self.regions[removed].boundary
        (self.regions[mergedto].boundary).update(boundary_removed)

        # update neighbouring regions for boundary of removed
        new_interior = set()
        for b in boundary_removed:
            self.global_boundary[b].remove(removed)
            self.global_boundary[b].add(mergedto)
            # clean boundary
            if self.global_boundary[b] == {mergedto}:
                new_interior.add(b)
        (self.regions[mergedto]).boundary_to_interior(new_interior)

        # transfer neighbors
        nbr_index = (self.neighbors_couples).index((mergedto, removed))
        del self.neighbors_couples[nbr_index]
        del self.neighbors[nbr_index]

        for couple in self.neighbors_couples:
            delete = []
            if couple[0] == removed:
                i = (self.neighbors_couples).index(couple)
                new_couple = (min(couple[1], mergedto), max(couple[1], mergedto))
                if new_couple in self.neighbors_couples:
                    i_new = (self.neighbors_couples).index(new_couple)
                    self.neighbors[i_new].merge(self.neighbors[i])
                    delete.append(couple)
                else:
                    (self.neighbors[i]).regions = new_couple
                    (self.neighbors_couples[i]) = new_couple

            if couple[1] == removed:
                i = (self.neighbors_couples).index(couple)
                new_couple = (min(couple[0], mergedto), max(couple[0], mergedto))
                if new_couple in self.neighbors_couples:
                    i_new = (self.neighbors_couples).index(new_couple)
                    self.neighbors[i_new].merge(self.neighbors[i])
                    delete.append(couple)
                else:
                    (self.neighbors[i]).regions = new_couple
                    (self.neighbors_couples[i]) = new_couple

        for couple in delete:
            i = (self.neighbors_couples).index(couple)
            del self.neighbors_couples[i]
            del self.neighbors[i]


def min_indices_from_loc_min(J):
    """returns ordered list of indices of local minima"""
    m = []
    for n in range(len(J)):
        if J[n] != 0:
            m.append((J[n], n))
    mm = sorted(m)
    mmm = [i[1] for i in mm]
    return mmm


def init_regions(I, J, W):
    """
    Builds up the initial Regions data from the mesh structured values I, the labeled minima J and the watershed W

    The algorithm visits once each vertex n and its neighbors if n is boundary.
    """

    n_reg = np.max(W)
    min_indices = min_indices_from_loc_min(J)
    min_values = [I[m].z for m in min_indices]
    n_ver = len(W)
    r = Regions(min_indices, min_values)

    for n in range(n_ver):
        # case n is an interior vertex of reg w[n]-1 (regions are indexed from zero):
        # add it to interior of this region
        # only add this region as neighboring the vertex
        if W[n] != 0:
            (r.regions[W[n] - 1]).add_interior({n})
            (r.global_boundary).append({W[n] - 1})
        # case n is a boundary vertex of reg w[j]-1 where j is a neighbor:
        # add the vertex to boundary of all neighboring regions
        # add all neighboring regions to boundary[n]
        if W[n] == 0:
            neighbouring = set()
            for j in I[n].vneighbors:
                if W[j] != 0:
                    (r.regions[W[j] - 1]).add_boundary({n})
                    neighbouring.add(W[j] - 1)
            (r.global_boundary).append(neighbouring)
            # add neighbouring regions as neighbors
            val = I[n].z
            k = len(neighbouring)  # number of regions touching vertex n
            neighlist = list(neighbouring)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    r1 = min(neighlist[i], neighlist[j])
                    r2 = max(neighlist[i], neighlist[j])
                    if (r1, r2) not in r.neighbors_couples:
                        (r.neighbors_couples).append((r1, r2))
                        new_neighbors = NeighborRegions(r1, r2, {n}, n, I[n].z)
                        r.add_neighbors(new_neighbors)
                    else:
                        pos = (r.neighbors_couples).index((r1, r2))
                        (r.neighbors[pos]).add_index(n, I[n].z)
    return r


def merge_algorithm(
    regions, barrier_condition=(lambda min, barr: barr > 1.5 * min), inclusive=True
):
    """
    Returns a Regions object obtained from merging non independent regions.

    Proceeds with initial barriers in order of height (so it might visit uselessly some of them, but willonly visit once).  Merges regions if barrier_condition isn't satisfied.

    The name of the region is retrieved at each step to get the current min.

    If inclusive is True regions that have asymetrically permeable barriers are merged into their measiest accessible neighbour. If set to False, they will have a spurious_tag and be added to irrelevant_minima.
    """

    # Create list containing all mountain passes (vertices' index) ordered by height
    bv = []
    dtype = [("min_index", int), ("min_value", float)]
    for nr in regions.neighbors:
        if (nr.min_index, nr.min_value) not in bv:
            bv.append((nr.min_index, nr.min_value))
    barrier_vertices = np.array(bv, dtype=dtype)
    np.ndarray.sort(barrier_vertices, order="min_value")

    for v in barrier_vertices:
        full_check = False
        # print('v is ', v)################################
        while not full_check:
            # retrieve neighbor regions
            flag_merge = False
            nbr = list(regions.global_boundary[v[0]])
            # print('nbr:',nbr)###############################################
            for i in range(len(nbr) - 1):
                for j in range(i + 1, len(nbr)):
                    r1 = min(nbr[i], nbr[j])
                    r2 = max(nbr[i], nbr[j])
                    # min of r1 is smaller than min of r2
                    min1 = (regions.regions[r1]).min_value
                    min2 = (regions.regions[r2]).min_value
                    if (not barrier_condition(min1, v[1])) and (
                        not barrier_condition(min2, v[1])
                    ):
                        regions.merge(r1, r2)
                        flag_merge = True
                        break
                    elif barrier_condition(min1, v[1]) and (
                        not barrier_condition(min2, v[1])
                    ):
                        if inclusive:
                            regions.merge(r1, r2)
                            flag_merge = True
                            break
                        else:
                            regions.remove(r2)
                if flag_merge:
                    break
            if not flag_merge:
                full_check = True

    for i_r in range(len(regions.regions)):
        reg = regions.regions[i_r]
        if reg.removed and reg.mergedwith is None:
            regions.add_irrelevant_min((i, reg.minima))
    return regions

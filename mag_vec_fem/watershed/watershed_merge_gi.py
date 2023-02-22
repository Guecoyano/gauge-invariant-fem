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
        indices of interior vertices
        indices of boundary indices

    Methods:

        add_interior:
            Extends the interior indices list with another list

        add_boundary:
            Extends the boundary indices list with another list

        boundary_to_interior:
            Transfers indices from the boundary list to the interior list.

        removed:
            indicates that region was removed by merging algo (and where to)
    """

    def __init__(self, min_index):
        self.minima = {min_index}
        self.interior = set()
        self.boundary = set()
        self.removed = False
        self.mergedwith = None

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

        indices: list of boundary indices

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
        self, index, I
    ):  # to be used for initialization from original watershed only
        (self.indices).add(index)
        if I[index].z < I[self.min_index].z:
            self.min_index = index

    def merge(self, nbreg):
        (self.indices).update(nbreg.indices)
        n = np.copy(nbreg.min_index)
        if nbreg.min_value < self.min_value:
            self.min_index = nbreg.min_index
            self.min_value = nbreg.min_value


class Regions:
    """
    Regions objects will contain all relevant information for the watershed merging algorithm. Region info will mainly contain the indices, and should be used together with the data info.

    Regions have attributes:

        regions: the list of regions which are region class objects.

        neighbors_couples: list of couples (name1,name2) of regions

        neighbors: a list of NeighborRegions objects

        irrelevant_min: list of minima discarded from Watershed

        boundary: a list giving tuples for each vertex for neighbouring regions
                  it is the same info as in each region but indexed by vertices
    """

    def __init__(self, min_indices):
        self.regions = []
        for min_index in min_indices:
            (self.regions).append(Region(min_index))
        self.neighbors_couples = []
        self.neighbors = []
        self.irrelevant_min = []
        self.boundary = []

    def add_neighbors(self, NR):
        (self.neighbors).append(NR)

    def add_irrelevant_min(self, list):
        (self.irrelevant_min).extend(list)

    def remove(self, name):
        (self.regions[name]).remove()
        self.irrelevant_min.append(name)

    def merge(self, removed, mergedto):  # removed and mergedto are the regions'number
        # transfer interior
        (self.regions)[removed].remove(mergedto)
        interior = self.regions[removed].interior
        for n in interior:
            self.boundary[n] = [mergedto]
        (self.regions)[mergedto].add_interior(interior)

        # transfer removed boundary completely to mergedto
        boundary_removed = set(self.regions[removed].boundary)
        boundary_mergedto = set(self.regions[mergedto].boundary)
        boundary_mergedto.update(boundary_removed)
        self.regions[mergedto].boundary = list(boundary_mergedto)

        #update neighbouring regions for boundary of removed
        new_interior = set()
        for b in boundary_removed:
            self.boundary[b].remove(removed)
            self.boundary[b].add(mergedto)
        # clean boundary
            if self.boundary[b] == {mergedto}:
                new_interior.add(b)
        (self.regions[mergedto]).boundary_to_interior(new_interior)

        # transfer neighbors
        for couple in self.neighbors_couples:
            if couple[0] == removed:
                i = (self.neighbors_couples).index(couple)
                new_couple = (min(couple[1], mergedto), max(couple[1], mergedto))
                if new_couple in self.neighbors_couples:
                    i_new = (self.neighbors_couples).index(new_couple)
                    self.neighbors[i_new].merge(i)
                else:
                    (self.neighbors[i]).regions = couple
                del self.neighbors[i]
            if couple[1] == removed:
                i = (self.neighbors_couples).index(couple)
                new_couple = (min(couple[0], mergedto), max(couple[0], mergedto))
                if new_couple in self.neighbors_couples:
                    i_new = (self.neighbors_couples).index(new_couple)
                    self.neighbors[i_new].merge(i)
                else:
                    (self.neighbors[i]).regions = couple
                del self.neighbors[i]


def init_regions(I, J, W):
    """
    Builds up the initial Regions data from the mesh structured values I, the labeled minima J and the watershed W

    The algorithm visits once each vertex n and its neighbors if n is boundary.
    """

    n_reg = np.max(W)
    min_indices = np.zeros(n_reg)
    for i in J:
        if i != 0:
            min_indices[J[i]] = i
    n_ver = len(W)
    r = Regions()
    for n in range(n_reg):
        j = Region(n, J[n])
        r.add_region(j)
    for n in range(n_ver):
        # case n is an interior vertex of reg w[n]:
        # add it to interior of this region
        # only add this region as neighboring the vertex
        if W[n] != 0:
            (r.regions[W[n]]).add_interior([n])
            (r.boundary).append(set([W[n]]))
        # case n is a boundary vertex of reg w[j] where j is a neighbor:
        # add the vertex to boundary of all neighboring regions
        # add all neighboring regions to boundary[n]
        if W[n] == 0:
            neighbouring = set()
            for j in I[n].vneighbors:
                if W[j] != 0:
                    (r.regions[W[j]]).add_boundary([n])
                    neighbouring.add(W[j])
            (r.boundary).append(neighbouring)
            # add neighbouring regions as neighbors
            val = I[n].z
            k = len(neighbouring)  # number of regions touching vertex n
            neighlist = list(neighbouring)
            for i in range(k - 1):
                for j in range(i + 1, k):
                    r1 = np.min(neighlist[i], neighlist[j])
                    r2 = np.max(neighlist[i], neighlist[j])
                    if (r1, r2) not in r.neighbors_couples:
                        (r.neighbors_couples).append((r1, r2))
                        new_neighbors = NeighborRegions(r1, r2, [n], n)
                        r.add_neighbors(new_neighbors)
                    else:
                        pos = (r.neighbors_couples).index((r1, r2))
                        (r.neighbors[pos]).add_index[n]
    return r


def merge_algorithm(
    regions, barrier_condition=lambda min, barr: barr > 1.5 * min, inclusive=True
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
        bv.append((nr.min_index, nr.min_value))
    barrier_vertices = np.array(bv, dtype=dtype)
    np.sort(np.array(barrier_vertices), order="min_value")

    for v in barrier_vertices:
        # retrieve neighbor regions
        nbr = list(regions.boundary[v[0]])
        for i in range(len(nbr) - 1):
            for j in range(i + 1, len(nbr)):
                r1 = np.min(nbr[i], nbr[j])
                r2 = np.max(nbr[i], nbr[j])
                pos = (regions.neighbors_couples).index((r1, r2))
                neighbor_regions = regions.neighbors[pos]
                # min of r1 is smaller than min of r2
                min1 = (regions.regions[r1]).min_value
                min2 = (regions.regions[r2]).min_value
                if (not barrier_condition(min1, v[1])) and (
                    not barrier_condition(min2, v[1])
                ):
                    regions.merge(r1, r2)
                elif barrier_condition(min1, v[1]):
                    if inclusive:
                        regions.merge(r1, r2)
                    else:
                        regions.remove(r2)
    for i_r in len(regions.regions):
        reg = regions.regions[i_r]
        if reg.removed and reg.mergedwith is None:
            regions.add_irrelevant_min((i, reg.min_index))
    return regions

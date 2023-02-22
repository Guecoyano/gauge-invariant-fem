import numpy as np
import watershed_utils
import watershed_merge_utils


# This program takes the effective potential stored as a txt file W_10.txt
# and returns an array of integers called independent_minima which are the indices
# of the subset of minima of the effective potential which are considered able to
# host localized eigenfunctions.

W = np.loadtxt("W_10.txt")
M = watershed_merge_utils.find_minima(W)
watershed_merge_utils.store_watershed_transform(M, W)
#
threshold = 0.1  # This is rhe threshold beyond which 2 points are considered
# not likely to be negihbors on a noramlized domain of size 1.
# This may need to be adjusted based on the potential.
# If no value if provided, it is 0.5 by default, but this will take long to execute.
watershed_merge_utils.find_neighbors(threshold=threshold)
watershed_merge_utils.construct_network(W)
independent_minima = watershed_merge_utils.merge_algorithm(W)

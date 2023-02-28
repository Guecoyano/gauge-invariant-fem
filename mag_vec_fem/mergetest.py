"""script to test watershed and merge """

import numpy as np
from fem_base.exploit_fun import *
import matplotlib.pyplot as plt
import watershed.watershed_utils_gi as ws
import watershed.watershed_merge_gi as wsm


print("setting vertex data")
vertex_values = [
    2,
    2,
    2,
    10,
    4,
    4,
    2,
    1,
    8,
    10,
    3,
    6,
    10,
    7,
    4,
    10,
    2,
    10,
    4,
    10,
    6,
    3,
    3,
    8,
    9,
    5,
]
positions = [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (0, -1),
    (1, -1),
    (2, -1),
    (3.3, -1),
    (4.5, -1),
    (1, -2),
    (2.2, -2.2),
    (3.5, -2),
    (5, -1.5),
    (0, -3),
    (1, -3),
    (1.7, -3),
    (3, -3),
    (4.2, -2.8),
    (5, -3),
    (0, -5),
    (1, -5),
    (2.5, -5),
    (4.2, -5),
    (5, -5),
]
vneighbors = [
    (0, 1),
    (0, 6),
    (1, 2),
    (1, 7),
    (1, 6),
    (2, 3),
    (2, 7),
    (2, 8),
    (3, 4),
    (3, 8),
    (3, 9),
    (4, 5),
    (4, 9),
    (4, 10),
    (5, 10),
    (5, 14),
    (6, 7),
    (6, 11),
    (6, 15),
    (7, 8),
    (7, 11),
    (7, 12),
    (8, 9),
    (8, 12),
    (8, 13),
    (9, 10),
    (9, 13),
    (10, 13),
    (10, 14),
    (10, 19),
    (11, 12),
    (11, 15),
    (11, 15),
    (11, 16),
    (11, 17),
    (12, 13),
    (12, 17),
    (12, 18),
    (13, 18),
    (13, 19),
    (14, 19),
    (14, 20),
    (15, 16),
    (15, 21),
    (16, 17),
    (16, 21),
    (16, 22),
    (17, 18),
    (17, 22),
    (17, 23),
    (18, 19),
    (18, 23),
    (18, 24),
    (19, 20),
    (19, 24),
    (20, 24),
    (20, 25),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 25),
]
nvert = len(vertex_values)

vertex_data = []
for n in range(nvert):
    vertex_data.append(ws.Vertex(n, positions[n][0], positions[n][1], vertex_values[n]))
for couple in vneighbors:
    n1, n2 = couple[0], couple[1]
    vertex_data[n1].add_neighbor(n2)
    vertex_data[n2].add_neighbor(n1)


print("find local minima")
M = ws.label_local_minima(vertex_data, mode="irr")

print("initial watershed")
W = ws.watershed(vertex_data, M, mode="irr")

print("initial numberof regions:", np.max(M))

print("Structuring data")
regions = wsm.init_regions(vertex_data, M, W)

# for r in regions.regions:
#    print(r.interior,'are inside of',r.boundary)

# print(regions.global_boundary)

print("neighbours", regions.neighbors_couples)
# for nbrr in regions.neighbors:
#    print('regions', nbrr.regions)
#    print('index',nbrr.min_index,'value',nbrr.min_value)

"""print("Plotting boundaries over effective potential")
x=[]
y=[]
for n in range(nvert):
    if len(regions.global_boundary[n])>=2:
        x.append(positions[n][0])
        y.append(positions[n][1])
plt.figure()
plt.clf()
plt.scatter(x,y,c='k',s=2)
plt.show()
plt.clf()
plt.close()"""

print("merging regions")


def cond(min, barr):
    return barr > 2.50 * min


# print(regions.neighbors_couples)
# for nnbbrr in regions.neighbors:
#    print(nnbbrr.regions)
# regions.merge(2,4)

merged = wsm.merge_algorithm(regions, barrier_condition=cond, inclusive=False)

final_min = 0
for r in regions.regions:
    if not r.removed:
        final_min += 1

print("final number of regions:", final_min)


print(regions.neighbors_couples)
for nnbbrr in regions.neighbors:
    print(nnbbrr.regions)

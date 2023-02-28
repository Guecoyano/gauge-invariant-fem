import numpy as np
import sys
from matplotlib.pyplot import *
from watershed.watershed_utils import *
import matplotlib.pyplot as plt
import fenics
import dolfin
import pickle


# 1) Convert W into a text file
# The first bit of the code takes the effective potential and converts it into a txt file.


def read_W_and_store_as_txt():
    # Define periodic boundary conditions.
    class PeriodicBoundary(SubDomain):
        # Left edge and bottom edge are target domain
        def inside(self, x, on_boundary):
            return (
                (near(x[0], 0) or near(x[1], 0.0))
                and not (near(x[0], 1.0) or near(x[1], 1.0))
            ) and on_boundary

        def map(self, x, y):
            # Map top right corner to origin.
            if near(x[0], 1.0) and near(x[1], 1.0):
                y[0] = x[0] - 1.0
                y[1] = x[1] - 1.0
            # Map right edge of domain to left edge.
            elif near(x[0], 1.0):
                y[0] = x[0] - 1.0
                y[1] = x[1]
            # Map top edge of domain to bottom edge.
            elif near(x[1], 1.0):
                y[0] = x[0]
                y[1] = x[1] - 1.0
            # Need else statement for some unknown reason.
            else:
                y[0] = -1.0
                y[1] = -1.0

    N = 800  # Potential is defined on [0, 1] x [0, 1] grid of N x N squares
    deg_fe = 1  # Degree of finite elements used to solve eigenvalue problem

    mesh = UnitSquareMesh(N, N)

    pbc = PeriodicBoundary()
    Vh = FunctionSpace(
        mesh, "CG", deg_fe, constrained_domain=pbc
    )  # Function space for FE solution

    W = Function(Vh)
    with XDMFFile(
        MPI.comm_world, "eff_" + str(N) + "_" + str(deg_fe) + ".xdmf"
    ) as fFile:
        fFile.read_checkpoint(W, "W", 0)

    u = Function(Vh)
    with XDMFFile(MPI.comm_world, "u_" + str(N) + "_" + str(deg_fe) + ".xdmf") as fFile:
        fFile.read_checkpoint(u, "u", 0)

    # Convert the W(x,y) to an array W_np[x*mesh_size][y*mesh_size]

    mesh_size = N
    W_np_fixed = np.zeros([mesh_size, mesh_size])
    u_np_fixed = np.zeros([mesh_size, mesh_size])

    for i in range(mesh_size):
        for j in range(mesh_size):
            W_np_fixed[i][j] = W(1.0 * i / mesh_size, 1.0 * j / mesh_size) / (const * q)
            u_np_fixed[i][j] = u(1.0 * i / mesh_size, 1.0 * j / mesh_size)

    np.savetxt("results/W.txt", W_np_fixed)
    np.savetxt("results/u.txt", u_np_fixed)


def find_minima(W_np_fixed):
    # filename = 'results/W.txt'
    # W_np_fixed = np.loadtxt(filename) # function on [0, 1] x [0, 1]
    n1, n2 = W_np_fixed.shape

    # corresponding x and y coordinates
    x = np.linspace(0.0, 1.0, n1, endpoint=False)
    y = np.linspace(0.0, 1.0, n2, endpoint=False)
    # label the local maxima
    M = label_local_minima(W_np_fixed, mode="wrap")
    no_of_minima = int(np.max(M))
    print("Number of minima", no_of_minima)

    W_ag = np.zeros(no_of_minima)
    x_ag = np.zeros(no_of_minima)
    y_ag = np.zeros(no_of_minima)
    x_min = np.zeros(no_of_minima)
    y_min = np.zeros(no_of_minima)

    for i in range(n1):
        for j in range(n2):
            if M[i, j] != 0:
                # print(i,j,M[i,j])
                W_ag[int(M[i][j]) - 1] = W_np_fixed[i][j]
                x_ag[int(M[i][j]) - 1] = i / (n1)
                y_ag[int(M[i][j]) - 1] = j / (n2)
                x_min[int(M[i][j]) - 1] = i
                y_min[int(M[i][j]) - 1] = j

    # print(W_ag)
    # print(x_ag)
    # print(y_ag)
    np.savetxt("x_W.txt", x_ag)
    np.savetxt("y_W.txt", y_ag)
    np.savetxt("E_W.txt", W_ag)
    print(
        "Storing the x,y coordinates of the minima normalized to a domain of size 1 in x_W.txt, y_W.txt"
    )
    print("Storing the value of the effective potential at the minima E_W.txt")

    return M


#


def find_borders(W_np_fixed, W):
    W_np_fixed[W == 0.0] = np.nan
    n1, n2 = W.shape

    #
    border_x = []
    border_y = []
    for i in range(n1):
        for j in range(n2):
            if W_np_fixed[i, j] != W_np_fixed[i, j]:
                border_x.append(i)
                border_y.append(j)

    fig1, ax1 = plt.subplots(figsize=(10, 10))

    plt.plot(border_x, border_y, ".")
    plt.show()


def store_watershed_transform(M, W_np_fixed):
    W = watershed(W_np_fixed, M, mode="wrap")
    np.savetxt("watershed.txt", W)
    print(
        "Storing the watershed transform (as defined in watershed_utils.py) in watershed.txt"
    )


# np.savetxt("W_ag.txt", W_ag)
def find_neighbors(threshold=0.5):
    print("Calculating the network of neighbouring subregions. This may take a while")
    W = np.loadtxt("watershed.txt")
    n1, n2 = W.shape

    x_min = np.loadtxt("x_W.txt") * n1
    y_min = np.loadtxt("y_W.txt") * n2
    x_ag = np.loadtxt("x_W.txt")
    y_ag = np.loadtxt("y_W.txt")

    no_of_minima = x_min.size

    neighbors = []

    for i in range(no_of_minima):
        neighbors.append([])
    for i in range(no_of_minima):
        # print(i)
        for j in range(i + 1, no_of_minima):
            # print(j)
            if i != j:
                dist_x = np.abs(x_ag[i] - x_ag[j])
                if dist_x > 0.5:
                    dist_x = 1 - dist_x
                dist_y = np.abs(y_ag[i] - y_ag[j])
                if dist_y > 0.5:
                    dist_y = 1 - dist_y

                dist = np.sqrt(dist_x**2 + dist_y**2)
                # print("dist", i,j, dist)
                if dist > threshold:
                    # print("Not checking")
                    continue
                # print("Checking")
                bd_i = boundary(W, int(x_min[i]), int(y_min[i]), "wrap")
                bd_j = boundary(W, int(x_min[j]), int(y_min[j]), "wrap")

                common_points = points_in_common(bd_i, bd_j)
                if len(common_points) != 0:
                    # print("Neighbors", i, j)
                    neighbors[i].append(j)
                    neighbors[j].append(i)

    with open("neighbors.data", "wb") as f:
        pickle.dump(neighbors, f)

    print("Storing the list of neighbors in neighbors.data")
    # with open('neighbors.data','rb') as f:
    #     new_data = pickle.load(f)

    # print(new_data)


def construct_network(W_np_fixed):
    print(
        "Evaluating the values of the effective potential at the borders of the sub-regions. This may take a while."
    )
    W = np.loadtxt("watershed.txt")
    n1, n2 = W.shape

    x_min = np.loadtxt("x_W.txt") * n1
    y_min = np.loadtxt("y_W.txt") * n2
    no_of_minima = x_min.size
    with open("neighbors.data", "rb") as f:
        neighbors = pickle.load(f)

    min_boundary = np.zeros([no_of_minima, no_of_minima])
    avg_boundary = np.zeros([no_of_minima, no_of_minima])

    for i in range(no_of_minima):
        # print(i)
        for j in neighbors[i]:
            if j > i:
                bd_i = boundary(W, int(x_min[i]), int(y_min[i]), "wrap")
                bd_j = boundary(W, int(x_min[j]), int(y_min[j]), "wrap")

                common_points = points_in_common(bd_i, bd_j)

                min_ij = np.min(W_np_fixed[tuple(np.array(common_points).T)])
                avg_ij = np.mean(W_np_fixed[tuple(np.array(common_points).T)])
                min_boundary[i][j] = min_ij
                min_boundary[j][i] = min_ij
                avg_boundary[i][j] = avg_ij
                avg_boundary[j][i] = avg_ij

    np.savetxt("min_boundary.txt", min_boundary)
    np.savetxt("avg_boundary.txt", avg_boundary)
    print(
        "Storing the minima of the boundary between different sub-regions in min_boundary.txt"
    )


def is_independent(i, W_ag, bound_val, neighbors):
    for j in neighbors[i]:
        if 1.5 * W_ag[i] > bound_val[i][j]:
            return False

    return True


# region
def merging_condition(i, j, W_ag, bound_val):
    if 1.5 * W_ag[i] > bound_val[i][j] and 1.5 * W_ag[j] > bound_val[j][i]:
        return True
    else:
        return False


def index_p(i, L):
    ii = (i + L) % L
    return ii


def put_in_order(i, nbrs):
    for j in range(len(nbrs)):
        if i < nbrs[j]:
            nbrs.insert(j, i)
            return

    nbrs.append(i)


def merge(i, j, x_min, y_min, W, bound_val, neighbors, plot_option=False):
    bd_i = boundary(W, int(x_min[i]), int(y_min[i]), "wrap")
    bd_j = boundary(W, int(x_min[j]), int(y_min[j]), "wrap")

    n1, n2 = W.shape
    if plot_option == True:
        fig1, ax1 = plt.subplots(figsize=(10, 10))

        for k in range(len(bd_i)):
            plt.plot(bd_i[k][0], bd_i[k][1], ".", color="tab:orange")

        plt.xlim(0, n1)
        plt.ylim(0, n2)
        plt.show()

        fig1, ax1 = plt.subplots(figsize=(10, 10))

        for k in range(len(bd_j)):
            plt.plot(bd_j[k][0], bd_j[k][1], ".", color="tab:orange")

        plt.xlim(0, n1)
        plt.ylim(0, n2)
        plt.show()

    common_points = points_in_common(bd_i, bd_j)
    region_points = np.argwhere(W == j + 1)

    # print(len(region_points))
    # fig1, ax1 = plt.subplots(figsize=(10,10))

    W[tuple(np.array(region_points).T)] = i + 1

    #     for k in range(len(region_points)):
    #         if(W[region_points[k][0]][region_points[k][1]] == i+1):
    # #         W[region_points[k][0]][region_points[k][1]] = i+1
    #             plt.plot(region_points[k][0],region_points[k][1], ".", color = "tab:blue")

    for k in range(len(common_points)):
        ii, jj = common_points[k]
        #         print(ii,jj)
        flag_boundary = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                if (
                    W[index_p(ii + x, n1)][index_p(jj + y, n2)] != i + 1
                    and W[index_p(ii + x, n1)][index_p(jj + y, n2)] != 0
                ):
                    flag_boundary = 1
                    break

        #                 if W[index_p(ii+x,L)][index_p(jj+y,L)] != i+1 and W[index_p(ii+x,L)][index_p(jj+y,L)] != j+1 and W[index_p(ii+x,L)][index_p(jj+y,L)] != 0:
        #                     flag_boundary = 1
        if flag_boundary == 0:
            W[common_points[k][0]][common_points[k][1]] = i + 1
            # plt.plot(common_points[k][0],common_points[k][1], ".", color = "tab:orange")

    # plt.xlim(0,800)
    # plt.ylim(0,800)
    # plt.title("Common")
    # plt.show()

    # fig1, ax1 = plt.subplots(figsize=(10,10))
    # plt.imshow(W)
    # plt.show()

    for k in range(len(neighbors)):
        if k != i:
            for l in neighbors[k]:
                if l == j:
                    neighbors[k].remove(j)
                    if i not in neighbors[k]:
                        put_in_order(i, neighbors[k])
                        bound_val[k][i] = bound_val[k][j]
                        bound_val[i][k] = bound_val[k][i]
                    else:
                        bound_val[k][i] = min(bound_val[k][i], bound_val[k][j])
                        bound_val[i][k] = bound_val[k][i]
                    break
        else:
            neighbors[k].remove(j)

    for k in neighbors[j]:
        if k != i:
            if k not in neighbors[i]:
                put_in_order(k, neighbors[i])

    if plot_option == True:
        bd_i = boundary(W, int(x_min[i]), int(y_min[i]), "wrap")

        region_points = np.argwhere(W == i + 1)

        fig1, ax1 = plt.subplots(figsize=(10, 10))

        for k in range(len(region_points)):
            plt.plot(region_points[k][0], region_points[k][1], ".", color="tab:blue")

        for k in range(len(bd_i)):
            plt.plot(bd_i[k][0], bd_i[k][1], ".", color="tab:orange")
        plt.title("After merge")
        plt.xlim(0, n1)
        plt.ylim(0, n2)
        plt.show()


def merge_algorithm(W_np_fixed):
    print(
        "Merging different sub-regions and eliminating spurious minima. This may take a while."
    )
    independent_list = []
    sub_region_final = []
    working_list = []
    spurious_list = []
    W = np.loadtxt("watershed.txt")
    n1, n2 = W.shape

    x_min = np.loadtxt("x_W.txt") * n1
    y_min = np.loadtxt("y_W.txt") * n2

    no_of_minima = x_min.size
    with open("neighbors.data", "rb") as f:
        neighbors = pickle.load(f)

    min_boundary = np.loadtxt("min_boundary.txt")
    W_ag = np.loadtxt("E_W.txt")

    for i in range(no_of_minima):
        working_list.append(i)

    for i in range(no_of_minima):
        # print(i)
        if is_independent(i, W_ag, min_boundary, neighbors):
            independent_list.append(i)
            sub_region_final.append(i)
            working_list.remove(i)
    # print(independent_list)

    merged_list = []
    independent_list_2 = []
    all_regions_not_checked = 1
    index = 0
    while all_regions_not_checked:
        # print("index", index)
        if index == len(working_list):
            all_regions_not_checked = 0
            break

        else:
            # print("working listindex", working_list[index])
            if is_independent(working_list[index], W_ag, min_boundary, neighbors):
                # print("is independent true", working_list[index])
                independent_list.append(working_list[index])
                working_list.pop(index)
                continue

            else:
                merge_flag = 0
                # print("neighbors", neighbors[working_list[index]])
                for j in neighbors[working_list[index]]:
                    if j not in independent_list:
                        merge_flag = 2
                        # print("j", j)
                        if merging_condition(
                            working_list[index], j, W_ag, min_boundary
                        ):
                            # print('merging condition verified for', j)
                            merge(
                                working_list[index],
                                j,
                                x_min,
                                y_min,
                                W,
                                min_boundary,
                                neighbors,
                                False,
                            )
                            merged_list.append([working_list[index], j])
                            working_list.remove(j)
                            #                             fig1, ax1 = plt.subplots(figsize=(10,10))
                            #                             plt.imshow(W)
                            #                             plt.show()
                            #                             find_borders(W_np_fixed, W)

                            merge_flag = 1
                            break

                # if(merge_flag == 2):
                #     print("is not independent and second case", working_list[index])
                #     independent_list.append(working_list[index])
                #     independent_list_2.append(working_list[index])
                #     working_list.pop(index)
                #     continue

                if merge_flag == 0 or merge_flag == 2:
                    # print("spurious", working_list[index])
                    spurious_list.append(working_list[index])
                    index += 1
    independent_array = np.sort(np.array(independent_list))
    # print(independent_array)
    np.savetxt("ind.txt", independent_array)
    # np.savetxt("ind2.txt", np.array(independent_list_2))
    # border_x = []
    # border_y = []
    # for i in range(n1):
    #     for j in range(n2):
    #         if W[i][j] == 0:
    #             border_x.append(i)
    #             border_y.append(j)
    #
    # fig1, ax1 = plt.subplots(figsize=(10,10))
    # plt.plot(x_min[independent_list], y_min[independent_list], ".")
    # # plt.plot(x_min[independent_list_2], y_min[independent_list_2], "x")
    #
    # plt.plot(border_x, border_y, ".")
    # plt.savefig("watershed_merged_ind.png", dpi = 150, bbox_inches = "tight")
    # plt.close()

    np.savetxt("watershed_merged.txt", W)

    with open("neighbors_merged.data", "wb") as f:
        pickle.dump(neighbors, f)
    print("Storing the merged watershed transform in watershed_merged.txt")

    return independent_array


# endregion

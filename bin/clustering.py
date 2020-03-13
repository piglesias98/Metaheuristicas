import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations


def read_file(file):
    with open(file, 'r') as f:
        line = [i.strip().split(',') for i in f]
        data = [[float(i) for i in l] for l in line]
        return np.array(data)


def build_restrictions_list(matrix):
    list = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if matrix[i][j] != 0:
                list.append([i, j, matrix[i][j]])
    return np.array(list)


def infeasibility(xi, ci, r_matrix, clusters):
    infeas = 0
    for i in range(r_matrix.shape[0]):
        if (ci == clusters[i] and r_matrix[xi][i] == -1) or (ci != clusters[i] and r_matrix[xi][i] == 1):
            infeas = infeas + 1
    return infeas


def compute_centroids(data, clusters, k):
    return [np.mean(data[np.where(clusters == i)]) for i in range(k)]



def greedy(data, r_matrix, r_list, k):
    # Number of items to cluster
    n = len(data)
    rsi = np.array(range(data.shape[0]))
    # Shuffle the indexes
    random.shuffle(rsi)
    # Calculate initial centroids
    centroids = data[:k]
    # Clusters
    clusters = np.full(n, -1)
    while True:
        old_clusters = np.copy(clusters)
        for i in rsi:
            infeas = [infeasibility(i, ci, r_matrix, clusters) for ci in range(k)]
            min_inf = np.where(infeas == np.min(infeas))[0]
            if len(min_inf) == 1:
                best_cluster = min_inf[0]
            else:
                distances = [distance.euclidean(data[i], centroids[c]) for c in min_inf]
                print(distances)
                best_cluster = np.argmin(np.array(distances))
            # Assign the element to the best cluster
            clusters[i] = best_cluster
        # Update centroid uk with the average instances of its associated cluster ci
        centroids = compute_centroids(data, clusters, k)
        if np.array_equal(old_clusters, clusters):
            break
    return clusters


def initial_solution(n, k):
    clusters = np.random.randint(0, k, n)
    # If the solution is feasible
    if len(np.unique(clusters)) == k:
        initial_solution(n, k)
    else:
        return clusters


def c(clusters, data, k):
    intra_cluster_deviation = [np.std(data[np.where(clusters == i)]) for i in range(k)]
    return np.mean(intra_cluster_deviation)


def infeasibility_total(k, clusters, r_list):
    return np.count_nonzero((r_list[i][2] == -1 and r_list[i][1] == clusters[i]) or (r_list[i][2] == 1 and r_list[i][1] != clusters[i]) for i in range(k))


def objective_function(clusters, data, k, list_res):
    return c(clusters, data, k) + infeasibility_total(len(data), k) * len(data)/len(list_res)

def change_neighbour(clusters, i, l):
    if np.count_nonzero(clusters == l) and clusters[i]!=l:
        clusters[i]=l
        return clusters
    else:
        change_neighbour(clusters, i, l)

# def lambda(data, r_list):
#

# def local_search(data, r_matrix, r_list, k):
#     n = len(data)
#     sol = initial_solution()
#     n = len(data)
#     rsi = np.array(range(data.shape[0]))
#     clusters = np.array(range(k))
#     while (iteration < 10000 and !no_change):
#         old_sol = np.copy(sol)
#         # hago shuffle y recorro el vecindario
#         # Shuffle the indexes
#         random.shuffle(rsi)
#         random.shuffle(clusters)
#         while (i< len(rsi) and !better)
#             while (c<len(clusters) and !better)
#                 sol = change_neighbour(sol, i, c)
#                 if (objective_function(sol) > objective_function(old_sol)):
#                     better = True
#                     old_sol = np.copy(sol)
#                 iteration += 1
#                 c += 1
#             iteration += 1
#             i += 1
#         if (better):
#             no_changes = True
#     return sol
#
#
#
# def local_search(data, k):
#     n = len(data)
#     sol = initial_solution(n, k)
#     iteration = 0
#     while True:
#         neighbourhood = generate_virtual_neighbourhood(n, k)
#         i = -1
#         while True:
#             i += 1
#             iteration += 1
#             possible_neighbour = generate_neighbour(sol, neighbourhood[i])
#             if len(np.unique(possible_neighbour)) == k:
#                 neighbour = possible_neighbour
#                 if objective(neighbour) > objective(sol) or neighbourhood:
#                     break
#         if objective(neighbour) > objective(sol):
#             sol = neighbour
#         iteration += 1
#         else objective(neighbour) <= objective (sol) or iterations:
#             break


def generate_neighbour(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour

def generate_virtual_neighbourhood(n, k):
    neighbourhood = []
    for c in range(k):
        for i in range(n):
            neighbourhood.append([i, c])
    random.shuffle(neighbourhood)
    return np.array(neighbourhood)


mi = [1,2,3,4]
comb = combinations(mi, 2)
for i in comb:
    print(i)
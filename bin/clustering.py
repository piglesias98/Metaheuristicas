import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time

random.seed(30)

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


def infeasibility(xi, ci, r_matrix, sol):
    infeas = np.count_nonzero([(ci == sol[i] and r_matrix[xi][i] == -1) or (ci != sol[i] and r_matrix[xi][i] == 1) for i in range(r_matrix.shape[0])])
    return infeas


def compute_centroids(data, sol, k):
    return [np.mean(data[np.where(sol == i)]) for i in range(k)]


def initial_solution(k, n):
    initial_sol = np.random.randint(0, k, n)
    # If the solution is not feasible try again
    if len(np.unique(initial_sol)) != k:
        initial_solution(k, n)
    else:
        return initial_sol


def generate_neighbour(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour


def generate_virtual_neighbourhood(n, k, sol):
    neighbourhood = []
    for c in range(k):
        for i in range(n):
            if len(np.unique(generate_neighbour((sol),[i,c]))) == k:
                neighbourhood.append([i, c])
    random.shuffle(neighbourhood)
    return np.array(neighbourhood)


def infeasibility_total(sol, r_list):
    return np.count_nonzero((i[2] == -1 and sol[i[0]] == sol[i[1]]) or (i[2] == 1 and sol[i[0]] != sol[i[1]]) for i in r_list)


def compute_lambda(data, r_list):
    index_list = np.array(range(data.shape[0]))
    comb = np.array(list(combinations(index_list, 2)))
    d = np.max([distance.euclidean(data[i[0]], data[i[1]]) for i in comb])
    r = len(r_list)
    return d/r


def c(sol, data, centroids, k):
    data_cluster = np.array([data[np.where(sol == c)] for c in range(k)])
    dis_cluster = [[distance.euclidean(data_cluster[c][i], centroids[c]) for i in range(data_cluster[c].shape[0])] for c in range(k)]
    intra_cluster_mean_distance = [np.mean(dis_cluster[c]) for c in range(k)]
    general_deviation = np.mean(intra_cluster_mean_distance)
    return general_deviation


def objective(sol, data, k, r_list, l):
    return c(sol, data, compute_centroids(data, sol, k), k) + infeasibility_total(sol, r_list) * l


def change_neighbour(clusters, i, l):
    if np.count_nonzero(clusters == l) and clusters[i]!=l:
        clusters[i]=l
        return clusters
    else:
        change_neighbour(clusters, i, l)


def initial_centroids(data, k):
    max = [np.max([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    min = [np.min([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    return [[random.uniform(max[i], min[i]) for i in range(data.shape[1])] for j in range(k)]


def greedy(data, r_matrix, k):
    # Number of items to cluster
    n = len(data)
    rsi = np.array(range(data.shape[0]))
    # Shuffle the indexes
    random.shuffle(rsi)
    # Compute initial centroids
    centroids = initial_centroids(data, k)
    # Clusters
    sol = np.full(n, -1)
    while True:
        old_sol = np.copy(sol)
        for i in rsi:
            infeas = [infeasibility(i, ci, r_matrix, sol) for ci in range(k)]
            min_inf = np.where(infeas == np.min(infeas))[0]
            if len(min_inf) == 1:
                best_cluster = min_inf[0]
            else:
                distances = np.array([[distance.euclidean(data[i], centroids[c]), c] for c in min_inf])
                best_cluster = distances[np.argmin(distances[:, 0]), 1]
            # Assign the element to the best cluster
            sol[i] = int(best_cluster)
        # Update centroid uk with the average instances of its associated cluster ci
        centroids = compute_centroids(data, sol, k)
        print("old",old_sol)
        print("new", sol)
        if np.array_equal(old_sol, sol):
            break
    return sol


def local_search(data, r_list, k):
    n = len(data)
    l = compute_lambda(data, r_list)
    sol = initial_solution(k, n)
    iteration = 0
    i = 0
    neighbourhood = generate_virtual_neighbourhood(n, k, sol)
    objective_sol = objective(sol, data, k, r_list, l)
    while iteration<100000 and i<len(neighbourhood):
        neighbour = generate_neighbour(sol, neighbourhood[i])
        i += 1
        objective_neighbour = objective(neighbour, data, k, r_list, l)
        iteration += 1
        # first neighbour that improves actual solution
        if objective_neighbour < objective_sol:
            sol = np.copy(neighbour)
            objective_sol = objective_neighbour
            neighbourhood = generate_virtual_neighbourhood(n, k, sol)
            i = 0
    return sol


data = read_file("bin/ecoli_set.dat")
r_matrix = read_file("bin/ecoli_set_const_20.const")
r_list = build_restrictions_list(r_matrix)
k = 8
start_time = time.time()
mi_sol = local_search(data, r_list, k)
# mi_sol = greedy(data, r_matrix, 3)
elapsed_time = time.time() - start_time
print(mi_sol)
print("tiempo", elapsed_time)
objetivo = objective(mi_sol, data, k, r_list, compute_lambda(data, r_list))
print("objetivo", objetivo)
c_rate = c(mi_sol, data, compute_centroids(data, mi_sol, k))
print("c_rate", c_rate)

import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import copy

'''
Read data file and convert it to np array
'''


def read_file(file):
    with open(file, 'r') as f:
        line = [i.strip().split(',') for i in f]
        data = [[float(i) for i in l] for l in line]
        return np.array(data)


'''
Build a np array from the restriction matrix
'''


def build_restrictions_list(matrix):
    list = []
    for i in range(matrix.shape[0]):
        for j in range(i+1, matrix.shape[1]):
            if matrix[i][j] != 0:
                list.append([i, j, int(matrix[i][j])])
    return np.array(list)


'''
Compute infeasibility of one adding one element to a cluster
'''


def infeasibility(xi, ci, r_matrix, sol):
    infeas = np.count_nonzero([(ci == sol[i] and r_matrix[xi][i] == -1) or (ci != sol[i] and r_matrix[xi][i] == 1) for i in range(r_matrix.shape[0])])
    return infeas


'''
Compute centroids from a solution and number of clusters
'''


def compute_centroids(data, sol, k):
    return [np.mean(data[np.where(sol == i)]) for i in range(k)]


'''
Computes random initial solution of n lenght with k clusters
'''


def initial_solution(k, n, seed):
    np.random.seed(seed)
    initial_sol = np.random.randint(0, k, n)
    # If the solution is not feasible try again
    if len(np.unique(initial_sol)) != k:
        initial_solution(k, n)
    else:
        return initial_sol


'''
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour


'''
Generate all possible changes from a solution sol
'''


def generate_virtual_neighbourhood (n, k, sol):
    neighbourhood = [[i, c] for c in range(k) for i in range(n) if sol[i] != c]
    return np.array(random.shuffle(neighbourhood))


'''
Computes global infeasibility for a solution
'''


def infeasibility_total(sol, r_list):
    return np.count_nonzero([ (i[2] == -1 and sol[i[0]] == sol[i[1]]) or (i[2] == 1 and sol[i[0]] != sol[i[1]]) for i in r_list])


'''
Compute lambda
'''


def compute_lambda(data, r_list):
    index_list = np.array(range(data.shape[0]))
    comb = np.array(list(combinations(index_list, 2)))
    d = np.max([distance.euclidean(data[i[0]], data[i[1]]) for i in comb])
    r = len(r_list)
    return d/r


'''
Compute c
'''


def c(sol, data, k):
    centroids = compute_centroids(data, sol, k)
    data_cluster = np.array([data[np.where(sol == c)] for c in range(k)])
    dis_cluster = [[distance.euclidean(data_cluster[c][i], centroids[c]) for i in range(data_cluster[c].shape[0])] for c in range(k)]
    intra_cluster_mean_distance = [np.mean(dis_cluster[c]) for c in range(k)]
    general_deviation = np.mean(intra_cluster_mean_distance)
    return general_deviation


'''
Compute objective function
'''


def objective(sol, data, k, r_list, l):
    obj = c(sol, data, k) + infeasibility_total(sol, r_list) * l
    return obj



def initial_centroids(data, k, seed):
    random.seed(seed)
    max = [np.max([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    min = [np.min([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    return [[random.uniform(max[i], min[j]) for i in range(data.shape[1])] for j in range(k)]


def greedy(data, r_matrix, k, seed):
    # Number of items to cluster
    n = len(data)
    rsi = np.array(range(data.shape[0]))
    # Shuffle the indexes
    random.seed(seed)
    random.shuffle(rsi)
    # Compute initial centroids
    centroids = initial_centroids(data, k, seed)
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
        if np.array_equal(old_sol, sol):
            break
    return sol


def local_search(data, r_list, k, seed):
    n = len(data)
    l = compute_lambda(data, r_list)
    sol = initial_solution(k, n, seed)
    iteration = 0
    i = 0
    neighbourhood = generate_virtual_neighbourhood(n, k, sol)
    objective_sol = objective(sol, data, k, r_list, l)
    while iteration<100000 and i<len(neighbourhood):
        neighbour = generate_neighbour(sol, neighbourhood[i])
        i += 1
        # If it is a feasible neighbour
        if len(np.unique(neighbour)) == k:
            objective_neighbour = objective(neighbour, data, k, r_list, l)
            iteration += 1
            # first neighbour that improves actual solution
            if objective_neighbour < objective_sol:
                sol = copy.deepcopy(neighbour)
                objective_sol = copy.deepcopy(objective_neighbour)
                neighbourhood = generate_virtual_neighbourhood(n, k, sol)
                i = 0
    return sol
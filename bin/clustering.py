import numpy as np
import random
from scipy.spatial import distance


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
    if
    return clusters



def c(clusters, data, k):
    intra_cluster_deviation = [np.std(data[np.where(clusters == i)]) for i in range(k)]
    return np.mean(intra_cluster_deviation)




def objective_function(data, list_res):
    return c() + infeasibility_total() * len(data)/len(list_res)

def local_search(data, r_matrix, r_list, k):
    n = len(data)
    clusters = initial_solution()




data = read_file('bin/iris_set.dat')
# r_matrix = read_file('bin/iris_set_const_10.const')
# print(r_matrix)
# r_list = build_restrictions_list(r_matrix)
#
# clusters = greedy(data, r_matrix, r_list, 3)
# print(clusters)
print(c(initial_solution(5,3), data, 3))
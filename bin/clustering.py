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


def infeasibility(xi, ci, r_list, clusters_index):
    concerning_xi = r_list[r_list[:,0] == xi]
    cl = np.count_nonzero((np.isin(concerning_xi[:,1], clusters_index[ci])) & (concerning_xi[:,2] == -1))
    ml = np.count_nonzero((np.isin(concerning_xi[:,1], np.concatenate(np.delete(clusters_index, ci))) & (concerning_xi[:,2] == 1)))
    return cl + ml


def greedy(data, r_matrix, r_list, k):
    rsi = np.array(range(data.shape[0]))
    # Shuffle the indexes
    random.shuffle(rsi)
    # Calculate initial centroids
    centroids = data[:k]
    # Clusters
    clusters = np.array([[-1], [-1], [-1]])
    clusters_index = np.array([[-1], [-1], [-1]])
    while True:
        old_clusters = np.copy(clusters)
        old_clusters_index = np.copy(clusters_index)
        for i in rsi:
            infs = [infeasibility(i, ci, r_list, clusters_index) for ci in range(len(clusters_index))]
            min_inf = np.where(infs == np.min(infs))
            if len(min_inf) == 1:
                best_cluster_index = min_inf
            else:
                distances = [distance.euclidean(data[i], cen) for cen in centroids]
                min_dist = np.argmin(distances)
                best_cluster_index = min_dist
            clusters[best_cluster_index].append(data[i])
            clusters_index[best_cluster_index].append(i)
        for i in k:
            # Update centroid uk with the average instances of its associated cluster ci
            centroids[i] = np.mean(clusters[i])
        if np.array_equal(np.sort(old_clusters_index), np.sort(clusters)):
            break
    return clusters


data = read_file('bin/iris_set.dat')
r_matrix = read_file('bin/iris_set_const_10.const')
r_list = build_restrictions_list(r_matrix)
k=3
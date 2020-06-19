import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time
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
        for j in range(i + 1, matrix.shape[1]):
            if matrix[i][j] != 0:
                list.append([i, j, int(matrix[i][j])])
    return np.array(list)


'''
Computes global infeasibility for a solution
'''


def infeasibility_total(sol, r_list):
    return np.count_nonzero([ (i[2] == -1 and sol[i[0]] == sol[i[1]]) or (i[2] == 1 and sol[i[0]] != sol[i[1]]) for i in r_list])


'''
Compute lambda factor
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


def c(sol, data, k, centroids):
    data_cluster = np.array([data[np.where(sol == c)] for c in range(k)])
    dis_cluster = [[distance.euclidean(data_cluster[c][i], centroids[c]) for i in range(data_cluster[c].shape[0])] for c in range(k)]
    intra_cluster_mean_distance = [np.mean(dis_cluster[c]) for c in range(k)]
    general_deviation = np.mean(intra_cluster_mean_distance)
    return general_deviation


'''
Compute objective function
'''


def objective(sol, data, k, r_list, l, centroids):
    obj = c(sol, data, k, centroids) + infeasibility_total(sol, r_list) * l
    return obj




'''
Compute initial centroids
'''


def compute_initial_centroids(data, k):
    max = [np.max([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    min = [np.min([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    return [[random.uniform(max[i], min[i]) for i in range(data.shape[1])] for j in range(k)]


'''
Get solution from centroids
'''


def solution_from_centroids(data, centroids):
    sol = [np.argmin([distance.euclidean(i, c) for c in centroids]) for i in data]
    return sol


def initial_fireflies(data, k, n_fireflies, l):
    initial_centroids = [compute_initial_centroids(data, k) for i in range(len(n_fireflies))]
    initial_sol = solution_from_centroids(data, initial_centroids)
    initial_obj = [[initial_sol[i], objective(initial_sol[i], data, k, r_list, l, initial_centroids[i])] for i in range(len(n_fireflies))]
    fireflies = [[initial_centroids[i], initial_centroids[i], initial_obj[i]] for i in range(len(n_fireflies))]
    return fireflies


dataset = "iris"
k = 3
data = read_file("bin/" + dataset + "_set.dat")
r = "10"
r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
r_list = build_restrictions_list(r_matrix)




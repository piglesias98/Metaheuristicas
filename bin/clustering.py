import numpy as np
import random

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


def infeasibility(xi, ci, matrix):
    infeas = 0
    for xj in range(matrix.shape[0]):
        ml = matrix[xi][xj] == 1 and xj not in ci
        cl = matrix[xi][xj] == -1 and xj in ci
        if ml or cl:
            infeas = infeas + 1
    return infeas

def greedy(data, matrix, k):
    rsi = np.array(range(data.shape[0]))
    # Shuffle the indexes
    random.shuffle(rsi)
    # Calculate initial centroids
    centroids = data[:k]
    # Clusters
    c = np.empty([3,1])
    while True:
        for i in rsi:
            infs = [infeasibility(i, ci, matrix) for ci in c]
            min_inf = np.where(infs == infs.min())
            if len(min_inf) == 1:
                c[min_inf].append(i)
            else:
                distances = [distance(i, ci, data) for ci in c]
                min_dist = np.min(distances)
                c[min_dist].append(i)
        for ci in range(k):
            # Update centroid uk with the average instaces of its
            # associated cluster ci
            update_dentroids(ci, data)
        if no_change:
            break
    return c


iris = read_file('bin/iris_set.dat')
iris_matrix = read_file('bin/iris_set_const_10.const')
# print(iris_matrix)
# print(iris_matrix.shape[0])
iris_list = build_restrictions_list(iris_matrix)
# print(iris_list)
# print(iris_list.shape[0])
# greedy(iris, iris_matrix)
import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import math
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
    return np.count_nonzero(
        [(i[2] == -1 and sol[i[0]] == sol[i[1]]) or (i[2] == 1 and sol[i[0]] != sol[i[1]]) for i in r_list])


'''
Compute lambda factor
'''


def compute_lambda(data, r_list):
    index_list = np.array(range(data.shape[0]))
    comb = np.array(list(combinations(index_list, 2)))
    d = np.max([distance.euclidean(data[i[0]], data[i[1]]) for i in comb])
    r = len(r_list)
    return d / r


'''
Compute c
'''


def compute_centroids(data, sol, k):
    return [np.mean(data[np.where(sol == i)]) for i in range(k)]


def c(sol, data, k):
    centroids = compute_centroids(data, sol, k)
    data_cluster = np.array([data[np.where(sol == c)] for c in range(k)])
    dis_cluster = [[distance.euclidean(data_cluster[c][i], centroids[c]) for i in range(data_cluster[c].shape[0])] for c
                   in range(k)]
    intra_cluster_mean_distance = [np.mean(dis_cluster[c]) for c in range(k)]
    general_deviation = np.mean(intra_cluster_mean_distance)
    return general_deviation


'''
Compute objective function
'''


def objective(sol, data, k, r_list, l):
    obj = c(sol, data, k) + infeasibility_total(sol, r_list) * l
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
    sol = np.array([np.argmin([distance.euclidean(i, c) for c in centroids]) for i in data])
    return sol


'''
Compute initial solution
'''


def compute_initial_solution(data, data_normalized, k):
    # Calculamos los centroides
    centroids = compute_initial_centroids(data_normalized, k)
    # Calculamos la solución en base a los centroides
    sol = solution_from_centroids(data_normalized, centroids)
    # Nos aseguramos de que sea una solución factible
    while len(np.unique(sol)) != k:
        # Calculamos los centroides
        centroids = compute_initial_centroids(data_normalized, k)
        # Calculamos la solución en base a los centroides
        sol = solution_from_centroids(data_normalized, centroids)
    return centroids, sol, objective(sol, data, k, r_list, l)


'''
Compute initial fireflies
'''


def initial_fireflies(n_fireflies, data, data_normalized, k):
    return [compute_initial_solution(data, data_normalized, k) for i in range(n_fireflies)]


def movement(move_from, move_to, beta, gamma, data, data_normalized, k, r_list, l):
    centroid_from = copy.deepcopy(np.array(move_from[0]))
    centroid_to = copy.deepcopy(np.array(move_to[0]))
    for c in range(k):
        r = distance.euclidean(centroid_from[c], centroid_to[c])
        suma = beta * math.exp(- gamma * r * r) * np.array(centroid_to[c] - centroid_from[c]) + np.array([random.uniform(-0.05,0.05) for i in range(centroid_from[c].shape[0])])
        # np.array([random.uniform(-1,1) for i in range(centroid_from[c].shape[0])]) * gamma
        centroid_from[c] = centroid_from[c] + suma

    sol = solution_from_centroids(data_normalized, centroid_from)
    # Update solution and light intensity only if it satisfies restrictions
    if len(np.unique(sol)) == k:
        move_from[0] = centroid_from
        move_from[1] = sol
        move_from[2] = objective(move_from[1], data, k, r_list, l)

    return move_from


def compute_gamma(data, k):
    mean = np.mean([np.mean([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])])
    return 1/np.sqrt(mean)


def normalize(data):
    data_normalized = copy.deepcopy(data)
    max = np.max([np.max([data_normalized[i][j] for i in range(data_normalized.shape[0])]) for j in range(data_normalized.shape[1])])
    min = np.min([np.min([data_normalized[i][j] for i in range(data_normalized.shape[0])]) for j in range(data_normalized.shape[1])])
    for i in range(data_normalized.shape[0]):
        for j in range(data_normalized.shape[1]):
            data_normalized[i][j] = (data_normalized[i][j]-min)/(max-min)
    # data = [[(data[i][j]-min)/(max-min) for i in range(data.shape[0])] for j in range(data.shape[1])]
    print("MAX" + str(max))
    print("MIN" + str(min))
    return data_normalized


dataset = "iris"
k = 3
data = read_file("bin/" + dataset + "_set.dat")
r = "10"
r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
r_list = build_restrictions_list(r_matrix)
n_fireflies = 50
l = compute_lambda(data, r_list)


# Inicializar to do
def fa_v1(data, r_list, k, l, n_fireflies):
    max_evaluations = 100000
    gamma = 1
    beta = 1
    data_normalized = normalize(data)
    fireflies = np.array(initial_fireflies(n_fireflies, data, data_normalized, k))
    evaluations = n_fireflies
    comb = list(combinations(range(n_fireflies), 2))
    fallos = 0
    best = 10000000000
    while evaluations < max_evaluations and fallos < n_fireflies:
        random.shuffle(comb)
        for i in comb:
           if fireflies[i[0]][2] < fireflies[i[1]][2]:
                # mover luciérnaga i hacia j
                # Variar atracción con la distancia via exp(-gamma r)
                # Evaluar las nuevas soluciones y actualizar la intensidad de la luz
                fireflies[i[1]] = movement(fireflies[i[1]], fireflies[i[0]], beta, gamma, data, data_normalized, k, r_list, l)
                evaluations = evaluations + 1
        if best > fireflies[np.argmin(fireflies[:, 2])][2]:
            best = fireflies[np.argmin(fireflies[:, 2])][2]
            fallos = 0
        else:
            fallos = fallos + 1
        print(str(best)+', ' + str(np.mean(fireflies[:, 2])) + ', ' + str(evaluations))
    # Ordenar las luciérnagas y buscar la más luminosa
    print('          FIIN          ')
    print(best)
    return best


f = fa_v1(data, r_list, k, l, n_fireflies)

# datasets = ["iris", "ecoli", "rand", "newthyroid"]
# clusters = [3, 8, 3, 3]
# restrictions = ["10", "20"]
#
# def run_in_parallel(d):
#     dataset = datasets[d]
#     k = clusters[d]
#     r = '10'
#     random.seed(d*10)
#     # Results file
#     f = open("fa_random05_" + dataset + '_' + str(d) + ".txt", "w")
#     data = read_file("bin/" + dataset + "_set.dat")
#     f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
#     f.write("SEED: " + str(d*10))
#     r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
#     r_list = build_restrictions_list(r_matrix)
#     f.write("\n\n--------> Restriction: " + r + "\n")
#     print("Restriction: ", r)
#     l = compute_lambda(data, r_list)
#     start_time = time.time()
#     sol = fa_v1(data, r_list, k, l, n_fireflies)
#     time_sol = time.time() - start_time
#     print(str(time_sol))
#     f.write("TIME: " + str(time_sol) + "\n\n")
#     obj_rate = objective(sol[1], data, k, r_list, compute_lambda(data, r_list))
#     print("OBJ_RATE: " + str(obj_rate) + "\n")
#     f.write("OBJ_RATE: " + str(obj_rate) + "\n")
#     c_rate = c(sol[1], data, k)
#     inf_rate = infeasibility_total(sol[1], r_list)
#     f.write("C_RATE: " + str(c_rate) + "\n")
#     print("C_RATE: " + str(c_rate) + "\n")
#     print("INF_RATE: " + str(inf_rate) + "\n")
#     f.write("INF_RATE: " + str(inf_rate) + "\n")
#
#
# from multiprocessing import Pool
#
# argument = [0,1,2,3]
#
#
# if __name__ == '__main__':
#     pool = Pool()
#     pool.map(run_in_parallel, argument)
#     pool.close()
#     pool.join()


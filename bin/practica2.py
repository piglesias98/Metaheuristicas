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


def initial_population(k,n, chromosomes, seed):
    population = []
    for i in range(chromosomes):
        population.append(initial_solution(k, n, seed))
    return population


def binary_tournament_agg(population, chromosomes):
    parents = []
    for i in range(chromosomes):
        indiviual_1 = random.choice(population)
        indiviual_2 = random.choice(population)
        if objective(indiviual_1, data, k, r_list, l) > objective(indiviual_2, data, k, r_list, l):
            parents.append(indiviual_1)
        else:
            parents.append(indiviual_2)
    return parents


def uniform_crossover(individual_1, individual_2, n):
    son = individual_1
    genes = np.random.randint(0, n, int(n/2))
    for i in genes:
        son[i] = individual_2[i]
    return son


def two_points_crossover(individual_1, individual_2, n):
    r = random.randint(0, n)
    v = random.randint(0, n)
    son = uniform_crossover(individual_1, individual_2, n)
    son[r, ((r+v) % n) - 1] = individual_1[r, ((r+v) % n) - 1]
    return son


# chromosomes = 50
def agg(data, r_list, k, seed, chromosomes):
    n = len(data)
    t = 0
    evaluations = 0
    # Initialize P(0)
    population = initial_population(k, n, chromosomes, seed)
    # Evaluate P(0)

    while (evaluations < 100000):
        # Selection
        parents = binary_tournament_agg(population, chromosomes)
        t = t+1
        seleccionar
        recombinar
        reemplazar
        evaluar

def memetic(data, r_list, k, seed, xi):
    n = len(data)
    l = compute_lambda(data, r_list)
    sol = initial_solution(k, n, seed)
    errors = 0
    improvement = True
    i = 0
    while (improvement or errors < xi) and i<n:
        improvement = False


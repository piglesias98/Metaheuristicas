import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time
import copy
import math


'''
Compute centroids from a solution and number of clusters
'''


def compute_centroids(data, sol, k):
    return [np.mean(data[np.where(sol == i)]) for i in range(k)]


'''
Computes random initial solution of n lenght with k clusters
'''


def initial_solution(k, n):
    # Se genera un array de tamaño n con números entre 0 y k,
    # donde k es el número de clústers
    initial_sol = np.random.randint(0, k, size=n)
    # Si la solución no es factible llamamos de nuevo a la función
    if len(np.unique(initial_sol)) != k:
        initial_solution(k, n)
    else:
        return initial_sol





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


def initial_temperature(mu, cost):
    return (mu*cost)/(-np.log(mu))


'''
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour(solution, n, k):
    sol = copy.deepcopy(solution)
    # El elemento a cambiar será una posición aleatoria de la solución
    to_change = random.randint(0, n - 1)
    # El valor será uno distinto al actual
    cluster = random.randint(0, k - 1)
    old_cluster = sol[to_change]
    sol[to_change] = cluster
    # Nos aseguramos de que sea una solución factible
    while len(np.unique(sol)) != k or old_cluster == cluster:
        sol[to_change] = old_cluster
        to_change = random.randint(0, n - 1)
        old_cluster = sol[to_change]
        cluster = random.randint(0, k - 1)
        sol[to_change] = cluster
    return sol


def cooling(temperature, initial_temperature, m, final_temperature):



def simulated_annealing(data, k, r_list, mu, final_temperature):
    l = compute_lambda(data, r_list)
    n = len(data)
    # Initial solution
    sol = initial_solution(k, n)
    obj_sol = objective(sol, data, k, r_list, l)
    evaluations = 1
    best_sol = sol
    obj_best = obj_sol
    # Initial temperature
    temperature = initial_temperature(mu, obj_sol)
    # L(T)
    max_vecinos = 10 * n
    max_exitos = 0.1 * max_vecinos
    while temperature <= final_temperature:
        vecinos = 0
        exitos = 0
        while vecinos<max_vecinos and exitos<max_exitos:
            new_sol = generate_neighbour(sol, n, k)
            obj_new = objective(new_sol, data, k, r_list, l)
            evaluations = evaluations + 1
            vecinos = vecinos + 1
            difference = obj_new - obj_sol
            if difference < 0 or random.random() <= math.exp(-difference/temperature):
                sol = new_sol
                obj_sol = obj_new
                if obj_sol < obj_best:
                    best_sol = sol
                    obj_best = obj_sol
                    exitos = exitos + 1
        temperature = cooling(temperature)
    return best_sol

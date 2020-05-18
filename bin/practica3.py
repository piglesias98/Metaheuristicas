import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time
import copy
import math


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


def compute_beta(initial_temperature, final_temperature, m):
    return (initial_temperature-final_temperature)/(m * initial_temperature * final_temperature)


def cooling(temperature, beta):
    return temperature/(1+beta*temperature)


def simulated_annealing(data, k, r_list, mu, final_temperature):
    l = compute_lambda(data, r_list)
    n = len(data)
    # Initial solution
    sol = initial_solution(k, n)
    obj_sol = objective(sol, data, k, r_list, l)
    evaluations = 1
    best_sol = copy.deepcopy(sol)
    obj_best = copy.deepcopy(obj_sol)
    # Initial temperature
    temperature = initial_temperature(mu, obj_sol)
    # L(T)
    max_vecinos = 10 * n
    max_exitos = 0.1 * max_vecinos
    # Cooling
    m = 100000/max_vecinos
    beta = compute_beta(temperature, final_temperature, m)
    while temperature > final_temperature and evaluations < 100000:
        vecinos = 0
        exitos = 0
        while vecinos<max_vecinos and exitos<max_exitos:
            new_sol = generate_neighbour(sol, n, k)
            obj_new = objective(new_sol, data, k, r_list, l)
            evaluations = evaluations + 1
            # print(evaluations)
            vecinos = vecinos + 1
            difference = obj_new - obj_sol
            if difference < 0 or random.random() <= math.exp(-difference/temperature):
                sol = copy.deepcopy(new_sol)
                obj_sol = copy.deepcopy(obj_new)
                if obj_sol < obj_best:
                    best_sol = copy.deepcopy(sol)
                    obj_best = copy.deepcopy(obj_sol)
                    exitos = exitos + 1
        temperature = cooling(temperature, beta)
    return best_sol




'''
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour_local_search(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour


'''
Local search algorithm
'''

def generate_virtual_neighbourhood (n, k, sol):
    neighbourhood = [[i, c] for c in range(k) for i in range(n) if sol[i] != c]
    random.shuffle(neighbourhood)
    return np.array(neighbourhood)


def local_search(data, r_list, k, sol=None):
    n = len(data)
    l = compute_lambda(data, r_list)
    if sol is None:
        sol = initial_solution(k, n)
    evaluations = 0
    i = 0
    neighbourhood = generate_virtual_neighbourhood(n, k, sol)
    objective_sol = objective(sol, data, k, r_list, l)
    while evaluations<100000 and i<len(neighbourhood):
        neighbour = generate_neighbour_local_search(sol, neighbourhood[i])
        i += 1
        # If it is a feasible neighbour
        if len(np.unique(neighbour)) == k:
            objective_neighbour = objective(neighbour, data, k, r_list, l)
            evaluations += 1
            # first neighbour that improves actual solution
            if objective_neighbour < objective_sol:
                sol = copy.deepcopy(neighbour)
                objective_sol = copy.deepcopy(objective_neighbour)
                neighbourhood = generate_virtual_neighbourhood(n, k, sol)
                i = 0
    return sol, objective_sol



'''
Búsqueda local Multiarranque
'''

def bmb(data, r_list, k):
    solutions = np.array([local_search(data, r_list, k) for i in range(10)])
    return solutions[np.argmin(solutions[:,1])][0]


def reparation(sol, n, k):
    # Comprobamos que cada cluster esté en el hijo
    for cluster in range(k):
        if cluster not in sol:
            # Si no se encuentra seleccionamos un elemento
            # aleatorio y lo asignamos a ese cluster
            sol[random.randint(0, n - 1)] = cluster
    return sol


def mutation(sol, n, k):
    # Inicio del segmento
    r = random.randint(0, n-1)
    # Tamaño del segmento
    v = int(0.1 * n)
    # Los índices que realizarán se copiarán del padre serán
    segment = np.arange(r, r+v) % n
    # Para el segmento realizamos la mutación
    sol[segment] = [random.randint(0, k - 1) for i in segment]
    # Comprobar factibilidad
    if len(np.unique(sol)) != k:
        sol[segment] = reparation(sol[segment], len(segment), k)
    return sol


'''
ILS
'''

def ils(data, r_list, k):
    n = len(data)
    sol0 = initial_solution(k, n)
    sol, obj_sol = local_search(data, r_list, k, sol0)
    for i in range(9):
        sol1 = mutation(sol, n, k)
        sol2, obj_sol2 = local_search(data, r_list, k, sol1)
        sol = sol if obj_sol<obj_sol2 else copy.deepcopy(sol2)
    return sol





data = read_file("bin/" + "ecoli" + "_set.dat")
r_matrix = read_file("bin/" + "ecoli" + "_set_const_" + "10" + ".const")
r_list = build_restrictions_list(r_matrix)
start_time = time.time()
sol = ils(data, r_list, 8)
time_sol = time.time() - start_time
print(str(time_sol))
obj_rate = objective(sol, data, 8, r_list, compute_lambda(data, r_list))
print("OBJ_RATE: " + str(obj_rate) + "\n")
c_rate = c(sol, data, 8)
inf_rate = infeasibility_total(sol, r_list)
print("C_RATE: " + str(c_rate) + "\n")
print("INF_RATE: " + str(inf_rate) + "\n")
# #
# #
#
# mu = 0.3
# final_temperature = 0.001
#
# executions = 5
# datasets = ["iris", "ecoli", "rand", "newthyroid"]
# clusters = [3, 8, 3, 3]
# restrictions = ["10", "20"]
#
#
# def run_in_parallel(d):
#     for i in range(4):
#         dataset = datasets[i]
#         k = clusters[i]
#         random.seed(d*10)
#         # Results file
#         f = open("bmb_" + dataset + '_' + str(d) + ".txt", "w")
#         data = read_file("bin/" + dataset + "_set.dat")
#         f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
#         f.write("SEED: " + str(d*10))
#         for r in restrictions:
#             r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
#             r_list = build_restrictions_list(r_matrix)
#             f.write("\n\n--------> Restriction: " + r + "\n")
#             print("Restriction: ", r)
#             start_time = time.time()
#             # sol = simulated_annealing(data, k, r_list, mu, final_temperature)
#             sol = bmb(data, r_list, k)
#             time_sol = time.time() - start_time
#             print(str(time_sol))
#             f.write("TIME: " + str(time_sol) + "\n\n")
#             obj_rate = objective(sol, data, k, r_list, compute_lambda(data, r_list))
#             print("OBJ_RATE: " + str(obj_rate) + "\n")
#             f.write("OBJ_RATE: " + str(obj_rate) + "\n")
#             c_rate = c(sol, data, k)
#             inf_rate = infeasibility_total(sol, r_list)
#             f.write("C_RATE: " + str(c_rate) + "\n")
#             print("C_RATE: " + str(c_rate) + "\n")
#             print("INF_RATE: " + str(inf_rate) + "\n")
#             f.write("INF_RATE: " + str(inf_rate) + "\n")
#
#
# from multiprocessing import Pool
#
# argument = [0,1,2,3,4]
#
#
# if __name__ == '__main__':
#     pool = Pool()
#     pool.map(run_in_parallel, argument)
#     pool.close()
#     pool.join()
#
#
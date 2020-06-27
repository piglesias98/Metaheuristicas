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
    # Calcular máximo de cada dimensión
    max = [np.max([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    # Calcular mínimo de cada dimensión
    min = [np.min([data[i][j] for i in range(data.shape[0])]) for j in range(data.shape[1])]
    # Devolver un vector de números aleatorios en el rango especificado para cada dimensión
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
def generate_firefly_from_solution(sol, data, data_normalized, k, r_list, l):
    return compute_centroid_from_solution(data_normalized, sol, k), sol, objective(sol, data, k, r_list, l)



def compute_centroid_from_solution(data, sol, k):
    dimensiones = len(data[0])
    elem_cluster = [data[np.where(sol == i)] for i in range(k)]
    centroides = [[np.mean(c[:,d]) for d in range(dimensiones)] for c in elem_cluster]
    return centroides



def initial_fireflies(n_fireflies, data, data_normalized, k):
    return [compute_initial_solution(data, data_normalized, k) for i in range(n_fireflies)]


def movement(move_from, move_to, beta, gamma, data, data_normalized, k, r_list, l, temperature):
    centroid_from = copy.deepcopy(np.array(move_from[0]))
    centroid_to = copy.deepcopy(np.array(move_to[0]))
    for c in range(k):
        r = distance.euclidean(centroid_from[c], centroid_to[c])
        suma = beta * math.exp(- gamma * r * r) * np.array(centroid_to[c] - centroid_from[c]) + temperature * np.array([random.uniform(-0.1,0.1) for i in range(centroid_from[c].shape[0])])
        centroid_from[c] = centroid_from[c] + suma

    sol = solution_from_centroids(data_normalized, centroid_from)
    # Update solution and light intensity only if it satisfies restrictions
    if len(np.unique(sol)) == k:
        move_from[0] = copy.deepcopy(centroid_from)
        move_from[1] = copy.deepcopy(sol)
        move_from[2] = objective(move_from[1], data, k, r_list, l)

    return move_from


def iefa_movement(move_from, move_to, beta, gamma, data, data_normalized, k, r_list, l, temperature):
    centroid_from = copy.deepcopy(np.array(move_from[0]))
    centroid_to = copy.deepcopy(np.array(move_to[0]))
    for c in range(k):
        r = distance.euclidean(centroid_from[c], centroid_to[c])
        suma =  np.array([random.uniform(0,1) for i in range(centroid_from[c].shape[0])]) * np.array(centroid_to[c] - centroid_from[c]) + temperature * np.array([random.uniform(-0.1,0.1) for i in range(centroid_from[c].shape[0])])
        centroid_from[c] = centroid_from[c] + suma

    sol = solution_from_centroids(data_normalized, centroid_from)
    # Update solution and light intensity only if it satisfies restrictions
    if len(np.unique(sol)) == k:
        move_from[0] = copy.deepcopy(centroid_from)
        move_from[1] = copy.deepcopy(sol)
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


'''
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour_local_search(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour


def generate_virtual_neighbourhood (n, k, sol):
    neighbourhood = [[i, c] for c in range(k) for i in range(n) if sol[i] != c]
    random.shuffle(neighbourhood)
    return np.array(neighbourhood)



'''
Local search
'''


def local_search(data, r_list, k, sol, max_evaluations):
    print("Búsqueda local")
    # n será la longitud de los datos
    n = len(data)
    l = compute_lambda(data, r_list)
    evaluations = 0
    i = 0
    neighbourhood = generate_virtual_neighbourhood(n, k, sol)
    objective_sol = objective(sol, data, k, r_list, l)
    while evaluations<max_evaluations and i<len(neighbourhood):
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
                # print("OBJ", objective_sol)
                neighbourhood = generate_virtual_neighbourhood(n, k, sol)
                i = 0
    return sol


'''
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour



def soft_local_search(data, r_list, k, sol, max_evaluations=None):
    n = len(sol)
    # Xi será el número máximo de fallos, un 10% del tamaño del cromosoma
    xi = int(0.1*n)
    # Recorreremos los elementos de forma aleatoria
    rsi = random.sample(range(n), n)
    errors = 0
    improvement = True
    i = 0
    evaluations = 0
    # Mientras que haya mejora
    # No se alcance el número máximo de errores
    # Se recorre como máximo una vez el cromosoma
    while (improvement or errors < xi) and (i < n):
        old_sol = copy.deepcopy(sol)
        improvement = False
        # Asignamos el mejor valor posible a sol[i]
        # Tiene que tratarse de una solución factible
        obj = np.array([[objective(generate_neighbour(sol, [rsi[i], ci]), data, k, r_list, l), ci]  for ci in range(k) if len(np.unique(generate_neighbour(sol, [rsi[i], ci]))) == k])
        best = np.argmin(obj[:,0])
        evaluations = evaluations + len(obj)
        # Asignar al elemento el mejor cluster
        sol[rsi[i]] = obj[best][1]
        if np.array_equal(old_sol, sol):
            errors = errors + 1
        else:
            improvement = True
        i = i + 1
    # Return sol and evaluations
    return sol, evaluations




# Inicializar to do
def fa_v1(data, r_list, k, l, n_fireflies, file=None):
    max_evaluations = 100000
    gamma = 1
    beta = 1
    temperature = 0.8
    data_normalized = normalize(data)
    fireflies = np.array(initial_fireflies(n_fireflies, data, data_normalized, k))
    evaluations = n_fireflies
    comb = list(combinations(range(n_fireflies), 2))
    fallos = 0
    best = 10000000000
    generations = 0
    max_eval_local = 100
    while evaluations < max_evaluations and fallos < n_fireflies * 10:
        random.shuffle(comb)
        for i in comb:
           if fireflies[i[0]][2] < fireflies[i[1]][2]:
                # mover luciérnaga i hacia j
                # Variar atracción con la distancia via exp(-gamma r)
                # Evaluar las nuevas soluciones y actualizar la intensidad de la luz
                fireflies[i[1]] = iefa_movement(fireflies[i[1]], fireflies[i[0]], beta, gamma, data, data_normalized, k, r_list, l ,temperature)
                evaluations = evaluations + 1
        # Memético
        generations = generations + 1
        if generations > 10:
            # Búsqueda local
            # subset = np.argpartition(population[:, 1], int(perc * chromosomes))[:int(perc * chromosomes)]
            # population[subset], new_evaluations = map(list, zip(
            #     *[soft_local_search(sol, data, k, r_list, l) for sol in population[subset]]))
            improved_sols, new_evaluations = map(list, zip(
                *[soft_local_search(data, r_list, k, f, max_eval_local) for f in fireflies[:,1]]))

            # improved_sols, new_evaluations = [soft_local_search(data, r_list, k, f, max_eval_local) for f in fireflies[:,1]]
            fireflies = np.array([generate_firefly_from_solution(i, data, data_normalized, k, r_list, l) for i in improved_sols])
            evaluations = evaluations + np.sum(new_evaluations)
            generations = 0
        if best > fireflies[np.argmin(fireflies[:, 2])][2]:
            best = fireflies[np.argmin(fireflies[:, 2])][2]
            fallos = 0
        else:
            fallos = fallos + 1
        # Cooling scheme
        # temperature = temperature * 0.997
        print(str(best)+', ' + str(np.mean(fireflies[:, 2])) + ', ' + str(evaluations) +  ', ' + str(temperature))
        file.write(str(best)+', ' + str(np.mean(fireflies[:, 2])) + ', ' + str(evaluations) +  ', ' + str(temperature) + str('\n'))
    # Ordenar las luciérnagas y buscar la más luminosa
    print('          FIIN          ')
    print(best)
    return fireflies[np.argmin(fireflies[:, 2])]


dataset = "rand"
k = 3


data = read_file("bin/" + dataset + "_set.dat")
r = "10"
r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
r_list = build_restrictions_list(r_matrix)
n_fireflies = 50
l = compute_lambda(data, r_list)
#
d = 0
f = open("fa_soft_local_search_sin_temp_" + dataset + '_' + str(d) + ".txt", "w")
data = read_file("bin/" + dataset + "_set.dat")
f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
f.write("SEED: " + str(d*10))
r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
r_list = build_restrictions_list(r_matrix)
f.write("\n\n--------> Restriction: " + r + "\n")
print("Restriction: ", r)
start_time = time.time()
sol = fa_v1(data, r_list, k, l, n_fireflies, f)
time_sol = time.time() - start_time
print(str(time_sol))
f.write("TIME: " + str(time_sol) + "\n\n")
c_rate = c(sol[1], data, k)
inf_rate = infeasibility_total(sol[1], r_list)
f.write("C_RATE: " + str(c_rate) + "\n")
print("C_RATE: " + str(c_rate) + "\n")
print("INF_RATE: " + str(inf_rate) + "\n")
f.write("INF_RATE: " + str(inf_rate) + "\n")

# datasets = ["iris", "ecoli", "rand", "newthyroid"]
# clusters = [3, 8, 3, 3]
# restrictions = ["10", "20"]

# def run_in_parallel(d):
#     dataset = datasets[d]
#     k = clusters[d]
#     r = '10'
#     random.seed((d+1)*10)
#     # Results file
#     f = open("fa_random_10_097_" + dataset + '_' + str(d) + ".txt", "w")
#     data = read_file("bin/" + dataset + "_set.dat")
#     f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
#     f.write("SEED: " + str(d*10))
#     r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
#     r_list = build_restrictions_list(r_matrix)
#     f.write("\n\n--------> Restriction: " + r + "\n")
#     print("Restriction: ", r)
#     l = compute_lambda(data, r_list)
#     start_time = time.time()
#     sol = fa_v1(data, r_list, k, l, n_fireflies, f)
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
# #

def initial_solution(k, n):
    # Se genera un array de tamaño n con números entre 0 y k,
    # donde k es el número de clústers
    initial_sol = np.random.randint(0, k, size=n)
    # Si la solución no es factible llamamos de nuevo a la función
    if len(np.unique(initial_sol)) != k:
        initial_solution(k, n)
    else:
        return initial_sol


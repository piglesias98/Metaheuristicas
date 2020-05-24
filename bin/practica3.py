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

'''
SIMULATED ANNEALING
'''

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


'''

Enfriamiento simulado

'''



def cooling(temperature, beta):
    return temperature/(1+beta*temperature)


def cooling_geometric(temperature):
    return temperature*0.9


def simulated_annealing(data, k, r_list, mu, final_temperature, sol=None, f=None):
    # Calculamos lambda ya que no varía en
    l = compute_lambda(data, r_list)
    n = len(data)
    # La solución inicial la generamos de forma aleatoria
    if sol is None:
        sol = initial_solution(k, n)
    obj_sol = objective(sol, data, k, r_list, l)
    # Inicializamos a 1 el número de evaluaciones
    evaluations = 1
    # Inicializamos la mejor solución
    best_sol = copy.deepcopy(sol)
    obj_best = copy.deepcopy(obj_sol)
    # Calculamos la temperatura inicial en función de mu y phi
    temperature = initial_temperature(mu, obj_sol)
    # L(T) Inicializamos las condiciones de parada del bucle interno
    max_vecinos = 10 * n
    max_exitos = 0.1 * max_vecinos
    # Calculamos beta para el enfriamiento de Cauchy ya que no varía en
    m = 100000/max_vecinos
    beta = compute_beta(temperature, final_temperature, m)
    # Bucle externo: mientras no se supere el número de evaluaciones
    # y la temperatura sea menor que la temperatura final
    while temperature > final_temperature and evaluations < 10000:
        vecinos = 0
        exitos = 0
        # Bucle interno: condiciones L(T)
        while vecinos<max_vecinos and exitos<max_exitos:
            # Generamos un vecino de forma aleatoria con el operador de vecino
            new_sol = generate_neighbour(sol, n, k)
            obj_new = objective(new_sol, data, k, r_list, l)
            # Aumentamos el número de evaluaciones
            evaluations = evaluations + 1
            # Aumentamos el número de vecinos generados
            vecinos = vecinos + 1
            # Calculamos la diferencia de función objetivo
            difference = obj_new - obj_sol
            # Si la diferencia es negativa o el número aleatorio menor que la expresión e^(-diferencia/temperatura)
            if difference < 0 or random.random() <= math.exp(-difference/temperature):
                # Actualizamos la solución actual
                sol = copy.deepcopy(new_sol)
                obj_sol = copy.deepcopy(obj_new)
                # Aumentamos el número de éxitos
                exitos = exitos + 1
                if obj_sol < obj_best:
                    # Si es mejor que la mejor solución, ésta se actualiza
                    best_sol = copy.deepcopy(sol)
                    obj_best = copy.deepcopy(obj_sol)
        f.write(str(evaluations) + "," + str(vecinos) + "," + str(exitos) + "," + str(temperature) + "," + str(obj_best) + "\n")
        print(str(evaluations) + "," + str(vecinos) + "," + str(exitos) + "," + str(temperature) + "," + str(obj_best))
        # Enfriamos la temperatura con el esquema de enfriamiento
        temperature = cooling_geometric(temperature)
        # temperature = cooling(temperature, beta)
    return best_sol, obj_best




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
    print("Búsqueda local")
    # n será la longitud de los datos
    n = len(data)
    l = compute_lambda(data, r_list)
    if sol is None:
        sol = initial_solution(k, n)
    evaluations = 0
    i = 0
    neighbourhood = generate_virtual_neighbourhood(n, k, sol)
    objective_sol = objective(sol, data, k, r_list, l)
    while evaluations<10000 and i<len(neighbourhood):
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
    return sol, objective_sol



'''
Búsqueda local Multiarranque
'''

def bmb(data, r_list, k):
    # Se genera un array con los resultados de 10 ejecuciones de búsqueda local
    solutions = np.array([local_search(data, r_list, k) for i in range(10)])
    # Se devuelve la solución con la mejor función objetivo
    return solutions[np.argmin(solutions[:,1])][0]




'''
ILS
'''

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
    # Calculamos el segmento
    segment = np.arange(r, r+v) % n
    # Para el segmento realizamos la mutación
    sol[segment] = [random.randint(0, k - 1) for i in segment]
    # Comprobar factibilidad
    if len(np.unique(sol)) != k:
        # Si no es factible se realiza una reparación en el segmento,
        # para conservar la aleatoriedad
        sol[segment] = reparation(sol[segment], len(segment), k)
    return sol



def ils(data, r_list, k, f):
    n = len(data)
    # Generamos la solución inicial de manera aleatoria
    sol0 = initial_solution(k, n)
    # Aplicamos la búsqueda local
    sol, obj_sol = local_search(data, r_list, k, sol0)
    # En total realizaremos 100000 evaluaciones, 10000 cada vez
    for i in range(9):
        print("--------------------------", i)
        # Aplicamos el operador de mutación
        sol1 = mutation(sol, n, k)
        # Aplicamos la búsqueda local
        sol2, obj_sol2 = local_search(data, r_list, k, sol1)
        # Nos quedamos con la mejor solución
        if obj_sol2 < obj_sol:
            print("Mejora")
            sol = copy.deepcopy(sol2)
            f.write(str(obj_sol2)+ " Mejora " + "\n")
        else:
            f.write(str(obj_sol) + " No Mejora " + "\n")
            print('No mejora')
        # sol = sol if obj_sol<obj_sol2 else copy.deepcopy(sol2)
    return sol


def ils_es(data, r_list, k, mu, final_temperature, f):
    n = len(data)
    sol0 = initial_solution(k, n)
    sol, obj_sol = simulated_annealing(data, k, r_list, mu, final_temperature, sol0)
    for i in range(9):
        print(i)
        sol1 = mutation(sol, n, k)
        sol2, obj_sol2 = simulated_annealing(data, k, r_list, mu, final_temperature, sol1)
        if obj_sol2 < obj_sol:
            print("Mejora")
            sol = copy.deepcopy(sol2)
            f.write(str(obj_sol2)+ " Mejora " + "\n")
        else:
            f.write(str(obj_sol) + " No Mejora " + "\n")
            print('No mejora')
        # sol = sol if obj_sol<obj_sol2 else copy.deepcopy(sol2)
    return sol





mu = 0.3
final_temperature = 0.001

executions = 3
datasets = ["iris", "ecoli", "rand", "newthyroid"]
clusters = [3, 8, 3, 3]
restrictions = ["10", "20"]


dataset= "ecoli"
k = 8
f = open("convergencia_simulated_annealing_geometric_updated_10.txt", "w")
data = read_file("bin/" + dataset + "_set.dat")
f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
r = "10"
r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
r_list = build_restrictions_list(r_matrix)
f.write("\n\n--------> Restriction: " + r + "\n")
print("Restriction: ", r)
start_time = time.time()
sol, obj = simulated_annealing(data, k, r_list, mu, final_temperature, sol = None, f =f)
# sol = ils(data, r_list, k, f)
time_sol = time.time() - start_time
print(str(time_sol))
f.write("TIME: " + str(time_sol) + "\n\n")
obj_rate = objective(sol, data, k, r_list, compute_lambda(data, r_list))
print("OBJ_RATE: " + str(obj_rate) + "\n")
f.write("OBJ_RATE: " + str(obj_rate) + "\n")
c_rate = c(sol, data, k)
inf_rate = infeasibility_total(sol, r_list)
f.write("C_RATE: " + str(c_rate) + "\n")
print("C_RATE: " + str(c_rate) + "\n")
print("INF_RATE: " + str(inf_rate) + "\n")
f.write("INF_RATE: " + str(inf_rate) + "\n")

# def run_in_parallel(d):
#     for i in range(3):
#         dataset = datasets[d]
#         k = clusters[d]
#         random.seed(i*10)
#         # Results file
#         f = open("bmb_10mil_" + dataset + '_' + str(i) + ".txt", "w")
#         data = read_file("bin/" + dataset + "_set.dat")
#         f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
#         f.write("SEED: " + str(i*10))
#         for r in restrictions:
#             r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
#             r_list = build_restrictions_list(r_matrix)
#             f.write("\n\n--------> Restriction: " + r + "\n")
#             print("Restriction: ", r)
#             start_time = time.time()
#             # sol = simulated_annealing(data, k, r_list, mu, final_temperature)
#             # sol = ils(data, r_list, k)
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
# argument = [0,1,2,3]
#
#
# if __name__ == '__main__':
#     pool = Pool()
#     pool.map(run_in_parallel, argument)
#     pool.close()
#     pool.join()
#

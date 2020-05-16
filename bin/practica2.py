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
Generate neighbour from sol changing values from to_change
'''


def generate_neighbour(sol, to_change):
    neighbour = np.copy(sol)
    neighbour[to_change[0]]=to_change[1]
    return neighbour


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


def initial_population(k,n, chromosomes):
    population = []
    for i in range(chromosomes):
        population.append(initial_solution(k, n))
    return population


def binary_tournament_agg(population, chromosomes):
    parents = []
    # Para cada cromosoma de la población
    for i in range(chromosomes):
        # Escogemos dos individuos aleatoriamente
        individual_1 = random.choice(population)
        individual_2 = random.choice(population)
        # Si la función objetivo del individuo 1 es mejor,
        # escogeremos este individuo
        if individual_1[1] < individual_2[1]:
            parents.append(individual_1)
        # Si es al contrario, escogeremos el individuo 2
        else:
            parents.append(individual_2)
    return np.array(parents)


def binary_tournament_age(population):
    parents = []
    for i in range(2):
        individual_1 = random.choice(population)
        individual_2 = random.choice(population)
        if individual_1[1] < individual_2[1]:
            parents.append(individual_1)
        else:
            parents.append(individual_2)
    return np.array(parents)


def uniform_crossover(individual_1, individual_2, n):
    # El hijo copiará en primer lugar todos
    # los genes del primer padre
    child = np.copy(individual_1)
    # Se generan n/2 números aleatorios distintos
    # en el rango {0,1, … , n-1}.
    genes = np.random.randint(0, n, int(n/2))
    # Asignamos al hijo los genes de estos números
    # generados del segundo padre
    for i in genes:
        child[i] = individual_2[i]
    return child


def two_points_crossover(individual_1, individual_2, n):
    # Inicio del segmento
    r = random.randint(0, n-1)
    # Tamaño del segmento
    v = random.randint(0, n-1)
    # Los índices que realizarán se copiarán del padre serán
    two_points = np.arange(r, r+v) % n
    # Por lo tanto los que realizan cruce uniforme
    uniform = np.ones(len(individual_1), dtype=bool)
    uniform[two_points] = False
    # Copiamos primero el padre
    child = copy.deepcopy(individual_1)
    # Realizamos el cruce uniforme con los que caen fuera del intervalo
    child[uniform] = uniform_crossover(individual_1[uniform], individual_2[uniform], n)
    return child


def mutation(sol, n, k):
    # El gen será una posición aleatoria del cromosoma
    gen = random.randint(0, n-1)
    # El valor será uno distinto al actual
    cluster = random.randint(0, k-1)
    old_cluster = sol[gen]
    sol[gen] = cluster
    # Nos aseguramos de que sea una solución factible
    while len(np.unique(sol)) != k or old_cluster == cluster:
        sol[gen] = old_cluster
        gen = random.randint(0, n-1)
        old_cluster = sol[gen]
        cluster = random.randint(0, k - 1)
        sol[gen] = cluster
    return sol


def evaluate_initial_population(population, data, k, r_list, l):
    evaluations = [[population[i], objective(population[i], data, k, r_list, l)] for i in range(len(population))]
    return evaluations

def reparation(child, n, k):
    repaired = np.copy(child)
    # Comprobamos que cada cluster esté en el hijo
    for cluster in range(k):
        if cluster not in child:
            # Si no se encuentra seleccionamos un elemento
            # aleatorio y lo asignamos a ese cluster
            repaired[random.randint(0, n - 1)] = cluster
    return repaired


# chromosomes = 50
def agg(data, r_list, k, chromosomes, crossover, prob_crossover, prob_mutation):
    n = len(data)
    n_crossovers = int(chromosomes * prob_crossover)
    n_mutations = int(n * chromosomes * prob_mutation)
    l = compute_lambda(data, r_list)
    # Initialize P(0)
    population = initial_population(k, n, chromosomes)
    # Evaluate P(0)
    population = evaluate_initial_population(population, data, k, r_list, l)
    evaluations = chromosomes
    population = np.array(population)
    best = population[np.argmin(population[:, 1])]
    while evaluations < 100000:
        print(evaluations)
        # Selection
        parents = binary_tournament_agg(population, chromosomes)
        # Crossover
        intermediate = np.copy(parents)
        i = 0
        # Only chromosomes * prob_crossover
        parents = parents[:n_crossovers]
        for j, q in zip(parents[0::2], parents[1::2]):
            for w in range(2):
                child = crossover(j[0], q[0], n)
                if len(np.unique(child)) != k:
                    # Reparation
                    child = reparation(child, n, k)
                intermediate[i][0] = child
                intermediate[i][1] = objective(child, data, k, r_list, l)
                i = i + 1
                evaluations = evaluations + 1
        # Mutation
        children = np.copy(intermediate)
        # Aplicamos la mutación
        for m in range(n_mutations):
            crom = random.randint(0, chromosomes-1)
            children[crom][0] = mutation(children[crom][0], n, k)
            # Calculamos la función objetivo
            children[crom][1] = objective(children[crom][0], data, k, r_list, l)
            evaluations = evaluations + 1
        # Insert offspring into the population
        # Los hijos reemplazan directamente a la población
        population = np.copy(children)
        # Calculamos la nueva mejor solución
        best_new = population[np.argmin(population[:, 1])]
        # Si la mejor solución anterior no ha sobrevivido
        if best[1] not in population[:, 1]:
            # Sustiuimos la peor solución con la mejor solución
            # de la población anterior
            population[np.argmax(population[:, 1])] = best
        # Reemplazamos la mejor solución anterior
        best = np.copy(best_new)

    return population


def age(data, r_list, k, chromosomes, crossover, prob_mutation):
    n = len(data)
    l = compute_lambda(data, r_list)
    # Initialize P(0)
    population = initial_population(k, n, chromosomes)
    # Evaluate P(0)
    population = evaluate_initial_population(population, data, k, r_list, l)
    evaluations = chromosomes
    population = np.array(population)
    while evaluations < 100000:
        print(evaluations)
        # Selection
        parents = binary_tournament_age(population)
        # Crossover
        intermediate = []
        for w in range(2):
            child = crossover(parents[0][0], parents[1][0], n)
            if len(np.unique(child)) != k:
                child  = reparation(child, n, k)
            intermediate.append(child)
        # Mutation
        children = np.copy(intermediate)
        for c in range(len(children)):
            # Con probabilidad prob_mutation,
            # mutamos un cromosoma
            if random.random() < prob_mutation * n:
                children[c][0] = mutation(child, n, k)
                children[c][1] = objective(child, data, k, r_list, l)
                evaluations = evaluations + 1
        # Insert offspring into the population
        # Worst two solutions
        # Calculamos las dos peores soluciones
        worst = np.argpartition(population[:, 1], -2)[-2:]
        # De las cuatro soluciones (dos peores + dos hijos),
        # calculamos las dos mejores para reemplazar a las dos peores
        possible = np.concatenate((population[worst], children))
        best = np.argpartition(possible[:,1], 2)[:2]
        # Las dos peores son reemplazadas por las dos mejores
        population[worst] = possible[best]
    return population



def soft_local_search(sol, data, k, r_list, l):
    n = len(sol[0])
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
        print(evaluations)
        old_sol = copy.deepcopy(sol)
        improvement = False
        # Asignamos el mejor valor posible a sol[i]
        # Tiene que tratarse de una solución factible
        obj = np.array([[objective(generate_neighbour(sol[0], [rsi[i], ci]), data, k, r_list, l), ci]  for ci in range(k) if len(np.unique(generate_neighbour(sol[0], [rsi[i], ci]))) == k])
        best = np.argmin(obj[:,0])
        evaluations = evaluations + len(obj)
        # Assign al elemento el mejor cluster
        sol[0][rsi[i]] = obj[best][1]
        sol[1] = obj[best][0]
        if np.array_equal(old_sol[0], sol[0]):
            errors = errors + 1
        else:
            improvement = True
        i = i + 1
    # Return sol and evaluations
    return sol, evaluations


def memetic (n_generations, perc, pick_best, data, r_list, k, chromosomes, crossover, prob_crossover, prob_mutation):
    n = len(data)
    n_crossovers = int(chromosomes * prob_crossover)
    n_mutations = int(n * chromosomes * prob_mutation)
    l = compute_lambda(data, r_list)
    # Initialize P(0)
    population = initial_population(k, n, chromosomes)
    # Evaluate P(0)
    population = evaluate_initial_population(population, data, k, r_list, l)
    evaluations = chromosomes
    population = np.array(population)
    best = population[np.argmin(population[:, 1])]
    generations = 0
    while evaluations < 100000:
        print(evaluations)
        # Selection
        parents = binary_tournament_agg(population, chromosomes)
        # Crossover
        intermediate = np.copy(parents)
        i = 0
        # Only chromosomes * prob_crossover
        parents = parents[:n_crossovers]
        for j, q in zip(parents[0::2], parents[1::2]):
            for w in range(2):
                child = crossover(j[0], q[0], n)
                if len(np.unique(child)) != k:
                    # Reparation
                    child = reparation(child, n, k)
                intermediate[i][0] = child
                intermediate[i][1] = objective(child, data, k, r_list, l)
                i = i + 1
                evaluations = evaluations + 1
        # Mutation
        children = np.copy(intermediate)
        # Aplicamos la mutación
        for m in range(n_mutations):
            crom = random.randint(0, chromosomes-1)
            children[crom][0] = mutation(children[crom][0], n, k)
            # Calculamos la función objetivo
            children[crom][1] = objective(children[crom][0], data, k, r_list, l)
            evaluations = evaluations + 1
        # Insert offspring into the population
        population = np.copy(children)
        best_new = population[np.argmin(population[:, 1])]
        if best[1] not in population[:, 1]:
            population[np.argmax(population[:, 1])] = best
        best = np.copy(best_new)
        generations = generations + 1
        # Apply local search
        if generations == n_generations:
            print('----Reach '+ str(n_generations) + ' generations!-----')
            generations = 0
            if pick_best:
                subset = np.argpartition(population[:, 1], int(perc * chromosomes))[:int(perc * chromosomes)]
            else:
                if perc == 1:
                    subset = np.arange(chromosomes)
                else:
                    subset = random.sample(range(chromosomes), int(perc*chromosomes))
            population[subset], new_evaluations = map(list, zip(*[soft_local_search(sol, data, k, r_list, l) for sol in population[subset]]))
            evaluations = evaluations + np.sum(new_evaluations)
    return population



executions = 5
datasets = ["iris", "ecoli", "rand", "newthyroid"]
clusters = [3, 8, 3, 3]
restrictions = ["10", "20"]


def run_in_parallel(d):
    for i in range(4):
        dataset = datasets[i]
        k = clusters[i]
        random.seed(d*10)
        # Results file
        f = open("am_10_0.1_v18_" + dataset + '_' + str(d) + ".txt", "w")
        data = read_file("bin/" + dataset + "_set.dat")
        f.write("\n\n------------------------------------  " + dataset + "  ------------------------------------\n")
        f.write("SEED: " + str(d*10))
        for r in restrictions:
            r_matrix = read_file("bin/" + dataset + "_set_const_" + r + ".const")
            r_list = build_restrictions_list(r_matrix)
            f.write("\n\n--------> Restriction: " + r + "\n")
            print("Restriction: ", r)
            start_time = time.time()
            # population = agg(data, r_list, clusters[d], 50, two_points_crossover, 0.7, 0.001)
            # population = age(data, r_list, k, 50, two_points_crossover, 0.001)
            population = memetic(n_generations=10, perc=0.1, pick_best=False, data=data, r_list=r_list, k=k, chromosomes=50, crossover=uniform_crossover, prob_crossover=0.7, prob_mutation=0.001)
            population  = np.array(population)
            time_sol = time.time() - start_time
            print(str(time_sol))
            f.write("TIME: " + str(time_sol) + "\n\n")
            p = population[np.argmin(population[:, 1])]
            print("OBJ_RATE: " + str(p[1]) + "\n")
            f.write("OBJ_RATE: " + str(p[1]) + "\n")
            c_rate = c(p[0], data, k)
            inf_rate = infeasibility_total(p[0], r_list)
            f.write("C_RATE: " + str(c_rate) + "\n")
            print("C_RATE: " + str(c_rate) + "\n")
            print("INF_RATE: " + str(inf_rate) + "\n")
            f.write("INF_RATE: " + str(inf_rate) + "\n")


from multiprocessing import Pool

argument = [0,1,2,3,4]


if __name__ == '__main__':
    pool = Pool()
    pool.map(run_in_parallel, argument)
    pool.close()
    pool.join()



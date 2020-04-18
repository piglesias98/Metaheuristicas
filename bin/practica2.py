import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time

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
    initial_sol = np.random.randint(0, k, size=n)
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


def initial_population(k,n, chromosomes):
    population = []
    for i in range(chromosomes):
        population.append(initial_solution(k, n))
    return population


def binary_tournament_agg(population, chromosomes):
    parents = []
    for i in range(chromosomes):
        individual_1 = random.choice(population)
        individual_2 = random.choice(population)
        if individual_1[1] > individual_2[1]:
            parents.append(individual_1)
        else:
            parents.append(individual_2)
    return np.array(parents)


def uniform_crossover(individual_1, individual_2, n):
    child = np.copy(individual_1)
    genes = np.random.randint(0, n, int(n/2))
    for i in genes:
        child[i] = individual_2[i]
    return child


def two_points_crossover(individual_1, individual_2, n):
    r = random.randint(0, n-1)
    v = random.randint(0, n-1)
    child = uniform_crossover(individual_1, individual_2, n)
    child[r, ((r+v) % n) - 1] = individual_1[r, ((r+v) % n) - 1]
    return child


def mutation(sol, n, k):
    mutated = np.copy(sol)
    mutated[random.randint(0, n-1)] = random.randint(0, k-1)
    return mutated


def evaluate_initial_population(population, data, k, r_list, l):
    evaluations = [[population[i], objective(population[i], data, k, r_list, l)] for i in range(len(population))]
    return evaluations


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
        # Only chromosomes * prob_crossover
        parents = parents[:n_crossovers]
        i = 0
        for j, q in zip(parents[0::2], parents[1::2]):
            for w in range(2):
                child = crossover(j[0], q[0], n)
                if len(np.unique(child)) != k:
                    # Reparation
                    for cluster in range(k):
                        if cluster not in child:
                            child[random.randint(0, n-1)] = cluster
                intermediate[i][0] = child
                intermediate[i][1] = objective(child, data, k, r_list, l)
                evaluations = evaluations + 1
                i = i + 1
            # Mutation
        children = np.copy(intermediate)
        for m in range(n_mutations):
            children[m][0] = mutation(children[m][0], n, k)
            if len(np.unique(children[m][0])) != k:
                # Reparation
                for cluster in range(k):
                    if cluster not in children[m][0]:
                        children[m][0][random.randint(0, n-1)] = cluster
            children[m][1] = objective(children[m][0], data, k, r_list, l)
            evaluations = evaluations + 1
        # Insert offspring into the population
        population = np.copy(children)
        best_new = population[np.argmin(population[:, 1])]
        if best[1] not in population[:, 1]:
            population[np.argmax(population[:, 1])] = best
        best = np.copy(best_new)

    return population


'''
def memetic(data, r_list, k, seed, xi):
    n = len(data)
    l = compute_lambda(data, r_list)
    sol = initial_solution(k, n, seed)
    errors = 0
    improvement = True
    i = 0
    while (improvement or errors < xi) and i<n:
        improvement = False
        
'''

data = read_file("bin/iris_set.dat")
r_matrix = read_file("bin/iris_set_const_10.const")
r_list = build_restrictions_list(r_matrix)
start_time = time.time()
sol = agg(data, r_list, 3, 50, uniform_crossover, 0.7, 0.001)
time_sol = time.time() - start_time
print(sol[0][0])
print(sol[0][1])

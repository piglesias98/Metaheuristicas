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
        if individual_1[1] < individual_2[1]:
            parents.append(individual_1)
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
    child = np.copy(individual_1)
    genes = np.random.randint(0, n, int(n/2))
    for i in genes:
        child[i] = individual_2[i]
    return child


def two_points_crossover(individual_1, individual_2, n):
    r = random.randint(0, n-1)
    v = random.randint(0, n-1)
    child = uniform_crossover(individual_1, individual_2, n)
    child[np.arange(r, r+v) % n] = individual_1[np.arange(r, r+v) % n]
    return child


def mutation(sol, n, k):
    mutated = np.copy(sol)
    mutated[random.randint(0, n-1)] = random.randint(0, k-1)
    return mutated


def evaluate_initial_population(population, data, k, r_list, l):
    evaluations = [[population[i], objective(population[i], data, k, r_list, l)] for i in range(len(population))]
    return evaluations

def reparation(child, n, k):
    # Reparation
    repaired = np.copy(child)
    for cluster in range(k):
        if cluster not in child:
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
        children = []
        for child in intermediate:
            if random.random() < prob_mutation * n:
                child= mutation(child, n, k)
                if len(np.unique(child)) != k:
                    # Reparation
                    child  = reparation(child, n, k)
            children.append([child, objective(child, data, k, r_list, l)])
            evaluations = evaluations + 1
        # Insert offspring into the population
        # Worst two solutions
        worst = np.argpartition(population[:, 1], -2)[-2:]
        possible = np.concatenate((population[worst], children))
        best = np.argpartition(possible[:,1], 2)[:2]
        population[worst] = possible[best]
    return population



def soft_local_search(sol, data, k, r_list, l):
    n = len(sol[0])
    xi = int(0.1*n)
    rsi = random.sample(range(n), n)
    errors = 0
    improvement = True
    i = 0
    while (improvement or errors < xi) and (i < n):
        old_sol = copy.deepcopy(sol)
        improvement = False
        # Assign best possible value to sol[i]objective(sol, data, k, r_list, l)
        obj = np.array([[objective(generate_neighbour(sol[0], [rsi[i], ci]), data, k, r_list, l), ci]  for ci in range(k) if len(np.unique(generate_neighbour(sol[0], [rsi[i], ci]))) == k])
        best = np.argmin(obj[:,0])
        # Assign the element to the best cluster
        sol[0][rsi[i]] = obj[best][1]
        sol[1] = obj[best][0]
        if np.array_equal(old_sol[0], sol[0]):
            errors = errors + 1
        else:
            improvement = True
        i = i + 1
    # Return sol and evaluations
    return sol, i*k


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
                            child[random.randint(0, n - 1)] = cluster
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
                        children[m][0][random.randint(0, n - 1)] = cluster
            children[m][1] = objective(children[m][0], data, k, r_list, l)
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
                    subset = np.random.randint(0, n, int(perc * chromosomes))
            population, new_evaluations = map(list, zip(*[soft_local_search(sol, data, k, r_list, l) for sol in population[subset]]))
            evaluations = evaluations + np.sum(new_evaluations)
    return population


executions = 5
datasets = ["iris", "ecoli", "rand", "newthyroid"]
clusters = [3, 8, 3, 3]
restrictions = ["10", "20"]



# # sol = initial_solution(8, len(data))
# # sol2   = soft_local_search(sol, 8, r_list, compute_lambda(data, r_list))
# start_time = time.time()
# population = memetic(n_generations= 10, perc= 1, pick_best= False, data = data,
#                      r_list = r_list, k = 3, chromosomes = 50, crossover = uniform_crossover,
#                      prob_crossover = 0.7, prob_mutation = 0.001)
# time_sol = time.time() - start_time
# for p in population:
#     print("SOL: " + str(p[0]) + "\n")
#     print("OBJ_RATE: " + str(p[1]) + "\n")
#     c_rate = c(p[0], data, 3)
#     inf_rate = infeasibility_total(p[0], r_list)
#     print("C_RATE: " + str(c_rate) + "\n")
#     print("INF_RATE: " + str(inf_rate) + "\n")
#
# def results():
#     for d in range(len(datasets)):
#         data = read_file("bin/"+datasets[d]+"_set.dat")
#         f.write("\n\n------------------------------------  "+datasets[d]+"  ------------------------------------\n")
#         print(datasets[d])
#         for r in restrictions:
#             r_matrix = read_file("bin/"+datasets[d]+"_set_const_"+r+".const")
#             r_list = build_restrictions_list(r_matrix)
#             f.write("\n\n--------> Restriction: " + r + "\n")
#             print("Restriction: ", r)
#             for i in range(executions):
#                 f.write("--EXECUTION: " + str(i) + "\n")
#                 print("Execution: ", i)
#                 start_time = time.time()
#                 # population = agg(data, r_list, clusters[d], 50, two_points_crossover, 0.7, 0.001)
#                 population = age(data, r_list, clusters[d], 50, uniform_crossover, 0.001)
#                 time_sol = time.time() - start_time
#                 f.write("TIME: " + str(time_sol) + "\n\n")
#                 for p in population:
#                     f.write("SOL: " + str(p[0]) + "\n")
#                     f.write("OBJ_RATE: " + str(p[1]) + "\n")
#                     c_rate = c(p[0], data, clusters[d])
#                     inf_rate = infeasibility_total(p[0], r_list)
#                     f.write("C_RATE: " + str(c_rate) + "\n")
#                     f.write("INF_RATE: " + str(inf_rate) + "\n")
#
#
# results()


def run_in_parallel(d):
    dataset = 'rand'
    k = 3
    random.seed(d*10)
    # Results file
    f = open("am_1_1_" + dataset + '_' + str(d) + ".txt", "w")
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
        # population = age(data, r_list, clusters[d], 50, two_points_crossover, 0.001)
        population = memetic(n_generations=10, perc=1, pick_best=False, data=data, r_list=r_list, k=k, chromosomes=50, crossover=uniform_crossover, prob_crossover=0.7, prob_mutation=0.001)
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
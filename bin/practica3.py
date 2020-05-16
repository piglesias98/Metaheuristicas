import numpy as np
import random
from scipy.spatial import distance
from itertools import combinations
import time
import copy

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



def simulated_annealing(data, k, initial_temperature, final_temperature):
    n = len(data)
    temperature = initial_temperature
    sol = initial_solution(k, n)
    while temperature < final_temperature:
        neighbourhood = generate_neighbourhood(temperature)
        for neighbour in neighbourhood:
            new_sol = generate_neighbour(sol)
            difference = objective(new_sol) - objective(sol)
            if difference < 0 or random.random() <= exp(-difference/temperature):
                sol = new_sol
                if obj(sol)<obj(best):
                    best = sol
        temperature = cooling(temperature)
    return best

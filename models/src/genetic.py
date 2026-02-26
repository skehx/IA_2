from .load_data import load_data
import pandas as np
import random

POPULATION_SIZE = 100
LOWER_BOUND     = -5000
UPPER_BOUND     = 5000
GENERATIONS     = 50
MUTATION_RATE   = 0.2
TOURNAMENT_SIZE = 3
ELITE_SIZE      = 1  
N_GENES         = 5


def fitness_function(individual, X, Y):
  
    weights = np.array(individual[:4])
    bias    = individual[4]
    y_pred  = X @ weights + bias
    mse     = np.mean((Y - y_pred) ** 2)
    return 1 / (1 + mse)


def create_population(pop_size, lower_bound, upper_bound):
    return np.random.uniform(lower_bound, upper_bound, (pop_size, 2))


def tournament_selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        contestants = np.random.choice(len(population), tournament_size, replace=False)
        best_contestant = contestants[np.argmax(fitnesses[contestants])]
        selected.append(population[best_contestant])
    return np.array(selected)


# Cruzamiento
def arithmetic_crossover(parent1, parent2):
    alpha = np.random.random()
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2


# Mutación
def mutate(individual, mutation_rate, lower_bound, upper_bound):
    if np.random.random() < mutation_rate:
        individual += np.random.normal(-2, 2, individual.shape)
        individual = np.clip(individual, lower_bound, upper_bound)
    return individual



import random
import numpy as np


def selection_chances(fitness_fn, population):
    pass


def reproduce(x, y):
    pass


def mutate(child, gene_pool):
    pass


def argmax(population, key):
    return population[np.argmax([key(x) for x in population])]


def genetic_algorithm(population, fitness_fn, gene_pool=[0, 1], f_thres=None, ngen=1000, pmut=0.1):
    for i in range(ngen):
        new_population = []
        random_selection = selection_chances(fitness_fn, population)
        for j in range(len(population)):
            x = random_selection()
            y = random_selection()
            child = reproduce(x, y)
            if random.uniform(0, 1) < pmut:
                child = mutate(child, gene_pool)
            new_population.append(child)

        population = new_population

        if f_thres:
            fittest_individual = argmax(population, key=fitness_fn)
            if fitness_fn(fittest_individual) >= f_thres:
                return fittest_individual

    return argmax(population, key=fitness_fn)

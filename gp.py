"""
Genetic programming loop for symbolic regression.

Evolves a population of expression trees using tournament selection,
subtree crossover, subtree mutation, and reproduction, and logs
per-generation statistics to a CSV file. All settings are read from a
JSON config file (e.g. dataset1_variables.json).

Run from the command line:
    python gp.py dataset1_variables.json
"""

import json
import random 
import numpy as np 

from fitness import fitness
from data import load_dataset, split_data
from tree import crossover, mutate, depth, size, to_string, generate_population


def tournament_select(population, scores, k):
    """
    Select one parent using tournament selection.

    Picks k distinct trees at random and returns the one with the lowest
    (best) fitness score. Tournament selection is one of the standard
    selection methods described in Koza's tutorial; Luke & Panait (2006)
    used a tournament size of 7.

    Parameters
    ----------
    population : list of Node
        Current population of trees.
    scores : list of float
        Fitness score of each tree, in the same order as population.
        Lower is better.
    k : int
        Tournament size (number of trees competing).

    Returns
    -------
    Node
        The winning tree.
    """
    contestants = random.sample(range(len(population)), k)

    winner = contestants[0]
    for i in contestants[1:]:
        if scores[i] < scores[winner]:
            winner = i
    
    return population[winner]

def make_children(population, scores, variables):
    """
    Create one or two children using crossover, mutation, or reproduction.

    The operation is chosen at random according to the rates in the
    config: crossover with probability crossover_rate, mutation with
    probability mutation_rate, and reproduction (an unchanged copy)
    otherwise. Default rates follow Koza's tutorial (about 90% crossover,
    1% mutation, the rest reproduction).

    Crossover children deeper than depth_limit are replaced by a copy of
    their parent (depth limiting, Luke & Panait 2006).

    Parameters
    ----------
    population : list of Node
        Current population of trees.
    scores : list of float
        Fitness score of each tree, in the same order as population.
    variables : dict
        Run settings loaded from the config file.

    Returns
    -------
    list of Node
        Two children for crossover, or one child for mutation or
        reproduction.
    """
    k = variables["tournament_size"]
    depth_limit = variables["depth_limit"]

    r = random.random()

    if r < variables["crossover_rate"]:
        parent1 = tournament_select(population, scores, k)
        parent2 = tournament_select(population, scores, k)
        child1, child2 = crossover(parent1, parent2)
        
        # Depth limit per Luke & Panait: too deep children
        # are replaced by a copy of their parent
        if depth(child1) > depth_limit:
            child1 = parent1
        if depth(child2) > depth_limit:
            child2 = parent2
        return [child1, child2]
    
    elif r < variables["crossover_rate"] + variables["mutation_rate"]:
        parent = tournament_select(population, scores, k)
        return [mutate(parent, depth_limit, variables["num_variables"], 
                       variables["integer_constants"])]
    else:
        parent = tournament_select(population, scores, k)
        return [parent]

def next_gen(population, scores, variables):
    """
    Build the next generation of the population.

    Repeatedly creates children with make_children until the new
    population reaches population_size, then trims any extra child
    (crossover can produce one more tree than is needed).

    Parameters
    ----------
    population : list of Node
        Current population of trees.
    scores : list of float
        Fitness score of each tree, in the same order as population.
    variables : dict
        Run settings loaded from the config file.

    Returns
    -------
    list of Node
        The new population, exactly population_size trees long.
    """
    new_population = []
    while len(new_population) < variables["population_size"]:
        new_population.extend(make_children(population, scores, variables))
    return new_population[:variables["population_size"]]

def diversity(population):
    """
    Measure population diversity as the fraction of unique trees.

    Two trees count as the same if their string forms are identical.
    A value of 1.0 means every tree is different; values near 0 mean
    the population has converged on a few trees (loss of diversity).

    Parameters
    ----------
    population : list of Node
        Current population of trees.

    Returns
    -------
    float
        Number of unique trees divided by population size, between 0 and 1.
    """
    unique = len(set(to_string(tree) for tree in population))
    return unique / len(population)

def run(variables_path):
    # Initialize with variables and splitting dataset
    variables = json.load(open(variables.path))
    random.seed(variables["seed"])
    x, y = load_dataset(variables["dataset"])
    x_train, y_train, x_test, y_test = split_data(x, y)
    # Starting population
    population = generate_population(len(population), 
                                     variables["init_max_depth"], 
                                     variables["num_variables"], 
                                     variables["integer_constants"])
    # Best-so-far tracking 
    best_so_far = None
    best_so_far_score = np.inf

    # Evolution
    for gen in range(0, len(population) - 1):
        # use fitness function to determine scores for each tree in a generation
        scores = set([fitness(tree, x_train, y_train, variables["complexity_weight"])] 
                     for tree in population)
        # iterate through the scores until the lowest is found
        best_index = 0
        if scores[best_index] < best_so_far_score:
            best_so_far_score = scores[best_index]
            best_so_far = population[best_index]
        

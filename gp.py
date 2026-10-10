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
import os
import csv
import sys

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
    variables: crossover with probability crossover_rate, mutation with
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
        
        
        # Depth limit per Luke & Panait: too deep children
        # are replaced by a copy of their parent
        child1, child2 = crossover(parent1, parent2, depth_limit)
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
    """
    Run one complete GP experiment.

    Loads the config and data, splits the data into training and test
    sets, creates the starting population, and evolves it for the
    configured number of generations. Fitness is computed on the
    training data only.

    The best tree seen in any generation is tracked and reported as the
    result of the run: "This best-so-far individual ... is designated as
    the result of the run" (Koza tutorial, p. 5). Its error on the
    held-out test data is computed only at the end, without the size
    penalty, so the reported numbers are plain MSE.

    Parameters
    ----------
    variables_path : str
        Path to the JSON config file (e.g. "dataset1_variables.json").

    Returns
    -------
    Node
        The best-so-far tree from the run.

    Notes
    -----
    Writes one row per generation to the CSV file named by the config's
    "log_file" setting, with columns: generation, best_fitness,
    best_so_far_fitness, mean_fitness, median_fitness, best_size,
    mean_size, mean_depth, and diversity. Mean and median fitness are
    computed over finite scores only, so invalid trees (score inf) do
    not distort them.

    Prints progress every 10 generations, then the best formula, its
    size, and its training and test MSE.
    """

    # Initialize with variables and splitting dataset
    variables = json.load(open(variables_path))
    random.seed(variables["seed"])
    x, y = load_dataset(variables["dataset"])
    x_train, y_train, x_test, y_test = split_data(x, y)
    # Starting population
    population = generate_population(variables["population_size"], 
                                     variables["init_max_depth"], 
                                     variables["num_variables"], 
                                     variables["integer_constants"])
    # Best-so-far tracking 
    best_so_far = None
    best_so_far_score = np.inf

    # Make sure the results folder exists
    # create the folder that log_file lives in, if missing
    os.makedirs("results", exist_ok=True)

    # Write one row per generation to the CSV file named by the config's
    # "log_file" setting, with various columns characteristic to each generation

    with open(variables["log_file"], "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow(["generation", "best_fitness", "best_so_far_fitness",
                         "mean_fitness", "median_fitness", "best_size", "mean_size",
                         "mean_depth", "diversity"])

        # Evolution
        for gen in range(variables["generations"]):
            # use fitness function to determine scores for each tree in a generation
            scores = [fitness(tree, x_train, y_train, variables["complexity_weight"]) 
                         for tree in population]
            
            # iterate through the scores until the lowest is found
            best_index = int(np.argmin(scores))
            if scores[best_index] < best_so_far_score:
                best_so_far_score = scores[best_index]
                best_so_far = population[best_index]

            # Mean and median fitness are computed over finite scores only, so 
            # invalid trees (score inf) do not distort them.
            finite = [score for score in scores if np.isfinite(score)]
            writer.writerow([
                gen,
                scores[best_index],
                best_so_far_score,
                np.mean(finite),
                np.median(finite),
                size(population[best_index]),
                np.mean([size(tree) for tree in population]),
                np.mean([depth(tree) for tree in population]),
                diversity(population),
                ])
            
            # Print progress every 10 generations
            if gen % 10 == 0:
                print("Generation", gen, "best so far:", best_so_far_score)
            # Run on next generation
            population = next_gen(population, scores, variables)
    
    # Determine error on the fitness of the best scores of trees, between training
    # and testing groups
    train_mse = fitness(best_so_far, x_train, y_train)
    test_mse = fitness(best_so_far, x_test, y_test)

    # Prints best formula, size, and its training and test MSE
    print("\nBest formula:", to_string(best_so_far))
    print("Size:", size(best_so_far))
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)

    return best_so_far

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "dataset1_variables.json"
    run(path)
    


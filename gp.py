"""
Genetic programming loop for symbolic regression.

Evolves a population of expression trees using tournament selection,
subtree crossover, subtree mutation, and reproduction, and logs
per-generation statistics to a CSV file. All settings are read from a
YAML config file (e.g. dataset1_variables.yaml).

Run from the command line:
    python gp.py dataset1_variables.yaml
"""

import yaml
import random 
import numpy as np 
import os
import csv
import sys
import time

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
    Run GP experiments for the seeds listed in a YAML configuration.

    Parameters
    ----------
    variables_path : str
        Path to the YAML config file.

    Notes
    -----
    Writes one combined generation CSV with a seed column. Prints each
    run's best formula, tree size, training/test MSE, and runtime.
    """

    # Initialize with variables and splitting dataset
    with open(variables_path) as f:
        variables = yaml.safe_load(f)

    # Normalize using training data only
    mse_scale = 1.0
    if variables.get("normalize_fitness", False):
        mse_scale = float(np.var(y_train))
        if not np.isfinite(mse_scale) or mse_scale <= 0:
            raise ValueError(
                "Normalized fitness requires positive, finite training-target variance."
            )

    seeds = variables["seeds"]
    x, y = load_dataset(variables["dataset"])
    variables["num_variables"] = x.shape[1]

    # Use the same training/test rows for every seed
    x_train, y_train, x_test, y_test = split_data(
        x,
        y,
        train_fraction=variables["train_fraction"],
        seed=variables["split_seed"],
    )

    # Build one results filename for the whole experiment
    dataset_name = os.path.splitext(
        os.path.basename(variables["dataset"])
    )[0]
    log_file = variables["log_file"].format(
        dataset=dataset_name,
        complexity_weight=variables["complexity_weight"],
        population_size=variables["population_size"],
        generations=variables["generations"],
    )

    # Create the folder that log_file lives in, if missing
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Write all seeds to one CSV
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "generation", "best_fitness", "best_so_far_fitness",
            "mean_fitness", "median_fitness", "best_size", "mean_size",
            "mean_depth", "diversity",
        ])

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            start = time.perf_counter()
            random.seed(seed)

            # Starting population
            population = generate_population(
                variables["population_size"],
                variables["init_max_depth"],
                variables["num_variables"],
                variables["integer_constants"],
            )

            # Best-so-far tracking resets for each seed
            best_so_far = None
            best_so_far_score = np.inf

            # Evolution
            for gen in range(variables["generations"]):
                # Use fitness to determine scores for each tree
                scores = [
                    fitness(
                        tree,
                        x_train,
                        y_train,
                        complexity_weight=variables["complexity_weight"],
                        mse_scale=mse_scale,
                    )
                    for tree in population
                ]

                # Find the current generation's best tree
                best_index = int(np.argmin(scores))
                if scores[best_index] < best_so_far_score:
                    best_so_far_score = scores[best_index]
                    best_so_far = population[best_index]

                # Compute mean and median over finite scores only
                finite = [score for score in scores if np.isfinite(score)]
                writer.writerow([
                    seed,
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
                    print(
                        "Generation", gen,
                        "best so far:", best_so_far_score,
                    )

                # Create another population only if needed
                if gen < variables["generations"] - 1:
                    population = next_gen(population, scores, variables)

            # Report plain MSE
            train_mse = fitness(best_so_far, x_train, y_train)
            test_mse = fitness(best_so_far, x_test, y_test)
            elapsed = time.perf_counter() - start

            print("\nBest formula:", to_string(best_so_far))
            print("Size:", size(best_so_far))
            print("Train MSE:", train_mse)
            print("Test MSE:", test_mse)
            print(f"Runtime: {elapsed:.2f} seconds")

            # Preserve completed rows before starting the next seed
            f.flush()

    print("\nResults saved to", log_file)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/dataset1.yaml"

    start = time.perf_counter()
    run(path)
    elapsed = time.perf_counter() - start

    print(f"Runtime: {elapsed:.2f} seconds ({elapsed / 60:.2f} minutes)")
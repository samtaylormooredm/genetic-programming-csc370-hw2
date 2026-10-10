import json
import random 
import numpy as np 

from fitness import fitness
from data import load_dataset, split_data
from tree import crossover, mutate, depth, size, to_string, generate_population


def tournament_select(population, scores, k):
    contestants = random.sample(range(len(population)), k)

    winner = contestants[0]
    for i in contestants[1:]:
        if scores[i] < scores[winner]:
            winner = i
    
    return population[winner]

def make_children(population, scores, variables):
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
    new_population = []
    while len(new_population) < variables["population_size"]:
        new_population.extend(make_children(population, scores, variables))
    return new_population[:variables["population_size"]]

def diversity(population):
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
        

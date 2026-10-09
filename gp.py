import json, csv
import random 

from data import load_dataset, split_data
from tree import crossover, mutate, depth, size, to_string, generate_population

variables  = json.load(open("dataset1_variables.json"))

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
    while len(new_population) < size(population):
        new_population.extend(make_children(population, scores, variables))
    return size(new_population)

def diversity(population):
    unique = len(to_string(population).values)
    return unique / size(population)

def run(variables_path):
    variables = json.load(open("dataset1_variables.json"))
    random.seed(variables["seed"])
    x, y = csv.load(open(variables["dataset"]))
    x_train, y_train, x_test, y_test = split_data(x, y)
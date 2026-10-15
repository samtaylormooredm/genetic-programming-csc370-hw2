"""
Run several independent GP runs and pick the best model by test error.

Runs the GP loop in gp.py once per random seed, records each run's
best-so-far tree with its training and test MSE, saves one summary row
per run, and reports the overall best model.

Usage
-----
    python multi_run.py variables_dataset1.json
    python multi_run.py variables_dataset2.json

If no config file is given, variables.json is used.

Outputs
-------
results/summary_<dataset>.csv
    One row per run with columns: seed, formula, size, train_mse,
    test_mse.
results/<dataset>_seed<seed>_penalty<weight>.csv
    Per-generation log for each run, written by gp.run.

Notes
-----
Model selection follows the overfitting procedure in the HW2 handout:
every run evolves on the training set only, the best model across
runs is chosen by test-set MSE, and that model's test MSE is reported
as an estimate of its error on unseen data.

When test MSEs are equal (within 1e-9), the smaller tree is chosen,
following the same preference for simple models as the complexity
penalty in fitness.py (our choice).

Each run uses a different seed, so each starts from a different random
population. Seeds are fixed so that results are reproducible.

References
----------
CSC 370 Homework 2 handout, "On Overfitting" section.
"""

import csv
import json
import os
import sys

from gp import run
from tree import to_string, size

config_path = sys.argv[1] if len(sys.argv) > 1 else "variables.json"
seeds = list(range(370, 380))

# Name the summary file after the dataset
with open(config_path) as f:
    variables = json.load(f)
dataset_name = os.path.splitext(os.path.basename(variables["dataset"]))[0]
summary_path = "results/summary_" + dataset_name + ".csv"

# Run GP once per seed and record each run's best tree
results = []
for seed in seeds:
    print("\n=== Starting run with seed", seed, "===")
    best_tree, train_mse, test_mse = run(config_path, seed)
    results.append([seed, to_string(best_tree), size(best_tree), train_mse, test_mse])

# Save one row per run
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "formula", "size", "train_mse", "test_mse"])
    writer.writerows(results)

# Pick the overall best by test MSE; break ties with the smaller tree
winner = results[0]
for row in results:
    clearly_better = row[4] < winner[4] - 1e-9
    tied_but_smaller = abs(row[4] - winner[4]) <= 1e-9 and row[2] < winner[2]
    if clearly_better or tied_but_smaller:
        winner = row

print("\nBest overall (picked by test MSE, ties broken by size):")
print("Seed:", winner[0])
print("Formula:", winner[1])
print("Size:", winner[2])
print("Test MSE:", winner[4])
print("Summary saved to", summary_path)

import csv
import json
import os
import sys

from gp import run
from tree import to_string, size

config_path = sys.argv[1] if len(sys.argv) > 1 else "variables.json"
seeds = list(range(370, 380))

# Name summary file after the dataset
with open(config_path) as f:
        variables = json.load(f)
dataset_name = os.path.splitext(
     os.path.basename(variables["dataset"]))[0]
summary_path = "results/summary_" + dataset_name + ".csv"

# Run GP once per seed, then record each run's best tree
results = []
for seed in seeds:
    print(seed)
    best_tree, train_mse, test_mse = run(config_path, seed)
    results.append([seed, to_string(best_tree), size(best_tree), train_mse, test_mse])
# Save one row per run
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "formula", "size", "train_mse", "test_mse"])
    writer.writerows(results)
# Pick overall best by test MSE
winner = results[0]
for row in results:
    if row[4] < winner[4]:
        winner = row

print("\nBest overall (picked by test MSE):")
print("Seed:", winner[0])
print("Formula:", winner[1])
print("Size:", winner[2])
print("Test MSE:", winner[4])
print("Summary saved to", summary_path)
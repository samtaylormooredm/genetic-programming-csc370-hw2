"""
Run GP experiments across multiple random seeds.

Notes
-----
Saves each run's formula, tree size, training/test MSE, and runtime
to a summary CSV. Selects the overall winner by training fitness.
"""

import csv
import yaml
import os
import sys
import time

from gp import run
from tree import to_string, size

config_path = sys.argv[1] if len(sys.argv) > 1 else "variables.json"
seeds = list(range(370, 380))

# Name summary file after the dataset
with open(config_path) as f:
    variables = yaml.safe_load(f)
dataset_name = os.path.splitext(
    os.path.basename(variables["dataset"]))[0]
summary_path = (
    f"results/summary_{dataset_name}"
    f"_pop{variables['population_size']}"
    f"_gen{variables['generations']}"
    f"_penalty{variables['complexity_weight']}.csv"
)

# Run GP once per seed, then record each run's best tree
results = []

for seed in seeds:
    print(seed)

    start = time.perf_counter()
    best_tree, train_mse, test_mse = run(config_path, seed)
    elapsed = time.perf_counter() - start

    print(f"Runtime: {elapsed:.2f} seconds")

    results.append([
        seed,
        to_string(best_tree),
        size(best_tree),
        train_mse,
        test_mse,
        elapsed,
    ])

# Save one row per run
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "seed", "formula", "size", "train_mse", "test_mse",
        "runtime_seconds",
    ])
    writer.writerows(results)
# Test using training fitness
winner = min(
    results,
    key=lambda row: row[3] + variables["complexity_weight"] * row[2],
)
for row in results:
    if row[4] < winner[4]:
        winner = row

print("\nBest overall (picked by training fitness):")
print("Seed:", winner[0])
print("Formula:", winner[1])
print("Size:", winner[2])
print("Test MSE:", winner[4])
print("Summary saved to", summary_path)
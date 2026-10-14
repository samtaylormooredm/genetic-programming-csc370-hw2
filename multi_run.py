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
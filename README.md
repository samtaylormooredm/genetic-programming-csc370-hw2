# Symbolic regression with genetic programming

CSC 370 Homework 2 evolves arithmetic expression trees to predict two datasets.
Dataset 1 consistently recovers **x² − 6x + 14**. Our best overall dataset-2
result, **seed 20**, comes from a **37-node tree** with test scaled error
**0.000923665**. Away from protected division near x3 = 0, it simplifies
approximately to:

```math
\hat{y} \approx 6.656 \times 10^{-11}
\left(
\frac{x_1 x_2 - 0.0987}{x_3^2}
- \frac{0.00161 x_1 x_2}{x_3^3}
+ \frac{0.3002}{x_3}
\right).
```

Simplifying further to its leading term approximates Newton's law of
gravitational force:

$$
\hat y \approx 6.656 \times 10^{-11}\frac{x_1x_2}{x_3^2}
$$

## Running the project

Install the dependencies and run from the repository root:

```bash
pip install -r requirements.txt
python -u gp.py configs/dataset1.yaml
python -u gp.py configs/dataset2.yaml
```

On the cluster, use `sbatch submit.sh configs/dataset2.yaml` (or dataset1).
CSV inputs live in `data/`. YAML controls the split, seeds, population, tree
limits, genetic operators, and complexity penalty. Dataset 2 now defaults to
seeds 20–22, 1,000 generations, and penalty `1e-5`; reruns use a separate log
filename to preserve the archived results. Dataset 1 defaults to seeds 370–372;
the saved ten-seed experiment below used 370–379.

## Trees, fitness, and bloat control

Each individual is an immutable binary `Node(kind, value, children)` tree.
Operators are `+`, `-`, `*`, and protected `/`; terminals are constants or input
columns (`x1` is column 0). Crossover swaps subtrees, mutation replaces a random
subtree, and reproduction keeps a parent. Random immigrants add fresh trees.

Every printed division means:

```text
pdiv(a, b) = a / b if abs(b) >= 1e-6, otherwise 1
```

Size counts all operator and terminal occurrences. Depth counts edges, so a
terminal has depth 0. All reported sizes refer to the evolved tree before
algebraic rewriting. The configured depth limit is 17.

Lower fitness is better:

```text
Dataset 1: mean((prediction - target)**2) + 0.0001 * size
Dataset 2: mean(((prediction - target)/(1 + abs(target)))**2) + 0.00001 * size
```

The scaled objective reduces the dominance of extreme targets without filtering
rows or scaling inputs. Printed **best so far** includes the size penalty;
final train/test errors exclude it. Earlier dataset-2 runs used penalty `1e-6`,
so their penalized fitness values are not directly comparable. Raw MSE remains
useful for showing large absolute errors.

## Dataset 1 results

**Settings:** population 500; 50 generations; seeds 370–379; initial maximum
depth 5; depth limit 17; tournament size 7; crossover 0.90; mutation 0.01;
reproduction 0.09; integer constants in [-10, 10]; 80/20 split, split seed 42.
Fitness is training MSE + 0.0001 * node count.

**Result files:**

- Model summary: `results/summary_dataset1_pop500_gen50_penalty0.0001.csv`.

All 10 formulas were independently expanded and verified to equal
**f(x) = x^2 - 6x + 14**. Predictions reproduce the reported errors using the
20,000/5,000 split and split seed 42.

| Seed | Best tree size | Runtime (s) |
|---|---:|---:|
| 370 | 13 | 5.85 |
| 371 | 11 | 5.85 |
| 372 | 11 | 4.95 |
| 373 | 11 | 4.56 |
| 374 | 11 | 4.63 |
| 375 | 15 | 7.50 |
| 376 | 15 | 6.33 |
| 377 | 15 | 8.30 |
| 378 | 9 | 5.07 |
| 379 | 9 | 6.48 |

For every seed, train MSE is approximately **0.0005953816485** and test MSE
approximately **0.0005941104340**. Seed 370's test RMSE is **0.02437438** and
median absolute error is **0.0109**. These are raw errors, without the penalty.
Seeds 378 and 379 have the smallest final trees (9 nodes); differences in MSE
at the last floating-point digits are not meaningful model differences.
The test prediction plot follows the identity line closely. Residual spread
varies with x; its cause is not established by this plot alone.

**Interpretation:** consistent recovery across all 10 seeds and similar
train/test errors support stable prediction on this split. Residual error
remains, so algebraic agreement of the evolved models does not prove the
noise-free generating function exactly.

### Final equations and representative trees

**Recovered model:**

$$
\hat y = x^2 - 6x + 14 = (x - 3)^2 + 5.
$$

The seed-370 tree used for the prediction analysis was:

```text
((9 - x1) + ((5 + (x1 * (x1 + -4))) - x1))
```

It contains **13 nodes**, with depth **5**: 6 operator nodes and 7 terminal
occurrences. Its root operator is `+`; both branches combine constants and
`x1`. The expression expands exactly to the quadratic above.

The smallest saved trees were seeds 378 and 379, each with **9 nodes**.
For example, seed 379 produced:

```text
((8 + 6) + (x1 * (x1 + -6)))
```

This tree has depth **3**, with 4 operators and 5 terminal occurrences.
Seed 378 has depth 4. Both are exactly the same polynomial as seed 370 and
have effectively identical prediction errors. Thus the 13-node seed-370 plot
and the 9-node compact example describe the same fitted function, but are
not the same stored tree. The rewritten quadratic is an interpretation of
the evolved expressions, not a formula inserted into the initial population.

## Dataset 2: selected seed-20 run

**Settings:** population 500; 1,000 generations; seeds 20–22; 70/30 split
(17,500/7,500 rows), split seed 42; initial maximum depth 5; depth limit 17;
tournament size 3; crossover 0.80; mutation 0.10; reproduction 0.10 among
operator events; random immigrants fill 10% of each population. Constants are
sampled equally from uniform [-10, 10] and signed 10^U with U uniform [-12, 12].
No row filtering or input scaling is applied.

**Saved artifacts:**

- Final formulas and metrics: [summary CSV](results/summary_dataset2_pop500_gen1000_penalty1e-05_seeds20-22.csv).
- Completed console output: [seeds 20–22](results/dataset2_pop500_gen1000_penalty1e-05_seeds20-22.out).
- Available generation rows: [CSV](results/dataset2_pop500_gen1000_penalty1e-05_seeds20-22.csv).
  The source trial CSV contains only 12 rows for seed 20, but all 1,000 rows
  for seeds 21 and 22. The console log preserves seed 20's progress every ten
  generations and its completed formula; missing generation rows were not reconstructed.
- Selected-model [metrics](results/analysis/dataset2_pop500_gen1000_penalty1e-05_seeds20-22_seed20/metrics.json),
  [test predictions](results/analysis/dataset2_pop500_gen1000_penalty1e-05_seeds20-22_seed20/test_predictions.csv),
  and [prediction plot](results/analysis/dataset2_pop500_gen1000_penalty1e-05_seeds20-22_seed20/prediction_analysis.png).

| Seed | Train scaled error | Test scaled error | Tree size | Runtime (s) |
|---|---:|---:|---:|---:|
| **20** | **0.0010521203** | **0.0009236646** | **37** | **518.98** |
| 21 | 0.0534034262 | 0.0547601242 | 127 | 1254.25 |
| 22 | 0.0012101184 | 0.0011328119 | 53 | 776.23 |

Seed 20 has the lowest scaled errors and smallest tree in this experiment.
Its training fitness is **0.0014221202806949625**, including the 0.00037 size
penalty. That value appears by printed generation 850; the exact first discovery
generation is unavailable in the partial CSV. Train raw MSE is
**1.474527712225556e18**; test raw MSE is **4.836701037213689e15**.
The training-fitted constant baseline is **1.82376892**, with test scaled
error **0.80716405**. All three seeds outperform it.

The recomputed median absolute test error is **0.12605** output
units. **94.09%** of test rows satisfy
`abs(prediction - actual) <= 0.01 * (1 + abs(actual))`; **98.92%**
satisfy the corresponding 0.05 tolerance. These are stated error tolerances,
not classification accuracy.

### Exact evolved equation

Every `/` below uses protected division. The tree has **37 nodes** (18 operators,
19 terminals), depth **12**, and a protected-division root. It uses all three inputs.

```text
(((((1.5533477335591037 - ((((x1 * x2) / (1.5533477335591037 / 4.847521942247518)) / (1.5533477335591037 / 7.93620801287236)) - ((((((x1 * x2) / x3) + 1.5533477335591037) / 7.93620801287236) - (x1 * x2)) / 4.847521942247518))) / x3) - 4.847521942247518) * -4.121528354621327e-12) / x3)
```

### Simplified inverse-square form

For `abs(x3) >= 1e-6`, all divisions in this tree use ordinary arithmetic.
Expanding and collecting terms gives the following form (coefficients rounded):

```math
\hat{y} =
\left(
6.656360438 \times 10^{-11}
- \frac{1.071335426 \times 10^{-13}}{x_3}
\right)
\frac{x_1 x_2}{x_3^2}
+ \frac{1.997919913 \times 10^{-11}}{x_3}
- \frac{6.568582374 \times 10^{-12}}{x_3^2}.
```

Factoring out approximately `6.656e-11` gives the rounded expression in the
introduction. The correction terms matter particularly near zero. This is an
algebraic interpretation of the evolved tree; all metrics use the original tree.

For `abs(x3) < 1e-6`, the original tree returns **1** because its outermost
division is protected. The ordinary inverse-square expression does not apply
there. All reported metrics and plots use the full protected tree. Severe
errors on extreme targets remain, especially around x3 = 0.

### Earlier experiments

Early raw-MSE runs converged to weak models or improved training error while
worsening test error. Dividing MSE by training-target variance changed its scale
without fixing the imbalance. The scaled objective, broader constants, smaller
tournaments, higher mutation, and immigrants were introduced together; their
individual effects were not isolated. Longer runs improved convergence.

The earlier population-500, 1,000-generation experiment used penalty `1e-6`:

| Seed | Train scaled error | Test scaled error | Size |
|---|---:|---:|---:|
| 370 | 0.00883580 | 0.00888537 | 93 |
| 371 | 0.00095168 | 0.00079944 | 139 |
| 372 | 0.01634359 | 0.01680159 | 301 |
| 373 | 0.00316467 | 0.00308737 | 151 |

Its formulas remain in
`results/slurm_dataset2_scaled_wide_constants_tourn3_mut10_imm10_pop500_gen1000_penalty1e-06.out`.
Seed 371 retains the lowest test scaled error of these compared models.
Seed 20 uses **73.4% fewer nodes**, with about **15.5% higher test scaled error**.
The older seed-371 analysis remains archived under `results/analysis/` and
belongs to that model, not seed 20. Trial and local-constant-mutation CSVs are
retained as intermediate search records, not the selected result.

Independent inspection of training rows suggested the relationship
`y ≈ 6.673e-11 * x1*x2/x3**2` away from zero. For x3 > 1, the median ratio
`y*x3**2/(x1*x2)` was approximately 6.673e-11. This informed the constant range;
it was not inserted into the population or verified as equivalent to the GP
model. Ordinary division leaves this candidate undefined at x3 = 0.

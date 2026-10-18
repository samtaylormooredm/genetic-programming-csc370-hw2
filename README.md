# Symbolic regression with genetic programming

CSC 370 Homework 2 explores how genetic programming can evolve arithmetic
expressions that predict two datasets. Each candidate equation is an expression
tree. Evolution selects trees using prediction error and a penalty for tree size.

Dataset 1 consistently recovers the quadratic **x² − 6x + 14**. Dataset 2
uses a scaled-error objective and broad constant sampling to model a target
spanning many orders of magnitude. Its selected model achieves a test scaled
error of **0.00079944** and a median absolute test error of **0.04167**.
The largest errors occur near x3 = 0.

## Running the project

Run commands from the repository root, with the project dependencies installed
in your Python environment. The code uses NumPy, scikit-learn, PyYAML, and
Matplotlib for analysis. Dataset CSV files are expected in `data/`.

```bash
python -u gp.py configs/dataset1.yaml
python -u gp.py configs/dataset2.yaml
```

On the cluster:

```bash
sbatch submit.sh configs/dataset1.yaml
sbatch submit.sh configs/dataset2.yaml
```

YAML files control the dataset, data split, random seeds, population size,
generation budget, genetic operations, and complexity penalty. To reproduce
the results below, use the recorded settings and matching implementation;
a different branch or current configuration may have different defaults.

## Expression trees

Each individual is an immutable binary expression tree represented by a frozen
`Node(kind, value, children)` object. Operator nodes contain `+`, `-`, `*`, or
protected `/` and have two children; terminal nodes contain an input variable
or a numeric constant and have no children. Displayed `x1` corresponds to
input-column index 0, and so on. Crossover swaps subtrees; mutation replaces a
subtree; both construct new trees rather than modifying the parents.

All `/` symbols in printed GP expressions mean protected division:

`pdiv(a, b) = a / b if abs(b) >= 1e-6, otherwise 1`.

Size counts every operator and terminal occurrence. Depth counts edges from
the root to its deepest terminal, so a one-node tree has depth 0. Reported
sizes/depths refer to the actual evolved trees, before any algebraic rewriting.

## Fitness and bloat control

Lower fitness is better. Dataset 1 uses:

```text
training MSE + 0.0001 * tree size
```

Dataset 2 uses:

```text
np.mean(((prediction - target) / (1 + np.abs(target))) ** 2) + 0.000001 * tree_size
```

The dataset-2 objective weights each squared residual by
`1 / (1 + abs(target))**2`. This reduces the dominance of very large targets;
it changes which errors the search prioritizes. It does not normalize the
inputs or remove outliers. Raw MSE is reported separately because large
absolute errors still matter.

Printed **best so far** is the lowest penalized training fitness found so far
within a seed. Final train/test errors below exclude the tree-size penalty.
Scores from different fitness objectives should not be compared directly.

Tree growth is controlled by a depth limit of 17 and a linear size penalty.
For dataset 1, adding the penalty reduced seed 370's tree from 125 to 13 nodes
(an 89.6% reduction), with effectively unchanged train/test MSE. Dataset 2
still produces large trees, so the penalty does not eliminate bloat.

## Dataset 1 results

**Settings:** population 500; 50 generations; seeds 370–379; initial maximum
depth 5; depth limit 17; tournament size 7; crossover 0.90; mutation 0.01;
reproduction 0.09; integer constants in [-10, 10]; 80/20 split, split seed 42.
Fitness is training MSE + 0.0001 * node count.

**Result files:**

- Model summary: `results/summary_dataset1_pop500_gen50_penalty0.0001.csv`.
- Generation log recorded for this experiment: `results/dataset1_pop500_gen50_penalty0.0001.csv`.

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

## Dataset 2 results

**Configuration:** population 500; 1,000 generations; four seeds 370-373;
70/30 split (17,500/7,500 rows), split seed 42; initial maximum depth 5;
depth limit 17; tournament size 3; crossover 0.80; mutation 0.10;
reproduction probability 0.10 among offspring-generation events; random
immigrants fill 10% of the population; wide constants described under Search configuration and development; scaled-error
fitness + 0.000001 * node count. No row filtering or input scaling.

**Result files:**

- **Generation CSV:**
  `results/dataset2_scaled_wide_constants_tourn3_mut10_imm10_pop500_gen1000_penalty1e-06.csv`
- **Completed output with formulas and final errors:**
  `results/slurm_dataset2_scaled_wide_constants_tourn3_mut10_imm10_pop500_gen1000_penalty1e-06.out`

| Seed | Train scaled error | Test scaled error | Best size | Best penalized model first found (generation) | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| 370 | 0.00883580 | 0.00888537 | 93 | 982 | 1231.37 |
| 371 | 0.00095168 | 0.00079944 | 139 | 891 | 983.95 |
| 372 | 0.01634359 | 0.01680159 | 301 | 991 | 1358.42 |
| 373 | 0.00316467 | 0.00308737 | 151 | 996 | 918.45 |

Generation indices are zero-based, as logged. Final errors exclude the size
penalty. Printed "best so far" includes it and tracks the lowest training
fitness ever found in that seed's run, not the current population's fitness.

| Seed | Train raw MSE | Test raw MSE |
|---|---:|---:|
| 370 | 1.474528e18 | 4.836753e15 |
| 371 | 1.474195e18 | 4.794879e15 |
| 372 | 1.474528e18 | 4.836750e15 |
| 373 | 1.474528e18 | 4.836709e15 |

**Best under scaled error:** seed 371. The training-fitted constant baseline
is 1.82376892, with test scaled error 0.80716405. All four runs outperform this
baseline. Seed 371 has the lowest training scaled error, test scaled error,
and raw test MSE among the four runs.

**Diversity and bloat:** final structural diversity is 97.4%, 99.2%, 97.8%,
and 97.4%. Distinct expression strings do not establish behavioral diversity.
Final population-average tree sizes are approximately 148, 146, 327, and 137;
growth remains substantial despite the size penalty and depth limit.

### Prediction performance: seed 371

The selected model predicts most test observations closely. Across all 7,500
test rows, its median absolute error is **0.04167 output units**; 75% of
absolute errors are at most **0.38643**, and 81.4% are at most 1 output unit.

| Statistic | Absolute prediction error |
|---|---:|
| 25th percentile | 0.00855 |
| **Median** | **0.04167** |
| 75th percentile | 0.38643 |
| 90th percentile | 5.91661 |
| 95th percentile | 53.15677 |
| 99th percentile | 4108.53330 |

**Scaled-error coverage:** 93.8% of test predictions have absolute error at
most `0.01 * (1 + abs(actual))`; 98.68% meet the corresponding 0.05 threshold.
These measure coverage under stated error tolerances, rather than classification
accuracy or ordinary percentage error.

**Extreme-target behavior:** the three test rows with x3 = 0 contribute
approximately 99.9984% of total squared error. Excluding only those rows from
the existing predictions reduces test MSE from **4.794879e15** to
**7.507193e10** across the remaining 7,497 rows. This is a diagnostic comparison,
not a retrained model or the primary evaluation. Large errors also remain for
some nonzero x3 values.

**Result:** GP produced a model with low scaled prediction error across most
observations and similar training and test scaled errors. Its 139-node
expression remains structurally complex, and raw MSE exposes severe errors
on extreme targets. These results establish the model's predictive performance
on this split; they do not establish recovery of the exact generating function.

### Final evolved equation and tree: seed 371

**Selected model under scaled error:** the full protected-arithmetic expression
below, evolved at seed 371 in the 1,000-generation experiment. Its best
penalized training fitness was first found at zero-based generation **891**.
This actual tree produced the reported test scaled error **0.0007994381**.

- **Size:** 139 nodes: 69 operators and 70 terminal occurrences.
- **Actual depth:** 16 (configured maximum: 17).
- **Root operator:** protected division.
- **Inputs used:** x1, x2, and x3; terminals also contain real-valued constants.
- **Representation:** the immutable binary `Node` tree described above.

**Exact printed evolved expression** (every `/` uses `pdiv`, not ordinary
unprotected division):

```text
(((x1 / ((((x3 * x1) * x2) + (x1 / x1)) + (((x2 - ((x3 + x1) / (92890484798.7518 + -2.002835191099754e-06))) / (x3 * 20.561749989710545)) / 20.561749989710545))) + (((((((6.194662092275518 + 0.9568895863753788) - (((-0.7878465889803703 - x3) + x3) + ((8.6654421403368 + 6.194662092275518) * x1))) - (((((((6.194662092275518 + ((20.561749989710545 + 0.9568895863753788) - x3)) * x1) + x1) + (20.561749989710545 + 6.194662092275518)) * ((x3 / 6.194662092275518) + (8.6654421403368 / -4.269752075015885))) / -1.0320722220226024) + ((6.346603816893701 + 6.346603816893701) * x1))) - (x3 + ((20.561749989710545 + 6.194662092275518) * x1))) / -3902149.0985513106) - 1.7081374851345555e-08) / x3)) / (((x3 + ((6.194662092275518 / (x3 * (x3 / (20.561749989710545 + 6.346603816893701)))) + (3981380.8245344167 - 6.194662092275518))) / x2) / (((8.865141113756462 / x3) + (6.194662092275518 / ((((20.561749989710545 + ((8.6654421403368 + 6.194662092275518) / x3)) - x3) / (((6.194662092275518 + 8.6654421403368) / (x3 * x3)) + 6.194662092275518)) + 6.346603816893701))) / (6.194662092275518 / 6.346603816893701))))
```

This is the saved GP model used for the reported metrics. Its full expression
preserves all protected-division branches. A compact equivalent has not been
verified, so tree size and predictive error are reported separately.

### Supporting analysis of the input-output relationship

Independently of the evolved tree, inspection of training rows suggested:

$$
\hat y \approx 6.673 \times 10^{-11}\frac{x_1 x_2}{x_3^2},
\qquad x_3 > 0.
$$

For training rows with x3 > 1, the ratio `y * x3**2 / (x1 * x2)` had median
approximately 6.673e-11, with 5th-95th percentiles approximately
6.651e-11 to 6.695e-11. This supports the candidate relationship away from zero.

**Relationship to the GP result:** this data-derived candidate provides
context for the observed input-output patterns. It is separate from the
evolved model and has not been verified as an equivalent expression. Ordinary
division leaves it undefined at x3 = 0. Neither the formula nor its coefficient
was inserted into the GP population; broad constant sampling was informed by
this analysis. The prediction metrics above belong to the saved GP tree.

## Search configuration and development

Early dataset-2 runs either converged to simple weak models or improved training
MSE while producing much worse test MSE. Dividing MSE by training-target variance
changed its scale but did not resolve the imbalance between ordinary rows and
extreme targets. The reported configuration uses:

- The per-row scaled-error objective defined above.
- Tournament size 3, crossover 0.80, mutation 0.10, and fresh random trees filling
  10% of each population. These changes were combined, so their separate effects
  were not isolated.
- Real-valued constants sampled with 50% probability uniformly from [-10, 10];
  otherwise as a random sign times 10^U, where U is uniform on [-12, 12].
  Dataset 1's integer-constant generation stayed unchanged.
- A larger generation budget after shorter runs were still improving.

For seed 370, test scaled error fell from 0.785328 at 50 generations to 0.477149
at 200, 0.270364 at 500, and 0.008885 at 1,000. These runs share the same initial
trajectory and represent convergence checkpoints, not independent replications.
Across the original three seeds, all had lower test scaled error at 1,000
than at 500 generations.

## Analyzing saved equations

Place `analyze_model.py` beside `gp.py`. It accepts a final-model summary CSV
containing formulas or a completed SLURM output containing printed formulas.
A generation CSV alone may not contain the final equation.

```bash
python analyze_model.py results/summary_dataset1_pop500_gen50_penalty0.0001.csv --dataset dataset1 --seed 370
python analyze_model.py results/slurm_dataset2_scaled_wide_constants_tourn3_mut10_imm10_pop500_gen1000_penalty1e-06.out --dataset dataset2 --seed 371
```

The script evaluates the saved tree without rerunning evolution. It recreates
the original unfiltered, unscaled split: dataset 1 defaults to 80/20 and dataset 2
to 70/30, both with split seed 42. Override those settings for experiments with
a different split; filtered or input-scaled models require matching preprocessing.

By default, output is written to:

```text
results/analysis/<model-file-stem>_seed<seed>/
    metrics.json
    test_predictions.csv
    prediction_analysis.png
```

The metrics file includes training and test errors. Both panels in the plot
show **test observations**:

- **Actual versus predicted:** points near the diagonal have accurate predictions.
- **Residual plot:** residuals are prediction minus actual, so positive means
  overprediction and negative means underprediction. Dataset 1 shows raw residuals
  against its input. Dataset 2 shows scaled residuals against x3 as a diagnostic;
  the equation itself uses all three inputs.

Dataset 2's scaled residual is `(prediction - actual) / (1 + abs(actual))`.
Its actual/predicted panel uses symmetric logarithmic axes to display its large
range. Dataset 1's residual spread narrows near x = 3, but the plot alone does
not establish why.

## Reproducibility and interpretation

Keep the matching configuration, implementation commit, generation CSV, and
final-model summary or completed SLURM output together. Generation logs show
search progress; summaries and completed output preserve the final equations.
An incomplete output file is not a resumable checkpoint.

Dataset 1 recovers a consistent compact model across ten seeds. Dataset 2
produces low scaled errors across all four runs, with seed 371 giving the
strongest predictive performance. Median error and error percentiles describe
performance across typical observations; raw MSE captures the impact of rare,
large misses. Both are retained in the evaluation.

**Evaluation scope:** the fixed test split was inspected during configuration
development, so the reported test metrics are not an
untouched final assessment. An independent holdout would be needed to measure
performance without that reuse.

## Dataset 1 baseline

**Parameters:** Seed = 370; population = 500; generations = 50;
initial max depth = 5; depth limit = 17; tournament size = 7;
crossover = 0.90; mutation = 0.01; reproduction = 0.09;
complexity weight = 0.0. Training/test split = 80%/20%.

**Results:**
- Training MSE: 0.00059538
- Test MSE: 0.00059411
- Best tree size: 125 nodes
- Simplified formula: x² − 6x + 14

**Observation:** Achieved near-final fitness by generation 10.
Similar training and test error suggests good generalization.
The tree contains substantial redundant structure, consistent with
having no complexity penalty.

**Next experiment:** Introduce a small complexity penalty and compare
tree size and error while keeping the other parameters fixed.


## Dataset 1 with complexity penalty

**Change from baseline:** complexity_weight = 0.0001.
All other parameters unchanged.

**Results:**
results/dataset1_seed370_penalty0001.csv
- Training MSE: 0.00059538
- Test MSE: 0.00059411
- Best tree size: 13 nodes
- Simplified formula: x² − 6x + 14
- Runtime: 5.31 seconds

**Observation:** The penalty reduced tree size by 89.6% without
meaningfully changing accuracy. Near-final fitness was reached
by generation 10.

**Next experiment:** Keep the penalty and change the seed to 371
to check whether another run finds a similarly accurate formula.

Notes on next experiments: Seeds 371 and 372 both recovered x² − 6x + 14 with essentially identical training and test MSE, producing 11-13-node trees at around 6 seconds per seed. Across all three seeds, the complexity penalty consistently produced compact, accurate formulas. 

## Dataset 1: Final configuration and results

**Recovered function:** f(x) = x² − 6x + 14.

### Final parameters

- Population: 500; generations: 50; independent runs: 3 (seeds 370–372).
- Initial maximum depth: 5; random trees with operator roots and
  randomly terminating branches.
- Building blocks: +, −, ×, protected ÷; integer constants from −10 to 10.
- Depth limit: 17; tournament size: 7
  (Luke & Panait, 2006, Sec. 3).
- Crossover probability: 0.90; mutation probability: 0.01
  (Koza tutorial); remaining 0.09 selects reproduction.
- Fitness: training MSE + 0.0001 × node count. The linear size-penalty
  method follows Luke & Panait (2006, Sec. 5); the weight was selected
  experimentally.
- Training/test split: 80%/20% (20,000/5,000 rows), split seed 42.

### Results

All three runs recovered equivalent forms of the same quadratic.
Training MSE was approximately 0.00059538 and test MSE approximately
0.00059411 for every run.

| Seed | Best tree size | Generation found | Runtime | Initial → final diversity |
|---|---:|---:|---:|---:|
| 370 | 13 | 8 | 5.70 s | 92.8% → 74.8% |
| 371 | 11 | 11 | 6.31 s | 89.4% → 73.2% |
| 372 | 11 | 7 | 6.16 s | 90.6% → 84.2% |

**Stability and overfitting:** All three seeds recovered the same
function quickly. Similar training and test errors provide no clear
evidence of overfitting on this split.

**Diversity:** Measured as the fraction of distinct expression strings.
Structural diversity remained substantial, with no population-wide
collapse into duplicates. No explicit diversity-preservation mechanism
was used; this measure does not establish behavioral diversity.

**Bloat:** The seed-370 baseline comparison reduced best-tree size from
125 to 13 nodes without meaningfully changing MSE. Population-average
sizes still temporarily reached about 46–62 nodes, so bloat was reduced
rather than eliminated.

**Results CSV:** results/dataset1_pop500_gen50_penalty0.0001.csv

## Dataset 2 initial run

**Settings:** Seed = 370; population = 500; generations = 50;
complexity weight = 0.0001.

**Results CSV:** results/dataset2_seed370_penalty0.0001.csv

**Results:** Training MSE = 1.07428e18; test MSE = 2.33995e17;
tree size = 145 nodes; runtime = 20.15 seconds.

**Observation:** Training fitness improved, but test performance was
worse than a constant-mean baseline. Extreme target values dominate
MSE, and the current complexity penalty is negligible at this scale.
Should test again with higher generation # to see if results improve.

## Dataset 2: Increased generations

**Change:** Generations increased from 50 to 200; seed 370 and
all other settings unchanged.

**Results:** Training MSE = 3.08975e17; test MSE = 3.40806e19;
tree size = 299 nodes; runtime = 136.49 seconds.

**Observation:** Training error decreased by 71%, but test error
increased approximately 146-fold and tree size roughly doubled.
Additional evolution improved training fit without improving
generalization. 0.0001 size penalty is effectively invisible at this scale.


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

Notes on next experiments: Seeds 371 and 372 both recovered x² − 6x + 14 with essentially identical training and test MSE, producing 11-node trees in 5.58 and 4.85 seconds. Across all three seeds, the complexity penalty consistently produced compact, accurate formulas. Next, test dataset 2.

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


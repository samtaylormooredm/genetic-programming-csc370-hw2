## Experiment 1: Dataset 1 baseline

**Parameters:** Seed = 370; population = 500; generations = 50;
initial max depth = 5; depth limit = 17; tournament size = 7;
crossover = 0.90; mutation = 0.01; reproduction = 0.09;
complexity weight = 0.0. Training/test split = 80%/20%.

**Results:**
- Training MSE: 0.00059538
- Test MSE: 0.00059411
- Best tree size: 125 nodes
- Simplified formula: x² − 6x + 14
- Runtime: Not recorded

**Observation:** Achieved near-final fitness by generation 10.
Similar training and test error suggests good generalization.
The tree contains substantial redundant structure, consistent with
having no complexity penalty.

**Next experiment:** Introduce a small complexity penalty and compare
tree size and error while keeping the other parameters fixed.
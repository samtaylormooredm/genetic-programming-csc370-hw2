from sklearn.metrics import mean_squared_error
from numpy import np

from tree import evaluate, size

# TODO: Currently start with weight of 0, update after initial tests
def fitness(node, x, y, complexity_weight=0):
    """
    Score a tree using mean squared error and a size penalty.
    Return a float with the fitness score. Lower = better.
    """
    predictions = evaluate(node, x)

    # Give trees with NaN or infinite predictions the worst possible fitness.
    if not np.all(np.isfinite(predictions)):
        return float("inf")
    
    mse = mean_squared_error(y, predictions)
    score = mse + complexity_weight * size(node)

    if not np.isfinite(score):
        return float("inf")

    return float(score)


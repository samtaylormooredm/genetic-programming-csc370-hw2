from sklearn.metrics import mean_squared_error
import numpy as np

from tree import evaluate, size, Node

# TODO: Currently start with weight of 0, update after initial tests
def fitness(node, x, y, complexity_weight=0):
    """
    Score a tree using mean squared error and a size penalty.
    Return a float with the fitness score. Lower = better.

    Implements linear parametric parsimony pressure (Luke & Panait 
    2006, Sec. 5, p. 10), whose general form is g = x*f + y*s, where
    f is the raw fitness and s is the tree size. We have that f = MSE,
    s = size(node), x = 1, and y = complexity_weight

    Luke & Panait found this penalty , combined with a depth limit of 
    17, reduced bloat without hurting fitness. The right weight depends 
    on the scale of the fitness values (Sec. 5, p. 10), so 
    complexity_weight is tuned separately for each dataset.

    Parameters
    ----------
    node : Node
        Root of the tree to score.
    x : numpy.ndarray
        Input data, one row per data point.
    y : numpy.ndarray
        True target values, one per row.
    complexity_weight : float, optional
        Penalty per node (y in Luke & Panait's formula). Defaults to 0,
        meaning no size penalty.

    Returns
    -------
    float 
        Penalized fitness. Lower is better; inf for invalid trees.
    """
    predictions = evaluate(node, x)

    # Give trees with NaN or infinite predictions the worst possible fitness.
    if not np.all(np.isfinite(predictions)):
        return float("inf")
    
    mse = mean_squared_error(y, predictions)

    # Size penalty: linear parametric parsimony pressure (Luke & Panait 2006)
    score = mse + complexity_weight * size(node)

    if not np.isfinite(score):
        return float("inf")

    return float(score)

if __name__ == "__main__":
    # Sample known function: y = x + 3
    x = np.array([[1.0], [2.0], [3.0]])
    y = np.array([4.0, 5.0, 6.0])

    correct_tree = Node(
        "operator",
        "+",
        (Node("variable", 0), Node("constant", 3)),
    )
    incorrect_tree = Node("variable", 0)

    # Should get 0 (perfect prediction)
    print("Correct tree:", fitness(correct_tree, x, y))

    # Predicts x, so every prediction is off by 3
    # MSE = 3^2 = 9, so should expect 9
    print("Incorrect tree:", fitness(incorrect_tree, x, y))
    print(
        "Correct tree with penalty:",
        fitness(correct_tree, x, y, complexity_weight=0.1),
    )
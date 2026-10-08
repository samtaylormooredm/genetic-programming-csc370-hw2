from sklearn.metrics import mean_squared_error
import numpy as np

from tree import evaluate, size, Node

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
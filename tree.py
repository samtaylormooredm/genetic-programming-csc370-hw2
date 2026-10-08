import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Node:
  """
    Represent a node in a symbolic expression tree.

    Parameters
    ----------
    kind : str
        Type of node: "operator", "variable", or "constant".
    value : object
        Value stored in the node. 
            Operators --> operator symbol. 
            Variables --> column index. 
            Constants --> this is the numeric value.
    children : tuple, optional
        Child nodes of this node. Operator nodes have two children,
        while variable and constant nodes have none.
    """
  kind: str
  value: object
  children: tuple = ()


OPERATORS = ["+", "-", "*", "/"]


def evaluate(node, x):
    """
    Evaluate an expression tree on input data.

    Parameters
    ----------
    node : Node
        Root node of the expression tree.
    x : numpy.ndarray
        Input data where each column corresponds to a variable.

    Returns
    -------
    numpy.ndarray
        Predicted values produced by evaluating the tree for each row
        of input data.
    """
    if node.kind == "variable":
        return x[:, node.value]
    
    if node.kind == "constant":
        return np.ones(x.shape[0]) * node.value
    
    if node.kind == "operator":
        left = evaluate(node.children[0], x)
        right = evaluate(node.children[1], x)

        if node.value == "+":
            return left + right
        if node.value == "-":
            return left - right
        if node.value == "*":
            return left * right
        if node.value == "/":
            return division_rule(left, right)


def generate_random_terminal(num_variables):
    """
    Generate a random terminal node.

    A terminal node is either a variable or an integer constant.
    Each type is selected with equal probability.

    Parameters
    ----------
    num_variables : int
        Number of variables available for selection.

    Returns
    -------
    Node
        Random variable or constant node.
    """
    if random.random() < 0.5:
        # Get random variable based on # of possible variables for that dataset
        variable = random.randrange(num_variables)
        return Node("variable", variable)

    # For constants, generate random # btwn -10 and 10
    # NOTE: Could be potential place to adjust for experimentation
    constant = random.randint(-10, 10)
    return Node("constant", constant)

def generate_random_tree(max_depth, num_variables, current_depth=0):
    """
    Generate a random symbolic expression tree recursively.

    Parameters
    ----------
    max_depth : int
        Maximum allowed depth of the generated tree.
    num_variables : int
        Number of variables available for terminal nodes.
    current_depth : int, optional
        Current recursion depth. Defaults to 0.

    Returns
    -------
    Node
        Root node of the generated expression tree.
    """
    # If we reach max depth, force terminal node
    if current_depth == max_depth:
        return generate_random_terminal(num_variables)
    
    # Otherwise randomly choose operator or terminal
    # NOTE: Currently set to 50% chance of being operator, potential area to adjust if needed
    if random.random() < 0.5:
        operator = random.choice(OPERATORS)

        left = generate_random_tree(
            max_depth,
            num_variables,
            current_depth + 1
        )
        right = generate_random_tree(
            max_depth,
            num_variables,
            current_depth + 1
        )
        return Node("operator", operator, (left, right))

    return generate_random_terminal(num_variables) 

def generate_population(num_trees, max_depth, num_variables):
    """
    Generate an initial population of random expression trees.

    Parameters
    ----------
    num_trees : int
        Number of trees to generate.
    max_depth : int
        Maximum allowed depth of each tree.
    num_variables : int
        Number of variables available for terminal nodes.

    Returns
    -------
    list of Node
        Population of randomly generated expression trees.
    """
    population = []
    for _ in range(num_trees):
        tree = generate_random_tree(max_depth, num_variables)
        population.append(tree)
    
    return population

def division_rule(a, b):
    """
    Perform protected element-wise division.

    Values with denominators close to zero return 1 instead of
    performing division.

    Parameters
    ----------
    a : numpy.ndarray
        Numerator values.
    b : numpy.ndarray
        Denominator values.

    Returns
    -------
    numpy.ndarray
        Result of protected element-wise division.
    """
    result = np.ones(len(b))
    for i in range(len(b)):
        if abs(b[i]) >= 1e-6:
            result[i] = a[i] / b[i]
    return result


def size(node):
    """
    Count the number of nodes in an expression tree.

    Parameters
    ----------
    node : Node
        Root node of the expression tree.

    Returns
    -------
    int
        Total number of nodes in the tree.
    """
    if node.kind == "variable" or node.kind == "constant":
        return 1
    if node.kind == "operator":
        return 1 + size(node.children[0]) + size(node.children[1])

def depth(node):
    """
    Calculate the depth of an expression tree.

    Terminal nodes have depth 0.

    Parameters
    ----------
    node : Node
        Root node of the expression tree.

    Returns
    -------
    int
        Maximum number of edges from the root to a leaf.
    """
    if node.kind == "variable" or node.kind == "constant":
        return 0
    if node.kind == "operator":
        return 1 + max(depth(node.children[0]), depth(node.children[1]))
    
def to_string(node):
    """
    Convert an expression tree to a readable mathematical expression.

    Parameters
    ----------
    node : Node
        Root node of the expression tree.

    Returns
    -------
    str
        String representation of the expression.
    """
    if node.kind == "variable":
        return "x" + str(node.value + 1)
    if node.kind == "constant":
        return str(node.value)
    if node.kind == "operator":
        return "(" + to_string(node.children[0]) + " " + node.value + " " + to_string(node.children[1]) + ")"
    
def print_tree(node, prefix="", is_left=True, is_root=True):
    """
    Print an expression tree in a sideways visual format.

    Parameters
    ----------
    node : Node
        Root node of the expression tree.
    prefix : str, optional
        Spacing and branch characters used for recursive formatting.
    is_left : bool, optional
        Whether the current node is a left child.
    is_root : bool, optional
        Whether the current node is the root of the tree.

    Returns
    -------
    None
        The tree is printed directly to standard output.
    """
    if node.kind == "variable":
        label = "x" + str(node.value + 1)
    else:
        label = str(node.value)

    if node.kind == "operator":
        print_tree(
            node.children[1],
            prefix + ("" if is_root else ("│   " if is_left else "    ")),
            False,
            False
        )

    if is_root:
        print(label)
    else:
        print(prefix + ("└── " if is_left else "┌── ") + label)

    if node.kind == "operator":
        print_tree(
            node.children[0],
            prefix + ("" if is_root else ("    " if is_left else "│   ")),
            True,
            False
        )

if __name__ == "__main__":
    x = np.array([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])

    # Test random tree generation
    random_tree = generate_random_tree(
        max_depth=4,
        num_variables=3
    )

    print("Random tree:")
    print(to_string(random_tree))
    print("evaluation:", evaluate(random_tree, x))
    print("size:", size(random_tree))
    print("depth:", depth(random_tree))
    print()

    # Test random population generation
    population = generate_population(
        num_trees=10,
        max_depth=5,
        num_variables=3
    )

    print("Random population:")
    for i, tree in enumerate(population, start=1):
        print(f"\nTree {i}")
        print("Expression:", to_string(tree))
        print("Size:", size(tree))
        print("Depth:", depth(tree))
        print_tree(tree)

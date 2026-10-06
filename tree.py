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


def generate_random_terminal(num_variables, integer_constants):
    """
    Generate a random terminal node: a variable or an ephemeral random
    constant (ERC), each with equal probability.

    integer_constants : bool
        True for whole-number constants (dataset 1),
        False for real-valued constants (dataset 2).
    """
    if random.random() < 0.5:
        return Node("variable", random.randrange(num_variables))

    # NOTE: constant ranges are a design choice; adjust during tuning
    if integer_constants:
        return Node("constant", random.randint(-10, 10))
    return Node("constant", random.uniform(-10, 10))


def generate_grow_tree(max_depth, num_variables, integer_constants, current_depth=0):
    """
    Grow method (Koza, 1992): each node is randomly an operator or a
    terminal, so branches end at different depths.
    """
    if current_depth == max_depth:
        return generate_random_terminal(num_variables, integer_constants)

    if random.random() < 0.5:
        operator = random.choice(OPERATORS)
        left = generate_grow_tree(max_depth, num_variables, integer_constants, current_depth + 1)
        right = generate_grow_tree(max_depth, num_variables, integer_constants, current_depth + 1)
        return Node("operator", operator, (left, right))

    return generate_random_terminal(num_variables, integer_constants)


def generate_full_tree(max_depth, num_variables, integer_constants, current_depth=0):
    """
    Full method (Koza, 1992): every branch reaches max_depth, so all
    leaves sit on the bottom level.
    """
    if current_depth == max_depth:
        return generate_random_terminal(num_variables, integer_constants)

    operator = random.choice(OPERATORS)
    left = generate_full_tree(max_depth, num_variables, integer_constants, current_depth + 1)
    right = generate_full_tree(max_depth, num_variables, integer_constants, current_depth + 1)
    return Node("operator", operator, (left, right))


def generate_population(num_trees, min_depth, max_depth, num_variables, integer_constants):
    """
    Ramped half-and-half (Koza, 1992): cycles through depths
    min_depth..max_depth, alternating full and grow on each pass so
    every depth gets both methods.
    """
    population = []
    depths = list(range(min_depth, max_depth + 1))
    for i in range(num_trees):
        d = depths[i % len(depths)]
        if (i // len(depths)) % 2 == 0:
            population.append(generate_full_tree(d, num_variables, integer_constants))
        else:
            population.append(generate_grow_tree(d, num_variables, integer_constants))
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
    random.seed(370)

    x = np.array([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])

    # Test random tree generation (grow method)
    random_tree = generate_grow_tree(
        max_depth=4,
        num_variables=3,
        integer_constants=True
    )

    print("Random tree:")
    print(to_string(random_tree))
    print("evaluation:", evaluate(random_tree, x))
    print("size:", size(random_tree))
    print("depth:", depth(random_tree))
    print()

    # Test ramped half-and-half population
    population = generate_population(
        num_trees=10,
        min_depth=2,
        max_depth=5,
        num_variables=3,
        integer_constants=False
    )

    print("Random population:")
    for i, tree in enumerate(population, start=1):
        print(f"\nTree {i}")
        print("Expression:", to_string(tree))
        print("Size:", size(tree))
        print("Depth:", depth(tree))
        print_tree(tree)
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

    Parameters
    ----------
    num_variables : int
        Number of variables available for selection.
    integer_constants : bool
        True for whole-number constants (dataset 1),
        False for real-valued constants (dataset 2).

    Returns
    -------
    Node
        Random variable or constant node.
    """
    if random.random() < 0.5:
        return Node("variable", random.randrange(num_variables))

    # NOTE: constant ranges are a design choice; adjust during tuning
    if integer_constants:
        return Node("constant", random.randint(-10, 10))
    return Node("constant", random.uniform(-10, 10))


def generate_grow_tree(max_depth, num_variables, integer_constants, current_depth=0):
    """
    Generate a random expression tree with the grow method, following
    the random construction process in Koza's symbolic regression
    tutorial: each node is randomly chosen to be an operator or a
    terminal, and choosing a terminal ends that branch. Branches are
    forced to end at max_depth.

    Parameters
    ----------
    max_depth : int
        Maximum allowed depth of the generated tree.
    num_variables : int
        Number of variables available for terminal nodes.
    integer_constants : bool
        True for whole-number constants, False for real-valued constants.
    current_depth : int, optional
        Current recursion depth. Defaults to 0.

    Returns
    -------
    Node
        Root node of the generated expression tree.
    """
    # If we reach max depth, force a terminal node
    if current_depth == max_depth:
        return generate_random_terminal(num_variables, integer_constants)

    # The root is always an operator (as in the tutorial's example, which
    # starts by choosing a function for the root), so no tree is a single
    # terminal. Below the root: 50% chance of an operator, 50% terminal.
    # NOTE: potential area to adjust if needed
    if current_depth == 0 or random.random() < 0.5:
        operator = random.choice(OPERATORS)
        left = generate_grow_tree(max_depth, num_variables, integer_constants, current_depth + 1)
        right = generate_grow_tree(max_depth, num_variables, integer_constants, current_depth + 1)
        return Node("operator", operator, (left, right))

    return generate_random_terminal(num_variables, integer_constants)


def generate_population(num_trees, max_depth, num_variables, integer_constants):
    """
    Generate an initial population of random expression trees, each
    built with the grow method.

    Parameters
    ----------
    num_trees : int
        Number of trees to generate.
    max_depth : int
        Maximum allowed depth of each tree.
    num_variables : int
        Number of variables available for terminal nodes.
    integer_constants : bool
        True for whole-number constants, False for real-valued constants.

    Returns
    -------
    list of Node
        Population of randomly generated expression trees.
    """
    population = []
    for _ in range(num_trees):
        tree = generate_grow_tree(max_depth, num_variables, integer_constants)
        population.append(tree)
    return population


def division_rule(a, b):
    """
    Perform protected element-wise division.

    Values with denominators close to zero return 1 instead of
    performing division (Koza's protected division convention).

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
    
## Mutations, crossovers, reproductions

## Returns subtree at a given number
def get_subtree(node, index):
    if index == 0:
        return node
    index = index - 1
    left_size = size(node.children[0])
    if index < left_size:
        return get_subtree(node.children[0], index)
    else:
        return get_subtree(node.children[1], index - left_size)
## Returns a new tree, with a subtree at a given index
def replace_subtree(node, index, new_subtree):
    if index == 0:
        return new_subtree
    index = index - 1
    left_size = size(node.children[0])
    if index < left_size:
        new_left = replace_subtree(node.children[0], index, new_subtree)
        return Node("operator", node.value, (new_left, node.children[1]))
    else:
        new_right = replace_subtree(node.children[1], index - left_size, new_subtree)
        return Node("operator", node.value, (node.children[0], new_right))
## Subtree mutation, max_new_depth should be relatively small value so its a modest change
def subtree_mutation(tree, max_new_depth, num_variables, integer_constants):
    index = random.randrange(size(tree))
    new_subtree = generate_grow_tree(max_new_depth, num_variables, integer_constants)
    return replace_subtree(tree, index, new_subtree)
## Subtree crossover
def subtree_crossover (parent1, parent2):
    i = random.randrange(size(parent1))
    j = random.randrange(size(parent2))
    child1 = replace_subtree(parent1, i, get_subtree(parent2, j))
    child2 = replace_subtree(parent2, j, get_subtree(parent1, i))
    return child1, child2



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

    # Test random population generation (grow method)
    population = generate_population(
        num_trees=10,
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

        # Test subtree helpers on a tree with known node numbers
    print("\nSubtree helper test:")
    ex = Node("operator", "+", (
        Node("operator", "*", (Node("variable", 0), Node("variable", 0))),
        Node("constant", 1)
    ))
    for i in range(size(ex)):
        print(i, to_string(get_subtree(ex, i)))
    print(to_string(replace_subtree(ex, 4, Node("constant", 7))))   # ((x1 * x1) + 7)

    # Immutability tests: operators must never change their parents
    for _ in range(100):
        p1 = generate_grow_tree(4, 3, True)
        p2 = generate_grow_tree(4, 3, True)
        before1, before2 = to_string(p1), to_string(p2)

        child1, child2 = subtree_crossover(p1, p2)
        mutant = subtree_mutation(p1, 2, 3, True)

        assert to_string(p1) == before1
        assert to_string(p2) == before2
        assert size(child1) + size(child2) == size(p1) + size(p2)
    print("100 immutability checks passed")
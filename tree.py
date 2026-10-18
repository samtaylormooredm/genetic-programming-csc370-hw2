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


def generate_random_terminal(num_variables, integer_constants=True):
    """
    Generate a random terminal node.

    A terminal node is either a variable or an integer constant.
    Each type is selected with equal probability. Koza'a tutorial suggests
    constants -5.0 to 5.0, where we use -10 to 10.

    Parameters
    ----------
    num_variables : int
        Number of variables available for selection.
    integer_constants : bool, optional
        True for whole number constants (dataset 1),
        False for real valued constants (dataset 2). Defaults to True

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
    if integer_constants:
        return Node("constant", random.randint(-10, 10))

    if random.random() < 0.5:
        value = random.uniform(-10, 10)
    else:
        sign = random.choice([-1.0, 1.0])
        value = sign * 10.0 ** random.uniform(-12, 12)

    return Node("constant", value)

def generate_random_tree(max_depth, num_variables, integer_constants=True, current_depth=0):
    """
    Generate a random symbolic expression tree recursively. Branches are forced
    to end at max_depth.

    Parameters
    ----------
    max_depth : int
        Maximum allowed depth of the generated tree.
    num_variables : int
        Number of variables available for terminal nodes.
    integer_constants : bool, optional
        True for whole number constants, False for real valued constants.
    current_depth : int, optional
        Current recursion depth. Defaults to 0.

    Returns
    -------
    Node
        Root node of the generated expression tree.
    """
    # If we reach max depth, force terminal node
    if current_depth >= max_depth:
        return generate_random_terminal(num_variables, integer_constants)
    
    # Otherwise randomly choose operator or terminal
    # NOTE: Currently set to 50% chance of being operator, potential area to adjust if needed
    if random.random() < 0.5:
        operator = random.choice(OPERATORS)

        left = generate_random_tree(
            max_depth,
            num_variables,
            integer_constants,
            current_depth + 1
        )
        right = generate_random_tree(
            max_depth,
            num_variables,
            integer_constants,
            current_depth + 1
        )
        return Node("operator", operator, (left, right))

    return generate_random_terminal(num_variables, integer_constants) 

def generate_population(num_trees, max_depth, num_variables, integer_constants=True):
    """
    Generate an initial population of random expression trees. Trees that come
    out as a single terminal at the root are regenerated so every starting tree
    has an operator at its root.

    Parameters
    ----------
    num_trees : int
        Number of trees to generate.
    max_depth : int
        Maximum allowed depth of each tree.
    num_variables : int
        Number of variables available for terminal nodes.
    integer_constants : bool, optional
        true for whole number constants, False for real
        valued constants

    Returns
    -------
    list of Node
        Population of randomly generated expression trees.
    """
    population = []
    for _ in range(num_trees):
        tree = generate_random_tree(max_depth, num_variables, integer_constants)
        while tree.kind != "operator":
            tree = generate_random_tree(max_depth, num_variables, integer_constants)
        population.append(tree)
    
    return population

def division_rule(a, b):
    """
    Perform protected element-wise division.

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
    result = np.ones_like(a, dtype=float)
    np.divide(a, b, out=result, where=np.abs(b) >= 1e-6)
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

def get_paths(node, current_path=()):
    """
    Collect paths to every node in the tree.

    Parameters
    ----------
    node : Node
        Root of the subtree.
    current_path : tuple of int, optional
        Path to this node. Defaults to ().

    Returns
    -------
    list of tuple of int
        Paths to this node and its descendants.
    """
    paths = [current_path]

    for i, child in enumerate(node.children):
        paths.extend(get_paths(child, current_path + (i,)))

    return paths


def get_subtree(node, path):
    """
    Retrieve the subtree at a given path.

    Parameters
    ----------
    node : Node
        Root of the tree.
    path : tuple of int
        Child indices to follow. () selects the root.

    Returns
    -------
    Node
        Selected subtree.
    """
    for child_index in path:
        node = node.children[child_index]

    return node


def replace_subtree(node, path, replacement):
    """
    Replace a subtree while preserving the original tree.

    Parameters
    ----------
    node : Node
        Root of the original tree.
    path : tuple of int
        Child indices to follow. () replaces the root.
    replacement : Node
        Replacement subtree.

    Returns
    -------
    Node
        Tree containing the replacement.
    """
    if not path:
        return replacement

    child_index = path[0]
    children = list(node.children)

    children[child_index] = replace_subtree(
        children[child_index],
        path[1:],
        replacement,
    )

    return Node(node.kind, node.value, tuple(children))

def mutate(node, max_depth, num_variables, integer_constants=True):
    """
    Mutate an expression while preserving the original tree.

    Integer-constant mode retains the original subtree mutation.
    Real-constant mode uses:
      - 30% constant mutation, when constants exist.
      - Otherwise, subtree replacement capped at depth 3.
    """

    paths = get_paths(node)

    # Dataset 2: sometimes adjust one constant without changing structure.
    if not integer_constants:
        constant_paths = [
            path
            for path in paths
            if get_subtree(node, path).kind == "constant"
        ]

        if constant_paths and random.random() < 0.3:
            path = random.choice(constant_paths)
            old_value = get_subtree(node, path).value

            if old_value == 0:
                new_value = random.gauss(0, 0.1)
            else:
                # Relative adjustment works for both tiny and large constants.
                new_value = old_value * 10.0 ** random.gauss(0, 0.1)

            if np.isfinite(new_value):
                return replace_subtree(
                    node,
                    path,
                    Node("constant", new_value),
                )

    # Subtree mutation: select a location and generate a replacement.
    path = random.choice(paths)
    remaining_depth = max_depth - len(path)

    # Keep dataset 1's original behavior.
    # For dataset 2, make newly generated branches more local.
    replacement_depth = (
        remaining_depth
        if integer_constants
        else min(remaining_depth, 3)
    )

    replacement = generate_random_tree(
        max_depth=replacement_depth,
        num_variables=num_variables,
        integer_constants=integer_constants,
    )

    return replace_subtree(node, path, replacement)

def crossover(parent1, parent2, max_depth):
    """
    Swap random subtrees between two parents.

    Parameters
    ----------
    parent1, parent2 : Node
        Roots of the parent trees.
    max_depth : int
        Maximum allowed offspring depth.

    Returns
    -------
    tuple of Node
        Two offspring. An offspring exceeding the depth limit
        is replaced with its original parent.
    """
    path1 = random.choice(get_paths(parent1))
    path2 = random.choice(get_paths(parent2))

    subtree1 = get_subtree(parent1, path1)
    subtree2 = get_subtree(parent2, path2)

    child1 = replace_subtree(parent1, path1, subtree2)
    child2 = replace_subtree(parent2, path2, subtree1)

    # Keep the original parent if its offspring grows too deep.
    if depth(child1) > max_depth:
        child1 = parent1

    if depth(child2) > max_depth:
        child2 = parent2

    return child1, child2

if __name__ == "__main__":
    random.seed(42)

    max_depth = 5
    num_variables = 1
    x = np.array([[1.0], [2.0], [3.0]])

    parent1 = Node(
        "operator",
        "+",
        (Node("variable", 0), Node("constant", 3)),
    )
    parent2 = Node(
        "operator",
        "*",
        (Node("variable", 0), Node("constant", 2)),
    )

    # Check evaluation and tree measurements.
    assert np.allclose(evaluate(parent1, x), [4, 5, 6])
    assert size(parent1) == 3
    assert depth(parent1) == 1

    # Check protected division.
    assert np.allclose(
        division_rule(np.array([6.0, 6.0]), np.array([2.0, 0.0])),
        [3, 1],
    )

    # Check subtree helpers.
    assert get_paths(parent1) == [(), (0,), (1,)]
    assert get_subtree(parent1, (1,)) == Node("constant", 3)
    replaced = replace_subtree(parent1, (1,), Node("constant", 7))
    assert to_string(replaced) == "(x1 + 7)"

    # Check generation, mutation, and crossover on varied trees.
    population = generate_population(20, max_depth, num_variables)
    assert len(population) == 20

    originals = tuple(population)

    for tree in population:
        assert depth(tree) <= max_depth
        assert evaluate(tree, x).shape == (len(x),)

        mutated = mutate(tree, max_depth, num_variables)
        assert depth(mutated) <= max_depth
        assert evaluate(mutated, x).shape == (len(x),)

        child1, child2 = crossover(
            tree, random.choice(population), max_depth
        )
        for child in (child1, child2):
            assert depth(child) <= max_depth
            assert evaluate(child, x).shape == (len(x),)

    # Check that the original trees remain unchanged.
    assert tuple(population) == originals
    assert to_string(parent1) == "(x1 + 3)"
    assert to_string(parent2) == "(x1 * 2)"

    # Display one example of each operation.
    mutated = mutate(parent1, max_depth, num_variables)
    child1, child2 = crossover(parent1, parent2, max_depth)

    for label, tree in [
        ("Original", parent1),
        ("Mutated", mutated),
        ("Crossover child 1", child1),
        ("Crossover child 2", child2),
    ]:
        print(f"\n{label}: {to_string(tree)}")
        print_tree(tree)

    print("\nAll tree checks passed.")

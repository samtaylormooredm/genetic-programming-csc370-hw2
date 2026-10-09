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
    Each type is selected with equal probability.

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
        return Node("constant", random.randint(-10,10))
    return Node("constant", random.uniform(-10,10))

def generate_random_tree(max_depth, num_variables, integer_constants=True, current_depth=0):
    """
    Generate a random symbolic expression tree recursively.

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
        return generate_random_terminal(num_variables)
    
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
    Generate an initial population of random expression trees.

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
    Replace a random subtree with a randomly generated tree.

    Parameters
    ----------
    node : Node
        Root of the original tree.
    max_depth : int
        Maximum depth allowed for the resulting tree.
    num_variables : int
        Number of available input variables.
    integer_constants : bool, optional
        True for whole-number constants, False for real-valued constants.

    Returns
    -------
    Node
        Mutated tree.
    """
    path = random.choice(get_paths(node))

    # Each step in the path uses one level of the depth limit.
    remaining_depth = max_depth - len(path)

    replacement = generate_random_tree(
        max_depth=remaining_depth,
        num_variables=num_variables,
        integer_constants = integer_constants
    )

    return replace_subtree(node, path, replacement)

def crossover(parent1, parent2):
    """
    Subtree crossover where a random point in each 
    parent is picked, then swap the subtrees there.

    Parameters
    ----------
    parent1, parent2: Node
        Roots of the two parent trees.
    
    Returns
    -------
    tuple of Node
        The two children.

    """
    path1 = random.choice(get_paths(parent1))
    path2 = random.choice(get_paths(parent2))

    child1 = replace_subtree(parent1, path1, get_subtree(parent2, path2))
    child2 = replace_subtree(parent2, path2, get_subtree(parent1, path1))

    return child1, child2

if __name__ == "__main__":
    # Test mutation with reproducible randomness.
    random.seed(42)
    max_depth = 5

    original = Node(
        "operator",
        "+",
        (Node("variable", 0), Node("constant", 3)),
    )
    original_expression = to_string(original)
    test_x = np.array([[1.0], [2.0], [3.0]])

    for i in range(1, 11):
        mutated = mutate(
            original,
            max_depth=max_depth,
            num_variables=1,
        )

        depth_ok = depth(mutated) <= max_depth
        original_unchanged = to_string(original) == original_expression

        print(f"\nMutation {i}")
        print("Original:", to_string(original))
        print("Mutated:", to_string(mutated))
        print("Size:", size(mutated))
        print("Depth:", depth(mutated))
        print("Predictions:", evaluate(mutated, test_x))
        print("Depth limit respected:", depth_ok)
        print("Original unchanged:", original_unchanged)

        assert depth_ok, "Mutation exceeded the depth limit."
        assert original_unchanged, "Mutation changed the original tree."

    print("\nAll mutation checks passed.")

    for _ in range(100):
        p1 = generate_random_tree(4, 3)
        p2 = generate_random_tree(4, 3)
        before1, before2 = to_string(p1), to_string(p2)
 
        child1, child2 = crossover(p1, p2)
 
        assert to_string(p1) == before1, "Crossover changed parent 1."
        assert to_string(p2) == before2, "Crossover changed parent 2."
        assert size(child1) + size(child2) == size(p1) + size(p2)
 
    print("All crossover checks passed.")
 
    # Test population: real-valued constants, no single-node trees
    population = generate_population(
        num_trees=20, max_depth=5, num_variables=3, integer_constants=False
    )
    assert all(tree.kind == "operator" for tree in population)
    print("Population example:", to_string(population[0]))
    print("All population checks passed.")    

        # Real-valued constants actually appear when requested
    def constants_in(node):
        if node.kind == "constant":
            return [node.value]
        found = []
        for child in node.children:
            found += constants_in(child)
        return found

    real_pop = generate_population(50, 5, 3, integer_constants=False)
    all_constants = [c for tree in real_pop for c in constants_in(tree)]
    assert any(isinstance(c, float) for c in all_constants), "No real-valued constants generated."
    print("All constant-type checks passed.")
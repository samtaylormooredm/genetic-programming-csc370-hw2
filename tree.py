from dataclasses import dataclass
import numpy as np
import random

## Node class, where a node can be a kind (operator, variable 
## or constant), the value of that node, and the children of 
## that node if that are any

## Function for immutability


@dataclass(frozen=True)
class Node:
  kind: str
  value: object
  children: tuple = ()


OPERATORS = ["+", "-", "*", "/"]
  
## Established a division rule so that division by zero doesn't
def division_rule(a, b):
    result = np.ones(len(b))
    for i in range(len(b)):
        if abs(b[i]) >= 1e-6:
            result[i] = a[i] / b[i]
    return result

## Determine predicted y-values based on an input node
## with domain of dataset
def evaluate(node, x):
    ## If node is a variable, return that variable's column of data
    if node.kind == "variable":
        return x[:, node.value]
    ## If node is a constant, return that number repeating for 
    ## number of rows
    if node.kind == "constant":
        return np.ones(x.shape[0]) * node.value
    ## If node is an operator, evaluate the children of the node
    if node.kind == "operator":
        left = evaluate(node.children[0], x)
        right = evaluate(node.children[1], x)
        ## Perform operations on left and right evaluations
        if node.value == "+":
            return left + right
        if node.value == "-":
            return left - right
        if node.value == "*":
            return left * right
        if node.value == "/":
            return division_rule(left, right)
## Return how many nodes are in the tree
def size(node):
    if node.kind == "variable" or node.kind == "constant":
        return 1
    if node.kind == "operator":
        return 1 + size(node.children[0]) + size(node.children[1])
## Return how many levels are in the tree
def depth(node):
    if node.kind == "variable" or node.kind == "constant":
        return 0
    if node.kind == "operator":
        return 1 + max(depth(node.children[0]), depth(node.children[1]))
## Return formula as readable text
def to_string(node):
    if node.kind == "variable":
        return "x" + str(node.value + 1)
    if node.kind == "constant":
        return str(node.value)
    if node.kind == "operator":
        return "(" + to_string(node.children[0]) + " " + node.value + " " + to_string(node.children[1]) + ")"

# Generate random terminal node
def generate_random_terminal(num_variables):
    if random.random() < 0.5:
        variable = random.randrange(num_variables)
        return Node("variable", variable)

    # NOTE: Possible range modification here
    constant = random.randint(-10, 10)
    return Node("constant", constant)

# Generate random trees recursively
    # make possible_operators a constant defined at top of class
def generate_random_tree(max_depth, num_variables, current_depth=0):
    # If we reach max depth, force terminal node
    if current_depth == max_depth:
        return generate_random_terminal(num_variables)
    
    # Otherwise randomly choose operator or terminal
    # NOTE: Currnently set to 50% chance of being operator, adjust for future?
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
    
if __name__ == "__main__":
    x = np.array([
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]
    ])

    # x1
    tree1 = Node("variable", 0)

    # x1 + 3
    tree2 = Node(
        "operator",
        "+",
        (Node("variable", 0), Node("constant", 3))
    )

    # x1 * x2
    tree3 = Node(
        "operator",
        "*",
        (Node("variable", 0), Node("variable", 1))
    )

    # (x1 + x2) * x3
    tree4 = Node(
        "operator",
        "*",
        (
            Node(
                "operator",
                "+",
                (Node("variable", 0), Node("variable", 1))
            ),
            Node("variable", 2)
        )
    )

    for tree in [tree1, tree2, tree3, tree4]:
        print(to_string(tree))
        print("evaluation:", evaluate(tree, x))
        print("size:", size(tree))
        print("depth:", depth(tree))
        print()

    # # Testing random tree generation
    tree = generate_random_tree(
    max_depth=4,
    num_variables=3
    )

    print(to_string(tree))
    print("size:", size(tree))
    print("depth:", depth(tree))
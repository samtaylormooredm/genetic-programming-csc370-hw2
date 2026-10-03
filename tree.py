from dataclasses import dataclass
import numpy as np

## Node class, where a node can be a kind (operator, variable 
## or constant), the value of that node, and the children of 
## that node if that are any

## Function for immutability
@dataclass(frozen=True)
class Node:
  kind: str
  value: object
  children: tuple = ()

## Established a division rule so that division by zero doesn't
## break fitness
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


    
if __name__ == "__main__":
    # Build the tree for ((x1 * x1) + 1)
    x1_a = Node("variable", 0)
    x1_b = Node("variable", 0)
    times = Node("operator", "*", (x1_a, x1_b))
    one = Node("constant", 1)
    tree = Node("operator", "+", (times, one))

    # Two data points from dataset 1, as a one-column table
    x = np.array([[-0.66], [2.99]])

    print("evaluate:", evaluate(tree, x))     # [1.4356 9.9401]
    print("size:", size(tree))                # 5
    print("depth:", depth(tree))              # 2
    print("to_string:", to_string(tree))      # ((x1 * x1) + 1)

    # Division tests
    zero_tree = Node("operator", "/", (Node("variable", 0), Node("constant", 0)))
    print("x1 / 0:", evaluate(zero_tree, x))  # [1. 1.]

    neg_tree = Node("operator", "/", (Node("variable", 0), Node("constant", -2)))
    print("x1 / -2:", evaluate(neg_tree, x))  # [ 0.33  -1.495]
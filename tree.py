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

    

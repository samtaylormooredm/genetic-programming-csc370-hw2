from tree import Node, print_tree


def constant(value):
    return Node("constant", value)


def variable():
    return Node("variable", 0)  # x1


def operator(symbol, left, right):
    return Node("operator", symbol, (left, right))


# Seed 370: (9 - x) + ((5 + x * (x + -4)) - x)
seed370 = operator(
    "+",
    operator("-", constant(9), variable()),
    operator(
        "-",
        operator(
            "+",
            constant(5),
            operator(
                "*",
                variable(),
                operator("+", variable(), constant(-4)),
            ),
        ),
        variable(),
    ),
)

# Seed 379: (8 + 6) + x * (x + -6)
seed379 = operator(
    "+",
    operator("+", constant(8), constant(6)),
    operator(
        "*",
        variable(),
        operator("+", variable(), constant(-6)),
    ),
)

print("Seed 370: 13 nodes")
print_tree(seed370)

print("\nSeed 379: 9 nodes")
print_tree(seed379)
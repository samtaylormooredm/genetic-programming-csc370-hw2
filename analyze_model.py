"""Analyze a saved GP formula using the original test split."""

import argparse
import ast
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from data import load_dataset, split_data
from tree import Node, evaluate, size


def parse_formula(formula):
    """Convert the printed arithmetic expression back into a Node tree."""
    operators = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
    }

    def convert(expression):
        if isinstance(expression, ast.Constant):
            if type(expression.value) not in (int, float):
                raise ValueError("Constants must be numeric.")
            return Node("constant", expression.value)

        if isinstance(expression, ast.Name):
            if expression.id not in ("x1", "x2", "x3"):
                raise ValueError(f"Unknown variable: {expression.id}")
            return Node("variable", int(expression.id[1:]) - 1)

        if isinstance(expression, ast.UnaryOp):
            child = convert(expression.operand)
            if child.kind != "constant":
                raise ValueError("Only signed constants are supported.")
            if isinstance(expression.op, ast.USub):
                return Node("constant", -child.value)
            if isinstance(expression.op, ast.UAdd):
                return child

        if isinstance(expression, ast.BinOp):
            operator = operators.get(type(expression.op))
            if operator is not None:
                return Node(
                    "operator",
                    operator,
                    (
                        convert(expression.left),
                        convert(expression.right),
                    ),
                )

        raise ValueError("Unsupported expression in formula.")

    return convert(ast.parse(formula, mode="eval").body)


def load_model(output_path, seed):
    """Extract one seed's best formula from a completed output log."""
    text = Path(output_path).read_text()

    sections = re.split(r"--- Seed (\d+) ---", text)
    for i in range(1, len(sections), 2):
        if int(sections[i]) == seed:
            match = re.search(
                r"^Best formula:\s*(.+)$",
                sections[i + 1],
                flags=re.MULTILINE,
            )
            if match:
                return parse_formula(match.group(1))
            raise ValueError(f"Seed {seed} has no completed formula.")

    raise ValueError(f"Seed {seed} was not found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_file", help="Completed SLURM/text output")
    parser.add_argument("--seed", type=int, default=371)
    args = parser.parse_args()

    model = load_model(args.output_file, args.seed)

    # Match the unfiltered, unscaled experiment exactly.
    x, y = load_dataset("dataset2.csv")
    x_train, y_train, x_test, y_test = split_data(
        x, y, train_fraction=0.7, seed=42
    )

    # evaluate() preserves the original protected-division behavior.
    predictions = evaluate(model, x_test)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model produced nonfinite test predictions.")

    residuals = predictions - y_test
    scaled_residuals = residuals / (1.0 + np.abs(y_test))

    weights = 1.0 / (1.0 + np.abs(y_train)) ** 2
    constant = np.sum(weights * y_train) / np.sum(weights)
    baseline_error = np.mean(
        ((constant - y_test) / (1.0 + np.abs(y_test))) ** 2
    )

    print("Seed:", args.seed)
    print("Tree size:", size(model))
    print("Test raw MSE:", np.mean(residuals ** 2))
    print("Test scaled error:", np.mean(scaled_residuals ** 2))
    print("Constant test scaled error:", baseline_error)
    print("Median absolute error:", np.median(np.abs(residuals)))

    # Show examples spread across the observed target range.
    indices = np.argsort(y_test)
    examples = indices[
        np.linspace(0, len(indices) - 1, 10, dtype=int)
    ]

    print("\nExamples across the target range:")
    print("Actual y | Predicted y | x3")
    for i in examples:
        print(
            f"{y_test[i]:.6g} | "
            f"{predictions[i]:.6g} | "
            f"{x_test[i, 2]:.6g}"
        )

    groups = [
        ("x3 = 0", x_test[:, 2] == 0),
        ("0 < x3 <= 0.1",
         (x_test[:, 2] > 0) & (x_test[:, 2] <= 0.1)),
        ("0.1 < x3 <= 1",
         (x_test[:, 2] > 0.1) & (x_test[:, 2] <= 1)),
        ("x3 > 1", x_test[:, 2] > 1),
    ]

    print("\nTest scaled error by x3:")
    for label, mask in groups:
        if np.any(mask):
            print(
                f"{label}: n={mask.sum()}, "
                f"error={np.mean(scaled_residuals[mask] ** 2):.6g}"
            )

    output_dir = Path("results") / (
        f"{Path(args.output_file).stem}_seed{args.seed}_analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save every test prediction for further inspection.
    np.savetxt(
        output_dir / "predictions.csv",
        np.column_stack((
            x_test, y_test, predictions,
            residuals, scaled_residuals,
        )),
        delimiter=",",
        header=(
            "x1,x2,x3,actual,predicted,"
            "prediction_minus_actual,scaled_residual"
        ),
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    zero_x3 = x_test[:, 2] == 0

    ax.scatter(
        y_test[~zero_x3], predictions[~zero_x3],
        s=10, alpha=0.3, label="x3 > 0",
    )
    ax.scatter(
        y_test[zero_x3], predictions[zero_x3],
        s=45, marker="x", color="orange", label="x3 = 0",
    )

    low = min(y_test.min(), predictions.min())
    high = max(y_test.max(), predictions.max())
    ax.plot([low, high], [low, high], "r--",
            label="Perfect prediction")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual y")
    ax.set_ylabel("Predicted y")
    ax.set_title(f"Dataset 2: seed {args.seed}, test predictions")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "actual_vs_predicted.png", dpi=200)
    plt.close(fig)

    print("\nAnalysis saved to:", output_dir)


if __name__ == "__main__":
    main()
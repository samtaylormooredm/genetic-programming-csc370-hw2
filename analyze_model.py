"""
Analyze a saved GP model without rerunning evolution.
"""

import argparse
import ast
import csv
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tree import apply_operator


def load_formula(path, seed):
    """Read one explicitly chosen seed; do not select using test results."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                if int(row["seed"]) == seed:
                    return row["formula"]
    else:
        sections = re.split(r"--- Seed (\d+) ---", path.read_text())
        for i in range(1, len(sections), 2):
            if int(sections[i]) == seed:
                match = re.search(r"^Best formula:\s*(.+)$", sections[i + 1], re.M)
                if match:
                    return match.group(1)
                raise ValueError(f"Seed {seed} has no completed formula.")
    raise ValueError(f"No formula found for seed {seed} in {path}.")


def predict(formula, x):
    """Interpret whitelisted arithmetic/functions using the training rules."""
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return np.full(len(x), float(node.value))
        if isinstance(node, ast.Name) and re.fullmatch(r"x[1-9]\d*", node.id):
            index = int(node.id[1:]) - 1
            if index >= x.shape[1]:
                raise ValueError(f"Variable {node.id} is absent from this dataset.")
            return x[:, index]
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -visit(node.operand)
            if isinstance(node.op, ast.UAdd):
                return visit(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
            symbol = symbols.get(type(node.op))
            if symbol is not None:
                return apply_operator(symbol, (left, right))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("sin", "cos", "exp", "log")
                and len(node.args) == 1 and not node.keywords):
            return apply_operator(node.func.id, (visit(node.args[0]),))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = visit(ast.parse(formula, mode="eval").body)
    if not np.all(np.isfinite(result)):
        raise ValueError("Model produced nonfinite predictions.")
    return result


def metrics(actual, predicted):
    residual = predicted - actual
    scaled = residual / (1.0 + np.abs(actual))
    return {
        "rows": len(actual),
        "mse": float(np.mean(residual ** 2)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "scaled_error": float(np.mean(scaled ** 2)),
        "fraction_within_0.01_times_1_plus_abs_y": float(np.mean(np.abs(scaled) <= .01)),
        "fraction_within_0.05_times_1_plus_abs_y": float(np.mean(np.abs(scaled) <= .05)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_file", help="Summary CSV or completed output log")
    parser.add_argument("--dataset", choices=("dataset1", "dataset2", "dataset3"), required=True)
    parser.add_argument("--seed", type=int, default=370)
    parser.add_argument("--data", type=Path, help="CSV path; default: data/DATASET.csv")
    parser.add_argument("--train-fraction", type=float)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    fraction = args.train_fraction
    if fraction is None:
        fraction = .8 if args.dataset == "dataset1" else .7
    if not 0 < fraction < 1:
        parser.error("--train-fraction must be between zero and one.")
    data_path = args.data or Path(__file__).resolve().parent / "data" / f"{args.dataset}.csv"
    values = np.loadtxt(data_path, delimiter=",", skiprows=1)
    x, y = values[:, :-1], values[:, -1]
    if not np.all(np.isfinite(values)):
        raise ValueError("Dataset contains nonfinite values.")
    expected_columns = 3 if args.dataset == "dataset2" else 1
    if x.shape[1] != expected_columns:
        raise ValueError(f"Expected {expected_columns} input columns.")

    indices = np.random.default_rng(args.split_seed).permutation(len(y))
    cutoff = int(len(y) * fraction)
    train, test = indices[:cutoff], indices[cutoff:]
    formula = load_formula(args.model_file, args.seed)
    train_prediction = predict(formula, x[train])
    test_prediction = predict(formula, x[test])
    report = {
        "dataset": args.dataset, "model_file": str(args.model_file),
        "seed": args.seed, "train_fraction": fraction,
        "split_seed": args.split_seed, "formula": formula,
        "preprocessing": "No filtering or input scaling",
        "train": metrics(y[train], train_prediction),
        "test": metrics(y[test], test_prediction),
    }
    if args.dataset == "dataset2":
        z = x[test, 2]
        groups = [
            ("x3 = 0", z == 0),
            ("0 < x3 <= 0.1", (z > 0) & (z <= .1)),
            ("0.1 < x3 <= 1", (z > .1) & (z <= 1)),
            ("x3 > 1", z > 1),
        ]
        report["test_by_x3"] = {
            label: metrics(y[test][mask], test_prediction[mask])
            for label, mask in groups if np.any(mask)
        }

    destination = args.output_dir or Path("results") / "analysis" / (
        f"{Path(args.model_file).stem}_seed{args.seed}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    residual = test_prediction - y[test]
    np.savetxt(
        destination / "test_predictions.csv",
        np.column_stack((test, x[test], y[test], test_prediction, residual,
                         residual / (1 + np.abs(y[test])))),
        delimiter=",", comments="",
        header=",".join(["source_row_index"] + [f"x{i+1}" for i in range(x.shape[1])]
                        + ["actual", "predicted", "prediction_minus_actual", "scaled_residual"]),
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    zero = x[test, 2] == 0 if args.dataset == "dataset2" else np.zeros(len(test), dtype=bool)
    axes[0].scatter(y[test][~zero], test_prediction[~zero], s=9, alpha=.35,
                    label="x3 > 0" if args.dataset == "dataset2" else "Test observations")
    if np.any(zero):
        axes[0].scatter(y[test][zero], test_prediction[zero], marker="x", color="orange",
                        s=40, label="x3 = 0")
    low = min(y[test].min(), test_prediction.min())
    high = max(y[test].max(), test_prediction.max())
    if low == high:
        low, high = low - 1, high + 1
    axes[0].plot([low, high], [low, high], "--", color="#b63c35", label="Perfect prediction")
    axes[0].set(xlabel="Actual y", ylabel="Predicted y", title="Actual versus predicted")
    axes[0].legend(fontsize=8)
    if args.dataset != "dataset2":
        axes[1].scatter(x[test, 0], residual, s=9, alpha=.35)
        axes[1].set(xlabel="Input x", ylabel="Prediction minus actual",
                    title="Residuals across the input range")
    else:
        axes[0].set_xscale("symlog", linthresh=1)
        axes[0].set_yscale("symlog", linthresh=1)
        axes[1].scatter(x[test, 2], residual / (1 + np.abs(y[test])), s=9, alpha=.35)
        axes[1].set(xlabel="Input x3", ylabel="Scaled residual",
                    title="Scaled residuals near zero")
    axes[1].axhline(0, color="#b63c35", linestyle="--")
    for axis in axes:
        axis.grid(alpha=.15)
    fig.suptitle(f"{args.dataset}: seed {args.seed}, test set only (n={len(test):,})")
    fig.tight_layout()
    fig.savefig(destination / "prediction_analysis.png", dpi=200)
    plt.close(fig)

    print(f"{args.dataset}, seed {args.seed}; split {len(train)}/{len(test)}")
    for label in ("train", "test"):
        item = report[label]
        print(f"{label}: MSE={item['mse']:.12g}, RMSE={item['rmse']:.8g}, "
              f"scaled error={item['scaled_error']:.12g}")
    print("Saved analysis to", destination)


if __name__ == "__main__":
    main()

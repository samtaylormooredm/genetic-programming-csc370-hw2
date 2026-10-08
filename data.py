from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent


def split_data(x, y, train_fraction=0.8, seed=42):
    """Split data into reproducible training and test sets."""
    random_gen = np.random.default_rng(seed)
    indices = random_gen.permutation(len(y))

    split_point = int(len(y) * train_fraction)
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    return (
        x[train_indices],
        y[train_indices],
        x[test_indices],
        y[test_indices],
    )
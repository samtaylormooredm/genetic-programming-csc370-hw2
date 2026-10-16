import numpy as np

from data import load_dataset, split_data

x, y = load_dataset("dataset2.csv")
x_train, y_train, _, _ = split_data(
    x, y, train_fraction=0.7, seed=42
)

x3 = x_train[:, 2]

# Check the values actually supplied in the CSV.
print("Training rows:", len(y_train))
print("Rows with x3 = 0:", np.sum(x3 == 0))
print("Smallest distinct x3 values:")
print(np.unique(x3)[:20])

# Inspect the observations with the largest targets.
largest = np.argsort(np.abs(y_train))[-10:][::-1]

print("\nTen largest targets:")
print("x1, x2, x3, y")
for i in largest:
    print(
        f"{x_train[i, 0]:.8g}, "
        f"{x_train[i, 1]:.8g}, "
        f"{x3[i]:.8g}, "
        f"{y_train[i]:.8g}"
    )

# Compare target distributions across ranges of x3.
groups = [
    ("x3 = 0", x3 == 0),
    ("0 < |x3| <= 0.01", (np.abs(x3) > 0) & (np.abs(x3) <= 0.01)),
    ("0.01 < |x3| <= 0.1", (np.abs(x3) > 0.01) & (np.abs(x3) <= 0.1)),
    ("0.1 < |x3| <= 1", (np.abs(x3) > 0.1) & (np.abs(x3) <= 1)),
    ("|x3| > 1", np.abs(x3) > 1),
]

print("\nTarget values grouped by x3:")
for label, mask in groups:
    values = y_train[mask]
    if values.size:
        print(
            f"{label}: n={values.size}, "
            f"median={np.median(values):.6g}, "
            f"max={np.max(values):.6g}"
        )
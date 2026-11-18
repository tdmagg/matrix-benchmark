
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

results = Path(__file__).resolve().parents[1] / "results"


def read_results(filename):
    with (results / filename).open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    n = np.array([int(row["n"]) for row in rows])
    dense = np.array([float(row["dense_ms"]) for row in rows])
    sparse = np.array([float(row["sparse_ms"]) for row in rows])

    return n, dense, sparse


fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for language, filename in (
    ("Python", "python_timings.csv"),
    ("C", "c_timings.csv")
):
    sizes, dense, sparse = read_results(filename)

    for ax, values in zip(axes, (dense, sparse)):
        ax.scatter(
            sizes, values,
            alpha=0.55,
            s=18,
            label=f"{language} measurements"
        )

        exponent, log_a = np.polyfit(
            np.log(sizes), np.log(values), 1
        )

        x = np.linspace(sizes.min(), sizes.max(), 300)
        trend = np.exp(log_a) * x ** exponent

        ax.plot(
            x, trend,
            label=f"{language} fitted trend"
        )


axes[0].set_title("Dense LU")
axes[1].set_title("Sparse LU")

for ax in axes:
    ax.set_xlabel("Matrix dimension n (n × n)")
    ax.set_ylabel("Median factorization time (ms)")
    ax.grid(alpha=0.25)
    ax.legend()

fig.suptitle("LU factorization: C versus Python")
fig.tight_layout()

output = results / "comparison.png"
fig.savefig(output, dpi=180)

plt.show()

print(f"Saved {output}")

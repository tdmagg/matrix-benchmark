import csv
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.linalg import lu_factor
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

TRIALS = 1000
SIZES = [25 + (225 * i) // 49 for i in range(50)]

def create_matrix(n):
    # creates a sparse matrix. close to a banded pattern, but has varying offsets
    A = 8.0 * np.eye(n)

    offsets = (1, 7, n // 3)

    for offset in offsets:
        i = np.arange(n - offset)
        A[i, i + offset] = -1.0
        A[i + offset, i] = -1.0

    return csc_matrix(A)


def measure_dense(case):
    np.copyto(case["work"], case["dense"]) # so we don't time/use the copy

    start = perf_counter()
    lu_factor(
        case["work"],
        overwrite_a=True,
        check_finite=False
    )
    return perf_counter() - start


def measure_sparse(case):
    start = perf_counter()

    factor = splu(case["sparse"], permc_spec="COLAMD")
    del factor

    return perf_counter() - start


def main():
    cases = []

    for n in SIZES:
        sparse = create_matrix(n)
        dense = sparse.toarray(order="F")

        case = {
            "n": n,
            "dense": dense,
            "sparse": sparse,
            "work": np.empty_like(dense, order="F"),
            "dense_times": [],
            "sparse_times": []
        }

        measure_dense(case)
        measure_sparse(case)

        cases.append(case)

    for trial in range(TRIALS):

        offset = (17 * trial) % len(cases)

        for slot in range(len(cases)):
            case = cases[(slot + offset) % len(cases)]

            case["dense_times"].append(
                measure_dense(case)
            )

            case["sparse_times"].append(
                measure_sparse(case)
            )

        if (trial + 1) % 100 == 0:
            print(
                f"Completed {trial + 1}/{TRIALS} trials per size"
            )

    results = Path(__file__).resolve().parents[1] / "results"
    results.mkdir(exist_ok=True)

    with (results / "python_timings.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["n", "dense_ms", "sparse_ms"])

        for case in cases:
            dense_ms = 1000 * median(case["dense_times"])
            sparse_ms = 1000 * median(case["sparse_times"])

            writer.writerow([
                case["n"],
                f"{dense_ms:.9f}",
                f"{sparse_ms:.9f}"
            ])

            print(
                f'{case["n"]:3d} x {case["n"]:<3d}  '
                f"Dense: {dense_ms:.4f} ms  "
                f"Sparse: {sparse_ms:.4f} ms  "
                f"Dense/Sparse: {dense_ms / sparse_ms:.2f}x"
            )

    print("Saved results/python_timings.csv")


if __name__ == "__main__":
    main()

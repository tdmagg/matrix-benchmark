import numpy as np
from scipy.linalg import lu_factor
from scipy.sparse import diags, kronsum
from scipy.sparse.linalg import splu
from time import perf_counter
from statistics import median, mean


def create_matrix(grid_size):
    """Create a sparse, PD matrix."""
    values = np.ones(grid_size) # creates a grid_size x 1 vector of ones

    # we create a grid_size x grid_size tridiagonal matrix 
    # with 2 on the diagonal and -1 on the off-diagonals.
    # we also store in a compressed format, useful for memory efficiency in sparse matrices
    T = diags(
        [-values[:-1], 2 * values, -values[:-1]],
        [-1, 0, 1],
        format="csc"
    ) 
 
    # we use T to form a bigger grid_size^2 x grid_size^2 matrix
    return kronsum(T, T, format="csc")


def benchmark_dense(A, trials=1000):
    times = []

    for _ in range(trials):
        work = A.copy(order="F") # we'll be working with column major arrays

        start = perf_counter()
        lu_factor(work, overwrite_a=True, check_finite=False) #overwrite prevents unnecessary copying, check_finite skips undefined checks
        end = perf_counter()

        times.append(end - start)

    return median(times)


def benchmark_sparse(A, trials=1000):
    times = []

    for _ in range(trials):
        start = perf_counter()
        splu(A)
        end = perf_counter()

        times.append(end - start)

    return median(times)


for grid_size in [5, 10, 15]:

    sparse_A = create_matrix(grid_size)
    dense_A = sparse_A.toarray(order="F")

    dense_time = benchmark_dense(dense_A)
    sparse_time = benchmark_sparse(sparse_A)

    n = dense_A.shape[0]

    print(f"\nMatrix size: {n} x {n}")
    print(f"Nonzero entries: {sparse_A.nnz}")
    print(f"Dense LU:  {dense_time:.6f} seconds")
    print(f"Sparse LU: {sparse_time:.6f} seconds")
    print(f"Dense / Sparse: {dense_time / sparse_time:.2f}x")
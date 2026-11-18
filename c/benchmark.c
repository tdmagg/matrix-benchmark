#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>
#include <slu_ddefs.h>

#define TRIALS 1000
#define NUM_SIZES 50
#define MIN_SIZE 25
#define MAX_SIZE 250
#define TRIAL_ORDER_STRIDE 17
#define PROGRESS_INTERVAL 100
#define CSV_PATH "results/c_timings.csv"

#define DIAGONAL_VALUE 8.0
#define OFF_DIAGONAL_VALUE -1.0
#define NEAR_OFFSET 1
#define MIDDLE_OFFSET 7
#define LONG_OFFSET_DIVISOR 3

#define MILLISECONDS_PER_SECOND 1000.0
#define NANOSECONDS_PER_SECOND 1000000000.0
#define SUPERLU_PANEL_SIZE_SELECTOR 1
#define SUPERLU_RELAX_SELECTOR 2

#ifdef _WIN32
#include <windows.h>
static double seconds(void) {
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}
#else
#include <time.h>
static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / NANOSECONDS_PER_SECOND;
}
#endif

typedef struct {
    int n;
    int nnz;
    double *dense;
    double *work;
    double *values;
    int_t *rows;
    int_t *colptr;
    lapack_int *ipiv;
    int_t *perm_c;
    int_t *perm_r;
    int_t *etree;
    SuperMatrix A;
    double dense_times[TRIALS];
    double sparse_times[TRIALS];
} Benchmark;

static void *checked_malloc(size_t bytes) {
    void *ptr = malloc(bytes);
    if (ptr == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }
    return ptr;
}

static int compare_doubles(const void *left, const void *right) {
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

static double median(double *times) {
    qsort(times, TRIALS, sizeof(*times), compare_doubles);
    if (TRIALS % 2 != 0)
        return times[TRIALS / 2];
    return (times[TRIALS / 2 - 1] + times[TRIALS / 2]) / 2.0;
}

static void init_matrix(Benchmark *b, int n) {
    const int long_offset = n / LONG_OFFSET_DIVISOR;
    if (long_offset <= MIDDLE_OFFSET || n <= long_offset) {
        fprintf(stderr, "Invalid matrix offsets for n=%d\n", n);
        exit(EXIT_FAILURE);
    }

    b->n = n;
    b->nnz = n + 2 * (n - NEAR_OFFSET)
               + 2 * (n - MIDDLE_OFFSET)
               + 2 * (n - long_offset);

    b->dense = calloc((size_t)n * n, sizeof(*b->dense));
    if (b->dense == NULL) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }
    b->work = checked_malloc((size_t)n * n * sizeof(*b->work));
    b->values = checked_malloc((size_t)b->nnz * sizeof(*b->values));
    b->rows = checked_malloc((size_t)b->nnz * sizeof(*b->rows));
    b->colptr = checked_malloc((size_t)(n + 1) * sizeof(*b->colptr));
    b->ipiv = checked_malloc((size_t)n * sizeof(*b->ipiv));
    b->perm_c = checked_malloc((size_t)n * sizeof(*b->perm_c));
    b->perm_r = checked_malloc((size_t)n * sizeof(*b->perm_r));
    b->etree = checked_malloc((size_t)n * sizeof(*b->etree));

    int index = 0;
    for (int col = 0; col < n; ++col) {
        b->colptr[col] = index;
        for (int row = 0; row < n; ++row) {
            int distance = abs(row - col);
            double value = 0.0;
            if (distance == 0)
                value = DIAGONAL_VALUE;
            else if (distance == NEAR_OFFSET ||
                     distance == MIDDLE_OFFSET ||
                     distance == long_offset)
                value = OFF_DIAGONAL_VALUE;

            if (value != 0.0) {
                b->dense[row + col * n] = value;
                b->rows[index] = row;
                b->values[index] = value;
                ++index;
            }
        }
    }
    b->colptr[n] = index;

    if (index != b->nnz) {
        fprintf(stderr, "Incorrect nonzero count for n=%d: %d != %d\n",
                n, index, b->nnz);
        exit(EXIT_FAILURE);
    }

    dCreate_CompCol_Matrix(&b->A, n, n, b->nnz,
                           b->values, b->rows, b->colptr,
                           SLU_NC, SLU_D, SLU_GE);
}

static double measure_dense(Benchmark *b) {
    memcpy(b->work, b->dense, (size_t)b->n * b->n * sizeof(*b->work));

    double start = seconds();
    lapack_int info = LAPACKE_dgetrf(LAPACK_COL_MAJOR, b->n, b->n,
                                     b->work, b->n, b->ipiv);
    double elapsed = seconds() - start;

    if (info != 0) {
        fprintf(stderr, "Dense LU failed at n=%d (info=%d)\n", b->n, (int)info);
        exit(EXIT_FAILURE);
    }
    return elapsed;
}

static double measure_sparse(Benchmark *b) {
    SuperMatrix AC, L, U;
    superlu_options_t options;
    SuperLUStat_t stats;
    GlobalLU_t glu = {0};
    int_t info = 0;

    set_default_options(&options);
    options.ColPerm = COLAMD;
    options.PrintStat = NO;
    StatInit(&stats);

    double start = seconds();
    get_perm_c(COLAMD, &b->A, b->perm_c);
    sp_preorder(&options, &b->A, b->perm_c, b->etree, &AC);
    dgstrf(&options, &AC,
           sp_ienv(SUPERLU_RELAX_SELECTOR),
           sp_ienv(SUPERLU_PANEL_SIZE_SELECTOR),
           b->etree, NULL, 0, b->perm_c, b->perm_r,
           &L, &U, &glu, &stats, &info);

    if (info != 0) {
        fprintf(stderr, "Sparse LU failed at n=%d (info=%lld)\n",
                b->n, (long long)info);
        exit(EXIT_FAILURE);
    }

    Destroy_SuperNode_Matrix(&L);
    Destroy_CompCol_Matrix(&U);
    Destroy_CompCol_Permuted(&AC);
    double elapsed = seconds() - start;

    StatFree(&stats);
    return elapsed;
}

static void free_matrix(Benchmark *b) {
    Destroy_SuperMatrix_Store(&b->A);
    free(b->dense);
    free(b->work);
    free(b->values);
    free(b->rows);
    free(b->colptr);
    free(b->ipiv);
    free(b->perm_c);
    free(b->perm_r);
    free(b->etree);
}

int main(void) {
    FILE *csv = fopen(CSV_PATH, "w");
    if (csv == NULL) {
        perror(CSV_PATH);
        return EXIT_FAILURE;
    }
    fputs("n,dense_ms,sparse_ms\n", csv);
    fflush(csv);

    Benchmark *cases = checked_malloc(NUM_SIZES * sizeof(*cases));
    for (int i = 0; i < NUM_SIZES; ++i) {
        int n = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * i / (NUM_SIZES - 1);
        init_matrix(&cases[i], n);
        measure_dense(&cases[i]);
        measure_sparse(&cases[i]);
    }

    printf("Prepared %d sizes: %d x %d through %d x %d\n",
           NUM_SIZES, cases[0].n, cases[0].n,
           cases[NUM_SIZES - 1].n, cases[NUM_SIZES - 1].n);
    fflush(stdout);

    for (int trial = 0; trial < TRIALS; ++trial) {
        int offset = (TRIAL_ORDER_STRIDE * trial) % NUM_SIZES;
        for (int slot = 0; slot < NUM_SIZES; ++slot) {
            int i = (slot + offset) % NUM_SIZES;
            cases[i].dense_times[trial] = measure_dense(&cases[i]);
            cases[i].sparse_times[trial] = measure_sparse(&cases[i]);
        }
        if ((trial + 1) % PROGRESS_INTERVAL == 0)
            fprintf(stderr, "Completed %d/%d trials per size\n",
                    trial + 1, TRIALS);
    }

    for (int i = 0; i < NUM_SIZES; ++i) {
        double dense_ms = MILLISECONDS_PER_SECOND * median(cases[i].dense_times);
        double sparse_ms = MILLISECONDS_PER_SECOND * median(cases[i].sparse_times);
        printf("%3d x %-3d  Dense: %.4f ms  Sparse: %.4f ms\n",
               cases[i].n, cases[i].n, dense_ms, sparse_ms);
        fprintf(csv, "%d,%.9f,%.9f\n", cases[i].n, dense_ms, sparse_ms);
        free_matrix(&cases[i]);
    }

    fclose(csv);
    free(cases);
    return EXIT_SUCCESS;
}

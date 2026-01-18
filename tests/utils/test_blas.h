#ifndef TEST_BLAS_H
#define TEST_BLAS_H

#include "../minunit.h"
#include "blas.h"
#include "utils/blas_wrappers.h"
#include <math.h>
#include <stdio.h>

/* BLAS dgemm prototype: C = alpha*A*B + beta*C */
extern void BLAS(gemm)(const char *transa, const char *transb, blas_int *m,
                       blas_int *n, blas_int *k, const double *alpha,
                       const double *a, blas_int *lda, const double *b,
                       blas_int *ldb, const double *beta, double *c, blas_int *ldc);

static const char *test_matrix_matrix_mult(void)
{
    /* Test: C = A * B where
     * A is 3x2:  [1  4]     B is 2x3:  [7   10  13]
     *            [2  5]                [8   11  14]
     *            [3  6]
     *
     * Expected C (3x3):  [39   54   69]
     *                    [54   75   96]
     *                    [69   96  123]
     */

    /* Matrices stored in column-major order */
    double A[6] = {1.0, 2.0, 3.0,  /* first column */
                   4.0, 5.0, 6.0}; /* second column */

    double B[6] = {7.0,  8.0,   /* first column */
                   10.0, 11.0,  /* second column */
                   13.0, 14.0}; /* third column */

    double C[9] = {0.0, 0.0, 0.0, /* initialize to zero */
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    /* Expected result (column-major) */
    double expected[9] = {39.0, 54.0, 69.0,   /* first column */
                          54.0, 75.0, 96.0,   /* second column */
                          69.0, 96.0, 123.0}; /* third column */

    /* Compute C = A * B using wrapper */
    /* A is 3x2, B is 2x3, C is 3x3 */
    mat_mat_mult(A, B, C, 3, 2, 3);

    /* Verify result */
    double tol = 1e-10;
    for (int i = 0; i < 9; i++)
    {
        double error = fabs(C[i] - expected[i]);
        if (error > tol)
        {
            printf("Error at index %d: got %f, expected %f (diff: %e)\n", i, C[i],
                   expected[i], error);
            return "Matrix multiplication result incorrect";
        }
    }

    return NULL; /* Test passed */
}

#endif /* TEST_BLAS_H */

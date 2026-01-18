#ifndef BLAS_WRAPPERS_H_GUARD
#define BLAS_WRAPPERS_H_GUARD

#ifdef __cplusplus
extern "C"
{
#endif

#include "blas.h"
#include <math.h>
#include <string.h>

    /*
     * BLAS wrapper functions for common linear algebra operations.
     *
     * When compiled with USE_BLAS, these use optimized BLAS routines.
     * Otherwise, they fall back to simple C implementations.
     */

    /* Vector operations */
    double vec_norm2(const double *x, int n);
    double vec_dot(const double *x, const double *y, int n);
    void vec_axpy(double alpha, const double *x, double *y,
                  int n);                           /* y += alpha*x */
    void vec_scale(double alpha, double *x, int n); /* x *= alpha */

    /* Matrix-vector operations */
    void mat_vec_mult(const double *A, const double *x, double *y, int m,
                      int n); /* y = A*x, A is m x n */

    /* Matrix-matrix operations */
    void mat_mat_mult(const double *X, const double *Y, double *Z, int m, int k,
                      int n); /* Z = X*Y, X is m x k, Y is k x n, Z is m x n */

#ifdef __cplusplus
}
#endif

#endif /* BLAS_WRAPPERS_H_GUARD */

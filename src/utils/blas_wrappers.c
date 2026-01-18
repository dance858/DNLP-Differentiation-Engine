#include "utils/blas_wrappers.h"

/* BLAS function prototypes */
#ifdef __cplusplus
extern "C"
{
#endif

    double BLAS(nrm2)(blas_int *n, const double *x, blas_int *incx);
    double BLAS(dot)(const blas_int *n, const double *x, const blas_int *incx,
                     const double *y, const blas_int *incy);
    void BLAS(axpy)(blas_int *n, const double *alpha, const double *x,
                    blas_int *incx, double *y, blas_int *incy);
    void BLAS(scal)(const blas_int *n, const double *alpha, double *x,
                    const blas_int *incx);
    void BLAS(gemv)(const char *trans, const blas_int *m, const blas_int *n,
                    const double *alpha, const double *a, const blas_int *lda,
                    const double *x, const blas_int *incx, const double *beta,
                    double *y, const blas_int *incy);
    void BLAS(gemm)(const char *transa, const char *transb, const blas_int *m,
                    const blas_int *n, const blas_int *k, const double *alpha,
                    const double *a, const blas_int *lda, const double *b,
                    const blas_int *ldb, const double *beta, double *c,
                    const blas_int *ldc);

#ifdef __cplusplus
}
#endif

/* BLAS implementations */

double vec_norm2(const double *x, int n)
{
    blas_int bn = (blas_int) n;
    blas_int inc = 1;
    return BLAS(nrm2)(&bn, x, &inc);
}

double vec_dot(const double *x, const double *y, int n)
{
    blas_int bn = (blas_int) n;
    blas_int inc = 1;
    return BLAS(dot)(&bn, x, &inc, y, &inc);
}

void vec_axpy(double alpha, const double *x, double *y, int n)
{
    blas_int bn = (blas_int) n;
    blas_int inc = 1;
    BLAS(axpy)(&bn, &alpha, x, &inc, y, &inc);
}

void vec_scale(double alpha, double *x, int n)
{
    blas_int bn = (blas_int) n;
    blas_int inc = 1;
    BLAS(scal)(&bn, &alpha, x, &inc);
}

void mat_vec_mult(const double *A, const double *x, double *y, int m, int n)
{
    /* y = A*x where A is m x n (column-major) */
    blas_int bm = (blas_int) m;
    blas_int bn = (blas_int) n;
    blas_int inc = 1;
    double alpha = 1.0;
    double beta = 0.0;
    char trans = 'N';

    BLAS(gemv)(&trans, &bm, &bn, &alpha, A, &bm, x, &inc, &beta, y, &inc);
}

void mat_mat_mult(const double *X, const double *Y, double *Z, int m, int k, int n)
{
    /* Z = X*Y where X is m x k, Y is k x n, Z is m x n (all column-major) */
    blas_int bm = (blas_int) m;
    blas_int bk = (blas_int) k;
    blas_int bn = (blas_int) n;
    double alpha = 1.0;
    double beta = 0.0;
    char transX = 'N';
    char transY = 'N';

    BLAS(gemm)(&transX, &transY, &bm, &bn, &bk, &alpha, X, &bm, Y, &bk, &beta, Z,
               &bm);
}

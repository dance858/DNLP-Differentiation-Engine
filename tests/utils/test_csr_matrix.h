#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "minunit.h"
#include "test_helpers.h"
#include "utils/CSR_Matrix.h"

const char *test_diag_csr_mult()
{
    /* Create a 3x3 CSR matrix A:
     * [1.0  2.0  0.0]
     * [0.0  3.0  4.0]
     * [5.0  0.0  6.0]
     */
    CSR_Matrix *A = new_csr_matrix(3, 3, 6);
    double Ax[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    int Ai[6] = {0, 1, 1, 2, 0, 2};
    int Ap[4] = {0, 2, 4, 6};
    memcpy(A->x, Ax, 6 * sizeof(double));
    memcpy(A->i, Ai, 6 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));
    double d[3] = {2.0, 3.0, 0.5};

    /* Result C should be diag(d) * A:
     * [2.0  4.0  0.0]
     * [0.0  9.0  12.0]
     * [2.5  0.0  3.0]
     */
    CSR_Matrix *C = new_csr_matrix(3, 3, 6);
    diag_csr_mult(d, A, C);

    double Ax_correct[6] = {2.0, 4.0, 9.0, 12.0, 2.5, 3.0};
    int Ai_correct[6] = {0, 1, 1, 2, 0, 2};
    int Ap_correct[4] = {0, 2, 4, 6};

    mu_assert("vals incorrect", cmp_double_array(C->x, Ax_correct, 6));
    mu_assert("cols incorrect", cmp_int_array(C->i, Ai_correct, 6));
    mu_assert("rows incorrect", cmp_int_array(C->p, Ap_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(C);

    return 0;
}

/*
[1  0  2]   [0 1 0]     [1 1 2]
[0  3  0] + [2 0 3] =   [2 3 3]
[4  0  5]   [0 6 0]     [4 6 5]
*/
const char *test_csr_sum()
{
    CSR_Matrix *A = new_csr_matrix(3, 3, 5);
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int Ai[5] = {0, 2, 1, 0, 2};
    int Ap[4] = {0, 2, 3, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    CSR_Matrix *B = new_csr_matrix(3, 3, 4);
    double Bx[4] = {1.0, 2.0, 3.0, 6.0};
    int Bi[4] = {1, 0, 2, 1};
    int Bp[4] = {0, 1, 3, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_Matrix *C = new_csr_matrix(3, 3, 9);
    sum_csr_matrices(A, B, C);

    double Cx_correct[9] = {1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 6.0, 5.0};
    int Ci_correct[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int Cp_correct[4] = {0, 3, 6, 9};

    mu_assert("C nnz incorrect", C->nnz == 9);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 9));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 9));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C);

    return 0;
}

/*
[1  0  2]   [0 1 0]     [1 1 2]
[0  0  3] + [2 0 3] =   [2 0 6]
[4  0  5]   [0 6 0]     [4 6 5]
*/
const char *test_csr_sum2()
{
    CSR_Matrix *A = new_csr_matrix(3, 3, 5);
    double Ax[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    int Ai[5] = {0, 2, 2, 0, 2};
    int Ap[4] = {0, 2, 3, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 4 * sizeof(int));

    CSR_Matrix *B = new_csr_matrix(3, 3, 4);
    double Bx[4] = {1.0, 2.0, 3.0, 6.0};
    int Bi[4] = {1, 0, 2, 1};
    int Bp[4] = {0, 1, 3, 4};
    memcpy(B->x, Bx, 4 * sizeof(double));
    memcpy(B->i, Bi, 4 * sizeof(int));
    memcpy(B->p, Bp, 4 * sizeof(int));

    CSR_Matrix *C = new_csr_matrix(3, 3, 8);
    sum_csr_matrices(A, B, C);

    double Cx_correct[8] = {1, 1, 2, 2, 6, 4, 6, 5};
    int Ci_correct[8] = {0, 1, 2, 0, 2, 0, 1, 2};
    int Cp_correct[4] = {0, 3, 5, 8};

    mu_assert("C nnz incorrect", C->nnz == 8);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 8));
    mu_assert("C cols incorrect", cmp_int_array(C->i, Ci_correct, 8));
    mu_assert("C rows incorrect", cmp_int_array(C->p, Cp_correct, 4));

    free_csr_matrix(A);
    free_csr_matrix(B);
    free_csr_matrix(C);

    return 0;
}

const char *test_transpose()
{
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    int iwork[5];
    CSR_Matrix *AT = transpose(A, iwork);
    double ATx_correct[5] = {1.0, 2.0, 3.0, 4.0, 1.0};
    int ATi_correct[5] = {0, 2, 1, 3, 0};
    int ATp_correct[6] = {0, 2, 4, 4, 4, 5};
    mu_assert("AT vals incorrect", cmp_double_array(AT->x, ATx_correct, 5));
    mu_assert("AT cols incorrect", cmp_int_array(AT->i, ATi_correct, 5));
    mu_assert("AT rows incorrect", cmp_int_array(AT->p, ATp_correct, 6));

    free_csr_matrix(A);
    free_csr_matrix(AT);

    return 0;
}

/* C = z^T A where

z = (1, 2, 3, 4) and
A = [1 0 0 0 1
     0 3 0 0 0
     2 0 0 0 0
     0 4 0 0 0]
*/
const char *test_csr_vecmat_values_sparse()
{
    CSR_Matrix *A = new_csr_matrix(4, 5, 5);
    double Ax[5] = {1.0, 1.0, 3.0, 2.0, 4.0};
    int Ai[5] = {0, 4, 1, 0, 1};
    int Ap[5] = {0, 2, 3, 4, 5};
    memcpy(A->x, Ax, 5 * sizeof(double));
    memcpy(A->i, Ai, 5 * sizeof(int));
    memcpy(A->p, Ap, 5 * sizeof(int));

    double z[4] = {1.0, 2.0, 3.0, 4.0};

    CSR_Matrix *C = new_csr_matrix(1, 3, 3);
    double Cx[3] = {0.0, 0.0, 0.0};
    int Ci[3] = {0, 1, 4};
    int Cp[2] = {0, 3};
    memcpy(C->x, Cx, 3 * sizeof(double));
    memcpy(C->i, Ci, 3 * sizeof(int));
    memcpy(C->p, Cp, 2 * sizeof(int));

    int iwork[5];

    CSR_Matrix *AT = transpose(A, iwork);

    csr_matvec_fill_values(AT, z, C);

    double Cx_correct[3] = {7.0, 22.0, 1.0};

    mu_assert("C nnz incorrect", C->nnz == 3);
    mu_assert("C vals incorrect", cmp_double_array(C->x, Cx_correct, 3));

    free_csr_matrix(A);
    free_csr_matrix(AT);
    free_csr_matrix(C);

    return 0;
}

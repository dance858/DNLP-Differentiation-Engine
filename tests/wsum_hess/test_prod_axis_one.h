#include <stdio.h>
#include <string.h>

#include "expr.h"
#include "minunit.h"
#include "other.h"
#include "test_helpers.h"

const char *test_wsum_hess_prod_axis_one_no_zeros()
{
    /* x is 2x3 variable, global index 1, total 8 vars
     * x = [1, 2, 3, 4, 5, 6] (column-major)
     *     [1, 3, 5]
     *     [2, 4, 6]
     * f = prod_axis_one(x) = [15, 48]
     * w = [1, 2]
     *
     * Hessian is block diagonal with two 3x3 blocks (one per row):
     * Row 0 block (scale = 1 * 15 = 15):
     *   cols (c0=1, c1=3, c2=5)
     *   off-diagonals: (0,1)=5, (0,2)=3, (1,0)=5, (1,2)=1, (2,0)=3, (2,1)=1; diag=0
     * Row 1 block (scale = 2 * 48 = 96):
     *   cols (c0=2, c1=4, c2=6)
     *   off-diagonals: (0,1)=12, (0,2)=8, (1,0)=12, (1,2)=4, (2,0)=8, (2,1)=4;
     * diag=0
     */
    double u_vals[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0};
    double w_vals[2] = {1.0, 2.0};
    expr *x = new_variable(2, 3, 1, 8);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[18] = {/* Block 0 */
                             0.0, 5.0, 3.0, 5.0, 0.0, 1.0, 3.0, 1.0, 0.0,
                             /* Block 1 */
                             0.0, 12.0, 8.0, 12.0, 0.0, 4.0, 8.0, 4.0, 0.0};

    /* Row pointers (per implementation ordering) */
    int expected_p[9] = {0, 0, 9, 3, 12, 6, 15, 18, 18};

    /* Column indices (block diagonal, repeated column sets per block) */
    int expected_i[18] = {/* Block 0 */
                          1, 3, 5, 1, 3, 5, 1, 3, 5,
                          /* Block 1 */
                          2, 4, 6, 2, 4, 6, 2, 4, 6};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 18));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 9));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 18));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_one_zero()
{
    /* x is 3x3 variable, global index 1, total 10 vars
     * x = [1, 2, 3, 4, 0, 6, 7, 8, 9] (column-major)
     *     [1, 4, 7]
     *     [2, 0, 8]
     *     [3, 6, 9]
     * f = prod_axis_one(x) = [28, 0, 162]
     * w = [1, 2, 3]
     *
     * Blocks are 3x3 (one per row):
     * Row 0 (no zeros, scale=1*28=28):
     *   cols (1,4,7): off-diag (0,1)=7, (0,2)=4, (1,0)=7, (1,2)=2, (2,0)=4, (2,1)=2
     * Row 1 (one zero at col 1, prod_nonzero=16, scale=2*16=32):
     *   only row/col 1 nonzero: (1,0)=32/2=16, (1,2)=32/8=4 (symmetric)
     * Row 2 (no zeros, scale=3*162=486):
     *   cols (3,6,9): off-diag (0,1)=27, (0,2)=18, (1,0)=27, (1,2)=9, (2,0)=18,
     * (2,1)=9
     */
    double u_vals[10] = {0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 9.0};
    double w_vals[3] = {1.0, 2.0, 3.0};
    expr *x = new_variable(3, 3, 1, 10);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[27];
    memset(expected_x, 0, sizeof(expected_x));

    /* Block 0 (row 0) */
    expected_x[0 * 3 + 1] = 7.0; /* (0,1) */
    expected_x[0 * 3 + 2] = 4.0; /* (0,2) */
    expected_x[1 * 3 + 0] = 7.0; /* (1,0) */
    expected_x[1 * 3 + 2] = 1.0; /* (1,2) = 28/(4*7) */
    expected_x[2 * 3 + 0] = 4.0; /* (2,0) */
    expected_x[2 * 3 + 1] = 1.0; /* (2,1) = 28/(7*4) */

    /* Block 1 (row 1, one zero at col 1) */
    int block1 = 9;
    expected_x[block1 + 1 * 3 + 0] = 16.0; /* (1,0) */
    expected_x[block1 + 0 * 3 + 1] = 16.0; /* (0,1) symmetric */
    expected_x[block1 + 1 * 3 + 2] = 4.0;  /* (1,2) */
    expected_x[block1 + 2 * 3 + 1] = 4.0;  /* (2,1) symmetric */

    /* Block 2 (row 2) */
    int block2 = 18;
    expected_x[block2 + 0 * 3 + 1] = 27.0;
    expected_x[block2 + 0 * 3 + 2] = 18.0;
    expected_x[block2 + 1 * 3 + 0] = 27.0;
    expected_x[block2 + 1 * 3 + 2] = 9.0;
    expected_x[block2 + 2 * 3 + 0] = 18.0;
    expected_x[block2 + 2 * 3 + 1] = 9.0;

    /* Row pointers (per implementation ordering) */
    int expected_p[11] = {0, 0, 9, 18, 3, 12, 21, 6, 15, 24, 27};

    /* Column indices (block diagonal) */
    int expected_i[27] = {/* Block 0 */
                          1, 4, 7, 1, 4, 7, 1, 4, 7,
                          /* Block 1 */
                          2, 5, 8, 2, 5, 8, 2, 5, 8,
                          /* Block 2 */
                          3, 6, 9, 3, 6, 9, 3, 6, 9};

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 27));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 11));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 27));

    free_expr(p);
    return 0;
}

const char *test_wsum_hess_prod_axis_one_mixed_zeros()
{
    /* x is 5x3 variable, global index 1, total 16 vars
     * Rows (axis=1 products):
     *   r0: [1, 2, 1] -> no zeros, prod=2
     *   r1: [1, 0, 0] -> two zeros (cols 1,2), prod_nonzero=1
     *   r2: [1, 3, 0] -> one zero (col 2), prod_nonzero=3
     *   r3: [1, 4, 2] -> no zeros, prod=8
     *   r4: [1, 5, 3] -> no zeros, prod=15
     * w = [1, 2, 3, 4, 5]
     * Blocks are 3x3 (one per row, block diagonal):
     */
    double u_vals[16] = {0.0,
                         /* col 0 (rows 0-4): 1,1,1,1,1 */
                         1.0, 1.0, 1.0, 1.0, 1.0,
                         /* col 1: 2,0,3,4,5 */
                         2.0, 0.0, 3.0, 4.0, 5.0,
                         /* col 2: 1,0,0,2,3 */
                         1.0, 0.0, 0.0, 2.0, 3.0};
    /* Actually store column-major:
     * col0: [1,1,1,1,1]
     * col1: [2,0,3,4,5]
     * col2: [1,0,0,2,3]
     */
    double w_vals[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    expr *x = new_variable(5, 3, 1, 16);
    expr *p = new_prod_axis_one(x);

    p->forward(p, u_vals);
    p->wsum_hess_init(p);
    p->eval_wsum_hess(p, w_vals);

    double expected_x[45];
    memset(expected_x, 0, sizeof(expected_x));

    /* Block offsets */
    int b0 = 0, b1 = 9, b2 = 18, b3 = 27, b4 = 36;

    /* Row 0 block (no zeros, scale = 1 * 2 = 2), x = [1,2,1] */
    expected_x[b0 + 0 * 3 + 1] = 1.0; /* 2/(1*2) */
    expected_x[b0 + 0 * 3 + 2] = 2.0; /* 2/(1*1) */
    expected_x[b0 + 1 * 3 + 0] = 1.0; /* symmetric */
    expected_x[b0 + 1 * 3 + 2] = 1.0; /* 2/(2*1) */
    expected_x[b0 + 2 * 3 + 0] = 2.0; /* symmetric */
    expected_x[b0 + 2 * 3 + 1] = 1.0; /* symmetric */

    /* Row 1 block (two zeros at cols 1,2), hess = w*prod_nonzero = 2*1 = 2 */
    expected_x[b1 + 1 * 3 + 2] = 2.0;
    expected_x[b1 + 2 * 3 + 1] = 2.0;

    /* Row 2 block (one zero at col 2), w_prod = 3 * 3 = 9, x = [1,3,0] */
    expected_x[b2 + 2 * 3 + 0] = 9.0; /* 9/1 */
    expected_x[b2 + 0 * 3 + 2] = 9.0; /* symmetric */
    expected_x[b2 + 2 * 3 + 1] = 3.0; /* 9/3 */
    expected_x[b2 + 1 * 3 + 2] = 3.0; /* symmetric */

    /* Row 3 block (no zeros, scale = 4 * 8 = 32), x = [1,4,2] */
    expected_x[b3 + 0 * 3 + 1] = 8.0;  /* 32/(1*4) */
    expected_x[b3 + 0 * 3 + 2] = 16.0; /* 32/(1*2) */
    expected_x[b3 + 1 * 3 + 0] = 8.0;  /* symmetric */
    expected_x[b3 + 1 * 3 + 2] = 4.0;  /* 32/(4*2) */
    expected_x[b3 + 2 * 3 + 0] = 16.0; /* symmetric */
    expected_x[b3 + 2 * 3 + 1] = 4.0;  /* symmetric */

    /* Row 4 block (no zeros, scale = 5 * 15 = 75), x = [1,5,3] */
    expected_x[b4 + 0 * 3 + 1] = 15.0; /* 75/(1*5) */
    expected_x[b4 + 0 * 3 + 2] = 25.0; /* 75/(1*3) */
    expected_x[b4 + 1 * 3 + 0] = 15.0; /* symmetric */
    expected_x[b4 + 1 * 3 + 2] = 5.0;  /* 75/(5*3) */
    expected_x[b4 + 2 * 3 + 0] = 25.0; /* symmetric */
    expected_x[b4 + 2 * 3 + 1] = 5.0;  /* symmetric */

    /* Row pointers (per implementation ordering) */
    int expected_p[17] = {0,  0,  9, 18, 27, 36, 3,  12, 21,
                          30, 39, 6, 15, 24, 33, 42, 45};

    /* Column indices (block diagonal) */
    int expected_i[45];
    for (int block = 0; block < 5; block++)
    {
        int offset = block * 9;
        int base = 1 + block; /* var_id + row index */
        for (int r = 0; r < 3; r++)
        {
            expected_i[offset + r * 3 + 0] = base;
            expected_i[offset + r * 3 + 1] = base + 5;
            expected_i[offset + r * 3 + 2] = base + 10;
        }
    }

    mu_assert("vals fail", cmp_double_array(p->wsum_hess->x, expected_x, 45));
    mu_assert("rows fail", cmp_int_array(p->wsum_hess->p, expected_p, 17));
    mu_assert("cols fail", cmp_int_array(p->wsum_hess->i, expected_i, 45));

    free_expr(p);
    return 0;
}

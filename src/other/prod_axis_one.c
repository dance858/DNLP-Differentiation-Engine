#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define IS_ZERO(x) (fabs((x)) < 1e-8)

static void forward(expr *node, const double *u)
{
    /* forward pass of child */
    expr *x = node->left;
    x->forward(x, u);

    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    int d1 = x->d1;
    int d2 = x->d2;

    /* initialize per-row statistics */
    for (int i = 0; i < d1; i++)
    {
        pnode->num_of_zeros[i] = 0;
        pnode->zero_index[i] = -1;
        pnode->prod_nonzero[i] = 1.0;
    }

    /* iterate over columns */
    for (int col = 0; col < d2; col++)
    {
        int start = col * d1;
        int end = start + d1;

        for (int idx = start; idx < end; idx++)
        {
            int row = idx - start;
            if (IS_ZERO(x->value[idx]))
            {
                pnode->num_of_zeros[row]++;
                pnode->zero_index[row] = col;
            }
            else
            {
                pnode->prod_nonzero[row] *= x->value[idx];
            }
        }
    }

    /* compute output values */
    for (int i = 0; i < d1; i++)
    {
        node->value[i] = (pnode->num_of_zeros[i] > 0) ? 0.0 : pnode->prod_nonzero[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;

    /* initialize child's jacobian */
    x->jacobian_init(x);

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        node->jacobian = new_csr_matrix(node->size, node->n_vars, x->size);

        /* set row pointers (each row has d2 nnzs) */
        for (int row = 0; row < x->d1; row++)
        {
            node->jacobian->p[row] = row * x->d2;
        }
        node->jacobian->p[x->d1] = x->size;

        /* set column indices */
        for (int row = 0; row < x->d1; row++)
        {
            int start = row * x->d2;
            for (int col = 0; col < x->d2; col++)
            {
                node->jacobian->i[start + col] = x->var_id + col * x->d1 + row;
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;

    double *J_vals = node->jacobian->x;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* process each row */
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];
            int start = row * x->d2;

            if (num_zeros == 0)
            {
                for (int col = 0; col < x->d2; col++)
                {
                    int idx = col * x->d1 + row;
                    J_vals[start + col] = node->value[row] / x->value[idx];
                }
            }
            else if (num_zeros == 1)
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
                J_vals[start + pnode->zero_index[row]] = pnode->prod_nonzero[row];
            }
            else
            {
                memset(J_vals + start, 0, sizeof(double) * x->d2);
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static void wsum_hess_init(expr *node)
{
    expr *x = node->left;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        /* Hessian has block diagonal structure: d1 blocks of size d2 x d2 */
        int nnz = x->d1 * x->d2 * x->d2;
        node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, nnz);
        CSR_Matrix *H = node->wsum_hess;

        /* fill row pointers for the variable's rows (block diagonal) */
        for (int i = 0; i < x->size; i++)
        {
            int block_idx = i % x->d1;    /* row of the matrix */
            int row_in_block = i / x->d1; /* column index within that row-block */
            H->p[x->var_id + i] = block_idx * x->d2 * x->d2 + row_in_block * x->d2;
        }

        /* fill row pointers for rows after the variable */
        for (int i = x->var_id + x->size; i <= node->n_vars; i++)
        {
            H->p[i] = nnz;
        }

        /* set column indices for each block */
        for (int block = 0; block < x->d1; block++)
        {
            int block_start = block * x->d2 * x->d2;
            int col_start = block; /* global column offset for this row-block */
            for (int row = 0; row < x->d2; row++)
            {
                for (int col = 0; col < x->d2; col++)
                {
                    /* variable indices: row-block fixed, column-major layout */
                    int var_col =
                        col * x->d1 +
                        block; /* global variable index offset from var_id */
                    H->i[block_start + row * x->d2 + col] = x->var_id + var_col;
                }
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static inline void wsum_hess_row_no_zeros(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    double *H = node->wsum_hess->x;

    int block_start = row * d2 * d2;
    double scale = w[row] * node->value[row];

    /* fill block for this row */
    for (int r = 0; r < d2; r++)
    {
        int idx_r = r * node->left->d1 + row; /* column-major index into x */
        for (int c = 0; c < d2; c++)
        {
            int idx_c = c * node->left->d1 + row;
            double val =
                (r == c) ? 0.0 : scale / (x->value[idx_r] * x->value[idx_c]);
            H[block_start + r * d2 + c] = val;
        }
    }
    (void) pnode; /* suppress unused warning */
}

static inline void wsum_hess_row_one_zero(expr *node, const double *w, int row,
                                          int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    double *H = node->wsum_hess->x;

    int block_start = row * d2 * d2;
    int p = pnode->zero_index[row]; /* zero column */
    double w_prod = w[row] * pnode->prod_nonzero[row];

    /* clear block */
    memset(&H[block_start], 0, sizeof(double) * d2 * d2);

    /* fill row p and column p */
    int col_offset = p * node->left->d1 + row;
    for (int c = 0; c < d2; c++)
    {
        if (c == p) continue;
        int idx_c = c * node->left->d1 + row;
        double hess_val = w_prod / x->value[idx_c];
        H[block_start + p * d2 + c] = hess_val;
        H[block_start + c * d2 + p] = hess_val;
    }
    (void) col_offset; /* suppress unused */
}

static inline void wsum_hess_row_two_zeros(expr *node, const double *w, int row,
                                           int d2)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    double *H = node->wsum_hess->x;

    int block_start = row * d2 * d2;

    /* clear block */
    memset(&H[block_start], 0, sizeof(double) * d2 * d2);

    /* find indices p and q where row has zeros */
    int p = -1, q = -1;
    for (int c = 0; c < d2; c++)
    {
        int idx = c * node->left->d1 + row;
        if (IS_ZERO(x->value[idx]))
        {
            if (p == -1)
            {
                p = c;
            }
            else
            {
                q = c;
                break;
            }
        }
    }
    assert(p != -1 && q != -1);

    double hess_val = w[row] * pnode->prod_nonzero[row];
    H[block_start + p * d2 + q] = hess_val;
    H[block_start + q * d2 + p] = hess_val;
}

static inline void wsum_hess_row_many_zeros(expr *node, int row, int d2)
{
    double *H = node->wsum_hess->x;
    int block_start = row * d2 * d2;
    memset(&H[block_start], 0, sizeof(double) * d2 * d2);
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *x = node->left;
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;

    /* if x is a variable */
    if (x->var_id != NOT_A_VARIABLE)
    {
        for (int row = 0; row < x->d1; row++)
        {
            int num_zeros = pnode->num_of_zeros[row];

            if (num_zeros == 0)
            {
                wsum_hess_row_no_zeros(node, w, row, x->d2);
            }
            else if (num_zeros == 1)
            {
                wsum_hess_row_one_zero(node, w, row, x->d2);
            }
            else if (num_zeros == 2)
            {
                wsum_hess_row_two_zeros(node, w, row, x->d2);
            }
            else
            {
                wsum_hess_row_many_zeros(node, row, x->d2);
            }
        }
    }
    else
    {
        assert(false && "child must be a variable");
    }
}

static bool is_affine(const expr *node)
{
    (void) node;
    return false;
}

static void free_type_data(expr *node)
{
    prod_axis_one_expr *pnode = (prod_axis_one_expr *) node;
    free(pnode->num_of_zeros);
    free(pnode->zero_index);
    free(pnode->prod_nonzero);
}

expr *new_prod_axis_one(expr *child)
{
    prod_axis_one_expr *pnode =
        (prod_axis_one_expr *) calloc(1, sizeof(prod_axis_one_expr));
    expr *node = &pnode->base;

    /* output is always a row vector 1 x d1 (one product per row) */
    init_expr(node, 1, child->d1, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, wsum_hess_init, eval_wsum_hess,
              free_type_data);

    /* allocate arrays to store per-row statistics */
    pnode->num_of_zeros = (int *) calloc(child->d1, sizeof(int));
    pnode->zero_index = (int *) calloc(child->d1, sizeof(int));
    pnode->prod_nonzero = (double *) calloc(child->d1, sizeof(double));

    node->left = child;
    expr_retain(child);

    return node;
}

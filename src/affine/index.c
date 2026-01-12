// SPDX-License-Identifier: Apache-2.0

#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>
#include <string.h>

/* Index/slicing: y = child[indices] where indices is a list of flattened positions */

static void forward(expr *node, const double *u)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;

    /* child's forward pass */
    child->forward(child, u);

    /* gather selected elements */
    for (int i = 0; i < idx->n_selected; i++)
    {
        node->value[i] = child->value[idx->indices[i]];
    }
}

static void jacobian_init(expr *node)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;

    /* initialize child's jacobian */
    child->jacobian_init(child);
    CSR_Matrix *J_child = child->jacobian;

    /* allocate mapping arrays */
    idx->jac_row_starts = (int *)malloc(idx->n_selected * sizeof(int));
    idx->jac_row_lengths = (int *)malloc(idx->n_selected * sizeof(int));

    /* count nnz and pre-compute row mapping */
    int nnz = 0;
    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        int row_len = J_child->p[row + 1] - J_child->p[row];
        idx->jac_row_starts[i] = nnz;
        idx->jac_row_lengths[i] = row_len;
        nnz += row_len;
    }

    /* allocate jacobian with computed sparsity */
    node->jacobian = new_csr_matrix(idx->n_selected, node->n_vars, nnz);

    /* fill sparsity pattern (p and i arrays) */
    node->jacobian->p[0] = 0;
    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        int row_len = idx->jac_row_lengths[i];
        /* copy column indices from child row */
        if (row_len > 0)
        {
            memcpy(&node->jacobian->i[idx->jac_row_starts[i]],
                   &J_child->i[J_child->p[row]], row_len * sizeof(int));
        }
        node->jacobian->p[i + 1] = idx->jac_row_starts[i] + row_len;
    }
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;
    CSR_Matrix *J_child = child->jacobian;

    /* evaluate child's jacobian */
    child->eval_jacobian(child);

    /* fast copy using pre-computed mapping */
    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        int row_len = idx->jac_row_lengths[i];
        if (row_len > 0)
        {
            memcpy(&node->jacobian->x[idx->jac_row_starts[i]],
                   &J_child->x[J_child->p[row]],
                   row_len * sizeof(double));
        }
    }
}

static void wsum_hess_init(expr *node)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;

    /* initialize child's wsum_hess */
    child->wsum_hess_init(child);

    /* allocate scatter buffer (zeroed) */
    idx->parent_w = (double *)calloc(child->size, sizeof(double));

    /* wsum_hess inherits from child (affine has no local Hessian) */
    /* We need to allocate our own to avoid aliasing issues */
    CSR_Matrix *child_hess = child->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);
    memcpy(node->wsum_hess->p, child_hess->p, (child_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, child_hess->i, child_hess->nnz * sizeof(int));
}

static void eval_wsum_hess(expr *node, const double *w)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;

    /* zero the scatter buffer */
    memset(idx->parent_w, 0, child->size * sizeof(double));

    /* scatter w to child size with accumulation (handles repeated indices) */
    for (int i = 0; i < idx->n_selected; i++)
    {
        idx->parent_w[idx->indices[i]] += w[i];
    }

    /* delegate to child */
    child->eval_wsum_hess(child, idx->parent_w);

    /* copy values from child */
    memcpy(node->wsum_hess->x, child->wsum_hess->x, child->wsum_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    index_expr *idx = (index_expr *)node;
    if (idx->indices)
    {
        free(idx->indices);
        idx->indices = NULL;
    }
    if (idx->jac_row_starts)
    {
        free(idx->jac_row_starts);
        idx->jac_row_starts = NULL;
    }
    if (idx->jac_row_lengths)
    {
        free(idx->jac_row_lengths);
        idx->jac_row_lengths = NULL;
    }
    if (idx->parent_w)
    {
        free(idx->parent_w);
        idx->parent_w = NULL;
    }
}

expr *new_index(expr *child, const int *indices, int n_selected)
{
    /* allocate type-specific struct */
    index_expr *idx = (index_expr *)calloc(1, sizeof(index_expr));
    expr *node = &idx->base;

    /* output shape is (n_selected, 1) - flattened */
    init_expr(node, n_selected, 1, child->n_vars, forward, jacobian_init,
              eval_jacobian, is_affine, free_type_data);

    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;
    node->left = child;
    expr_retain(child);

    /* copy indices */
    idx->indices = (int *)malloc(n_selected * sizeof(int));
    memcpy(idx->indices, indices, n_selected * sizeof(int));
    idx->n_selected = n_selected;

    /* mapping arrays allocated lazily in jacobian_init */
    idx->jac_row_starts = NULL;
    idx->jac_row_lengths = NULL;
    idx->parent_w = NULL;

    return node;
}

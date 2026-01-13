// SPDX-License-Identifier: Apache-2.0

#include "affine.h"
#include "subexpr.h"
#include <stdlib.h>
#include <string.h>

/* Index/slicing: y = child[indices] where indices is a list of flattened positions */

/* Check if indices array contains duplicates using a bitmap.
 * Returns true if duplicates exist, false otherwise. */
static bool check_for_duplicates(const int *indices, int n_selected, int max_idx)
{
    bool *seen = (bool *)calloc(max_idx, sizeof(bool));
    bool has_dup = false;
    for (int i = 0; i < n_selected && !has_dup; i++)
    {
        if (seen[indices[i]])
        {
            has_dup = true;
        }
        seen[indices[i]] = true;
    }
    free(seen);
    return has_dup;
}

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

    child->jacobian_init(child);
    CSR_Matrix *J_child = child->jacobian;

    /* count nnz */
    int nnz = 0;
    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        nnz += J_child->p[row + 1] - J_child->p[row];
    }

    node->jacobian = new_csr_matrix(idx->n_selected, node->n_vars, nnz);
    CSR_Matrix *J = node->jacobian;

    /* fill p and i arrays in one pass */
    J->p[0] = 0;
    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        int src = J_child->p[row];
        int len = J_child->p[row + 1] - src;
        memcpy(J->i + J->p[i], J_child->i + src, len * sizeof(int));
        J->p[i + 1] = J->p[i] + len;
    }
}

static void eval_jacobian(expr *node)
{
    expr *child = node->left;
    index_expr *idx = (index_expr *)node;

    child->eval_jacobian(child);

    CSR_Matrix *J = node->jacobian;
    CSR_Matrix *J_child = child->jacobian;

    for (int i = 0; i < idx->n_selected; i++)
    {
        int row = idx->indices[i];
        int len = J->p[i + 1] - J->p[i];
        memcpy(J->x + J->p[i], J_child->x + J_child->p[row], len * sizeof(double));
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

    if (idx->has_duplicates)
    {
        /* slow path: must zero and accumulate for repeated indices */
        memset(idx->parent_w, 0, child->size * sizeof(double));
        for (int i = 0; i < idx->n_selected; i++)
        {
            idx->parent_w[idx->indices[i]] += w[i];
        }
    }
    else
    {
        /* fast path: direct write (no memset needed, no accumulation) */
        for (int i = 0; i < idx->n_selected; i++)
        {
            idx->parent_w[idx->indices[i]] = w[i];
        }
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

    /* detect duplicates for Hessian optimization */
    idx->has_duplicates = check_for_duplicates(indices, n_selected, child->size);

    /* parent_w allocated lazily in wsum_hess_init */
    idx->parent_w = NULL;

    return node;
}

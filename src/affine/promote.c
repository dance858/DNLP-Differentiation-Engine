#include "affine.h"
#include <stdlib.h>
#include <string.h>

/* Promote broadcasts a scalar expression to a vector/matrix shape.
 * This matches CVXPY's promote atom which only handles scalars. */

static void forward(expr *node, const double *u)
{
    node->left->forward(node->left, u);

    /* broadcast scalar value to all output elements */
    double val = node->left->value[0];
    for (int i = 0; i < node->size; i++)
    {
        node->value[i] = val;
    }
}

static void jacobian_init(expr *node)
{
    node->left->jacobian_init(node->left);

    /* Each output row copies the single row from child's jacobian */
    CSR_Matrix *child_jac = node->left->jacobian;
    int nnz = node->size * child_jac->nnz;
    if (nnz == 0) nnz = 1;

    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);
    CSR_Matrix *jac = node->jacobian;

    /* Build sparsity pattern by replicating child's single row */
    jac->nnz = 0;
    for (int row = 0; row < node->size; row++)
    {
        jac->p[row] = jac->nnz;
        for (int k = child_jac->p[0]; k < child_jac->p[1]; k++)
        {
            jac->i[jac->nnz++] = child_jac->i[k];
        }
    }
    jac->p[node->size] = jac->nnz;
}

static void eval_jacobian(expr *node)
{
    node->left->eval_jacobian(node->left);

    CSR_Matrix *child_jac = node->left->jacobian;
    CSR_Matrix *jac = node->jacobian;
    int child_nnz = child_jac->p[1] - child_jac->p[0];

    /* Copy child's row values to each output row */
    for (int row = 0; row < node->size; row++)
    {
        memcpy(&jac->x[row * child_nnz], &child_jac->x[child_jac->p[0]],
               child_nnz * sizeof(double));
    }
}

static void wsum_hess_init(expr *node)
{
    node->left->wsum_hess_init(node->left);

    /* same sparsity as child since we're summing weights */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    node->wsum_hess = new_csr_matrix(child_hess->m, child_hess->n, child_hess->nnz);

    /* copy sparsity pattern */
    memcpy(node->wsum_hess->p, child_hess->p, (child_hess->m + 1) * sizeof(int));
    memcpy(node->wsum_hess->i, child_hess->i, child_hess->nnz * sizeof(int));
    node->wsum_hess->nnz = child_hess->nnz;
}

static void eval_wsum_hess(expr *node, const double *w)
{
    /* Sum all weights (they all correspond to the same scalar child) */
    double sum_w = 0.0;
    for (int i = 0; i < node->size; i++)
    {
        sum_w += w[i];
    }

    /* evaluate child's wsum_hess with summed weight */
    node->left->eval_wsum_hess(node->left, &sum_w);

    /* copy values */
    CSR_Matrix *child_hess = node->left->wsum_hess;
    memcpy(node->wsum_hess->x, child_hess->x, child_hess->nnz * sizeof(double));
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

expr *new_promote(expr *child, int d1, int d2)
{
    expr *node = new_expr(d1, d2, child->n_vars);
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    node->is_affine = is_affine;
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;

    node->left = child;
    expr_retain(child);

    return node;
}

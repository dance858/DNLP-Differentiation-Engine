#include "expr.h"
#include "utils/int_double_pair.h"
#include <stdlib.h>
#include <string.h>

void init_expr(expr *node, int d1, int d2, int n_vars, forward_fn forward,
               jacobian_init_fn jacobian_init, eval_jacobian_fn eval_jacobian,
               is_affine_fn is_affine, wsum_hess_init_fn wsum_hess_init,
               wsum_hess_fn eval_wsum_hess, free_type_data_fn free_type_data)
{
    node->d1 = d1;
    node->d2 = d2;
    node->size = d1 * d2;
    node->n_vars = n_vars;
    node->refcount = 0;
    node->value = (double *) calloc(d1 * d2, sizeof(double));
    node->var_id = NOT_A_VARIABLE;
    node->forward = forward;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;
    node->is_affine = is_affine;
    node->wsum_hess_init = wsum_hess_init;
    node->eval_wsum_hess = eval_wsum_hess;
    node->free_type_data = free_type_data;
}

void free_expr(expr *node)
{
    if (node == NULL) return;

    node->refcount--;
    if (node->refcount > 0) return; /* Still referenced elsewhere */

    /* recursively free children */
    free_expr(node->left);
    free_expr(node->right);
    node->left = NULL;
    node->right = NULL;

    /* free type-specific data */
    if (node->free_type_data)
    {
        node->free_type_data(node);
    }

    /* free value array and jacobian */
    free(node->value);
    free_csr_matrix(node->jacobian);
    free_csr_matrix(node->wsum_hess);
    free(node->dwork);
    free(node->iwork);
    node->value = NULL;
    node->jacobian = NULL;
    node->wsum_hess = NULL;
    node->dwork = NULL;
    node->iwork = NULL;

    /* free the node itself */
    free(node);
}

void expr_retain(expr *node)
{
    if (node == NULL) return;
    node->refcount++;
}

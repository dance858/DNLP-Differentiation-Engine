#include "affine.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    memcpy(node->value, u + node->var_id, node->d1 * node->d2 * sizeof(double));
}

static void jacobian_init(expr *node)
{
    node->jacobian = new_csr_matrix(node->size, node->n_vars, node->size);
    for (int j = 0; j < node->size; j++)
    {
        node->jacobian->p[j] = j;
        node->jacobian->i[j] = j + node->var_id;
        node->jacobian->x[j] = 1.0;
    }
    node->jacobian->p[node->size] = node->size;
}

static void eval_jacobian(expr *node)
{
    /* Variable jacobian never changes - nothing to evaluate */
    (void) node;
}

static void wsum_hess_init(expr *node)
{
    /* Variables have zero Hessian */
    node->wsum_hess = new_csr_matrix(node->n_vars, node->n_vars, 0);
}

static void wsum_hess_eval(expr *node, const double *w)
{
    /* Variables have zero Hessian */
    (void) node;
    (void) w;
}

static bool is_affine(const expr *node)
{
    (void) node;
    return true;
}

expr *new_variable(int d1, int d2, int var_id, int n_vars)
{
    expr *node = (expr *) calloc(1, sizeof(expr));
    init_expr(node, d1, d2, n_vars, forward, jacobian_init, eval_jacobian, is_affine,
              wsum_hess_init, wsum_hess_eval, NULL);
    node->var_id = var_id;

    return node;
}

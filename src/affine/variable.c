#include "affine/variable.h"
#include <stdlib.h>
#include <string.h>

static void forward(expr *node, const double *u)
{
    memcpy(node->value, u + node->var_id, node->m * sizeof(double));
}

static void jacobian_init(expr *node)
{
    node->jacobian = new_csr_matrix(node->m, node->n_vars, node->m);
    for (int j = 0; j < node->m; j++)
    {
        node->jacobian->p[j] = j;
        node->jacobian->i[j] = j + node->var_id;
        node->jacobian->x[j] = 1.0;
    }
    node->jacobian->p[node->m] = node->m;
}

static bool is_affine(expr *node)
{
    (void) node;
    return true;
}

expr *new_variable(int m, int var_id, int n_vars)
{
    expr *node = new_expr(m, n_vars);
    if (!node) return NULL;

    node->forward = forward;
    node->var_id = var_id;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    // node->jacobian = NULL;

    return node;
}

#include "affine/variable.h"
#include <string.h>

static void variable_forward(expr *node, const double *u)
{
    memcpy(node->value, u, node->m * sizeof(double));
}

expr *new_variable(int m)
{
    expr *node = new_expr(m);
    if (!node) return NULL;

    node->forward = variable_forward;

    return node;
}

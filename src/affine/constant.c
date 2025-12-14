#include "affine/constant.h"
#include <string.h>

static void constant_forward(expr *node, const double *u)
{
    /* Constants don't depend on u; values are already set */
    (void) node;
    (void) u;
}

expr *new_constant(int m, const double *values)
{
    expr *node = new_expr(m);
    if (!node) return NULL;

    /* Copy constant values */
    memcpy(node->value, values, m * sizeof(double));

    node->forward = constant_forward;

    return node;
}

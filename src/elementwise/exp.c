#include "elementwise.h"
#include <math.h>

static void exp_forward(expr *node, const double *u)
{
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* local forward pass */
    for (int i = 0; i < node->m; i++)
    {
        node->value[i] = exp(node->left->value[i]);
    }
}

expr *new_exp(expr *child)
{
    if (!child) return NULL;

    expr *node = new_expr(child->m);
    if (!node) return NULL;

    node->left = child;
    node->forward = exp_forward;

    return node;
}

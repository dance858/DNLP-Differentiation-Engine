#include "elementwise.h"
#include <math.h>

static void log_forward(expr *node, const double *u)
{
    /* child's forward pass */
    if (node->left)
    {
        node->left->forward(node->left, u);
    }

    /* local forward pass */
    for (int i = 0; i < node->m; i++)
    {
        node->value[i] = log(node->left->value[i]);
    }
}

expr *new_log(expr *child)
{
    if (!child) return NULL;

    expr *node = new_expr(child->m);
    if (!node) return NULL;

    node->left = child;
    node->forward = log_forward;

    return node;
}

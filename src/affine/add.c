#include "affine/add.h"

static void add_forward(expr *node, const double *u)
{
    /* children's forward passes */
    node->left->forward(node->left, u);
    node->right->forward(node->right, u);

    /* add left and right values */
    for (int i = 0; i < node->m; i++)
    {
        node->value[i] = node->left->value[i] + node->right->value[i];
    }
}

expr *new_add(expr *left, expr *right)
{
    if (!left || !right) return NULL;
    if (left->m != right->m) return NULL; /* Dimension mismatch */

    expr *node = new_expr(left->m);
    if (!node) return NULL;

    node->left = left;
    node->right = right;
    node->forward = add_forward;

    return node;
}

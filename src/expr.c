#include "expr.h"
#include <stdlib.h>
#include <string.h>

expr *new_expr(int m)
{
    expr *node = (expr *) malloc(sizeof(expr));
    if (!node) return NULL;

    node->m = m;
    node->value = (double *) calloc(m, sizeof(double));
    if (!node->value)
    {
        free(node);
        return NULL;
    }

    node->left = NULL;
    node->right = NULL;
    node->forward = NULL;

    return node;
}

void free_expr(expr *node)
{
    if (!node) return;

    /* recursively free children */
    free_expr(node->left);
    free_expr(node->right);

    /* free value array */
    free(node->value);

    /* free the node itself */
    free(node);
}

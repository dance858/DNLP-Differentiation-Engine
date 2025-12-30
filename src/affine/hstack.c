#include "affine.h"
#include <assert.h>
#include <string.h>

static void forward(expr *node, const double *u)
{

    /* children's forward passes */
    for (int i = 0; i < node->n_args; i++)
    {
        node->args[i]->forward(node->args[i], u);
    }

    /* concatenate values horizontally */
    int offset = 0;
    for (int i = 0; i < node->n_args; i++)
    {
        expr *child = node->args[i];
        memcpy(node->value + offset, child->value, child->size * sizeof(double));
        offset += child->size;
    }
}

static void jacobian_init(expr *node)
{
    /* initialize children's jacobians */
    int nnz = 0;
    for (int i = 0; i < node->n_args; i++)
    {
        node->args[i]->jacobian_init(node->args[i]);
        nnz += node->args[i]->jacobian->nnz;
    }

    node->jacobian = new_csr_matrix(node->size, node->n_vars, nnz);
}

static void eval_jacobian(expr *node)
{
    /* evaluate children's jacobians */
    int row_offset = 0;
    CSR_Matrix *A = node->jacobian;
    A->nnz = 0;

    for (int i = 0; i < node->n_args; i++)
    {
        expr *child = node->args[i];
        child->eval_jacobian(child);
        CSR_Matrix *B = child->jacobian;

        /* copy columns and values */
        memcpy(A->x + A->nnz, B->x, B->nnz * sizeof(double));
        memcpy(A->i + A->nnz, B->i, B->nnz * sizeof(int));

        /* set row pointers */
        for (int r = 0; r < child->size; r++)
        {
            A->p[row_offset + r] = A->nnz + B->p[r];
        }

        A->nnz += B->nnz;
        row_offset += child->size;
    }
    A->p[node->size] = A->nnz;
}

static bool is_affine(expr *node)
{
    for (int i = 0; i < node->n_args; i++)
    {
        if (!node->args[i]->is_affine(node->args[i]))
        {
            return false;
        }
    }
    return true;
}

expr *new_hstack(expr **args, int n_args, int n_vars)
{
    /* compute second dimension */
    int d2 = 0;
    for (int i = 0; i < n_args; i++)
    {
        d2 += args[i]->d2;
    }

    expr *node = new_expr(args[0]->d1, d2, n_vars);
    if (!node) return NULL;
    node->args = args;
    node->n_args = n_args;

    for (int i = 0; i < n_args; i++)
    {
        expr_retain(args[i]);
    }

    node->forward = forward;
    node->is_affine = is_affine;
    node->jacobian_init = jacobian_init;
    node->eval_jacobian = eval_jacobian;

    return node;
}

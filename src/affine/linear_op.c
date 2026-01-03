#include "affine.h"
#include <stdlib.h>

static void forward(expr *node, const double *u)
{
    expr *x = node->left;
    /* child's forward pass */
    node->left->forward(node->left, u);

    /* y = A * x */
    csr_matvec(node->jacobian, x->value, node->value, x->var_id);
}

static bool is_affine(const expr *node)
{
    return node->left->is_affine(node->left);
}

static void free_type_data(expr *node)
{
    linear_op_expr *lin_node = (linear_op_expr *) node;
    free_csr_matrix(lin_node->A_csr);
    free_csc_matrix(lin_node->A_csc);
}

expr *new_linear(expr *u, const CSR_Matrix *A)
{
    /* Allocate the type-specific struct */
    linear_op_expr *lin_node = (linear_op_expr *) calloc(1, sizeof(linear_op_expr));
    if (!lin_node) return NULL;

    expr *node = &lin_node->base;

    /* Initialize base fields */
    init_expr(node, A->m, 1, u->n_vars, forward, NULL, NULL, is_affine,
              free_type_data);

    /* Set left child */
    node->left = u;
    expr_retain(u);

    /* Check if allocation succeeded */
    if (!node->value)
    {
        free(lin_node);
        return NULL;
    }

    /* allocate jacobian and copy A into it */
    // TODO: this should eventually be removed
    node->jacobian = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, node->jacobian);

    /* Initialize type-specific fields */
    lin_node->A_csr = new_csr_matrix(A->m, A->n, A->nnz);
    copy_csr_matrix(A, lin_node->A_csr);
    lin_node->A_csc = csr_to_csc(A);

    return node;
}

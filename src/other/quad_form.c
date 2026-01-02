#include "other.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

/* Note: Q is not freed here because it's owned by the caller */

static void forward(expr *node, const double *u)
{
    expr *x = node->left;

    /* children's forward passes */
    x->forward(x, u);

    /* local forward pass  */
    quad_form_expr *qnode = (quad_form_expr *) node;
    csr_matvec(qnode->Q, x->value, node->dwork, 0);

    node->value[0] = 0.0;

    for (int i = 0; i < x->d1; i++)
    {
        node->value[0] += x->value[i] * node->dwork[i];
    }
}

static void jacobian_init(expr *node)
{
    expr *x = node->left;
    node->dwork = (double *) malloc(x->d1 * sizeof(double));

    /* if x is a variable */
    if (x->var_id != -1)
    {
        node->jacobian = new_csr_matrix(1, node->n_vars, x->d1);
        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = x->d1;

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->i[j] = x->var_id + j;
        }
    }
    else /* x is not a variable */
    {
        /* compute required allocation and allocate jacobian */
        bool *col_nz = (bool *) calloc(
            node->n_vars, sizeof(bool)); /* TODO: could use iwork here instead*/
        int nonzero_cols = count_nonzero_cols(x->jacobian, col_nz);
        node->jacobian = new_csr_matrix(1, node->n_vars, nonzero_cols + 1);

        /* precompute column indices */
        node->jacobian->nnz = 0;
        for (int j = 0; j < node->n_vars; j++)
        {
            if (col_nz[j])
            {
                node->jacobian->i[node->jacobian->nnz] = j;
                node->jacobian->nnz++;
            }
        }
        assert(nonzero_cols == node->jacobian->nnz);
        free(col_nz);

        node->jacobian->p[0] = 0;
        node->jacobian->p[1] = node->jacobian->nnz;

        /* store A^T of child's A to simplify chain rule computation */
        node->iwork = (int *) malloc(x->jacobian->n * sizeof(int));
        node->CSR_work = transpose(x->jacobian, node->iwork);
    }
}

static void eval_jacobian(expr *node)
{
    expr *x = node->left;
    quad_form_expr *qnode = (quad_form_expr *) node;
    CSR_Matrix *Q = qnode->Q;

    /* if x is a variable */
    if (x->var_id != -1)
    {
        csr_matvec(Q, x->value, node->jacobian->x, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->jacobian->x[j] *= 2.0;
        }
    }
    else /* x is not a variable */
    {
        /* local jacobian */
        csr_matvec(Q, x->value, node->dwork, 0);

        for (int j = 0; j < x->d1; j++)
        {
            node->dwork[j] *= 2.0;
        }

        /* chain rule */
        csr_matvec_fill_values(node->CSR_work, node->dwork, node->jacobian);
    }
}

/* Helper function to initialize a quad_form expr */
void init_quad_form(expr *node, expr *child)
{
    node->d1 = child->d1;
    node->d2 = 1;
    node->size = child->d1 * 1;
    node->n_vars = child->n_vars;
    node->var_id = -1;
    node->refcount = 1;
    node->left = child;
    node->right = NULL;
    node->dwork = NULL;
    node->iwork = NULL;
    node->value = (double *) calloc(node->size, sizeof(double));
    node->jacobian = NULL;
    node->wsum_hess = NULL;
    node->CSR_work = NULL;
    node->jacobian_init = jacobian_init;
    node->wsum_hess_init = NULL;
    node->eval_jacobian = eval_jacobian;
    node->eval_wsum_hess = NULL;
    node->local_jacobian = NULL;
    node->local_wsum_hess = NULL;
    node->is_affine = NULL;
    node->forward = forward;
    node->free_type_data = NULL; /* Q is owned by caller */

    expr_retain(child);
}

expr *new_quad_form(expr *left, CSR_Matrix *Q)
{
    /* Allocate the type-specific struct */
    quad_form_expr *qnode = (quad_form_expr *) malloc(sizeof(quad_form_expr));
    if (!qnode) return NULL;

    expr *node = &qnode->base;

    /* Initialize base quad_form fields */
    init_quad_form(node, left);

    /* Check if allocation succeeded */
    if (!node->value)
    {
        free(qnode);
        return NULL;
    }

    /* Set type-specific field */
    qnode->Q = Q;

    return node;
}

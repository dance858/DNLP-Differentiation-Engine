#ifndef EXPR_H
#define EXPR_H

#include "utils/CSC_Matrix.h"
#include "utils/CSR_Matrix.h"
#include <stdbool.h>
#include <stddef.h>

#define JAC_IDXS_NOT_SET -1

/* Forward declarations */
struct expr;
struct int_double_pair;

/* Function pointer types */
typedef void (*forward_fn)(struct expr *node, const double *u);
typedef void (*jacobian_init_fn)(struct expr *node);
typedef void (*wsum_hess_init_fn)(struct expr *node);
typedef void (*eval_jacobian_fn)(struct expr *node);
typedef void (*wsum_hess_fn)(struct expr *node, double *w);
typedef void (*local_jacobian_fn)(struct expr *node, double *out);
typedef void (*local_wsum_hess_fn)(struct expr *node, double *out, double *w);
typedef bool (*is_affine_fn)(struct expr *node);
typedef void (*free_type_data_fn)(struct expr *node);

/* Base expression node structure - contains only common fields */
typedef struct expr
{
    // ------------------------------------------------------------------------
    //                         general quantities
    // ------------------------------------------------------------------------
    int d1, d2, size, n_vars, refcount, var_id;
    struct expr *left;
    struct expr *right;
    double *dwork;
    int *iwork;

    // ------------------------------------------------------------------------
    //                     oracle related quantities
    // ------------------------------------------------------------------------
    double *value;
    CSR_Matrix *jacobian;
    CSR_Matrix *wsum_hess;
    forward_fn forward;
    jacobian_init_fn jacobian_init;
    wsum_hess_init_fn wsum_hess_init;
    eval_jacobian_fn eval_jacobian;
    wsum_hess_fn eval_wsum_hess;

    // ------------------------------------------------------------------------
    //                      other things
    // ------------------------------------------------------------------------
    CSR_Matrix *CSR_work;
    is_affine_fn is_affine;
    local_jacobian_fn local_jacobian;   /* used by elementwise univariate atoms*/
    local_wsum_hess_fn local_wsum_hess; /* used by elementwise univariate atoms*/
    free_type_data_fn free_type_data;   /* Cleanup for type-specific fields */

} expr;

/* Type-specific expression structures that "inherit" from expr */

/* Linear operator: y = A * x */
typedef struct linear_op_expr
{
    expr base; /* MUST be first member for casting to work */
    CSC_Matrix *A_csc;
    CSR_Matrix *A_csr;
} linear_op_expr;

/* Power: y = x^p */
typedef struct power_expr
{
    expr base; /* MUST be first member for casting to work */
    int p;
} power_expr;

/* Quadratic form: y = x'*Q*x */
typedef struct quad_form_expr
{
    expr base; /* MUST be first member for casting to work */
    CSR_Matrix *Q;
} quad_form_expr;

/* Sum reduction along an axis */
typedef struct sum_expr
{
    expr base; /* MUST be first member for casting to work */
    int axis;
    struct int_double_pair *int_double_pairs; /* for sorting jacobian entries */
} sum_expr;

/* Horizontal stack (concatenate) */
typedef struct hstack_expr
{
    expr base; /* MUST be first member for casting to work */
    expr **args;
    int n_args;
} hstack_expr;

expr *new_expr(int d1, int d2, int n_vars);
void free_expr(expr *node);

/* Reference counting helpers */
void expr_retain(expr *node);

#endif /* EXPR_H */

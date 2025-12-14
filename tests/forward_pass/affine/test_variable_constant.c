#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../minunit.h"
#include "../test_helpers.h"
#include "affine/constant.h"
#include "affine/variable.h"
#include "expr.h"

const char *test_variable()
{
    printf("Test: Variable node\n");

    double u[3] = {1.0, 2.0, 3.0};
    expr *var = new_variable(3);

    var->forward(var, u);

    mu_assert("Variable test failed", compare_values(var, u, 3));
    free_expr(var);
    return 0;
}

const char *test_constant()
{
    printf("Test: Constant node\n");

    double c[2] = {5.0, 10.0};
    double u[2] = {0.0, 0.0}; /* Input doesn't matter for constants */
    expr *const_node = new_constant(2, c);

    const_node->forward(const_node, u);

    mu_assert("Constant test failed", compare_values(const_node, c, 2));
    free_expr(const_node);
    return 0;
}

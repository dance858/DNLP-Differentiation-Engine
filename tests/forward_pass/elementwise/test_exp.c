#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../minunit.h"
#include "../test_helpers.h"
#include "affine/variable.h"
#include "elementwise.h"
#include "expr.h"

const char *test_exp()
{
    printf("Test: Exponential (exp(variable))\n");

    double u[2] = {0.0, 1.0};
    expr *var = new_variable(2);
    expr *exp_node = new_exp(var);

    exp_node->forward(exp_node, u);

    double expected[2] = {exp(0.0), exp(1.0)};
    mu_assert("Exponential test failed", compare_values(exp_node, expected, 2));
    free_expr(exp_node);
    return 0;
}

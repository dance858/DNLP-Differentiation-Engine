#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../minunit.h"
#include "../test_helpers.h"
#include "affine/add.h"
#include "affine/constant.h"
#include "affine/variable.h"
#include "elementwise.h"
#include "expr.h"

const char *test_composite()
{
    printf("Test: Composite expression (log(exp(x) + c))\n");

    double u[2] = {1.0, 2.0};
    double c[2] = {1.0, 1.0};

    /* Build tree: log(exp(x) + c) */
    expr *var = new_variable(2);
    expr *exp_node = new_exp(var);
    expr *const_node = new_constant(2, c);
    expr *sum = new_add(exp_node, const_node);
    expr *log_node = new_log(sum);

    log_node->forward(log_node, u);

    double expected[2] = {log(exp(1.0) + 1.0), log(exp(2.0) + 1.0)};
    mu_assert("Composite test failed", compare_values(log_node, expected, 2));

    free_expr(log_node);
    return 0;
}

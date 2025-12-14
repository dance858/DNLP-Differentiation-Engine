#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../minunit.h"
#include "../test_helpers.h"
#include "affine/variable.h"
#include "elementwise.h"
#include "expr.h"

const char *test_log()
{
    printf("Test: Logarithm (log(variable))\n");

    double u[2] = {1.0, 2.718281828}; /* e */
    expr *var = new_variable(2);
    expr *log_node = new_log(var);

    log_node->forward(log_node, u);

    double expected[2] = {log(1.0), log(2.718281828)};
    mu_assert("Logarithm test failed", compare_values(log_node, expected, 2));
    free_expr(log_node);
    return 0;
}

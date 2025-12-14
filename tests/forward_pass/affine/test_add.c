#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../minunit.h"
#include "../test_helpers.h"
#include "affine/add.h"
#include "affine/constant.h"
#include "affine/variable.h"
#include "expr.h"

const char *test_addition()
{
    printf("Test: Addition (variable + constant)\n");

    double u[2] = {3.0, 4.0};
    double c[2] = {1.0, 2.0};

    expr *var = new_variable(2);
    expr *const_node = new_constant(2, c);
    expr *sum = new_add(var, const_node);

    sum->forward(sum, u);

    double expected[2] = {4.0, 6.0};
    mu_assert("Addition test failed", compare_values(sum, expected, 2));
    free_expr(sum);
    return 0;
}

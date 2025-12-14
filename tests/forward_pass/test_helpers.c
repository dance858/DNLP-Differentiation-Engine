#include <math.h>
#include <stdio.h>

#include "expr.h"

#define EPSILON 1e-9

int compare_values(expr *node, const double *expected, int size)
{
    for (int i = 0; i < size; i++)
    {
        if (fabs(node->value[i] - expected[i]) > EPSILON)
        {
            printf("  FAILED: node->value[%d] = %f, expected %f\n", i,
                   node->value[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

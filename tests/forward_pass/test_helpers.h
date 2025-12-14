#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include "expr.h"

/* Compare actual values with expected values
 * Returns 1 if all values match, 0 otherwise */
int compare_values(expr *node, const double *expected, int size);

#endif /* TEST_HELPERS_H */

#include <stdio.h>

#include "minunit.h"

/* Forward declarations for all tests */
const char *test_variable(void);
const char *test_constant(void);
const char *test_addition(void);
const char *test_exp(void);
const char *test_log(void);
const char *test_composite(void);

int main(void)
{
    printf("=== Running Forward Pass Tests ===\n\n");

    int tests_run = 0;

    printf("--- Variable & Constant ---\n");
    mu_run_test(test_variable, tests_run);
    mu_run_test(test_constant, tests_run);

    printf("--- Affine Operations ---\n");
    mu_run_test(test_addition, tests_run);

    printf("--- Elementwise Operations ---\n");
    mu_run_test(test_exp, tests_run);
    mu_run_test(test_log, tests_run);

    printf("--- Composite Expression ---\n");
    mu_run_test(test_composite, tests_run);

    printf("\n=== All %d tests passed ===\n", tests_run);

    return 0;
}

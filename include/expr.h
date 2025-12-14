#ifndef EXPR_H
#define EXPR_H

#include <stddef.h>

/* Forward declaration */
struct expr;

/* Function pointer type for forward pass computation */
typedef void (*forward_fn)(struct expr *node, const double *u);

/* Expression node structure */
typedef struct expr
{
    int m;              /* Output dimension */
    double *value;      /* Preallocated output value array */
    struct expr *left;  /* Left child (can be NULL) */
    struct expr *right; /* Right child (can be NULL) */
    forward_fn forward; /* Forward pass function */
} expr;

/* Memory management functions */
expr *new_expr(int m);
void free_expr(expr *node);

#endif /* EXPR_H */

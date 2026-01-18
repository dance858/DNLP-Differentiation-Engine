#ifndef DNLP_BLAS_H_GUARD
#define DNLP_BLAS_H_GUARD

#ifdef __cplusplus
extern "C"
{
#endif

/* Default to underscore suffix for BLAS/LAPACK function names */
#ifndef BLAS_SUFFIX
#define BLAS_SUFFIX _
#endif

/* Handle BLAS function name mangling */
#if defined(NO_BLAS_SUFFIX) && NO_BLAS_SUFFIX > 0
/* No suffix version */
#define BLAS(x) d##x
#define BLASI(x) id##x
#else
/* Macro indirection needed for BLAS_SUFFIX to work correctly */
#define _stitch(pre, x, post) pre##x##post
#define _stitch2(pre, x, post) _stitch(pre, x, post)
/* Add suffix (default: underscore) */
#define BLAS(x) _stitch2(d, x, BLAS_SUFFIX)
#define BLASI(x) _stitch2(id, x, BLAS_SUFFIX)
#endif

/* BLAS integer type */
#ifdef BLAS_64
#include <stdint.h>
    typedef int64_t blas_int;
#else
typedef int blas_int;
#endif

#ifdef __cplusplus
}
#endif

#endif /* DNLP_BLAS_H_GUARD */

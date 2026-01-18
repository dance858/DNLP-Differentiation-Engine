# BLAS Integration

This project **requires** BLAS/LAPACK libraries for optimized linear algebra operations.

## Building

BLAS and LAPACK are **mandatory dependencies**. CMake will search for them and fail if not found.

### Standard Build

```bash
mkdir -p build
cd build
cmake ..
make
```

## Installing BLAS/LAPACK (Required)

You must install BLAS and LAPACK before building:

**Ubuntu/Debian:**
```bash
sudo apt-get install libblas-dev liblapack-dev
```

**macOS:**
```bash
brew install openblas lapack
```

**Alternative Libraries:**
- **OpenBLAS** (recommended): Fast, open-source BLAS
- **Intel MKL**: Optimized for Intel processors
- **Apple Accelerate**: Built-in on macOS
- **ATLAS**: Automatically tuned

## Configuration Options

```bash
# Use 64-bit integers for BLAS (if your BLAS library uses 64-bit ints)
cmake -DBLAS_64=ON ..

# If BLAS functions have no underscore suffix
cmake -DNO_BLAS_SUFFIX=ON ..
```

## Using BLAS in Code

Include the wrapper header:

```c
#include "utils/blas_wrappers.h"

// These use optimized BLAS routines
double norm = vec_norm2(x, n);
double dot_product = vec_dot(x, y, n);
vec_axpy(2.0, x, y, n);  // y += 2.0 * x
vec_scale(0.5, x, n);    // x *= 0.5
mat_vec_mult(A, x, y, m, n);  // y = A*x
```

## Direct BLAS Usage

For advanced usage, include the BLAS header directly:

```c
#include "blas.h"

// Declare the BLAS function you need
extern double BLAS(nrm2)(blas_int *n, const double *x, blas_int *incx);

void my_function(double *x, int n) {
    blas_int bn = (blas_int)n;
    blas_int inc = 1;
    double norm = BLAS(nrm2)(&bn, x, &inc);
}
```

The `BLAS(name)` macro handles function name mangling (e.g., `BLAS(nrm2)` → `dnrm2_`).

## Available BLAS Functions

**Level 1 (vector-vector):**
- `BLAS(nrm2)` - L2 norm
- `BLAS(dot)` - dot product
- `BLAS(axpy)` - y = alpha*x + y
- `BLAS(scal)` - x = alpha*x
- `BLASI(amax)` - index of max absolute value

**Level 2 (matrix-vector):**
- `BLAS(gemv)` - matrix-vector multiply

**Level 3 (matrix-matrix):**
- `BLAS(gemm)` - matrix-matrix multiply

**LAPACK:**
- `BLAS(gesv)` - solve linear system
- `BLAS(syevr)` - symmetric eigenvalue decomposition
- Many more...

## Performance Notes

BLAS provides significant speedups for:
- Large vector operations (n > 1000)
- Matrix-vector products
- Matrix-matrix products

For small vectors (n < 100), the overhead may outweigh the benefits.

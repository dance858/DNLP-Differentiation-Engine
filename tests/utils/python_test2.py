import scipy.sparse as sp
import numpy as np

np.random.seed(42)
m = 15
n = 10
density = 0.1
# A = sp.random(m, n, density=density, format="csc", dtype=float)
# A.data = np.round(A.data, 2) + 0.1
# d = np.round(np.random.randn(m), 2) + 0.1


# Ap = [0 2 4 6 6 9 12 12 14 14 15]
# Ai = [9 12 3 4 1 6 4 8 13 1 3 7 5 13 6]
# Ax =
#    [0.99 0.9 0.51 0.64 0.39 0.29 0.26 0.91 0.35 0.18 0.33 0.73 0.97 0.86 1.03]
# d =
#        [-0.6 - 0.23 - 0.29 - 1.36 0.4 0.36 0.11 - 0.13 - 1.32 - 0.32 - 0.24 - 0.7 -
#            0.06 0.5 1.99] Cp = [0 1 4 7 7 10 13 13 15 15 17] Ci =
#            [0 1 4 5 2 5 9 1 4 7 1 2 5 4 7 2 9] Cx =
#                [-0.362232 - 0.189896 0.06656 - 0.228888 - 0.025732 -
#                    0.016146 0.032857 0.06656 - 1.004802 0.1505 - 0.228888 -
#                    0.016146 - 0.224833 0.1505 0.708524 0.032857 0.116699] *
#                */


Ap = np.array([0, 2, 4, 6, 6, 9, 12, 12, 14, 14, 15])
Ai = np.array([9, 12, 3, 4, 1, 6, 4, 8, 13, 1, 3, 7, 5, 13, 6])
Ax = np.array(
    [
        0.99,
        0.9,
        0.51,
        0.64,
        0.39,
        0.29,
        0.26,
        0.91,
        0.35,
        0.18,
        0.33,
        0.73,
        0.97,
        0.86,
        1.03,
    ]
)

d = np.array(
    [
        -0.6,
        -0.23,
        -0.29,
        -1.36,
        0.4,
        0.36,
        0.11,
        -0.13,
        -1.32,
        -0.32,
        -0.24,
        -0.7,
        -0.06,
        0.5,
        1.99,
    ]
)
A = sp.csc_matrix((Ax, Ai, Ap), shape=(m, n))

C = A.T @ sp.diags(d) @ A
C.sort_indices()

Ap = A.indptr
Ai = A.indices
Ax = A.data

Cp = C.indptr
Ci = C.indices
Cx = C.data

# set precision for printing
np.set_printoptions(precision=10, suppress=True)

print("Ap =", Ap)
print("Ai =", Ai)
print("Ax =", Ax)
print("d =", d)
print("Cp =", Cp)
print("Ci =", Ci)
print("Cx =", Cx)

import pdb

pdb.set_trace()

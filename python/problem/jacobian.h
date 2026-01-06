#ifndef PROBLEM_JACOBIAN_H
#define PROBLEM_JACOBIAN_H

#include "common.h"

static PyObject *py_problem_jacobian(PyObject *self, PyObject *args)
{
    PyObject *prob_capsule;
    PyObject *u_obj;
    if (!PyArg_ParseTuple(args, "OO", &prob_capsule, &u_obj))
    {
        return NULL;
    }

    problem *prob =
        (problem *) PyCapsule_GetPointer(prob_capsule, PROBLEM_CAPSULE_NAME);
    if (!prob)
    {
        PyErr_SetString(PyExc_ValueError, "invalid problem capsule");
        return NULL;
    }

    if (prob->n_constraints == 0)
    {
        /* Return empty CSR components */
        npy_intp zero = 0;
        npy_intp one = 1;
        PyObject *data = PyArray_SimpleNew(1, &zero, NPY_DOUBLE);
        PyObject *indices = PyArray_SimpleNew(1, &zero, NPY_INT32);
        PyObject *indptr = PyArray_SimpleNew(1, &one, NPY_INT32);
        ((int *) PyArray_DATA((PyArrayObject *) indptr))[0] = 0;
        return Py_BuildValue("(OOO(ii))", data, indices, indptr, 0, prob->n_vars);
    }

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    CSR_Matrix *jac = problem_jacobian(prob, (const double *) PyArray_DATA(u_array));

    npy_intp nnz = jac->nnz;
    npy_intp m_plus_1 = jac->m + 1;

    PyObject *data = PyArray_SimpleNew(1, &nnz, NPY_DOUBLE);
    PyObject *indices = PyArray_SimpleNew(1, &nnz, NPY_INT32);
    PyObject *indptr = PyArray_SimpleNew(1, &m_plus_1, NPY_INT32);

    if (!data || !indices || !indptr)
    {
        Py_XDECREF(data);
        Py_XDECREF(indices);
        Py_XDECREF(indptr);
        Py_DECREF(u_array);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *) data), jac->x, nnz * sizeof(double));
    memcpy(PyArray_DATA((PyArrayObject *) indices), jac->i, nnz * sizeof(int));
    memcpy(PyArray_DATA((PyArrayObject *) indptr), jac->p, m_plus_1 * sizeof(int));

    Py_DECREF(u_array);
    return Py_BuildValue("(OOO(ii))", data, indices, indptr, jac->m, jac->n);
}

#endif /* PROBLEM_JACOBIAN_H */

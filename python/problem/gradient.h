#ifndef PROBLEM_GRADIENT_H
#define PROBLEM_GRADIENT_H

#include "common.h"

static PyObject *py_problem_gradient(PyObject *self, PyObject *args)
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

    PyArrayObject *u_array =
        (PyArrayObject *) PyArray_FROM_OTF(u_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!u_array)
    {
        return NULL;
    }

    double *grad = problem_gradient(prob, (const double *) PyArray_DATA(u_array));

    npy_intp size = prob->n_vars;
    PyObject *out = PyArray_SimpleNew(1, &size, NPY_DOUBLE);
    if (!out)
    {
        Py_DECREF(u_array);
        return NULL;
    }
    memcpy(PyArray_DATA((PyArrayObject *) out), grad, size * sizeof(double));

    Py_DECREF(u_array);
    return out;
}

#endif /* PROBLEM_GRADIENT_H */

#include "utils/mini_numpy.h"
#include <string.h>

void repeat(double *result, const double *a, int len, int repeats)
{
    int idx = 0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < repeats; j++)
        {
            result[idx++] = a[i];
        }
    }
}

void tile_double(double *result, const double *a, int len, int tiles)
{
    for (int i = 0; i < tiles; i++)
    {
        memcpy(result + i * len, a, len * sizeof(double));
    }
}

void tile_int(int *result, const int *a, int len, int tiles)
{
    for (int i = 0; i < tiles; i++)
    {
        memcpy(result + i * len, a, len * sizeof(int));
    }
}

void scaled_ones(double *result, int size, double value)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = value;
    }
}

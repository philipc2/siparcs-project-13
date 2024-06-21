import argparse
import time
import numpy as np
from numba import njit

import dask.array as da


def daxpy_list(v1, v2):
    """DAXPY implemented in pure Python, using a simple iteration with list appends"""
    v3 = []
    for i in range(len(v1)):
       v3.append(3 * v1[i] + v2[i])
    return v3

def daxpy_list_comp(v1, v2):
    """DAXPY implemented in pure Python, using list comprehension"""
    return [3 * v1[i] + v2[i] for i in range(len(v1))]

def daxpy_numpy(v1, v2):
    """DAXPY implemented using Numpy"""
    return 3 * v1 + v2

@njit
def daxpy_numba(v1, v2):
    """DAXPY implemented using Numba"""
    v3 = np.empty(len(v1))
    for i in range(len(v1)):
        v3[i] = 3 * v1[i] + v2[i]
    return v3

def daxpy_dask(v1, v2):
    """DAXPY implemented using Dask"""
    return 3 * v1 + v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=True, help="Vector Size")
    parser.add_argument('-m', type=str, required=True, help="Method")

    args = parser.parse_args()
    n = args.n
    m = args.m

    if m == "list":
        v1 = [float(i) for i in range(n)]
        v2 = [float(2 * i) for i in range(n)]
        start = time.time()
        v3 = daxpy_list(v1, v2)
        end = time.time()
    if m == "list_comp":
        v1 = [float(i) for i in range(n)]
        v2 = [float(2 * i) for i in range(n)]

        start = time.time()
        v3 = daxpy_list_comp(v1, v2)
        end = time.time()
    if m == "numpy":
        v1 = np.arange(0, n, dtype=np.double)
        v2 = 2 * np.arange(0, n, dtype=np.double)

        start = time.time()
        v3 = daxpy_numpy(v1, v2)
        end = time.time()

    if m == "numba":
        v1 = np.arange(0, n, dtype=np.double)
        v2 = 2 * np.arange(0, n, dtype=np.double)

        start = time.time()
        v3 = daxpy_numba(v1, v2)
        end = time.time()

    if m == "dask":
        v1 = da.arange(0, n, dtype=np.double)
        v2 = 2 * da.arange(0, n, dtype=np.double)

        start = time.time()
        v3 = daxpy_dask(v1, v2)
        end = time.time()

    # print the execution time in seconds
    print(end - start)
    return end-start

if __name__ == "__main__":
    main()

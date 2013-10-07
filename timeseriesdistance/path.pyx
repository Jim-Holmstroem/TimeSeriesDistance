from __future__ import print_function, division

from operator import itemgetter
from functools import partial
from itertools import product, starmap

from cython import boundscheck, wraparound

import numpy as np
cimport numpy as np

@boundscheck(False)
@wraparound(False)
cdef inline lpc(
    np.ndarray[np.float64_t, ndim=2] D,
    Py_ssize_t i, Py_ssize_t j,
    double C_D, double C_HV
): # local path constraint
    return (
        ((i - 1, j - 1), C_D  * D[i - 1, j - 1]),
        ((i    , j - 1), C_HV * D[i    , j - 1]),
        ((i - 1, j    ), C_HV * D[i - 1, j    ]),
    )

@boundscheck(False)
@wraparound(False)
def D_from_c(
    np.ndarray[np.float64_t, ndim=2] c,
    double C_D, double C_HV
):
    """
    c : float matrix
    """
    D = np.copy(c)

    D[0, 0] = 0
    np.cumsum(D[:, 0], out=D[:, 0])
    np.cumsum(D[0, :], out=D[0, :])

    #NOTE couldn't find any vectorization that solved this
    # tried with stuff like np.minimum(D[1:, i], D[:-1, i], out=D[1:, i]) to basically try todo cummin
    def foo(D, i, j):  # NOTE sideeffect only, and order dependent
        # int, int -> () , Monad how?
        D[i, j] += min(
            lpc(D, i, j, C_D, C_HV),
            key=itemgetter(1)
        )[1]

    list(starmap(
        partial(foo, D),
        product(*map(partial(xrange, 1), D.shape))
    ))

    return D

@boundscheck(False)
@wraparound(False)
def min_path(
    np.ndarray[np.float64_t, ndim=2] D,
    double C_D, double C_HV
):
    """
    D : float matrix
    """
    cdef Py_ssize_t i = D.shape[0] - 1
    cdef Py_ssize_t j = D.shape[1] - 1

    path = [(i, j)]
    while not (i == 0 and j == 0):
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            i, j = min(
                lpc(D, i, j, C_D, C_HV),
                key=itemgetter(1)
            )[0]
        path.append((i, j))

    return path

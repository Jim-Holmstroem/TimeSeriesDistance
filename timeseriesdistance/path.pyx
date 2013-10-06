from __future__ import print_function, division

from operator import itemgetter
from functools import partial
from itertools import product, starmap

import numpy as np
cimport numpy as np

cdef inline lpc(
    np.ndarray[np.float64_t, ndim=2] D,
    Py_ssize_t i, Py_ssize_t j,
    double C_D, double C_HV
): # local path constraint
    return (
        ((i-1, j-1), C_D *D[i-1, j-1]),
        ((i  , j-1), C_HV*D[i  , j-1]),
        ((i-1, j  ), C_HV*D[i-1, j  ]),
    )

def D_from_c(c, C_D, C_HV):
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

def min_path(D, C_D, C_HV):
    """
    D : float matrix
    """
    i, j = map(lambda size: size-1, D.shape)
    path = [(i, j)]
    while not (i == 0 and j == 0):
        assert(i>=0 and j>=0)
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

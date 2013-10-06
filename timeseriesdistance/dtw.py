from __future__ import print_function, division

from itertools import starmap, product
from functools import partial
from operator import itemgetter

import numpy as np
from scipy import sparse

import matplotlib.pyplot as plt

from timeseriesdistance.metric import Metric


def path_2_matrix(path, shape):
    """
    """
    matrix = sparse.coo_matrix(
        (
            np.ones(len(path)),
            zip(*path),
        ),
        shape=shape,
    ).todense()

    return matrix


class DTW(Metric):
    """

    [1] Dynamic Time Warping Algorithm Review, Pavel Senin
    [2] Time Warping, Springer, http://www.springer.com/978-3-540-74047-6

    Parameters
    ----------
    f : ufunc
        Inner distance (or what it's called)
    """
    def __init__(self, C_HV=1, C_D=10, f=np.square, verbose=False):
        self.f = f
        self.verbose = verbose
        self.C_HV = C_HV
        self.C_D = C_D

    def __call__(self, a, b):
        a, b = map(np.asarray, [a, b])
        #local_cost_matrix
        c = self.f(np.subtract.outer(a, b))  # f(|..|)
        # Optimal warping path p_i = (m_i, n_i):
        #
        # Boundary condition: p_1 = (1,1), p_K = (M, N)
        # Monotonicity condition: i<j => n_i<n_j AND m_i<m_j
        # Step size  condition: p_{l-1}-p_l \in {(1,1), (1,0), (0,1)}  # NOTE just for now


        D = np.copy(c)

        D[0, 0] = 0
        np.cumsum(D[:, 0], out=D[:, 0])
        np.cumsum(D[0, :], out=D[0, :])

        def lpc(D, i, j): # local path constraint
            return (
                ((i-1, j-1), self.C_D *D[i-1, j-1]),
                ((i  , j-1), self.C_HV*D[i  , j-1]),
                ((i-1, j  ), self.C_HV*D[i-1, j  ]),
            )

        #NOTE couldn't find any vectorization that solved this
        # tried with stuff like np.minimum(D[1:, i], D[:-1, i], out=D[1:, i]) to basically try todo cummin
        def foo(D, i, j):  # NOTE sideeffect only, and order dependent
            # int, int -> () , Monad how?
            min_ = lpc(D, i, j)
            D[i, j] += min(
                min_,
                key=itemgetter(1)
            )[1]

        list(starmap(
            partial(foo, D),
            product(*map(partial(xrange, 1), map(len, [a, b])))
        ))


        # NOTE expanded recursion
        # TODO write down the recursive algorithm
        # find the minimum path in a gready manner (is this really the minimum? TODO)
        i, j = map(lambda size: size-1, c.shape)
        path = [(i, j)]
        while not (i == 0 and j == 0):
            assert(i>=0 and j>=0)
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_ = lpc(D, i, j)
                i, j = min(
                    min_,
                    key=itemgetter(1)
                )[0]
            path.append((i, j))

        path_matrix = path_2_matrix(path, c.shape)

        total_cost = np.sum(map(c.__getitem__, path))

        if self.verbose:
            plt.hold(True)
            plt.imshow(c.T, interpolation='nearest')
            plt.plot(*zip(*path), color='red', linewidth=1.0)
            plt.title("total_cost={}".format(total_cost))
            plt.show()

        return total_cost

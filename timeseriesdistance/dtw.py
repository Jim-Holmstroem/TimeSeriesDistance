from __future__ import print_function, division

from itertools import starmap, product
from functools import partial
from operator import itemgetter

from abc import abstractmethod, ABCMeta

import numpy as np
import scipy as sp

class Metric(object):  # TODO move it out
    __metaclass__ = ABCMeta
    @abstractmethod
    def __call__(self, a, b):
        """a, b is time series wich have the same time sampling.

        Parameters
        ----------
        a : array, shape (M, 1)

        b : array, shape (N, 1)

        """
        pass

class DTW(Metric):
    """

    [1] Dynamic Time Warping Algorithm Review, Pavel Senin

    Parameters
    ----------
    f : ufunc
        Inner distance (or what it's called)
    """
    def __init__(self, f=None):
        self.f = f

    def __call__(self, a, b):
        a, b = map(np.asarray, [a, b])
        #local_cost_matrix
        c = np.square(np.subtract.outer(a, b))  # f(|..|)
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
                ((i-1, j-1), D[i-1, j-1]),
                ((i  , j-1), D[i  , j-1]),
                ((i-1, j  ), D[i-1, j  ]),
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

        print("D=\n", D)

        # NOTE expanded recursion
        # TODO write down the recursive algorithm
        # find the minimum path in a gready manner (is this really the minimum? TODO)
        i, j = map(lambda size: size-1, c.shape)
        path = [(i, j)]
        while i > 0 and j > 0:
            from warnings import warn
            warn('somethings fishy here')j
            if i == 0:
                j -= 0
            elif j == 0:
                i -= 0
            else:
                min_ = lpc(D, i, j)
                i, j = min(
                    min_,
                    key=itemgetter(1)
                )[0]
            path.append((i, j))
        print(i, j)
        print("path=\n",
            sp.sparse.coo_matrix(
                (
                    np.ones(len(path)),
                    zip(*path),
                ),
                shape=c.shape,
            ).todense()
        )
        return path

DTW()(range(10), range(8))
DTW()(range(10), range(10))

ts1 =   [1,1,2,3,2,0]
ts2 = [0,1,1,2,3,2,1]
DTW()(ts1, ts2)


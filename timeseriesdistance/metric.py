from __future__ import print_function, division

from abc import abstractmethod, ABCMeta

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

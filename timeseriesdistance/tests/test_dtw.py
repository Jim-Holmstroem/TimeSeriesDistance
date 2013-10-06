from __future__ import print_function, division

from operator import itemgetter

from nose.tools import assert_equal, assert_greater

import numpy  as np
from scipy.io import wavfile

from ..dtw import DTW

def downsample(data, w=16):
    """ could use np.add.reduceat but a bit quirky
    """
    data = data[:w * (data.size // w)]
    downsampled_data = np.mean(data.reshape((-1, w)), axis=-1)

    return downsampled_data

def test_wav():
    # http://labrosa.ee.columbia.edu/matlab/dtw/

    test1, test2 = map(downsample, map(itemgetter(1), map(
        wavfile.read,
        map(
            "../data/{}.wav".format,
            ["test1", "test2"]
        )
    )))

    assert_greater(DTW(verbose=True)(test1, test2), 0)

def test_trivial():
    assert_equal(DTW()(range(10), range(10)), 0)

def test_unequal_length():
    assert_greater(DTW()(range(10), range(8)), 0)

def test_simple():
    ts1 =   [1,1,2,3,2,0]
    ts2 = [0,1,1,2,3,2,1]
    assert_greater(DTW()(ts1, ts2), 0)

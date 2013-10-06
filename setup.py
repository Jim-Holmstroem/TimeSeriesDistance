from __future__ import print_function, division

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

VERSION = '0.05'

extensions = [
    Extension(
        "timeseriesdistance.path",
        ['timeseriesdistance/path.pyx']
    ),
]

setup(
    name='TimeSeriesDistance',
    description='Distance measures for time series.',
    author='Jim Holmstrom',
    author_email='jim.holmstroem@gmail.com',
    packages=['timeseriesdistance', 'timeseriesdistance.tests'],
    include_package_data=True,
    version=VERSION,
    zip_safe=False,
    install_requires=['numpy>=1.7',],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
)

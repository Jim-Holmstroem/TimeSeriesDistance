from __future__ import print_function, division

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

#from setuptools import find_packages

VERSION = '0.02'

extensions = [
   Extension("timeseriesdistance.hellocython", ['timeseriesdistance/hellocython.pyx']),
]

print(map(repr, extensions))

setup(
    name='TimeSeriesDistance',
    description='Distance measures for time series.',
    author='Jim Holmstrom',
    author_email='jim.holmstroem@gmail.com',
    packages=['timeseriesdistance'],
    include_package_data=True,
    version=VERSION,
    zip_safe=False,
    install_requires=['numpy>=1.7',],
    cmdclass={'build_ext': build_ext},
    ext_modules=extensions,
)

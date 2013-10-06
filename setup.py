from setuptools import setup, find_packages

VERSION = '0.01'

setup(
    name='TimeSeriesDistance',
    packages=find_packages(),
    include_package_data=True,
    version=VERSION,
    zip_safe=False,
    install_requires=[
        'numpy>=1.7',
    ]
)

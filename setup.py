# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import pybind11

ext_modules = [
    Extension(
        'example',  # Name of the module
        ['example.cpp'],
        include_dirs=[
            pybind11.get_include(),  # Path to pybind11 headers
        ],
        language='c++'
    ),
]

setup(
    name='example',
    version='0.0.1',
    author='Your Name',
    description='A Python package with C++ extension (using pybind11)',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)

# setup.py
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import pathlib

# Get the directory containing this file
this_dir = pathlib.Path(__file__).parent.resolve()

# Path to the local pybind11 headers
pybind11_include = str(this_dir / 'extern' / 'pybind11' / 'include')

def get_cpp_flag():
    if sys.platform == 'win32':
        return '/std:c++14'  # Change to '/std:c++17' if using C++17
    else:
        return '-std=c++14'  # Change to '-std=c++17' if using C++17

ext_modules = [
    Extension(
        'example',  # Name of the module
        ['example.cpp'],
        include_dirs=[
            pybind11_include,
        ],
        language='c++',
        extra_compile_args=[get_cpp_flag()],
    ),
]

setup(
    name='example',
    version='0.0.1',
    author='Your Name',
    description='A Python package with a C++ extension (using pybind11)',
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)

cmake_minimum_required(VERSION 3.30)
project(MRHMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

add_subdirectory("extern/pybind11" REQUIRED)

pybind11_add_module(mrhmm alphabetacompress.cpp)
install(TARGETS mrhmm DESTINATION .)
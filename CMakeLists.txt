cmake_minimum_required(VERSION 3.30)
project(MRHMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selction")  # or 11, 14, 17,20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

add_subdirectory("extern/pybind11" REQUIRED)
#message("${CMAKE_SOURCE_DIR}/extern/eigen")
#add_subdirectory("extern/eigen/Eigen" REQUIRED)
include_directories("extern/eigen")
#find_package(Eigen3 3.4 EXACT REQUIRED PATHS "${CMAKE_SOURCE_DIR}/extern/eigen" NO_DEFAULT_PATH)
#get_target_property(Eigen3_INCLUDE_DIRS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
#message(STATUS "Eigen3 include directories: ${Eigen3_INCLUDE_DIRS}")


pybind11_add_module(mrhmm alphabetacompress.cpp)
#target_link_libraries(mrhmm PRIVATE Eigen3::Eigen)
#target_include_directories(mrhmm PRIVATE "extern/eigen")
install(TARGETS mrhmm DESTINATION .)
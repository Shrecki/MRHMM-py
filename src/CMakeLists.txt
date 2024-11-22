
pybind11_add_module(mrhmm alphabetacompress.cpp)
#target_link_libraries(mrhmm PRIVATE Eigen3::Eigen)
#target_include_directories(mrhmm PRIVATE "extern/eigen")
set_target_properties(mrhmm PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
install(TARGETS mrhmm DESTINATION .)
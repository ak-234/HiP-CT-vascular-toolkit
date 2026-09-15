cmake_minimum_required(VERSION 3.22)
project(coronary_sdf_cgal LANGUAGES CXX)

find_package(CGAL 6.2 CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(coronary_sdf_cgal src/module.cpp)
target_compile_features(coronary_sdf_cgal PRIVATE cxx_std_17)
target_link_libraries(coronary_sdf_cgal PRIVATE CGAL::CGAL)
target_compile_definitions(coronary_sdf_cgal PRIVATE NOMINMAX)

if(MSVC)
  target_compile_options(coronary_sdf_cgal PRIVATE /O2 /EHsc /bigobj)
else()
  target_compile_options(coronary_sdf_cgal PRIVATE -O3)
endif()

install(TARGETS coronary_sdf_cgal LIBRARY DESTINATION . RUNTIME DESTINATION .)

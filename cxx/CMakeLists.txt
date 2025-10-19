cmake_minimum_required (VERSION 3.11)

project (parallel_pwmig Fortran C CXX)

#set (default_build_type "Release")
set (default_build_type "Debug")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message (STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set (CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif ()
message (STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS_DEBUG "-Wall  -Wno-sign-compare -O0 -g")
set (CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
set (PYBIND11_CPP_STANDARD -std=c++17)

set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN 1)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
  set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE ON)
  message (STATUS "Enabled Coverage")
endif ()

#find_program(MY_PYTHON_EXECUTABLE NAMES python3 python)
#
#if(NOT MY_PYTHON_EXECUTABLE)
#    message(WARNING "Python interpreter not found.")
#    find_package(Python COMPONENTS Interpreter Development)
#else()
#
#    # Get the version of the Python interpreter
#    execute_process(
#        COMMAND ${MY_PYTHON_EXECUTABLE} --version
#        OUTPUT_VARIABLE PYTHON_VERSION_OUTPUT
#        ERROR_VARIABLE PYTHON_VERSION_OUTPUT
#        OUTPUT_STRIP_TRAILING_WHITESPACE
#        ERROR_STRIP_TRAILING_WHITESPACE
#    )
#
#    string(REGEX MATCH "Python ([0-9]+\\.[0-9]+)" _ ${PYTHON_VERSION_OUTPUT})
#    set(MY_PYTHON_VERSION ${CMAKE_MATCH_1})
#    
#    message(STATUS "Found Python version: ${MY_PYTHON_VERSION}")
#    find_package(Python ${MY_PYTHON_VERSION} EXACT COMPONENTS Interpreter Development)
#endif()
#
#if (Python_FOUND)
#  find_package (pybind11)
#  if (NOT pybind11_FOUND)
#    message (STATUS "Building Pybind11")
#    include (cmake/pybind11.cmake)
#    fetch_pybind11 (
#      ${PROJECT_SOURCE_DIR}/cmake
#      ${PROJECT_BINARY_DIR}/pybind11
#      )
#  endif ()
#  message (STATUS "Python_LIBRARIES = ${Python_LIBRARIES}")
#  message (STATUS "Python_EXECUTABLE = ${Python_EXECUTABLE}")
#  message (STATUS "Python_INCLUDE_DIRS = ${Python_INCLUDE_DIRS}")
#  message (STATUS "pybind11_LIBRARIES = ${pybind11_LIBRARIES}")
#  message (STATUS "pybind11_INCLUDE_DIR= ${pybind11_INCLUDE_DIR}")
#  
#  # Disable pybind11's strict GIL checks to prevent segfaults in CI environments
#  # where background threads (e.g. from PyMongo) may interact with pybind11 objects
#  add_compile_definitions(PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF)
#  
#  add_subdirectory (python)
#else ()
#  message (WARNING "Python not found")
#endif ()

find_package (PythonInterp 3)

if (PYTHONINTERP_FOUND)
  find_package (pybind11)
  if (NOT pybind11_FOUND)
    message (STATUS "Building Pybind11")
    include (cmake/pybind11.cmake)
    fetch_pybind11 (
      ${PROJECT_SOURCE_DIR}/cmake
      ${PROJECT_BINARY_DIR}/pybind11
      )
  endif ()
  message (STATUS "PYTHON_LIBRARIES = ${PYTHON_LIBRARIES}")
  message (STATUS "PYTHON_EXECUTABLE = ${PYTHON_EXECUTABLE}")
  message (STATUS "PYTHON_INCLUDE_DIRS = ${PYTHON_INCLUDE_DIRS}")
  message (STATUS "pybind11_LIBRARIES = ${pybind11_LIBRARIES}")
  message (STATUS "pybind11_INCLUDE_DIR= ${pybind11_INCLUDE_DIR}")
  add_subdirectory (python)
else ()
  message (WARNING "Python not found")
endif ()

find_package (Boost 1.86.0 COMPONENTS serialization)
if (NOT Boost_FOUND)
  message (STATUS "Building Boost")
  include (cmake/boost.cmake)
  fetch_boost (
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/boost
    )
endif ()
message (STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}")
message (STATUS "Boost_LIBRARY_DIRS = ${Boost_LIBRARY_DIRS}")
message (STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}")

find_package (GSL)
if (NOT GSL_FOUND)
  message (STATUS "Building GNU Scientific Library (gsl)")
  include (cmake/gsl.cmake)
  fetch_gsl (
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/gsl
    )
endif ()
message (STATUS "GSL_INCLUDE_DIRS=${GSL_INCLUDE_DIRS}")
message (STATUS "GSL_LIBRARIES=${GSL_LIBRARIES}")

find_package (BLAS)
if (NOT BLAS_FOUND)
  message (STATUS "Building OpenBLAS")
  include (cmake/openblas.cmake)
  fetch_openblas (
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/openblas
    )
  set (BLAS_LIBRARIES ${PROJECT_BINARY_DIR}/openblas/lib/libopenblas.a)
endif ()

find_package (LAPACK)
if (NOT LAPACK_FOUND AND NOT BLAS_FOUND)
  message (STATUS "Use OpenBLAS for LAPACK")
  set (BLAS_LIBRARIES ${PROJECT_BINARY_DIR}/openblas/lib/libopenblas.a)
  set (LAPACK_LIBRARIES ${PROJECT_BINARY_DIR}/openblas/lib/libopenblas.a)
elseif (NOT LAPACK_FOUND)
  message (STATUS "Building OpenBLAS for LAPACK")
  include (cmake/openblas.cmake)
  fetch_openblas (
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/openblas
    )
  set (LAPACK_LIBRARIES ${PROJECT_BINARY_DIR}/openblas/lib/libopenblas.a)
endif ()

find_package(OpenMP)

include (FortranCInterface)
list (APPEND FORTRAN_FUNCTIONS ddot dscal daxpy dcopy dnrm2)
list (APPEND FORTRAN_FUNCTIONS dgetrf dgetri)
list (APPEND FORTRAN_FUNCTIONS dlamch dstebz dstein)
FortranCInterface_HEADER (include/FC.h
  MACRO_NAMESPACE "FC_"
  SYMBOLS ${FORTRAN_FUNCTIONS}
  )

# Set up mspass for now with this environment variable
if (DEFINED ENV{MSPASS_HOME})
   set(MSPASSHOME $ENV{MSPASS_HOME})
else ()
   set(MSPASSHOME "/usr/local/mspass")
endif ()
set(MSPASS_LIBRARIES ${MSPASSHOME}/lib/libmspass.a)
message(STATUS "Assuming top level directory for MsPASS is ${MSPASSHOME}")

add_subdirectory (src)
add_subdirectory (include)

#enable_testing ()
#add_subdirectory (test)

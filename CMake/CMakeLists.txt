cmake_minimum_required(VERSION 3.7)
project(rs CXX C CUDA)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}")

find_package(OpenSSL)
find_package(Threads REQUIRED)
find_package(Curses REQUIRED)
find_package(GMP REQUIRED)
find_package(FLINT REQUIRED)
find_package(MATH REQUIRED)
find_package(ARB REQUIRED)
find_package(MPI)

file(WRITE ../src/config.h "\n") 

if(MPI_CXX_FOUND)
   file(APPEND ../src/config.h "#define HAS_MPI 1\n")
 endif()

if(OPENSSL_FOUND)
  file(APPEND ../src/config.h "#define HAVE_SSL 1\n")
else()
  set(OPENSSL_LIBRARIES "")
endif()

find_package(CUDA REQUIRED)


if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE ../CMake/cuda_script) # No suffix required
  set(CUDAFILE ../CMake/check_cuda.cu)
  execute_process(COMMAND nvcc -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
  execute_process(COMMAND ${OUTPUTFILE}
    RESULT_VARIABLE CUDA_RETURN_CODE
    OUTPUT_VARIABLE ARCH)

  if(${CUDA_RETURN_CODE} EQUAL 0)
    set(CUDA_SUCCESS "TRUE")
  else()
    set(CUDA_SUCCESS "FALSE")
  endif()

  if (${CUDA_SUCCESS})
    message(STATUS "CUDA Architecture: ${ARCH}")
    message(STATUS "CUDA Version: ${CUDA_VERSION_STRING}")
    message(STATUS "CUDA Path: ${CUDA_TOOLKIT_ROOT_DIR}")
    message(STATUS "CUDA Libararies: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA Performance Primitives: ${CUDA_npp_LIBRARY}")
  else()
    message(WARNING ${ARCH})
  endif()
endif()


file(APPEND ../src/config.h "#define CUDA_VERSION ${CUDA_VERSION_MAJOR}\n")  

message("Including CMAKE_INCLUDE_PATH = $ENV{CMAKE_INCLUDE_PATH}\n")
message("Including CMAKE_LIBRARY_PATH = $ENV{CMAKE_LIBRARY_PATH}\n")

include_directories(SYSTEM ${MPI_INCLUDE_PATH} $ENV{CMAKE_INCLUDE_PATH})

#add_definitions("-O2")

set(CMAKE_CUDA_FLAGS "-Xptxas -O2 ${ARCH}" )
set(CMAKE_C_FLAGS "-O2 -march=native -Wall")
set(CMAKE_CXX_FLAGS "-O2 -march=native -Wall")

enable_language("CUDA")

file(GLOB code
  "../src/*.c"
  "../src/*.h"
  "../src/*.cu"
  )

add_executable(rs ${code})

target_link_libraries(rs
  ${MPI_C_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
  ${CURSES_LIBRARIES}
  ${ARB_LIBRARIES}
  ${FLINT_LIBRARIES}
  ${GMP_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${MATH_LIBRARIES}
  ${OPENSSL_LIBRARIES})

# Find CUDA
find_package(CUDA)

if (CUDA_FOUND)
  #Get CUDA compute capability
  set(OUTPUTFILE cuda_script) # No suffix required
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

    set(CUDA_NVCC_FLAGS "${ARCH}")
  else()
    message(WARNING ${ARCH})
  endif()
endif()

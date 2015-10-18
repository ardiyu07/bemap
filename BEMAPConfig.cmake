# Set environment variables for building purpose
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${TOPDIR}/common/cmake/)
SET(BEMAP_DATAFILES ${TOPDIR}/common/data/)
SET(BEMAP_INCLUDE_DIRECTORIES ${TOPDIR}/common/inc/)

# Set binary directory
IF (NOT EXISTS ${TOPDIR}/bin/)
  FILE(MAKE_DIRECTORY ${TOPDIR}/bin/)
ENDIF()
SET(BEMAP_BINARY_DIR ${TOPDIR}/bin/)

# Set CUDA variables
IF(BUILD_CUDA)
  FIND_PATH(CUDA_INCLUDE_DIRS
    NAMES cuda.h cuda_runtime_api.h
    HINTS /usr/local/cuda/include/ 
    )
  
  # Set NVCC Flags
  SET(CUDA_NVCC_FLAGS -O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -lpthread) 
ENDIF()

# Set flag
IF(UNIX OR APPLE)
  SET (CMAKE_C_FLAGS "-O3 -g -w")
  SET (CMAKE_CXX_FLAGS "-O3 -g -w")
ELSEIF(WIN32)
  SET (CMAKE_C_FLAGS "/Ox /EHsc")
  SET (CMAKE_CXX_FLAGS "/Ox /EHsc")
  SET (CMAKE_C_FLAGS_DEBUG "/Ox /Zi /EHsc")
  SET (CMAKE_CXX_FLAGS_DEBUG "/Ox /Zi /EHsc")
  SET (CMAKE_C_FLAGS_RELEASE "/Ox /w /EHsc")
  SET (CMAKE_CXX_FLAGS_RELEASE "/Ox /w /EHsc")
  SET (CMAKE_C_FLAGS_RELWITHDEBINFO "/Ox /w /Z7 /EHsc")
  SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "/Ox /w /Z7 /EHsc")
  SET (CMAKE_C_FLAGS_MINSIZEREL "/Ox")
  SET (CMAKE_CXX_FLAGS_MINSIZEREL "/Ox")
ELSE()
  SET (CMAKE_C_FLAGS "")
  SET (CMAKE_CXX_FLAGS "")
ENDIF()
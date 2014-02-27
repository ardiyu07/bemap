CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(BEMAP)

SET(TOPDIR ${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE(BEMAPConfig.cmake)

# Build Samples
ADD_SUBDIRECTORY(src/BlackScholes)
ADD_SUBDIRECTORY(src/GrayScale)
ADD_SUBDIRECTORY(src/GaussianFilter)
ADD_SUBDIRECTORY(src/MonteCarlo)
ADD_SUBDIRECTORY(src/BackProjection)

# User options
# Set default build to specific platform to false
# Hence only build original (single thread) version
OPTION(BUILD_OPENCL "OpenCL Build" OFF)
OPTION(BUILD_CUDA "CUDA Build" OFF)
OPTION(BUILD_OPENMP "OpenMP Build" OFF)

MARK_AS_ADVANCED(
  BUILD_OPENCL
  BUILD_CUDA
  BUILD_OPENMP
  )

# Make Clean
IF(UNIX OR APPLE)
	ADD_CUSTOM_TARGET(clean_all
	  ${CMAKE_COMMAND} -E remove -f CMakeCache.txt
	  ${CMAKE_COMMAND} -E remove -f ${TOPDIR}/bin/*
	  COMMAND make clean
	  COMMENT "Cleaning CMakeCache.txt and binaries")
ENDIF()
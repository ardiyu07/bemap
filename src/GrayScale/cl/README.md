GrayScale, OpenCL Implementation
===================================

* Read LICENSE first
* Input data is provided in the common/ folder 

* Prerequisites
  * CMake 2.8.x
  * OpenCL SDK
    * Intel  : http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
    * AMD    : http://developer.amd.com/zones/openclzone/Pages/default.aspx
    * NVIDIA : http://developer.nvidia.com/opencl

* Compilation (all projects) in build/
```
$ cd /path/to/bemap
$ cd build/
$ cmake ../
$ make
```

* Compilation (only this project) in TestRun
```
$ cd /path/to/GrayScale/cl/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./GrayScale_ocl -h
./GrayScale_ocl [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--kernel|-k NUMBER] [--workitems|-w NUMBER]
     [--use-gpu|-g] [--choose-dev|-d] [--choose-plat|-p DEV]
     [--dev-info] [--prep-time] [--comp-result]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --kernel=KERNEL            Kernel mode (0, 1) -- default = 0
                                      [0] Scalar
                                      [1] SIMD = Single Instruction Multiple Data
 --workitems=NUMBER         Number of (local) workitems for Scalar mode
 --use-gpu                  Use GPU as the CL device
 --choose-dev               Choose which OpenCL device to use
 --choose-plat=DEV          Choose which OpenCL platform to use (0, 1, 2)
                                       [0] Advanced Micro Devices, Inc.
                                       [1] NVIDIA Corporation
                                       [2] Intel(R) Corporation
                                       default: Any CPU device
 --dev-info                 Show Device Info
 --prep-time                Show initialization, memory preparation and copyback time
 --comp-result              Compare native and OpenCL results

 * Examples *
./GrayScale_ocl [OPTS...] -v -w 256 test_data.ppm
./GrayScale_ocl [OPTS...] --output=test_output.ppm test_data.ppm
```

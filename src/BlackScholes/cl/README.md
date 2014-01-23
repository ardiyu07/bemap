BlackScholes, OpenCL Implementation
===================================

* Read LICENSE first

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
$ cd /path/to/BlackScholes/cl/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:

```
./BlackScholes_ocl -h
./BlackScholes_ocl [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--optnum|-O NUMBER] [--riskfree|-R NUMBER] [--volatility|-V NUMBER]
     [--kernel|-k NUMBER] [--width|-W NUMBER] [--workitems|-w NUMBER]
     [--use-gpu|-g] [--choose-dev|-d] [--choose-plat|-p DEV]
     [--dev-info] [--prep-time] [--comp-result]

* Options *
 --verbose             Be verbose
 --help                Print this message
 --output=NAME         Write results to this file
 --optnum=NUMBER       Number of elements in the data array -- default = 50 * 1024 * 1024
 --riskfree=NUMBER     The annualized risk-free interest rate, continuously compounded -- default = 0.02
 --volatility=NUMBER   The volatility of stock's returns -- default = 0.30
 --kernel=KERNEL       Kernel mode (0, 1, [2, 3]) -- default = 0
                                 [0] Scalar
                                 [1] SIMD = Single Instruction Multiple Data
 --width=NUMBER        Data width (for SIMD mode only, must be a multiple of 8)
 --workitems=NUMBER    Number of (local) workitems for Scalar mode
 --use-gpu             Use GPU as the CL device
 --choose-dev          Choose which OpenCL device to use
 --choose-plat=DEV     Choose which OpenCL platform to use (0, 1, 2)
                                  [0] Advanced Micro Devices, Inc.
                                  [1] NVIDIA Corporation
                                  [2] Intel(R) Corporation
                                  default: Any CPU device
 --dev-info            Show Device Info
 --prep-time           Show initialization, memory preparation and copyback time
 --comp-result         Compare GPU and CPU results

 * Examples *
./BlackScholes_ocl [OPTS...] -v -k 0
./BlackScholes_ocl [OPTS...] -v --workitems=128
```
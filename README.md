BEMAP
====

INTRODUCTION
------------
Bemap is a benchmark used to measure parallel hardware performance.

All OpenCL code benchmarks covered in this project are done 
step-by-step along with hand-tunning. Each tuning step executional 
time are measured in details with a comprehensive user interface and 
help option. The exact implementation in native code (C++) is also 
provided in each project folder for reference.

By analyzing these benchmarks, we may analyze:
- How to tune/optimize using the given Parallel Programming API
- Whether the appropriate optimization procedure is provided within the compiler/hardware

This project is derived from [SourceForge site of bemap](http://sourceforge.net/projects/bemap/).

PACKAGES
--------

This package includes:
- Blackscholes
- GrayScale
- GaussianFilter

REQUIREMENTS
------------
* CMake 2.8.x

* OpenCL SDK
  * Intel  : http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
  * AMD    : http://developer.amd.com/zones/openclzone/Pages/default.aspx
  * NVIDIA : http://developer.nvidia.com/opencl

* After the SDK installation, make sure you have set the SDK path correctly
  * For Intel SDK (Below is just an example using default Intel OpenCL SDK path)
    `export INTELOCLSDKROOT=/opt/intel/opencl/`
  * For AMD SDK (Below is just an example using default AMD APP SDK path)
    `export AMDAPPSDKROOT=/opt/AMDAPP/`
  * For NVIDIA SDK (Below is just an example using default CUDA Path)
    `export CUDA_PATH=/usr/local/cuda/`

ALL WORKLOADS COMPILATION
-------------------------

To build all projects in one compilation, do the following:
(You have to ensure that at least cmake 2.8.x is installed)
```
$ mkdir build
$ cd build
$ cmake ../
$ make
````

After the above commands, all binaries will then be installed
to bin/ folder

### Building in Windows

Building in Windows with CMake needs the MS Visual Studio cmake files generator (and also requires Visual Studio to be installed). 

```
$ cmake -G XX
```

**XX** is MSVC's version. (xx64 is for 64-bit build)
Which may be:
* 6
* 7
* 7 .NET 2003
* 8 2005
* 8 2005 Win64 
* 9 2008
* 9 2008 IA64
* 9 2008 Win64 
* 10
* 10 IA64
* 10 Win64
* 11 Win64

The above command will create a Visual Studio solution file.
It can also be done by using `nmake`, for example for building in a Cygwin environment.

```
$ cmake -G NMake\ Makefiles
```

OTHERS
------

Please refer to each project's README.md for more information

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

# Building in Windows

Building in Windows with CMake needs the MS Visual Studio cmake files generator (and also requires Visual Studio to be installed). 

For 32-bit build, bemap can be compiled using the following commands.
```
$ cmake -G Visual\ Studio\ XX
```

For 64-bit build,
```
$ cmake -G Visual\ Studio\ XX\ Win64
```
**XX** is MSVC's version. 
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

or by using `nmake`

```
$ cmake -G NMake\ Makefiles
```

The above command will create a Visual Studio solution file.

REQUIREMENTS
------------
* CMake 2.8.x

* OpenCL SDK
  * Intel  : http://software.intel.com/en-us/articles/vcsource-tools-opencl-sdk/
  * AMD    : http://developer.amd.com/zones/openclzone/Pages/default.aspx
  * NVIDIA : http://developer.nvidia.com/opencl

OTHERS
------

Please refer to each project's README.md for more information

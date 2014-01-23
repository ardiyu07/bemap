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

    $ mkdir build
    $ cd build
    $ cmake ../
    $ make

After the above commands, all binaries will then be installed
to bin/ folder

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

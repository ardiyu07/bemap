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
- How to tune/optimize the auto-parallelizer's compiler
- Whether the appropriate optimization procedure is provided within the compiler

This project is derived from [sourceForge site of bemap](http://sourceforge.net/projects/bemap/).

PACKAGES
--------

This package includes:
- Gaussian Blur
- Black Scholes
- Linear Search
- Grayscale
- SIFT
- Runlength
- Back Projection
- Monte Carlo

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
CMake 2.8.x

OTHERS
------

Please refer to each project's README for more information

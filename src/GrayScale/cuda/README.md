GrayScale, CUDA Implementation
==============================

* Read LICENSE first

* Prerequisites
  * CMake 2.8.x
  * CUDA Toolkit

* Compilation (all projects) in build/
```
$ cd /path/to/bemap
$ cd build/
$ cmake ../
$ make
```

* Compilation (only this project) in TestRun
```
$ cd /path/to/BlackScholes/cuda/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./GrayScale_cuda -h
./GrayScale_cuda [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--prep-time] [--comp-result]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --prep-time                Show initialization, memory preparation and copyback time
 --comp-result              Compare native and OpenCL results

 * Examples *
./GrayScale_cuda [OPTS...] -v -w 256 test_data.ppm
./GrayScale_cuda [OPTS...] --output=test_output.ppm test_data.ppm
```
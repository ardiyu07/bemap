GaussianFilter, CUDA Implementation
===================================

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

** Usage:
```
./GaussianFilter_cuda -h
./GaussianFilter_cuda [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--kernel|-k NUMBER] [--prep-time] [--comp-result]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --kernel=KERNEL            Kernel mode (0, 1, 2) -- default = 0
                                      [0] Scalar
                                      [1] Scalar Fast no Shared Memory (Using convolution separable matrix)
                                      [2] Scalar Fast with Shared Memory (Using convolution separable matrix)
 --prep-time                Show initialization, memory preparation and copyback time
 --comp-result              Compare native and OpenCL results

 * Examples *
./GaussianFilter_cuda [OPTS...] -v -w 256 test_data.pgm
./GaussianFilter_cuda [OPTS...] --output=test_output.ppm test_data.ppm
```
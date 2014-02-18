BackProjection, CUDA Implementation
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

* Usage:
```
./BackProjection_cuda -h
./BackProjection_cuda [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--kernel|-k NUMBER] [--prep-time]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --rows=NUMBER              Number of rows in the data array -- default = 1024
 --columns=NUMBER      		Number of columns in the data array -- default = 1024
 --output=NAME              Write results to this file
 --prep-time                Show initialization, memory preparation and copyback time

 * Examples *
./BackProjection_cuda [OPTS...] -r 512 -c 512 -v
```

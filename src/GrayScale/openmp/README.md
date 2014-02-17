GrayScale, OpenMP Implementation
================================

* Read LICENSE first
* Input data is provided in the common/ folder 
* Compilation (all projects) in build/ 
```
$ cd /path/to/bemap
$ cd build/
$ cmake ../
$ make
```

* Compilation (only this project) in TestRun
```
$ cd /path/to/GaussianFilter/openmp/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./GrayScale_omp -h
./GrayScale_omp [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--prep-time]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --prep-time                Show initialization, memory preparation and copyback time

 * Examples *
./GrayScale_omp [OPTS...] -v test_data.ppm
./GrayScale_omp [OPTS...] --output=test_output.ppm test_data.ppm
```
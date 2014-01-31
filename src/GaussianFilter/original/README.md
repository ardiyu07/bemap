GaussianFilter, Reference Single Thread C++
===========================================

* Read LICENSE.txt first
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
$ cd /path/to/GaussianFilter/original/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./GaussianFilter_ref -h
./GaussianFilter_ref [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--kernel|-k NUMBER] [--prep-time]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --kernel=KERNEL            Kernel mode (0, 1, [2, 3]) -- default = 0
                                      [0] Slow (Using traditional four nested loops)
                                      [1] Fast (Using convolution separable method)
 --prep-time                Show initialization, memory preparation and copyback time

 * Examples *
./GaussianFilter_ref [OPTS...] -v test_data.pgm
./GaussianFilter_ref [OPTS...] --output=test_output.ppm test_data.ppm
```

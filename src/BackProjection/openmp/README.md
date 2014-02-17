BackProjection, CUDA Implementation
===================================

* Read LICENSE first
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
./BackProjection_omp.exe [--verbose|-v] [--help|-h]
     [--kernel|-k NUMBER] [--rows|-r NUMBER] [--columns|-c NUMBER]
     [--prep-time]

* Options *
 --verbose             Be verbose
 --help                Print this message
 --rows=NUMBER         Number of rows in the data array -- default = 1024
 --columns=NUMBER      Number of columns in the data array -- default = 1024
 --prep-time           Show initialization, memory preparation and copy_back time

 * Examples *
./BackProjection_omp.exe [OPTS...] -v -r 512 -c 512
./BackProjection_omp.exe [OPTS...] -v --workitems=128
```
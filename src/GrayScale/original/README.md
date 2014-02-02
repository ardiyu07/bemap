GrayScale, Reference Single Thread C++
=========================================

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
$ cd /path/to/GrayScale/original/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./GrayScale_ref -h
./GrayScale_ref [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--prep-time]
     FILENAME

* Options *
 --verbose                  Be verbose
 --help                     Print this message
 --output=NAME              Write results to this file
 --prep-time                Show initialization, memory preparation and copyback time

 * Examples *
./GrayScale_ref [OPTS...] -v test_data.ppm
./GrayScale_ref [OPTS...] --output=test_output.ppm test_data.ppm
```

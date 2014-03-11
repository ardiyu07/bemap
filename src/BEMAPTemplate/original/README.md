BEMAPTemplate, Reference Single Thread C++
=========================================

* Read LICENSE first

* Prerequisites
CMake - Cross Platform Make
- http://www.cmake.org/
 
* Compilation (all projects) in build/

```
$ cd /path/to/bemap
$ cd build/
$ cmake ../
$ make
```

* Compilation (only this project) in TestRun

```
$ cd /path/to/BEMAPTemplate/original/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./BEMAPTemplate_ref [--verbose|-v] [--help|-h] [--output|-o FILENAME]
     [--alpha|-A NUMBER] [--beta|-B NUMBER] [--partnum|-N INT]
     [--prep-time]

* Options *
 --verbose             Be verbose
 --help                Print this message
 --output=NAME         Write to this file
 --alpha=NUMBER        Alpha for dummy calculation -- default - 0.02
 --beta=NUMBER         Beta for dummy calculation -- default = 0.30
 --num=INT             Number of elements -- default = 1M Element
 --prep-time           Show initialization, memory preparation and copyback time

 * Examples *
./BEMAPTemplate_ref [OPTS...] -v
./BEMAPTemplate_ref [OPTS...] -v -A 1.50
```
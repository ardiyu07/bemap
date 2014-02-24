MonteCarlo, CUDA Implementation
===============================

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
./MonteCarlo_cuda -h
./MonteCarlo_cuda [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--pathnum|-P NUMBER]
     [--optnum|-O NUMBER] [--riskfree|-R NUMBER] [--volatility|-V NUMBER]
     [--prep-time] [--comp-result]

* Options *
 --verbose             Be verbose
 --help                Print this message
 --output=NAME         Write to this file
 --pathnum=NUMBER      Number of paths -- default = 8 * 1024 * 1024
 --optnum=NUMBER       Number of elements in the data array -- default = 128
 --riskfree=NUMBER     The annualized risk-free interest rate, continuously compounded -- default = 0.02
 --volatility=NUMBER   The volatility of stock's returns -- default = 0.30
 --prep-time           Show initialization, memory preparation and copyback time
 --comp-result         Compare GPU and CPU results

 * Examples *
./MonteCarlo_cuda [OPTS...] -v
./MonteCarlo_cuda [OPTS...] -v -O 4194304
```

MonteCarlo, Reference Single Thread C++
=======================================

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
$ cd /path/to/GrayScale/original/
$ cd TestRun
$ cmake ../
$ make
```

* Usage:
```
./MonteCarlo_ref -h
./MonteCarlo_ref [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--pathnum|-P NUMBER]
     [--optnum|-O NUMBER] [--riskfree|-R NUMBER] [--volatility|-V NUMBER]
     [--prep-time]

* Options *
 --verbose             Be verbose
 --help                Print this message
 --output=NAME         Write to this file
 --pathnum=NUMBER      Number of paths -- default = 8 * 1024 * 1024
 --optnum=NUMBER       Number of elements in the data array -- default = 128
 --riskfree=NUMBER     The annualized risk-free interest rate, continuously compounded -- default = 0.02
 --volatility=NUMBER   The volatility of stock's returns -- default = 0.30
 --prep-time           Show initialization, memory preparation and copyback time

 * Examples *
./MonteCarlo_ref [OPTS...] -v
./MonteCarlo_ref [OPTS...] -v -O 4194304
```
#ifndef __BLACKSCHOLES_HPP
#define __BLACKSCHOLES_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MonteCarlo_gold.hpp"

/* user parameters */
extern int verbose;
extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class MonteCarlo {

public:
    MonteCarlo(int _pathNum,
               int _optNum, float _riskFree, float _volatility);

    ~MonteCarlo();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

    /* MonteCarlo variables */
private:

    float *call;
    float *random;
    float *confidence;
    float *stockPrice;
    float *optionStrike;
    float *optionYears;

    int pathNum;
    int optNum;
    float riskFree;
    float volatility;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_gr, t_bm, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

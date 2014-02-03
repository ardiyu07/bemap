#ifndef __BLACKSCHOLES_HPP
#define __BLACKSCHOLES_HPP

#pragma once

/* include utilities */
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>
#include <cuda_util.hpp>

#include "BlackScholes_gold.hpp"
#include "common.hpp"

/* CUDA variables */
extern int nThreads;
extern int nBlocks;

/* user parameters */
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class BlackScholesCUDA {
public:

    BlackScholesCUDA(int _optNum, float _riskFree, float _volatility);

    ~BlackScholesCUDA();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* host variables */
private:

    float *call;
    float *put;
    float *stockPrice;
    float *optionStrike;
    float *optionYears;

    int optNum;
    float riskFree;
    float volatility;

    /* cuda variables */
private:

    cudaEvent_t start, end;
    float elapsed;

    float *d_call;
    float *d_put;
    float *d_stockPrice;
    float *d_optionStrike;
    float *d_optionYears;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

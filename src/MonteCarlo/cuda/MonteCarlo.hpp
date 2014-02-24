#ifndef __MONTECARLO_HPP
#define __MONTECARLO_HPP

#pragma once

/* include utilities */
#include <cuda_util.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MonteCarlo_gold.hpp"

/*************************/
/* @brief CUDA variables */
/*************************/
extern int nThreads;
extern int nBlocks;

/* user parameters */
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;
extern int choose;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class MonteCarloCUDA {

public:
    MonteCarloCUDA(int _pathNum,
                   int _optNum, float _riskFree, float _volatility);

    ~MonteCarloCUDA();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* MonteCarloCUDA variables */
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

    /* cl variables */
private:

    cudaEvent_t start, end;
    float elapsed;

    float *d_call;
    float *d_random;
    float *d_confidence;
    float *d_stockPrice;
    float *d_optionStrike;
    float *d_optionYears;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_gr, t_bm, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

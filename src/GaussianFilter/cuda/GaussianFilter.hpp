#ifndef __GAUSSIANFILTER_HPP
#define __GAUSSIANFILTER_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>
#include <cuda_util.hpp>

#include "GaussianFilter_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SCALAR_FAST,
    SCALAR_FAST_SHM,
} kernelVersion;

/* Kernel names */
static const char *kernelStr[] = {
    "gaussian_scalar",
    "gaussian_scalar_fast_no_shm",
    "gaussian_scalar_fast",
};

/* User parameters */
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;
extern int isPGM;
extern int isFast;
extern int choose;

extern std::string realName;
extern std::string inputFile;

extern kernelVersion kernelVer;

/* CUDA variables */
extern int nThreads;
extern int nBlocks;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class GaussianFilterCUDA {
public:

    GaussianFilterCUDA(imgStream _inp);

    ~GaussianFilterCUDA();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* gaussian filter variables */
private:

    imgStream inp;
    imgStream out;
    int nElem;
    int width;
    int height;

    /* cl variables */
private:
    cudaEvent_t start, end;
    float elapsed;

    pixel_uc *d_inp_r;
    pixel_uc *d_inp_g;
    pixel_uc *d_inp_b;
    pixel_uc *d_out_r;
    pixel_uc *d_out_g;
    pixel_uc *d_out_b;

    pixel_uc *d_buf_r;
    pixel_uc *d_buf_g;
    pixel_uc *d_buf_b;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

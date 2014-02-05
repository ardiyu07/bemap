#ifndef __GRAYSCALE_HPP
#define __GRAYSCALE_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>
#include <cuda_util.hpp>

#include "GrayScale_gold.hpp"

/* User parameters */
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;
extern int isPGM;
extern int choose;

extern std::string realName;
extern std::string inputFile;

/* CUDA variables */
extern int nThreads;
extern int nBlocks;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class GrayScaleCUDA {
public:

    GrayScaleCUDA(imgStream _inp);

    ~GrayScaleCUDA();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* grayscale variables */
private:

    imgStream inp;
    imgStream out;
    int nElem;
    int width;
    int height;

    /* cuda variables */
private:
    cudaEvent_t start, end;
    float elapsed;

    pixel_uc *d_inp_r;
    pixel_uc *d_inp_g;
    pixel_uc *d_inp_b;
    pixel_uc *d_out;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

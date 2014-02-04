#ifndef __GAUSSIANFILTER_HPP
#define __GAUSSIANFILTER_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "GaussianFilter_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SLOW,
    FAST
} kernelVersion;

/* Kernel names */
static const char *kernelStr[] = {
    "Traditional Four Nested Loops",
    "Convolution Separable",
};

/* User parameters */
extern int verbose;
extern int showPrepTime;
extern int isPGM;

extern std::string realName;
extern std::string inputFile;

extern kernelVersion kernelVer;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class GaussianFilter {
public:

    GaussianFilter(imgStream _inp);
    ~GaussianFilter();

public:

    void init();
    void prep_memory();
    void execute();
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

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

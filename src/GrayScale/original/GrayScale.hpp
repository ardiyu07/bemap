#ifndef __GRAYSCALE_HPP
#define __GRAYSCALE_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "GrayScale_gold.hpp"

/* User parameters */
extern int verbose;
extern int showPrepTime;
extern int isPGM;

extern std::string realName;
extern std::string inputFile;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class GrayScale {
public:

    GrayScale(imgStream _inp);

    ~GrayScale();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
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

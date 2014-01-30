#ifndef __GRAYSCALE_HPP
#define __GRAYSCALE_HPP

#pragma once

/* include utilities */
#include <cl_util_drvapi.hpp>
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "GrayScale_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SIMD,
} kernelVersion;

/* Kernel names */
static const char *kernelStr[] = {
    "grayscale_scalar",
    "grayscale_simd"
};

/* platform and device param */
typedef enum platforms {
    AMD,
    NVIDIA,
    INTEL
} platforms;

static const char *platform[] = {
    "Advanced Micro Devices, Inc.",
    "NVIDIA Corporation",
    "Intel(R) Corporation"
};

/**************************/
/* @brief User parameters */
/**************************/
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;
extern int isPGM;
extern int choose;

extern std::string realName;
extern std::string inputFile;

/************************/
/* @brief CL variables  */
/************************/
extern std::string sourceName;
extern std::string compileOptions;

/* Constants */
extern size_t globalWorkSize[3];
extern size_t localWorkSize[3];
extern size_t nLocals;
extern size_t nGlobals;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class GrayScaleCL {
public:

    GrayScaleCL(imgStream _inp,
                std::string _platformId,
                cl_device_type _deviceType, kernelVersion _kernelVer);

    ~GrayScaleCL();

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
    cl_int errNum;
    cl_event perfEvent;
    cl_ulong start;
    cl_ulong end;

    std::string platformId;
    cl_device_type deviceType;
    kernelVersion kernelVer;

    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;

    cl_kernel kernel;

    cl_mem d_inp_r;
    cl_mem d_inp_g;
    cl_mem d_inp_b;
    cl_mem d_out;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

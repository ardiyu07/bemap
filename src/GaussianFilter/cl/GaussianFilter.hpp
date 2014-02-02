#ifndef __GAUSSIANFILTER_HPP
#define __GAUSSIANFILTER_HPP

#pragma once

/* include utilities */
#include <cl_util_drvapi.hpp>
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "GaussianFilter_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SIMD,
    SCALAR_FAST_SHM,
    SCALAR_FAST,
    SIMD_FAST,
    STSD,
    STAD,
    SCALAR_FAST_C_SHM,
    SCALAR_FAST_C,
    SIMD_FAST_C
} kernelVersion;

/* Kernel names */
/* STSD = Single Thread Single Data */
/* STAD = Single Thread All Data */
static const char *kernelStr[] = {
    "gaussian_scalar",
    "gaussian_simd",

    /* Conv Separable row */
    "gaussian_scalar_fast",
    "gaussian_scalar_fast_no_shm",
    "gaussian_simd_fast",

    /* KERNELS FOR DEBUGGING PURPOSE */
    "gaussian_stsd",
    "gaussian_stad",

    /* Conv Separable column */
    "gaussian_scalar_fast_column",
    "gaussian_scalar_fast_column_no_shm",
    "gaussian_simd_fast_column"
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

/* CL variables  */
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

class GaussianFilterCL {
public:

    GaussianFilterCL(imgStream _inp,
                     std::string _platformId,
                     cl_device_type _deviceType, kernelVersion _kernelVer);

    ~GaussianFilterCL();

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

    cl_kernel kernel_r;
    cl_kernel kernel_g;
    cl_kernel kernel_b;

    /* For convolution separable kernel */
    cl_kernel kernel_col_r;
    cl_kernel kernel_col_g;
    cl_kernel kernel_col_b;

    cl_mem memobj_inp_r;
    cl_mem memobj_inp_g;
    cl_mem memobj_inp_b;
    cl_mem memobj_out_r;
    cl_mem memobj_out_g;
    cl_mem memobj_out_b;

    cl_mem memobj_buf_r;
    cl_mem memobj_buf_g;
    cl_mem memobj_buf_b;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

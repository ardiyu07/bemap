#ifndef __MONTECARLO_HPP
#define __MONTECARLO_HPP

#pragma once

/* include utilities */
#include <cl_util_drvapi.hpp>
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MonteCarlo_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SIMD,
    SHM
} kernelVersion;

/**
 * TODO:
 * Make GenRand and BoxMuller SIMD
 */


/* Kernel names */
/* static const char* kernelStrGR[] = { */
/*   "genrand_scalar", */
/*   "genrand_simd" */
/* }; */

static const char *kernelStrBM[] = {
    "boxmuller",
    "boxmuller",
    "boxmuller"
};

static const char *kernelStrMC[] = {
    "montecarlo_scalar",
    "montecarlo_simd",
    "montecarlo_scalar_shm"
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

/* user parameters */
extern int verbose;
extern int showPrepTime;
extern int showDevInfo;
extern int compResult;
extern int choose;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class MonteCarloCL {

public:
    MonteCarloCL(int _pathNum,
                 int _optNum,
                 float _riskFree,
                 float _volatility,
                 int _width,
                 int _height,
                 std::string _platformId,
                 cl_device_type _deviceType, kernelVersion _kernelVer);

    ~MonteCarloCL();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* MonteCarloCL variables */
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

    int width;                  /// Data elements per thread (SIMD mode)
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

    unsigned int deviceId;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel kernel_gr;        /// GenRand
    cl_kernel kernel_bm;        /// BoxMuller
    cl_kernel kernel_mc;        /// MonteCarlo
    cl_program program;

    cl_mem memobj_call;
    cl_mem memobj_random;
    cl_mem memobj_confidence;
    cl_mem memobj_stockPrice;
    cl_mem memobj_optionStrike;
    cl_mem memobj_optionYears;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_gr, t_bm, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

#ifndef __BLACKSCHOLES_HPP
#define __BLACKSCHOLES_HPP

/* include utilities */
#include <cl_util_drvapi.hpp>
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "BlackScholes_gold.hpp"

/* Constants */
#define coef1     0.31938153f
#define coef2    -0.356563782f
#define coef3     1.781477937f
#define coef4    -1.821255978f
#define coef5     1.330274429f
#define RSQRT2PI  0.3989422804f

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SIMD,
    STSD,
    STAD
} kernelVersion;

/* Kernel names */
/* STSD = Single Thread Single Data */
/* STAD = Single Thread All Data */
static const char *kernelStr[] = {
    "black_scholes_scalar",
    "black_scholes_simd",

    /* KERNEL FOR DEBUGGING PURPOSE */
    "black_scholes_stsd",
    "black_scholes_stad"
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

/* CL variables  */
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

class BlackScholesCL {
public:

    BlackScholesCL(int _optNum,
                   float _riskFree,
                   float _volatility,
                   int _width,
                   int _height,
                   std::string _platformId,
                   cl_device_type _deviceType, 
                   kernelVersion _kernelVer);

    ~BlackScholesCL();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void *param);
    void clean_mem();
    void finish();

    /* blackscholes variables */
private:

    float *call;
    float *put;
    float *stockPrice;
    float *optionStrike;
    float *optionYears;

    int optNum;
    float riskFree;
    float volatility;

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

    unsigned int deviceId;
    cl_device_id device;
    cl_context context;
    cl_command_queue commandQueue;
    cl_kernel kernel;
    cl_program program;

    cl_mem memobj_call;
    cl_mem memobj_put;
    cl_mem memobj_stockPrice;
    cl_mem memobj_optionStrike;
    cl_mem memobj_optionYears;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

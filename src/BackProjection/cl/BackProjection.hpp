#ifndef __BACKPROJECTION_HPP
#define __BACKPROJECTION_HPP

/* include utilities */
#include <cl_util_drvapi.hpp>
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "BackProjection_gold.hpp"

/* Kernels */
typedef enum kernelVersion {
    SCALAR,
    SIMD
} kernelVersion;

/* Kernel names */
static const char* kernelStr[] = {
    "BackProjection_scalar",
    "BackProjection_simd",
};

/* platform and device param */
typedef enum platforms {
    AMD,
    NVIDIA,
    INTEL
} platforms;

static const char* platform[] = {
    "Advanced Micro Devices, Inc.",
    "NVIDIA Corporation",
    "Intel(R) Corporation"
};

/**************************/
/* @brief User parameters */
/**************************/
extern int  verbose;
extern int  showPrepTime;
extern int  showDevInfo;
extern int  choose;
extern int  naive;

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

class BackProjectionCL
{
public:
  
    BackProjectionCL(int _rows,
                     int _columns,
                     std::string _platformId,
                     cl_device_type _deviceType,
                     kernelVersion _kernelVer);

    ~BackProjectionCL();

public:

    void init();
    void prep_memory();
    void execute();
    void copyback();
    void compare_to_cpu();
    void output(void* param);  
    void clean_mem();
    void finish();

private:
    /* create and release problem */
    void create_prob();
    void release_prob();
        
    /* cl variables */
private:
    cl_int errNum;

    unsigned int deviceId;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue command_queue;
    cl_command_queue command_queue2;
    cl_program       program;

    cl_kernel k_BackProjection;
    cl_kernel k_findmax;
    cl_kernel k_decreaseproj;
    cl_kernel k_init;
 
    cl_mem memobj_ori, memobj_ori_id;
    cl_mem memobj_image, memobj_image2, memobj_guess, memobj_zero;
    cl_mem memobj_rproj, memobj_cproj, memobj_uproj, memobj_dproj;
    cl_mem memobj_rband, memobj_cband, memobj_uband, memobj_dband;
    cl_mem memobj_rscore, memobj_cscore, memobj_uscore, memobj_dscore;
    cl_mem memobj_index, memobj_index2;
    cl_mem memobj_max, memobj_maxId;

    size_t init_nglobals;
    size_t init_nlocals;
    size_t BackProjection_nglobals[2];
    size_t BackProjection_nlocals[2];
    size_t findmax_nglobals;
    size_t findmax_nlocals;
    size_t decreaseproj_nglobals;
    size_t decreaseproj_nlocals;

    std::string    platformId;
    cl_device_type deviceType;
    kernelVersion  kernelVer;

    /* tmp? variable */
    unsigned char *guess;
    int *rproj, *cproj, *uproj, *dproj, *rband, *cband, *uband, *dband;
    F_TYPE *rscore, *cscore, *uscore, *dscore;
    unsigned char *input;
    F_TYPE *image;
    int *maxId_p;

    int r, c;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

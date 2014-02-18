#ifndef __BACKPROJECTION_HPP
#define __BACKPROJECTION_HPP

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>
#include <cuda_util.hpp>

#include "BackProjection_gold.hpp"

extern int nThreads;
extern int nBlocks;

extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class BackProjectionCUDA {
public:

    BackProjectionCUDA(int _rows, int _columns);
    ~BackProjectionCUDA();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

private:
    /* create and release problem */
    void create_prob();
    void release_prob();

private:
    /* testrun */
    void testrun();

private:

    /* BackProjection variables */
    unsigned char *guess;
    int *rproj, *cproj, *uproj, *dproj, *rband, *cband, *uband, *dband;
    F_TYPE *rscore, *cscore, *uscore, *dscore;
    unsigned char *input;
    F_TYPE *image;
    int *maxId_p;

    int r, c;

    /* cuda variables */
private:

	cudaStream_t *streams;
    cudaEvent_t start, end;
    float elapsed;

	int findmax_nglobals;
	int findmax_nlocals;
	dim3 dBlocks;
	dim3 dThreads;
	int init_nBlocks;
	int init_nThreads;	

    float* d_ori;
	float* d_ori_id;
    float* d_image;
	float* d_image2;
	unsigned char* d_guess;
	float* d_zero;
    int* d_rproj;
	int* d_cproj;
	int* d_uproj;
	int* d_dproj;
    int* d_rband;
	int* d_cband;
	int* d_uband;
	int* d_dband;
    float* d_rscore;
	float* d_cscore;
	float* d_uscore;
	float* d_dscore;
    int* d_index;
	int* d_index2;
    int* d_max;
	int* d_maxId;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_cpy, t_clean;
    StopWatch t_all;

};

#endif

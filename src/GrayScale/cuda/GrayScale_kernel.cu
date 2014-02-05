#include "GrayScale.hpp"

#define cudaEnumIdx (threadIdx.x + blockIdx.x * blockDim.x)
#define cudaEnumIdy (threadIdx.y + blockIdx.y * blockDim.y)

typedef uchar pixel_uc;
typedef float number;

#define R_RATE 0.298912f
#define G_RATE 0.586611f
#define B_RATE 0.114478f

__global__
void grayscale_scalar (pixel_uc* out, 
                       pixel_uc* inp_r,
                       pixel_uc* inp_g, 
                       pixel_uc* inp_b,
                       int img_size,
                       int blockOffset)
{
    int tid = cudaEnumIdx + blockOffset;
    int idx;
    number rr,gg,bb;
    number v;

    if (tid < img_size) {
        rr = (number)inp_r[tid];
        gg = (number)inp_g[tid];
        bb = (number)inp_b[tid];
        v = R_RATE*rr+G_RATE*gg+B_RATE*bb;

        out[tid] = (pixel_uc)v;
    }
}

void GrayScaleCUDA::execute()
{
    cudaEventCreate(&start); CUDA_CHECK_ERROR();
    cudaEventCreate(&end); CUDA_CHECK_ERROR();

    nBlocks = (nElem + nThreads - 1) / nThreads;
    int itr = (nBlocks - 1) / MAX_NBLOCKS + 1;
    nBlocks = min(nBlocks, MAX_NBLOCKS);

    cudaEventRecord(start, 0);
    for (int i = 0; i < itr; ++i)
        grayscale_scalar<<<nBlocks, nThreads>>>
            (d_out, d_inp_r, d_inp_g, d_inp_b, width * height, i * nThreads * nBlocks);
    cudaThreadSynchronize(); CUDA_CHECK_ERROR();
    cudaEventRecord(end, 0); CUDA_CHECK_ERROR();
    cudaEventSynchronize(start); CUDA_CHECK_ERROR();
    cudaEventSynchronize(end); CUDA_CHECK_ERROR();
    cudaEventElapsedTime(&elapsed, start, end); CUDA_CHECK_ERROR();

    t_kernel += elapsed;
}


#include "GaussianFilter.hpp"
#include "common.hpp"

#define cudaEnumIdx threadIdx.x + blockIdx.x * blockDim.x
#define cudaEnumIdy threadIdx.y + blockIdx.y * blockDim.y

__device__ __constant__ float filter[121] =  // arbitrary circularly symmetric filter -> AT * A
	{
		1.6287563414420342e-8f, 2.712122801646884e-7f, 0.00000241728874041747f, 0.000011532239604305994f, 0.0000294486054814111f, 0.00004025147128665112f, 0.0000294486054814111f, 0.000011532239604305994f, 0.00000241728874041747f, 2.712122801646884e-7f, 1.6287563414420342e-8f, 
		2.712122801646884e-7f, 0.00000451608991723132f, 0.00004025147128665112f, 0.00019202902969023768f, 0.0004903633058590361f, 0.0006702471714075378f, 0.0004903633058590361f, 0.00019202902969023768f, 0.00004025147128665112f, 0.00000451608991723132f, 2.712122801646884e-7f, 
		0.00000241728874041747f, 0.00004025147128665112f, 0.000358757458428415f, 0.0017115361112027578f, 0.004370560570661311f, 0.005973848012178047f, 0.004370560570661311f, 0.0017115361112027578f, 0.000358757458428415f, 0.00004025147128665112f, 0.00000241728874041747f, 
		0.000011532239604305994f, 0.00019202902969023768f, 0.0017115361112027578f, 0.008165282117850574f, 0.02085077833825263f, 0.02849963493572827f, 0.02085077833825263f, 0.008165282117850574f, 0.0017115361112027578f, 0.00019202902969023768f, 0.000011532239604305994f, 
		0.0000294486054814111f, 0.0004903633058590361f, 0.004370560570661311f, 0.02085077833825263f, 0.05324432775696791f, 0.07277636733051646f, 0.05324432775696791f, 0.02085077833825263f, 0.004370560570661311f, 0.0004903633058590361f, 0.0000294486054814111f, 
		0.00004025147128665112f, 0.0006702471714075378f, 0.005973848012178047f, 0.02849963493572827f, 0.07277636733051646f, 0.09947350008815053f, 0.07277636733051646f, 0.02849963493572827f, 0.005973848012178047f, 0.0006702471714075378f, 0.00004025147128665112f, 
		0.0000294486054814111f, 0.0004903633058590361f, 0.004370560570661311f, 0.02085077833825263f, 0.05324432775696791f, 0.07277636733051646f, 0.05324432775696791f, 0.02085077833825263f, 0.004370560570661311f, 0.0004903633058590361f, 0.0000294486054814111f, 
		0.000011532239604305994f, 0.00019202902969023768f, 0.0017115361112027578f, 0.008165282117850574f, 0.02085077833825263f, 0.02849963493572827f, 0.02085077833825263f, 0.008165282117850574f, 0.0017115361112027578f, 0.00019202902969023768f, 0.000011532239604305994f, 
		0.00000241728874041747f, 0.00004025147128665112f, 0.000358757458428415f, 0.0017115361112027578f, 0.004370560570661311f, 0.005973848012178047f, 0.004370560570661311f, 0.0017115361112027578f, 0.000358757458428415f, 0.00004025147128665112f, 0.00000241728874041747f, 
		2.712122801646884e-7f, 0.00000451608991723132f, 0.00004025147128665112f, 0.00019202902969023768f, 0.0004903633058590361f, 0.0006702471714075378f, 0.0004903633058590361f, 0.00019202902969023768f, 0.00004025147128665112f, 0.00000451608991723132f, 2.712122801646884e-7f, 
		1.6287563414420342e-8f, 2.712122801646884e-7f, 0.00000241728874041747f, 0.000011532239604305994f, 0.0000294486054814111f, 0.00004025147128665112f, 0.0000294486054814111f, 0.000011532239604305994f, 0.00000241728874041747f, 2.712122801646884e-7f, 1.6287563414420342e-8f
	};

__device__ __constant__ float filter_src[11] = 
	{ 
		0.0001276227386260785f, 0.0021251093894741795f, 0.018940893812817154f, 0.09036195060892929f, 0.23074732448496107f, 0.31539419793038453f, 0.23074732448496107f, 0.09036195060892929f, 0.018940893812817154f, 0.0021251093894741795f, 0.0001276227386260785f
	};

__device__ const int FILTER_SIDE_SIZE = 11;
__device__ const int FILTER_SRC_SIZE = 11;

__global__ void gaussian_scalar (pixel_uc *src, pixel_uc *dst, int w, int h)
{  
    const int fil_half  = (FILTER_SIDE_SIZE - 1) / 2;
    int tid       = cudaEnumIdx ;
    int i         = tid / w;
    int j         = tid % w;
    float img_tmp = 0.0f;
    int ii, jj;

    if (tid < w*h) {
		int fil_idx   = 0;
		for (ii = i - fil_half; ii <= i + fil_half; ii++) {
			for (jj = j - fil_half; jj <= j + fil_half; jj++) {	
				int iii = ii;
				int jjj = jj;
				if (ii < 0) iii = 0;
				else if (ii >= h) iii = h;
				else iii = ii;
				if (jj < 0) jjj = 0;
				else if (jj >= w) jjj = w;
				else jjj = jj;
				img_tmp += (float)src[iii * w + jjj] * filter[fil_idx];
				fil_idx++;
			}
		}	  
		img_tmp = MAX(0.0f, MIN(255.0f, img_tmp));
		dst[tid] = (pixel_uc)img_tmp;
    }
}

__global__ 
void gaussian_scalar_fast_no_shm (pixel_uc *src, pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int tid = cudaEnumIdx;
    int i   = tid / w;
    int j   = tid % w;
    int k, fil_idx;
    float v;

    v = 0.0f;
    fil_idx = 0;

    if (i < h && j < w) {
		if ( j >= s && j < w - s) {
			for ( k = j - s; k <= j + s; k++, fil_idx++)
				v += (float)src[i*w+k] * filter_src[fil_idx];
		}
		else if (j < s) { /* filter out of frame */
			for (k = j - s; k < 0; k++, fil_idx++) v += (float)src[i*w] * filter_src[fil_idx]; /* below zero */
			for (k = 0; fil_idx < 2*s+1; k++, fil_idx++) v += (float)src[i*w+k] * filter_src[fil_idx];
		}
		else { /* filter out of frame */
			for (k = j - s; k < w - 1; k++, fil_idx++) v += (float)src[i*w+k] * filter_src[fil_idx];
			for (; fil_idx < 2*s+1; fil_idx++) v += (float)src[i*w+k] * filter_src[fil_idx]; /* more than w */
		}
		v = MAX(0.0f, MIN(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }
}

__global__ 
void gaussian_scalar_fast_column_no_shm(pixel_uc *src, pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int tid = cudaEnumIdx;
    int i   = tid / w;
    int j   = tid % w;
    int k, fil_idx;
    float v;

    v = 0.0f;
    fil_idx = 0;

    if (i < h && j < w) {
		if ( i >= s && i < h - s) {
			for ( k = i - s; k <= i + s; k++, fil_idx++)
				v += src[k*w+j] * filter_src[fil_idx];
		}
		else if (i < s) { /* filter out of frame */
			for (k = i - s; k < 0; k++, fil_idx++) v += src[j] * filter_src[fil_idx]; /* below zero */
			for (k = 0; fil_idx < 2*s+1; k++, fil_idx++) v += src[k*w+j] * filter_src[fil_idx];
		}
		else { /* filter out of frame */
			for (k = i - s; k < h - 1; k++, fil_idx++) v += src[k*w+j] * filter_src[fil_idx];
			for (; fil_idx < 2*s+1; fil_idx++) v += src[k*w+j] * filter_src[fil_idx]; /* more than w */
		}
		v = MAX(0.0f, MIN(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }
}

__global__ 
void gaussian_scalar_fast(pixel_uc *src, pixel_uc *dst, int width, int height)
{
    __shared__ pixel_uc data[ROWS_BLOCKDIM_X + FILTER_SRC_SIZE - 1];

    int s = FILTER_SRC_SIZE >> 1;

    const int CACHE_WIDTH = ROWS_BLOCKDIM_X + FILTER_SRC_SIZE - 1;
    const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2)/ ROWS_BLOCKDIM_X;

    int bcol = blockIdx.x * ROWS_BLOCKDIM_X;
    int col =  bcol + threadIdx.x;
    int index_MIN = blockIdx.y * width;
    int index_MAX = index_MIN + width - 1;
    int src_index = index_MIN + bcol - s + threadIdx.x;
    int cache_index = threadIdx.x;
    float value = 0.0f;

    for(int j = 0; j < CACHE_COUNT; ++j) {
		if(cache_index < CACHE_WIDTH) {
			int fetch_index = src_index < index_MIN? index_MIN : (src_index > index_MAX ? index_MAX : src_index);
			data[cache_index] = src[fetch_index];
			src_index += ROWS_BLOCKDIM_X;
			cache_index += ROWS_BLOCKDIM_X;
		}
    }

    __syncthreads();

    if(col >= width && index_MIN >= height) return;
    for(int i = 0; i < FILTER_SRC_SIZE; ++i) {
		value += (data[threadIdx.x + i]* filter_src[i]);
    }
    value = MAX(0.0f, MIN(255.0f, value));
    dst[index_MIN+col] = (pixel_uc)value;
}

__global__ 
void gaussian_scalar_fast_column(pixel_uc *src, pixel_uc *dst, int width, int height)
{
    __shared__ pixel_uc data[(COLUMNS_BLOCKMEM_Y + FILTER_SRC_SIZE + 16) * COLUMNS_BLOCKDIM_X];

    int s = FILTER_SRC_SIZE >> 1;
    int CACHE_WIDTH = FILTER_SRC_SIZE + COLUMNS_BLOCKMEM_Y - 1;
    int TEMP = CACHE_WIDTH & 0xf;
    int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
    int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;
    int CACHE_COUNT = (CACHE_WIDTH + COLUMNS_BLOCKDIM_Y - 1) / COLUMNS_BLOCKDIM_Y;

    int row_block_first = blockIdx.y * COLUMNS_BLOCKMEM_Y;
    int col = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    int row_first = row_block_first - s;
    int data_index_MAX = (height-1)*width + col;
    int cache_col_start = threadIdx.y;	
    int cache_row_start = threadIdx.x * CACHE_TRUE_WIDTH;
    int cache_index = cache_col_start + cache_row_start;
    int data_index = (row_first + cache_col_start) * width + col;

    if(col < width) {
		for(int i = 0; i < CACHE_COUNT; ++i) {
			if(cache_col_start < CACHE_WIDTH - i * COLUMNS_BLOCKDIM_Y) {
				int fetch_index = data_index < col ? col : (data_index > data_index_MAX? data_index_MAX : data_index);
				data[cache_index + i * COLUMNS_BLOCKDIM_Y] = src[fetch_index];
				data_index += (COLUMNS_BLOCKDIM_Y * width);
			}
		}
    }

    __syncthreads();
	
    if(col >= width) return;

    int row = row_block_first + threadIdx.y;
    int index_start = cache_row_start + threadIdx.y;

    for(int i = 0; i < COLUMNS_RESULT_STEPS; ++i, 
			row += COLUMNS_BLOCKDIM_Y, index_start += COLUMNS_BLOCKDIM_Y) {
		if(row < height) {
			int index_dest = row * width + col;
			float value = 0.0f;

			for(int i = 0; i < FILTER_SRC_SIZE; ++i) {
				value += (data[index_start + i] * filter_src[i]);
			}
			value = MAX(0.0f, MIN(255.0f, value));
			dst[index_dest] = value;
		}
    }
}

/* function pointer */
void (*kernel)(pixel_uc*, pixel_uc*, int, int);
void (*kernel_c)(pixel_uc*, pixel_uc*, int, int);

void GaussianFilterCUDA::execute()
{
    cudaEventCreate(&start); CUDA_CHECK_ERROR();
    cudaEventCreate(&end); CUDA_CHECK_ERROR();

    /* dims */
    dim3 dThreads;
    dim3 dBlocks;

    pixel_uc *d_p_r, *d_p_g, *d_p_b;

    switch (kernelVer) {
    case SCALAR:
		{
			dThreads.x = nThreads;
			dThreads.y = 1;
			dBlocks.x  = (nElem + nThreads - 1) / nThreads;
			dBlocks.y  = 1;
			kernel   = &gaussian_scalar;
			kernel_c = NULL;
			d_p_r    = d_out_r;
			d_p_g    = d_out_g;
			d_p_b    = d_out_b;
		}
		break;
    case SCALAR_FAST:
		{
			dThreads.x = nThreads;
			dThreads.y = 1;
			dBlocks.x  = (nElem + nThreads - 1) / nThreads;
			dBlocks.y  = 1;
			kernel   = &gaussian_scalar_fast_no_shm;
			kernel_c = &gaussian_scalar_fast_column_no_shm;
			d_p_r    = d_buf_r;
			d_p_g    = d_buf_g;
			d_p_b    = d_buf_b;
		}
		break;
    case SCALAR_FAST_SHM:
		{
			dThreads.x = ROWS_BLOCKDIM_X;
			dThreads.y = ROWS_BLOCKDIM_Y;
			dBlocks.x  = (width + ROWS_BLOCKDIM_X - 1) / ROWS_BLOCKDIM_X;
			dBlocks.y  = (height + ROWS_BLOCKDIM_Y - 1) / ROWS_BLOCKDIM_Y;
			kernel   = &gaussian_scalar_fast;
			kernel_c = &gaussian_scalar_fast_column;
			d_p_r    = d_buf_r;
			d_p_g    = d_buf_g;
			d_p_b    = d_buf_b;
		}
		break;
    }

    float tmp = 0.0f;

    /* For BW channeled image */
    cudaEventRecord(start, 0);
    if (isPGM) {
		kernel<<<dBlocks, dThreads>>>(d_inp_r, d_p_r, width, height);
    } 

    /* For RGB Channeled image */
    else {
		kernel<<<dBlocks, dThreads>>>(d_inp_r, d_p_r, width, height);
		kernel<<<dBlocks, dThreads>>>(d_inp_g, d_p_g, width, height);
		kernel<<<dBlocks, dThreads>>>(d_inp_b, d_p_b, width, height);
    }

    cudaThreadSynchronize(); CUDA_CHECK_ERROR();
    cudaEventRecord(end, 0); CUDA_CHECK_ERROR();
    cudaEventSynchronize(start); CUDA_CHECK_ERROR();
    cudaEventSynchronize(end); CUDA_CHECK_ERROR();
    cudaEventElapsedTime(&elapsed, start, end); CUDA_CHECK_ERROR();
  
    tmp += elapsed;

    /* We have anoter step for fast kernels -- Column step */
    if (isFast) {

		if (kernelVer == SCALAR_FAST_SHM) {
			dThreads.x = COLUMNS_BLOCKDIM_X;
			dThreads.y = COLUMNS_BLOCKDIM_Y;
			dBlocks.x  = (width + COLUMNS_BLOCKDIM_X - 1)/ COLUMNS_BLOCKDIM_X;
			dBlocks.y  = (height + COLUMNS_BLOCKMEM_Y - 1)/COLUMNS_BLOCKMEM_Y;
		}

		cudaEventRecord(start, 0);

		/* For BW channeled image */
		if (isPGM) {
			kernel_c<<<dBlocks, dThreads>>>(d_buf_r, d_out_r, width, height);
		}     

		/* For RGB Channeled image */
		else {
			kernel_c<<<dBlocks, dThreads>>>(d_buf_r, d_out_r, width, height);
			kernel_c<<<dBlocks, dThreads>>>(d_buf_g, d_out_g, width, height);
			kernel_c<<<dBlocks, dThreads>>>(d_buf_b, d_out_b, width, height);
		}
		cudaThreadSynchronize(); CUDA_CHECK_ERROR();
		cudaEventRecord(end, 0); CUDA_CHECK_ERROR();
		cudaEventSynchronize(start); CUDA_CHECK_ERROR();
		cudaEventSynchronize(end); CUDA_CHECK_ERROR();
		cudaEventElapsedTime(&elapsed, start, end); CUDA_CHECK_ERROR();
  
		tmp += elapsed;
    }
    t_kernel += tmp;
}


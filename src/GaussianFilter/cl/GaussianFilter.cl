/* common types */
#include "GaussianFilter_common.hpp"

/* Enable uchar byte addressing */
#ifdef cl_khr_byte_addressable_store
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
#endif

/* enumerators */
#define clGlobIdx get_global_id(0)
#define clGlobIdy get_global_id(1)
#define clLocIdx  get_local_id(0)
#define clLocIdy  get_local_id(1)
#define clGrpIdx  get_group_id(0)
#define clGrpIdy  get_group_id(1)

/* GAUSSIAN FILTER */
/* Sigma = 2 */
__constant float filter[121] =  // arbitrary circularly symmetric filter -> AT * A
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

__constant float filter_src[11] = 
	{ 
		0.0001276227386260785f, 0.0021251093894741795f, 0.018940893812817154f, 0.09036195060892929f, 0.23074732448496107f, 0.31539419793038453f, 0.23074732448496107f, 0.09036195060892929f, 0.018940893812817154f, 0.0021251093894741795f, 0.0001276227386260785f
	};

#define FILTER_SIDE_SIZE 11
#define FILTER_SRC_SIZE  11

__kernel 
void gaussian_scalar (__global pixel_uc *src, __global pixel_uc *dst, int w, int h)
{  
    const int fil_half  = (FILTER_SIDE_SIZE - 1) / 2;
    int tid       = clGlobIdx ;
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
				else if (ii >= h) iii = h - 1;
				else iii = ii;
				if (jj < 0) jjj = 0;
				else if (jj >= w) jjj = w - 1;
				else jjj = jj;
				img_tmp += (float)src[iii * w + jjj] * filter[fil_idx];
				fil_idx++;
			}
		}	  
		img_tmp = max(0.0f, min(255.0f, img_tmp));
		dst[tid] = (pixel_uc)img_tmp;
    }
}

__kernel
void gaussian_scalar_fast (__global pixel_uc* src, __global pixel_uc* dst, int width, int height)
{
    __local pixel_uc data[ROWS_BLOCKDIM_X + FILTER_SRC_SIZE - 1];

    int s = FILTER_SRC_SIZE >> 1;

    const int CACHE_WIDTH = ROWS_BLOCKDIM_X + FILTER_SRC_SIZE - 1;
    const int CACHE_COUNT = 2 + (CACHE_WIDTH - 2)/ ROWS_BLOCKDIM_X;

    int bcol = clGrpIdx * ROWS_BLOCKDIM_X;
    int col =  bcol + clLocIdx;
    int index_min = clGrpIdy * width;
    int index_max = index_min + width - 1;
    int src_index = index_min + bcol - s + clLocIdx;
    int cache_index = clLocIdx;
    float value = 0.0f;

    for(int j = 0; j < CACHE_COUNT; ++j) {
		if(cache_index < CACHE_WIDTH) {
			int fetch_index = src_index < index_min? index_min : (src_index > index_max ? index_max : src_index);
			data[cache_index] = src[fetch_index];
			src_index += ROWS_BLOCKDIM_X;
			cache_index += ROWS_BLOCKDIM_X;
		}
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(col >= width && index_min >= height) return;
    for(int i = 0; i < FILTER_SRC_SIZE; ++i) {
		value += (data[clLocIdx + i]* filter_src[i]);
    }
    value = max(0.0f, min(255.0f, value));
    dst[index_min+col] = (pixel_uc)value;
}

__kernel
void gaussian_scalar_fast_column (__global pixel_uc* src, __global pixel_uc* dst, int width, int height)
{
    __local pixel_uc data[(COLUMNS_BLOCKMEM_Y + FILTER_SRC_SIZE + 16) * COLUMNS_BLOCKDIM_X];

    int s = FILTER_SRC_SIZE >> 1;
    int CACHE_WIDTH = FILTER_SRC_SIZE + COLUMNS_BLOCKMEM_Y - 1;
    int TEMP = CACHE_WIDTH & 0xf;
    int EXTRA = (TEMP == 1 || TEMP == 0) ? 1 - TEMP : 15 - TEMP;
    int CACHE_TRUE_WIDTH = CACHE_WIDTH + EXTRA;
    int CACHE_COUNT = (CACHE_WIDTH + COLUMNS_BLOCKDIM_Y - 1) / COLUMNS_BLOCKDIM_Y;

    int row_block_first = clGrpIdy * COLUMNS_BLOCKMEM_Y;
    int col = clGrpIdx * COLUMNS_BLOCKDIM_X + clLocIdx;
    int row_first = row_block_first - s;
    int data_index_max = (height-1)*width + col;
    int cache_col_start = clLocIdy;	
    int cache_row_start = clLocIdx * CACHE_TRUE_WIDTH;
    int cache_index = cache_col_start + cache_row_start;
    int data_index = (row_first + cache_col_start) * width + col;

    if(col < width) {
		for(int i = 0; i < CACHE_COUNT; ++i) {
			if(cache_col_start < CACHE_WIDTH - i * COLUMNS_BLOCKDIM_Y) {
				int fetch_index = data_index < col ? col : (data_index > data_index_max? data_index_max : data_index);
				data[cache_index + i * COLUMNS_BLOCKDIM_Y] = src[fetch_index];
				data_index += (COLUMNS_BLOCKDIM_Y * width);
			}
		}
    }

    barrier(CLK_LOCAL_MEM_FENCE);
	
    if(col >= width) return;

    int row = row_block_first + clLocIdy;
    int index_start = cache_row_start + clLocIdy;

    for(int i = 0; i < COLUMNS_RESULT_STEPS; ++i, 
			row += COLUMNS_BLOCKDIM_Y, index_start += COLUMNS_BLOCKDIM_Y) {
		if(row < height) {
			int index_dest = row * width + col;
			float value = 0.0f;

			for(int i = 0; i < FILTER_SRC_SIZE; ++i) {
				value += (data[index_start + i] * filter_src[i]);
			}
			value = max(0.0f, min(255.0f, value));
			dst[index_dest] = value;
		}
    }
}


__kernel 
void gaussian_scalar_fast_no_shm (__global pixel_uc *src, __global pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int tid = clGlobIdx;
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
		v = max(0.0f, min(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }
}

__kernel 
void gaussian_scalar_fast_column_no_shm(__global pixel_uc *src, __global pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int tid = clGlobIdx;
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
		v = max(0.0f, min(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }
}

/* With the use of Vector Data Types */
__constant float8 f00 = (float8)(0.012477641543232876f);
__constant float8 f01 = (float8)(0.02641516735431067f);
__constant float8 f02 = (float8)(0.03391774626899505f);
__constant float8 f03 = (float8)(0.02641516735431067f);
__constant float8 f04 = (float8)(0.012477641543232876f);

__constant float8 f10 = (float8)(0.02641516735431067f);
__constant float8 f11 = (float8)(0.05592090972790157f);
__constant float8 f12 = (float8)(0.07180386941492609f);
__constant float8 f13 = (float8)(0.05592090972790157f);
__constant float8 f14 = (float8)(0.02641516735431067f);

__constant float8 f20 = (float8)(0.03391774626899505f); 
__constant float8 f21 = (float8)(0.07180386941492609f); 
__constant float8 f22 = (float8)(0.09219799334529226f); 
__constant float8 f23 = (float8)(0.07180386941492609f); 
__constant float8 f24 = (float8)(0.03391774626899505f);

#define f30 f10
#define f31 f11
#define f32 f12
#define f33 f13
#define f34 f14

#define f40 f00
#define f41 f01
#define f42 f02
#define f43 f03
#define f44 f04

//filter source
__constant float8 ff0 = (float8)(0.11170336406408216f);
__constant float8 ff1 = (float8)(0.23647602357935057f);
__constant float8 ff2 = (float8)(0.30364122471313454f);
__constant float8 ff3 = (float8)(0.23647602357935057f);
__constant float8 ff4 = (float8)(0.11170336406408216f);

#define OUT_OF_RANGE(yy, xx)										\
    v = 0.0f;														\
    fil_idx = 0;													\
    for (ii = (yy) - fil_half; ii <= (yy) + fil_half; ii++) {		\
		for (jj = (xx) - fil_half; jj <= (xx) + fil_half; jj++) {	\
			int iii = ii;											\
			int jjj = jj;											\
			if (ii < 0) iii = 0;									\
			else if (ii >= h) iii = h - 1;							\
			else iii = ii;											\
			if (jj < 0) jjj = 0;									\
			else if (jj >= w) jjj = w - 1;							\
			else jjj = jj;											\
			v += (float)in[iii * w + jjj] * filter[fil_idx];		\
			fil_idx++;												\
		}															\
    }																\
    v = max(0.0f, min(255.0f, v))
				 
__kernel 
void gaussian_simd(__global pixel_uc *in, __global pixel_uc *out, int w, int h)
{
    int y = clGlobIdx ;
    __global pixel_uc *in_line = in + w * y;
    __global pixel_uc *out_line = out + w * y;
  
    int fil_half = (FILTER_SIDE_SIZE - 1) / 2;

    int fil_idx = 0;
    int ii, jj;
    float v;
								
    OUT_OF_RANGE(y, 0);
    out_line[0] = (pixel_uc)v;
    OUT_OF_RANGE(y, 1);
    out_line[1] = (pixel_uc)v;
    OUT_OF_RANGE(y, w-2);
    out_line[w-2] = (pixel_uc)v;
    OUT_OF_RANGE(y, w-1);
    out_line[w-1] = (pixel_uc)v;

    if ((y >= fil_half) && (y < (h - fil_half)))
    {
		int nloop = w - 2;
		int nloop_simd = nloop - 7;
		int x = 2;

		for (; x<nloop_simd; x+=8)
		{
			float8 img_tmp = (float8)(0.0f);

			img_tmp += convert_float8(vload8(0, &in_line[x+(-2*w-2)])) * f00;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-2*w-1)])) * f01;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-2*w+0)])) * f02;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-2*w+1)])) * f03;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-2*w+2)])) * f04;

			img_tmp += convert_float8(vload8(0, &in_line[x+(-1*w-2)])) * f10;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-1*w-1)])) * f11;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-1*w+0)])) * f12;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-1*w+1)])) * f13;
			img_tmp += convert_float8(vload8(0, &in_line[x+(-1*w+2)])) * f14;

			img_tmp += convert_float8(vload8(0, &in_line[x+(+0*w-2)])) * f20;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+0*w-1)])) * f21;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+0*w+0)])) * f22;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+0*w+1)])) * f23;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+0*w+2)])) * f24;

			img_tmp += convert_float8(vload8(0, &in_line[x+(+1*w-2)])) * f30;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+1*w-1)])) * f31;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+1*w+0)])) * f32;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+1*w+1)])) * f33;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+1*w+2)])) * f34;

			img_tmp += convert_float8(vload8(0, &in_line[x+(+2*w-2)])) * f40;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+2*w-1)])) * f41;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+2*w+0)])) * f42;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+2*w+1)])) * f43;
			img_tmp += convert_float8(vload8(0, &in_line[x+(+2*w+2)])) * f44;

			img_tmp = max((float8)(0.0f), img_tmp);
			img_tmp = min((float8)(255.0f), img_tmp);

			vstore8(convert_uchar8_rtn(img_tmp),
					0, out_line+x);
		}

		for (; x<nloop; x++)
		{
			float img_tmp = 0;
			img_tmp += in_line[x+(-2*w-2)] * filter[0];
			img_tmp += in_line[x+(-2*w-1)] * filter[1];
			img_tmp += in_line[x+(-2*w-0)] * filter[2];
			img_tmp += in_line[x+(-2*w+1)] * filter[3];
			img_tmp += in_line[x+(-2*w+2)] * filter[4];

			img_tmp += in_line[x+(-1*w-2)] * filter[5];
			img_tmp += in_line[x+(-1*w-1)] * filter[6];
			img_tmp += in_line[x+(-1*w-0)] * filter[7];
			img_tmp += in_line[x+(-1*w+1)] * filter[8];
			img_tmp += in_line[x+(-1*w+2)] * filter[9];

			img_tmp += in_line[x+(+0*w-2)] * filter[10];
			img_tmp += in_line[x+(+0*w-1)] * filter[11];
			img_tmp += in_line[x+(+0*w-0)] * filter[12];
			img_tmp += in_line[x+(+0*w+1)] * filter[13];
			img_tmp += in_line[x+(+0*w+2)] * filter[14];

			img_tmp += in_line[x+(+1*w-2)] * filter[15];
			img_tmp += in_line[x+(+1*w-1)] * filter[16];
			img_tmp += in_line[x+(+1*w-0)] * filter[17];
			img_tmp += in_line[x+(+1*w+1)] * filter[18];
			img_tmp += in_line[x+(+1*w+2)] * filter[19];

			img_tmp += in_line[x+(+2*w-2)] * filter[20];
			img_tmp += in_line[x+(+2*w-1)] * filter[21];
			img_tmp += in_line[x+(+2*w-0)] * filter[22];
			img_tmp += in_line[x+(+2*w+1)] * filter[23];
			img_tmp += in_line[x+(+2*w+2)] * filter[24];

			if (img_tmp < 0)
			{
				img_tmp = 0;
			}

			if (img_tmp > 255)
			{
				img_tmp = 255;
			}

			out_line[x] = (pixel_uc)img_tmp;
		}
    }
    else
    {
		for (int x=2; x<w-2; x++) {
			OUT_OF_RANGE(y, x);
			out_line[x] = (pixel_uc)v;
		}
    }
}


__kernel 
void gaussian_simd_fast(__global pixel_uc *src, __global pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int i = clGlobIdx;
    int j = 0;
    int k, fil_idx;

    /* filter out of frame left side */
    for (j = 0; j < s; j++) {
		float v = 0.0f;
		fil_idx = 0;
		for (k = j - s; k < 0 && fil_idx < 2*s+1; k++, fil_idx++) v += src[i*w] * filter_src[fil_idx]; /* below zero */
		for (k = 0; fil_idx < 2*s+1; k++, fil_idx++) v += src[i*w+k] * filter_src[fil_idx];
		v = max(0.0f, min(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }

    for (j = s; j < w - s - 7; j+=8) {
		float8 v = (float8)(0.0f);
		fil_idx = 0;
		for ( k = j - s; k <= j + s; k++, fil_idx++) {
			v+= convert_float8(vload8(0, &src[i*w+k])) * (float8)filter_src[fil_idx];
		}
		v = max((float8)0.0f, min((float8)255.0f, v));
		vstore8(convert_uchar8_rtn(v),
				0, &dst[i*w+j]);
    }

    /* filter out of frame right side */
    for (; j < w; j++) {
		float v = 0.0f;
		fil_idx = 0;
		for (k = j - s; k < w - 1 && fil_idx < 2*s+1; k++, fil_idx++)
			v += src[i*w+k] * filter_src[fil_idx];
		for (; fil_idx < 2*s+1; fil_idx++)
			v += src[i*w+k]*filter_src[fil_idx]; /* more than w */
		v = max(0.0f, min(255.0f, v));
		dst[i * w + j] = (pixel_uc)v;
    }
}
				 
__kernel 
void gaussian_simd_fast_column(__global pixel_uc *src, __global pixel_uc *dst, int w, int h)
{  
    const int s = (FILTER_SRC_SIZE - 1) / 2;
    int i = clGlobIdx;
    int j = 0;
    int k, fil_idx;

    for (j = 0; j < w - 7; j+=8) {
		float8 v = (float8)(0.0f);
		fil_idx = 0;
		if ( i >= s && i < h - s) {
			for ( k = i - s; k <= i + s; k++, fil_idx++)
				v += convert_float8(vload8(0, &src[k*w+j])) * (float8)filter_src[fil_idx];
		}
		else if (i < s) { /* filter out of frame */
			for (k = i - s; k < 0; k++, fil_idx++) 
				/* below zero */
				v += convert_float8(vload8(0, &src[j])) * (float8)filter_src[fil_idx];
			for (k = 0; fil_idx < 2*s+1; k++, fil_idx++) 
				v += convert_float8(vload8(0, &src[k*w+j])) * (float8)filter_src[fil_idx];
		}
		else { /* filter out of frame */
			for (k = i - s; k < h - 1; k++, fil_idx++) 
				v += convert_float8(vload8(0, &src[k*w+j])) * (float8)filter_src[fil_idx];
			for (; fil_idx < 2*s+1; fil_idx++) 
				/* more than w */
				v += convert_float8(vload8(0, &src[k*w+j])) * (float8)filter_src[fil_idx];
		}

		v = max((float8)0.0f, min((float8)255.0f, v));
		vstore8(convert_uchar8_rtn(v),
				0, &dst[i*w+j]);
    }

    /* filter out of frame right side */
    for (; j < w; j++) {
		float v = 0.0f;
		fil_idx = 0;

		if ( i >= s && i < h - s) {
			for ( k = i - s; k <= i + s; k++, fil_idx++)
				v += src[k*w+j] * filter[fil_idx];
		}
		else if (i < s) { /* filter out of frame */
			for (k = i - s; k < 0; k++, fil_idx++) v += src[j] * filter_src[fil_idx]; /* below zero */
			for (k = 0; fil_idx < 2*s+1; k++, fil_idx++) v += src[k*w+j] * filter_src[fil_idx];
		}
		else { /* filter out of frame */
			for (k = i - s; k < h - 1; k++, fil_idx++) v += src[k*w+j] * filter_src[fil_idx];
			for (; fil_idx < 2*s+1; fil_idx++) v += src[k*w+j] * filter_src[fil_idx]; /* more than w */
		}

		v = max(0.0f, min(255.0f, v));
		dst[i*w+j] = (pixel_uc)v;
    }
}

/* KERNELS FOR DEBUGGING PURPOSE */
__kernel 
void gaussian_stsd(__global pixel_uc *in, __global pixel_uc *out, int h, int w)
{  
    int tid       = clGlobIdx ;
    int i         = tid / w;
    int j         = tid % w;
    int fil_idx   = 0;
    int fil_half  = (FILTER_SIDE_SIZE - 1) / 2;
    float img_tmp = 0;

    int ii, jj;

    if (tid == 0) {
		if (i >= fil_half && i < h - fil_half && j >= fil_half && j < w - fil_half) {
			for (ii = i - fil_half; ii <= i + fil_half; ii++) {
				for (jj = j - fil_half; jj <= j + fil_half; jj++) {
					img_tmp += (float)in[ii * w + jj] * filter[fil_idx];
					fil_idx++;
				}
			}	  
			img_tmp = max(0.0f, min(255.0f, img_tmp));
		} else {
			img_tmp = in[tid]; // Frame pixels stay the same
		}
		out[tid] = (pixel_uc)img_tmp;
    }
}

__kernel 
void gaussian_stad(__global pixel_uc *in, __global pixel_uc *out, int h, int w)
{  
    int tid       = clGlobIdx ;
    int fil_idx   = 0;
    int fil_half  = (FILTER_SIDE_SIZE - 1) / 2;
    float img_tmp = 0;

    int i, j, ii, jj;

    if (tid == 0) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				if (i >= fil_half && i < h - fil_half && j >= fil_half && j < w - fil_half) {
					img_tmp = 0;
					fil_idx = 0;
					for (ii = i - fil_half; ii <= i + fil_half; ii++) {
						for (jj = j - fil_half; jj <= j + fil_half; jj++) {
							img_tmp += (float)in[ii * w + jj] * filter[fil_idx];
							fil_idx++;
						}
					}	  
					img_tmp = max(0.0f, min(255.0f, img_tmp));
				} else {
					img_tmp = in[i*w+j]; // Frame pixels stay the same
				}
				out[i*w+j] = (pixel_uc)img_tmp;
			}
		}
    }
}

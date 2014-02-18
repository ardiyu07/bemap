#include "BackProjection_common.hpp"
#include "BackProjection.hpp"

#define cudaEnumIdx threadIdx.x + blockIdx.x * blockDim.x
#define cudaEnumIdy threadIdx.y + blockIdx.y * blockDim.y
#define cudaLocIdx threadIdx.x
#define cudaLocIdy threadIdx.y
#define cudaGrpIdx blockIdx.x
#define cudaGrpIdy blockIdx.y

__global__ 
void dev_init (
			   F_TYPE *image,
			   int *index
			   )
{
	image[cudaEnumIdx] = 0;
}

__global__ 
void dev_backprojection_scalar (
								int r,
								int c,
								F_TYPE *image,
								unsigned char *guess,
								F_TYPE *rscore,
								F_TYPE *cscore,
								F_TYPE *uscore,
								F_TYPE *dscore,
								int *index
								)
{
	__shared__ F_TYPE l_image[64];
	__shared__ F_TYPE l_index[64];

	__shared__ F_TYPE r_local[128];
	__shared__ F_TYPE u_local[128];
	__shared__ F_TYPE d_local[128];

	int lj, li, lsize, i, j, index_result;
	F_TYPE c_tmp, image_result;
	F_TYPE image_tmp;

	lj = cudaLocIdx;
	lsize = blockDim.x;

	i = cudaEnumIdy * blockDim.x;
	j = cudaGrpIdx * blockDim.x;

	image_result = 0.0;
	index_result = -1;

	if (j + lj < c)
		c_tmp = cscore[ j + lj ];
	if (i + lj < r)
		r_local[lj] = rscore[ i + lj ];
	if (i + j + lj < r + c - 1)
		u_local[lj] = uscore[ i + j + lj ];
	if (i + j + lj + lsize < r + c - 1)
		u_local[lj + lsize] = uscore[ i + j + lj + lsize];
	if (i - j + c - lsize + lj < r + c - 1)
		d_local[lj] = dscore[ i - j + c - lsize + lj ]; 
	if (i - j + c + lj < r + c - 1)
		d_local[lj + lsize] = dscore[ i - j + c + lj ];

	__syncthreads();

	j = cudaEnumIdx;

	if (j < c || c_tmp <= 0) {
		for (li = 0; li < lsize && i + li < r; li++) {
			image_tmp = r_local[li];
			if ( image_tmp <= 0.0 ) {
				continue;
			}
			if ( guess[ (i + li) * c + j ] == TRUE ) {
				continue;
			} else {
				image_tmp = image_tmp * c_tmp * u_local[lj + li] * d_local[lsize - (lj + 1) + li];
			}
			if (image_tmp <= 0.0) {
				continue;
			}
			if (image_result < image_tmp) {
				image_result = image_tmp;
				index_result = (i + li) * c + j;
			}
		}
	}

	l_image[lj] = image_result;
	l_index[lj] = index_result;
	__syncthreads();

	for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
		if ( lj < i ) {
			if (l_image[ lj ] < l_image[ lj + i ]) {
				l_image[ lj ] = l_image[ lj + i ];
				l_index[ lj ] = l_index[ lj + i ];
			}
		}
	}

	__syncthreads();

	if( lj == 0 ) {
		image[ cudaGrpIdy * gridDim.y + cudaGrpIdx] = l_image[0];
		index[ cudaGrpIdy * gridDim.y + cudaGrpIdx] = l_index[0];
	}
}

__global__ 
void dev_findmax (
				  F_TYPE *image,
				  int *index,
				  F_TYPE *image2,
				  int *index2
				  )
{
	__shared__ F_TYPE l_image[128];
	__shared__ int l_index[128];

	int tid = cudaEnumIdx;
	int lid = cudaLocIdx;
	int i;
  
	l_image[ lid ] = image[ tid ];
	l_index[ lid ] = index[ tid ];
	__syncthreads();
 
	for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
		if ( lid < i ) {
			if (l_image[ lid ] < l_image[ lid + i ]) {
				l_image[ lid ] = l_image[ lid + i ];
				l_index[ lid ] = l_index[ lid + i ];
			}
		}
	}

	__syncthreads();
	if ( lid == 0 ) {
		image2[ cudaGrpIdx ] = l_image[0];
		if (l_image[0] <= 0.0)
			index2[ cudaGrpIdx ] = -1;
		else
			index2[ cudaGrpIdx ] = l_index[0];
	}
}

__global__ 
void dev_decreaseproj (
					   int *index,
					   int c,
					   unsigned char *guess,
					   F_TYPE *rscore,
					   F_TYPE *cscore,
					   F_TYPE *uscore,
					   F_TYPE *dscore,
					   int *rproj,
					   int *rband,
					   int *cproj,
					   int *cband,
					   int *uproj,
					   int *uband,
					   int *dproj,
					   int *dband,
					   int *maxId,
					   F_TYPE *image
					   )
{
	__shared__ float l_image[128];
	__shared__ int l_index[128];

	int tid = cudaEnumIdx;
	int lid = cudaLocIdx;
	int i, j, idx, proj, band;
  
	l_image[ lid ] = image[ tid ];
	l_index[ lid ] = index[ tid ];
	__syncthreads();
 
	for ( i = blockDim.x / 2; i > 0; i /= 2 ) {
		if ( lid < i ) {
			if (l_image[ lid ] < l_image[ lid + i ]) {
				l_image[ lid ] = l_image[ lid + i ];
				l_index[ lid ] = l_index[ lid + i ];
			}
		}
	}
	__syncthreads();

	if ( lid == 0 ) {  
		maxId[ 0 ] = l_index[0];
		image[ 0 ] = l_image[0];

		if(l_image[0] > 0.0) {
			idx = l_index[0];
			i = idx / c;
			j = idx % c;

			guess[idx] = TRUE;

			proj = rproj[ i ] - 1;
			rproj[i] = proj;
			band = rband[ i ] - 1;
			rband[i] = band;
			rscore[i] = (F_TYPE) proj / (F_TYPE) band;

			proj = cproj[ j ] - 1;
			cproj[j] = proj;
			band = cband[ j ] - 1;
			cband[j] = band;
			cscore[j] = (F_TYPE) proj / (F_TYPE) band;

			proj = uproj[ i+j ] - 1;
			uproj[i+j] = proj;
			band = uband[ i+j ] - 1;
			uband[i+j] = band;
			uscore[i+j] = (F_TYPE) proj / (F_TYPE) band;

			proj = dproj[ i-j+c-1 ] - 1;
			dproj[i-j+c-1] = proj;
			band = dband[ i-j+c-1 ] - 1;
			dband[i-j+c-1] = band;
			dscore[i-j+c-1] = (F_TYPE) proj / (F_TYPE) band;
		}
	}
}

void BackProjectionCUDA::execute()
{
	int i;
    float *swapf, *image_p, *image2_p;
	int *swapi, *index_p, *index2_p;

    t_kernel.start();

	dBlocks.x = (c + nThreads - 1) / nThreads;
	dBlocks.y = (r + nThreads - 1) / nThreads;
	dThreads.x = nThreads; 
	dThreads.y = 1;

	init_nBlocks = (((r * c / nThreads) + nThreads - 1) / nThreads);
	init_nThreads = nThreads;

	dev_init<<<init_nBlocks, init_nThreads>>>(d_image, d_index);
	dev_init<<<init_nBlocks, init_nThreads>>>(d_image2, d_index2);

	for (i = 0;; i++)
	{
		dev_backprojection_scalar<<<dBlocks, dThreads>>>(r,
														 c,
														 d_image,
														 d_guess,
														 d_rscore,
														 d_cscore,
														 d_uscore,
														 d_dscore,
														 d_index
														 );

		image_p = d_image;
		image2_p = d_image2;
		index_p = d_index;
		index2_p = d_index2;

		if (i != 0) {
			cudaMemcpy(maxId_p, d_maxId, sizeof(int), cudaMemcpyDeviceToHost);
			CUDA_CHECK_ERROR();
			if (*maxId_p < 0)
			{
				break;
			} 
			else 
			{
				guess[*maxId_p] = TRUE;
			}
		}
		for (findmax_nglobals = (((dBlocks.x * dBlocks.y * nThreads) + nThreads - 1) / nThreads) * nThreads,
				 findmax_nlocals = nThreads;
			 findmax_nglobals > nThreads;
			 findmax_nglobals = (((findmax_nglobals / nThreads) + nThreads - 1) / nThreads) * nThreads) {			
			dev_findmax<<<(findmax_nglobals / nThreads), findmax_nlocals>>>(image_p, index_p, image2_p, index2_p);

			swapf = image_p;
			image_p = image2_p;
			image2_p = swapf;
			swapi = index_p;
			index_p = index2_p;
			index2_p = swapi;
		}
		dev_decreaseproj<<<(findmax_nglobals / nThreads), nThreads>>>(index_p,
																	  c, 
																	  d_guess,
																	  d_rscore,
																	  d_cscore,
																	  d_uscore,
																	  d_dscore,
																	  d_rproj,
																	  d_rband,
																	  d_cproj,
																	  d_cband,
																	  d_uproj,
																	  d_uband,
																	  d_dproj,
																	  d_dband,
																	  d_maxId,
																	  image_p
																	  );
	}
	cudaMemcpy(guess, d_guess, r * c * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	CUDA_CHECK_ERROR();		  

	t_kernel.stop();	
}

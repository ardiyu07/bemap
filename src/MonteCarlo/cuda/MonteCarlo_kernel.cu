#include "common.hpp"
#include "MonteCarlo.hpp"

extern "C" {
#include "mt19937ar.h"
}

#define cudaEnumIdx threadIdx.x + blockIdx.x * blockDim.x
#define cudaEnumIdy threadIdx.y + blockIdx.y * blockDim.y
#define cudaLocIdx threadIdx.x
#define cudaGrpIdx blockIdx.x

__global__
void montecarlo_scalar(float *call, 
					   float *random, 
					   float *confidence, 
					   float *stockPrice, 
					   float *optionStrike, 
					   float *optionYears, 
					   float R, float V, int optN, int pathN) 
{
	int glid = cudaEnumIdx;

	float S, X, T;

	if (glid < optN) {
    
		S = stockPrice[glid];
		X = optionStrike[glid];
		T = optionYears[glid];

		float sum = 0.0f, sum2 = 0.0f;
		float VBySqrtT = V * sqrt(T);
		float MuByT = (R - 0.5f * V * V) * T;
		float r, path, tmp_path;

		for (int i = 0; i < pathN; i++)
		{
			r = random[i];
			tmp_path = S * exp(MuByT + VBySqrtT * r) - X;
			path     = (tmp_path > 0.0f) ? (tmp_path) : (0.0f);

			sum  += path;
			sum2 += path*path;
		}

		call[glid] = (float)(exp(- R * T) * sum / (float)pathN);
		float stdDev = sqrt(((float)pathN * sum2 - sum * sum) / ((float)pathN * (float)(pathN - 1)));
		confidence[glid] = (float)(exp(- R * T) * 1.96f * stdDev / sqrt((float)pathN));

	}
}

__global__
void montecarlo_shm(float *call, 
					float *random, 
					float *confidence, 
					float *stockPrice, 
					float *optionStrike, 
					float *optionYears, 
					float R, float V, int optN, int pathN) 
{

	int lcid = cudaLocIdx;
	int grid = cudaGrpIdx;

	float S, X, T;

	__shared__ float shared1[NTHREADS]; /// path
	__shared__ float shared2[NTHREADS]; /// path squared
  
	shared1[lcid] = 0.0f;
	shared2[lcid] = 0.0f;
	__syncthreads();

	if (grid < optN) {
    
		S = stockPrice[grid];
		X = optionStrike[grid];
		T = optionYears[grid];

		float sum = 0.0f, sum2 = 0.0f;
		float VBySqrtT = V * sqrt(T);
		float MuByT = (R - 0.5f * V * V) * T;
		float r, path, tmp_path;

		for (int i = lcid; i < pathN; i+=NTHREADS)
		{
			r = random[i];
			tmp_path = S * exp(MuByT + VBySqrtT * r) - X;
			path     = (tmp_path > 0.0f) ? (tmp_path) : (0.0f);

			shared1[lcid] += path;
			shared2[lcid] += path*path;
		}

		__syncthreads();

		if (lcid < 64) {
			shared1[lcid] += shared1[lcid + 64];
			shared2[lcid] += shared2[lcid + 64];
		}

		__syncthreads();

		if (lcid < 32) {
			shared1[lcid] += shared1[lcid + 32];
			shared2[lcid] += shared2[lcid + 32];
		}

		__syncthreads();

		if (lcid < 16) {
			shared1[lcid] += shared1[lcid + 16];
			shared2[lcid] += shared2[lcid + 16];
		}

		__syncthreads();

		if (lcid < 8) {
			shared1[lcid] += shared1[lcid + 8];
			shared2[lcid] += shared2[lcid + 8];
		}

		__syncthreads();

		if (lcid < 4) {
			shared1[lcid] += shared1[lcid + 4];
			shared2[lcid] += shared2[lcid + 4];
		}

		__syncthreads();

		if (lcid < 2) {
			shared1[lcid] += shared1[lcid + 2];
			shared2[lcid] += shared2[lcid + 2];
		}

		__syncthreads();

		sum  = shared1[0] + shared1[1];
		sum2 = shared2[0] + shared2[1];
  
		call[grid] = (float)(exp(- R * T) * sum / (float)pathN);
		float stdDev = sqrt(((float)pathN * sum2 - sum * sum) / ((float)pathN * (float)(pathN - 1)));
		confidence[grid] = (float)(exp(- R * T) * 1.96f * stdDev / sqrt((float)pathN));
	}
}

__device__
void boxmuller_calc(float *u1, float *u2)
{
	float r = sqrt(-2.0f * log(*u1));
	float phi = M_PI2 * *u2;

	*u1 = r * cos(phi);
	*u2 = r * sin(phi);
}

__global__
void boxmuller(float *random, int pathN)
{
	int tid = cudaEnumIdx;
	int glid = tid*2;
  
	if (glid < pathN) {
		boxmuller_calc(&random[glid], &random[glid+1]);
	}
}

void MonteCarloCUDA::execute()
{ 

	/// GenRand
	t_gr.start();
	for (int i = 0; i < pathNum; i++) {
		random[i] = genrand_real2();
	}
	t_gr.stop();

	/// BoxMuller

	t_bm.start();

	cudaMemcpy(d_random, random, pathNum * sizeof(float), cudaMemcpyHostToDevice); CUDA_CHECK_ERROR();

	nThreads = NTHREADS;
	nBlocks = (pathNum/2+1 + nThreads - 1) / nThreads;
	int itr = (nBlocks - 1) / MAX_NBLOCKS + 1;
	nBlocks = min(nBlocks, MAX_NBLOCKS);

	for (int i = 0; i < itr; ++i)
		boxmuller<<<nBlocks, nThreads>>>
			(d_random, pathNum);

	t_bm.stop();

	/// MonteCarlo European Call Simulation

	cudaEventCreate(&start); CUDA_CHECK_ERROR();
	cudaEventCreate(&end); CUDA_CHECK_ERROR();

	nBlocks = optNum;
	cudaEventRecord(start, 0);

	montecarlo_shm<<<nBlocks, nThreads>>>
		(d_call, d_random, d_confidence, d_stockPrice, d_optionStrike, d_optionYears, riskFree, volatility, optNum, pathNum);

	/* nThreads = 1; */
	/* nBlocks = optNum; */
	/* cudaEventRecord(start, 0); */

	/* montecarlo_scalar<<<nBlocks, nThreads>>> */
	/*   (d_call, d_random, d_confidence, d_stockPrice, d_optionStrike, d_optionYears, riskFree, volatility, optNum, pathNum); */

	cudaThreadSynchronize(); CUDA_CHECK_ERROR();
	cudaEventRecord(end, 0); CUDA_CHECK_ERROR();
	cudaEventSynchronize(start); CUDA_CHECK_ERROR();
	cudaEventSynchronize(end); CUDA_CHECK_ERROR();
	cudaEventElapsedTime(&elapsed, start, end); CUDA_CHECK_ERROR();

	t_kernel += elapsed;

	cudaEventDestroy(start); CUDA_CHECK_ERROR();
	cudaEventDestroy(end); CUDA_CHECK_ERROR();
}


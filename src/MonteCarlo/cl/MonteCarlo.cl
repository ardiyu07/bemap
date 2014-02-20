#include "MonteCarlo_common.hpp"

__kernel
void montecarlo_simd(__global float *call, 
					 __global float *random, 
					 __global float *confidence, 
					 __global float *stockPrice, 
					 __global float *optionStrike, 
					 __global float *optionYears, 
					 float R, float V, int optN, int pathN)
{
	int glid = clGlobIdx;
	float S, X, T;

	if (glid < optN) {
    
		S = stockPrice[glid];
		X = optionStrike[glid];
		T = optionYears[glid];

		float8 sum = 0.0f, sum2 = 0.0f;
		float8 VBySqrtT = V * sqrt(T);
		float8 MuByT = (R - (float8)0.5f * V * V) * T;
		float8 r, path, tmp_path;

		for (int i = 0; i < pathN; i+=8)
		{
			r = vload8(0, &random[i]);
			tmp_path = S * exp(MuByT + VBySqrtT * r) - X;
			path     = (tmp_path > 0.0f) ? (tmp_path) : (0.0f);

			sum  += path;
			sum2 += path*path;
		}

		float dsum = 
			sum.s0 +
			sum.s1 +
			sum.s2 +
			sum.s3 +
			sum.s4 +
			sum.s5 +
			sum.s6 +
			sum.s7;

		float dsum2 = 
			sum2.s0 +
			sum2.s1 +
			sum2.s2 +
			sum2.s3 +
			sum2.s4 +
			sum2.s5 +
			sum2.s6 +
			sum2.s7;
      
		float stdDev = sqrt(((float)pathN * dsum2 - dsum * dsum) / ((float)pathN * (float)(pathN - 1)));

		call[glid] = (float)(exp(- R * T) * dsum / (float)pathN);
		confidence[glid] = (float)(exp(- R * T) * (float)1.96f * stdDev / sqrt((float)pathN));
	}
}

__kernel
void montecarlo_scalar(__global float *call, 
					   __global float *random, 
					   __global float *confidence, 
					   __global float *stockPrice, 
					   __global float *optionStrike, 
					   __global float *optionYears, 
					   float R, float V, int optN, int pathN) 
{
	int glid = clGlobIdx;

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

__kernel
void montecarlo_scalar_shm(__global float *call, 
						   __global float *random, 
						   __global float *confidence, 
						   __global float *stockPrice, 
						   __global float *optionStrike, 
						   __global float *optionYears, 
						   float R, float V, int optN, int pathN) 
{

	int lcid = clLocIdx;
	int grid = clGrpIdx;

	float S, X, T;

	__local float shared1[NTHREADS]; /// path
	__local float shared2[NTHREADS]; /// path squared
  
	shared1[lcid]          = 0.0f;
	shared2[lcid]          = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);

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

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 64) {
			shared1[lcid] += shared1[lcid + 64];
			shared2[lcid] += shared2[lcid + 64];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 32) {
			shared1[lcid] += shared1[lcid + 32];
			shared2[lcid] += shared2[lcid + 32];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 16) {
			shared1[lcid] += shared1[lcid + 16];
			shared2[lcid] += shared2[lcid + 16];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 8) {
			shared1[lcid] += shared1[lcid + 8];
			shared2[lcid] += shared2[lcid + 8];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 4) {
			shared1[lcid] += shared1[lcid + 4];
			shared2[lcid] += shared2[lcid + 4];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		if (lcid < 2) {
			shared1[lcid] += shared1[lcid + 2];
			shared2[lcid] += shared2[lcid + 2];
		}

		barrier(CLK_LOCAL_MEM_FENCE);

		sum  = shared1[0] + shared1[1];
		sum2 = shared2[0] + shared2[1];
  
		call[grid] = (float)(exp(- R * T) * sum / (float)pathN);
		float stdDev = sqrt(((float)pathN * sum2 - sum * sum) / ((float)pathN * (float)(pathN - 1)));
		confidence[grid] = (float)(exp(- R * T) * 1.96f * stdDev / sqrt((float)pathN));
	}
}

void boxmuller_calc(__global float *u1, __global float *u2)
{
	float r = sqrt(- 2.0f * log(*u1));
	float phi = M_PI2 * *u2;

	*u1 = r * cos(phi);
	*u2 = r * sin(phi);
}

__kernel
void boxmuller(__global float *random, int pathN, int width, int height)
{
	int glid = clGlobIdx*2;
  
	if (glid < pathN) {
		boxmuller_calc(&random[glid], &random[glid+1]);
	}
}

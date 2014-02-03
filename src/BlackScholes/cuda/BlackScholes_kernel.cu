#include "common.hpp"
#include "BlackScholes.hpp"

#if 1
#define FABS fabsf
#define SQRT sqrtf
#define LOG  __logf
#define EXP  __expf
#else
#define FABS fabs
#define SQRT sqrt
#define LOG  log
#define EXP  exp
#endif

#define cudaEnumIdx threadIdx.x + blockIdx.x * blockDim.x
#define cudaEnumIdy threadIdx.y + blockIdx.y * blockDim.y

__device__
float d_cnd( float d )
{
    float K = 1.0f / ( 1.0f + 0.2316419f * FABS( d ) );
    float cnd = RSQRT2PI * EXP( - 0.5f * d * d ) * ( K * ( coef1 + K * ( coef2 + K * ( coef3 + K * ( coef4 + K * coef5 ) ) ) ) );

    if ( d > 0.0f )
    {
	cnd = 1.0f - cnd;
    }


    return cnd;
}

__device__
void d_black_scholes_equation( float *call, float *put, float *stock_price, float *option_strike, float *option_years, float riskless_rate, float volatility_rate, int opt )
{
    float sqrt_t, exp_rt;
    float d1, d2, cnd1, cnd2;

    sqrt_t = SQRT( option_years[ opt ] );
    d1 = ( LOG( stock_price[ opt ] / option_strike[ opt ] ) + ( riskless_rate + 0.5f * volatility_rate * volatility_rate ) * option_years[ opt ] ) / ( volatility_rate * sqrt_t );
    d2 = d1 - volatility_rate * sqrt_t;

    cnd1 = d_cnd( d1 );
    cnd2 = d_cnd( d2 );

    exp_rt = EXP( - riskless_rate * option_years[ opt ] );
    call[ opt ] = stock_price[ opt ] * cnd1 - option_strike[ opt ] * exp_rt * cnd2;
    put[ opt ]  = option_strike[ opt ] * exp_rt * ( 1.0f - cnd2 ) - stock_price[ opt ] * ( 1.0f - cnd1 );

}

__global__
void black_scholes_scalar( float *call, float *put, float *stockprice, float *optionstrike, float *optionyears, float riskfree, float volatility, int optN, int blockOffset)
{
    if (cudaEnumIdx < optN)
	d_black_scholes_equation( call, put, stockprice, optionstrike, optionyears, riskfree, volatility, cudaEnumIdx + blockOffset);
}

void BlackScholesCUDA::execute()
{
    cudaEventCreate(&start); CUDA_CHECK_ERROR();
    cudaEventCreate(&end); CUDA_CHECK_ERROR();

    nBlocks = (optNum + nThreads - 1) / nThreads;
    int itr = (nBlocks - 1) / MAX_NBLOCKS + 1;
    nBlocks = min(nBlocks, MAX_NBLOCKS);

    cudaEventRecord(start, 0);
    for (int i = 0; i < itr; ++i)
	black_scholes_scalar<<<nBlocks, nThreads>>>
	    (d_call, d_put, d_stockPrice, d_optionStrike, d_optionYears, riskFree, volatility, optNum, i * nThreads * nBlocks);
    cudaThreadSynchronize(); CUDA_CHECK_ERROR();
    cudaEventRecord(end, 0); CUDA_CHECK_ERROR();
    cudaEventSynchronize(start); CUDA_CHECK_ERROR();
    cudaEventSynchronize(end); CUDA_CHECK_ERROR();
    cudaEventElapsedTime(&elapsed, start, end); CUDA_CHECK_ERROR();
  
    t_kernel += elapsed;
}


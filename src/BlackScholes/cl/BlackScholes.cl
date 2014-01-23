#define clGlobIdx get_global_id(0)
#define clGlobIdy get_global_id(1)
#define clLocIdx  get_local_id(0)
#define clLocIdy  get_local_id(1)
#define clGrpIdx  get_group_id(0)
#define clGrpIdy  get_group_id(1)

__constant float coef11 = 0.31938153f;
__constant float coef22 = -0.356563782f;
__constant float coef33 = 1.781477937f;
__constant float coef44 = -1.821255978f;
__constant float coef55 = 1.330274429f;
__constant float r_sqrt2pii = 0.3989422804f;

float cnd(float d)
{
    float K = 1.0f / (1.0f + 0.2316419f * fabs(d));
    float cnd = r_sqrt2pii * exp(-0.5f*d*d) * (K*(coef11 + K * (coef22 + K * (coef33 + K * (coef44 + K * coef55)))));

    if (d > 0.0f) {
		cnd = 1.0f - cnd;
    }

    return cnd;
}

void black_scholes_equation(__global float *call, 
                            __global float *put, 
                            __global float *stock_price, 
                            __global float *option_strike, 
                            __global float *option_years, 
                            float riskless_rate, 
                            float volatility_rate, 
                            int opt)
{
    float sqrt_t, exp_rt;
    float d1, d2, cnd1, cnd2;

    sqrt_t = sqrt(option_years[opt]);
    d1 = (log(stock_price[opt] / option_strike[opt]) 
          + (riskless_rate + 0.5f * volatility_rate * volatility_rate) * option_years[opt]) / (volatility_rate * sqrt_t);
    d2 = d1 - volatility_rate * sqrt_t;

    cnd1 = cnd(d1);
    cnd2 = cnd(d2);

    exp_rt = exp(-riskless_rate * option_years[opt]);
    call[opt] = stock_price[opt] * cnd1 - option_strike[opt] * exp_rt * cnd2;
    put[opt]  = option_strike[opt] * exp_rt * (1.0f - cnd2) - stock_price[opt] * (1.0f - cnd1);
}

__kernel 
void black_scholes_scalar(__global float *call, 
                          __global float *put, 
                          __global float *stockprice, 
                          __global float *optionstrike, 
                          __global float *optionyears, 
                          float riskfree, 
                          float volatility, 
                          int optN, 
                          int heightN, 
                          int widthN)
{
    int tid = clGlobIdx;
    if (tid >= optN) return;

    black_scholes_equation(call, put, stockprice, optionstrike, optionyears, riskfree, volatility, tid);
}

/* With the use of Vector Data Type */
__constant float8 coef1 = (float8)(0.31938153f);
__constant float8 coef2 = (float8)(- 0.356563782f);
__constant float8 coef3 = (float8)(1.781477937f);
__constant float8 coef4 = (float8)(-1.821255978f);
__constant float8 coef5 = (float8)(1.330274429f);
__constant float8 r_sqrt2pi = (float8)(0.3989422804f);

__kernel void black_scholes_simd(__global float *call, 
								 __global float *put, 
								 __global float *stockprice, 
								 __global float *optionstrike, 
								 __global float *optionyears, 
								 float riskfree,
                                 float volatility,
                                 int optN,
                                 int heightN,
                                 int widthN)
{
    const int y = clGlobIdx;
    int ind = widthN * y;
    int nloop = widthN;
    int nloop_simd = nloop - 7;
    int i = 0;

    float8 sqrt_t, exp_rt, d1, d2, cnd1, cnd2, K1, K2;
    float8 in_stockprice, in_optionstrike, in_optionyears, out_put, out_call;

    float8 riskfree8 = (float8)(riskfree);
    float8 volatility8 = (float8)(volatility);

    for (; i < nloop_simd; i+=8)
    {
		in_stockprice = convert_float8(vload8(0, &stockprice[i + ind]));
		in_optionstrike = convert_float8(vload8(0, &optionstrike[i + ind]));
		in_optionyears = convert_float8(vload8(0, &optionyears[i + ind]));

		sqrt_t = sqrt(in_optionyears);
		d1 = (log(in_stockprice / in_optionstrike) + 
			  (riskfree8 + (float8)(0.5f) * volatility8 * volatility8) 
			  * in_optionyears) / (volatility8 * sqrt_t);
		d2 = d1 - volatility8 * sqrt_t;

		K1 = (float8)(1.0f) / ((float8)(1.0f) + (float8)(0.2316419f) * fabs(d1));
		cnd1 = r_sqrt2pi * exp((float8)(-0.5f) * d1 * d1) * (K1 * (coef1 + K1 * (coef2 + K1 * (coef3 + K1 * (coef4 + K1 * coef5)))));

		K2 = (float8)(1.0f) / ((float8)(1.0f) + (float8)(0.2316419f) * fabs(d2));
		cnd2 = r_sqrt2pi * exp((float8)(-0.5f) * d2 * d2) * (K2 * (coef1 + K2 * (coef2 + K2 * (coef3 + K2 * (coef4 + K2 * coef5)))));

		cnd1 = select(cnd1, (float8)(1.0f) - cnd1, (d1 > (float8)(0.0f)));
		cnd2 = select(cnd2, (float8)(1.0f) - cnd2, (d2 > (float8)(0.0f)));

		exp_rt = exp(-riskfree8 * in_optionyears);
		out_call = in_stockprice * cnd1 - in_optionstrike * exp_rt * cnd2;
		out_put = in_optionstrike * exp_rt * ((float8)(1.0f) - cnd2) - in_stockprice * ((float8)(1.0f) - cnd1);
	  
		vstore8(out_call, 0, call + i + ind);
		vstore8(out_put, 0, put + i + ind);	  
    }

    for (; i < nloop; i++)
    {
		black_scholes_equation(call, put, stockprice, optionstrike, optionyears, riskfree, volatility, ind + i);	  
    }
}

/* KERNELS FOR DEBUGGING PURPOSE ONLY */
__kernel void black_scholes_stsd(__global float *call, __global float *put, __global float *stockprice, __global float *optionstrike, __global float *optionyears, float riskfree, float volatility, int optN, int heightN, int widthN)
{
    if(clGlobIdx == 0)
		black_scholes_equation(call, put, stockprice, optionstrike, optionyears, riskfree, volatility, clGlobIdx);
}

__kernel void black_scholes_stad(__global float *call, __global float *put, __global float *stockprice, __global float *optionstrike, __global float *optionyears, float riskfree, float volatility, int optN, int heightN, int widthN)
{
    int i;
    if(clGlobIdx == 0)
		for (i = 0; i < optN; ++i)
			black_scholes_equation(call, put, stockprice, optionstrike, optionyears, riskfree, volatility, i);
}

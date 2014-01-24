#include <math.h>

void black_scholes_gold(float *call, float *put, float *stockprice,
                        float *optionstrike, float *optionyears,
                        float riskfree, float volatility, int opt_num);

#include "BlackScholes_gold.hpp"

inline float cnd(float d)
{
    float K = 1.0f / (1.0f + 0.2316419f * fabs(d));
    float cnd =
        RSQRT2PI * expf(-0.5f * d * d) * (K *
                                          (coef1 +
                                           K * (coef2 +
                                                K * (coef3 +
                                                     K * (coef4 +
                                                          K * coef5)))));

    if (d > 0.0f) {
        cnd = 1.0f - cnd;
    }

    return cnd;
}

inline
void black_scholes_equation(float *call, float *put, float stock_price,
                            float option_strike, float option_years,
                            float riskless_rate, float volatility_rate)
{
    float sqrt_t, exp_rt;
    float d1, d2, cnd1, cnd2;


    sqrt_t = sqrtf(option_years);
    d1 = (logf(stock_price / option_strike) +
          (riskless_rate +
           0.5f * volatility_rate * volatility_rate) * option_years) /
        (volatility_rate * sqrt_t);
    d2 = d1 - volatility_rate * sqrt_t;

    cnd1 = cnd(d1);
    cnd2 = cnd(d2);

    exp_rt = expf(-riskless_rate * option_years);
    *call = stock_price * cnd1 - option_strike * exp_rt * cnd2;
    *put =
        option_strike * exp_rt * (1.0f - cnd2) - stock_price * (1.0f -
                                                                cnd1);
}

void black_scholes_gold(float *call, float *put, float *stockprice,
                        float *optionstrike, float *optionyears,
                        float riskfree, float volatility, int opt_num)
{
    int i, iMax;

    iMax = opt_num;
    for (i = 0; i < iMax; i++) {
        black_scholes_equation(&call[i], &put[i], stockprice[i],
                               optionstrike[i], optionyears[i], riskfree,
                               volatility);
    }
}

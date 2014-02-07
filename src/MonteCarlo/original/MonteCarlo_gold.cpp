#include <stdio.h>
#include "MonteCarlo_gold.hpp"

void montecarlo_calc(float *call, float *confidence, float S, float X,
                     float T, float R, float V, float *random, int pathN)
{
    float sum = 0.0, sum2 = 0.0;
    float VBySqrtT = V * sqrtf(T);
    float MuByT = (R - 0.5f * V * V) * T;
    int i;
    float r, path, tmp_path;


    for (i = 0; i < pathN; i++) {
        r = random[i];
        tmp_path = S * expf(MuByT + VBySqrtT * r) - X;
        (tmp_path > 0.0f) ? (path = tmp_path) : (path = 0.0f);

        sum += path;
        sum2 += path * path;
    }

    *call = (float) (exp(-R * T) * sum / (float) pathN);

    float stdDev =
        sqrt(((float) pathN * sum2 -
              sum * sum) / ((float) pathN * (float) (pathN - 1)));
    *confidence =
        (float) (exp(-R * T) * 1.96 * stdDev / sqrt((float) pathN));
}


void boxmuller(float *u1, float *u2)
{
    float r = sqrtf(-2.0f * logf(*u1));
    float phi = 2.0f * PI * *u2;

    *u1 = r * cosf(phi);
    *u2 = r * sinf(phi);
}


void boxmuller_calculation(float *random, int pathN)
{
    int i;

    for (i = 0; i < pathN - 1; i += 2) {
        boxmuller(&random[i], &random[i + 1]);
    }
}

void montecarlo_gold(float *call, float *confidence, float *S, float *X,
                     float *T, float R, float V, float *random, int pathN,
                     int optN)
{
    for (int i = 0; i < optN; i++) {
        montecarlo_calc(&call[i],
                        &confidence[i],
                        S[i], X[i], T[i], R, V, random, pathN);
    }
}

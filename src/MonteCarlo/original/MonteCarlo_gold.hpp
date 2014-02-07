#include <math.h>

/* Constants */
#define RSQRT2PI  0.3989422804f
#define PI  3.141592653589793f
#define PI2 6.283185307179586f


void montecarlo_gold(float *call, float *confidence, float *S, float *X,
                     float *T, float R, float V, float *random, int pathN,
                     int optN);
void montecarlo_calc(float *call, float *confidence, float S, float X,
                     float T, float R, float V, float *h_Random,
                     int pathN);
void boxmuller(float *u1, float *u2);
void boxmuller_calculation(float *random, int pathN);

#ifndef __BLACKSCHOLES_HPP
#define __BLACKSCHOLES_HPP

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "BlackScholes_gold.hpp"

/* Constants */
#define coef1     0.31938153f
#define coef2    -0.356563782f
#define coef3     1.781477937f
#define coef4    -1.821255978f
#define coef5     1.330274429f
#define RSQRT2PI  0.3989422804f

/* user parameters */
extern int verbose;
extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class BlackScholes {

public:
    BlackScholes(int _optNum, float _riskFree, float _volatility);
    ~BlackScholes();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

    /* blackscholes variables */
private:

    float *call;
    float *put;
    float *stockPrice;
    float *optionStrike;
    float *optionYears;

    int optNum;
    float riskFree;
    float volatility;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

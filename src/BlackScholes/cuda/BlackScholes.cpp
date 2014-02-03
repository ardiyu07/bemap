#include <iostream>
#include <iomanip>
#include <fstream>

#include "BlackScholes.hpp"

BlackScholesCUDA::BlackScholesCUDA(int _optNum,
                                   float _riskFree, float _volatility)
{
    optNum = _optNum;
    riskFree = _riskFree;
    volatility = _volatility;

    call = NULL;
    put = NULL;
    stockPrice = NULL;
    optionStrike = NULL;
    optionYears = NULL;

    d_call = NULL;
    d_put = NULL;
    d_stockPrice = NULL;
    d_optionStrike = NULL;
    d_optionYears = NULL;
}

BlackScholesCUDA::~BlackScholesCUDA()
{
    /* nothing */
}

void BlackScholesCUDA::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    cudaDeviceReset();
    CUDA_CHECK_ERROR();
    cudaSetDevice(0);
    CUDA_CHECK_ERROR();

    t_init.stop();
}

void BlackScholesCUDA::prep_memory()
{
    t_mem.start();

    srand(time(NULL));

    /* Allocate host memory */
    call = new float[optNum];
    put = new float[optNum];
    stockPrice = new float[optNum];
    optionStrike = new float[optNum];
    optionYears = new float[optNum];
    ERROR_HANDLER((optionYears != NULL || optionStrike != NULL
                   || stockPrice != NULL || put != NULL
                   || call != NULL),
                  "Error in allocation memory for parameters");

    /* Initialize variables */
    for (int i = 0; i < optNum; i++) {
        call[i] = 0.0f;
        put[i] = 0.0f;
        stockPrice[i] = rand_float(10.0f, 100.0f);
        optionStrike[i] = rand_float(1.0f, 100.0f);
        optionYears[i] = rand_float(0.25f, 5.0f);
    }

    cudaMalloc((void**)&d_call, optNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_put, optNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_stockPrice, optNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_optionStrike, optNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_optionYears, optNum * sizeof(float));
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_stockPrice, stockPrice, optNum * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(d_optionStrike, optionStrike, optNum * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(d_optionYears, optionYears, optNum * sizeof(float),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    t_mem.stop();
}

void BlackScholesCUDA::copyback()
{
    t_cpy.start();
    cudaMemcpy(call, d_call, optNum * sizeof(float),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaMemcpy(put, d_put, optNum * sizeof(float), cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    t_cpy.stop();
}

void BlackScholesCUDA::compare_to_cpu()
{
    float *t_call;
    float *t_put;
    float sum_delta_c = 0.0f;
    float sum_delta_p = 0.0f;
    float max_delta_c = 0.0f;
    float max_delta_p = 0.0f;
    float sum_call = 0.0f;
    float sum_put = 0.0f;
    float d_c, d_p, L1c, L1p;
    StopWatch t_cpu;

    if (compResult) {
        t_call = new float[optNum * sizeof(float)];
        t_put = new float[optNum * sizeof(float)];

        verbose
            && std::cerr << "Comparing to CPU (Single thread) results .." <<
            std::endl;
        t_cpu.start();
        black_scholes_gold(t_call, t_put, stockPrice, optionStrike,
                           optionYears, riskFree, volatility, optNum);
        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int i = 0; i < optNum; ++i) {
            sum_call += fabs(t_call[i]);
            sum_put += fabs(t_put[i]);
            d_c = fabs(t_call[i] - call[i]);
            d_p = fabs(t_put[i] - put[i]);
            sum_delta_c += d_c;
            sum_delta_p += d_p;
            max_delta_c = max(max_delta_c, d_c);
            max_delta_p = max(max_delta_p, d_p);
        }

        L1c = sum_delta_c / sum_call;
        L1p = sum_delta_p / sum_put;
        verbose && std::cerr << std::fixed << std::setprecision(20)
                             << "Relative L1 norm (call)    = " << L1c << std::endl
                             << "Relative L1 norm (put)     = " << L1p << std::endl
                             << "Max absolute error (call)  = " << max_delta_c << std::endl
                             << "Max absolute error (put)   = " << max_delta_p << std::endl;
        std::cerr << ((L1c < 1e-6 && L1p < 1e-6) ? ("TEST PASSED.")
                      : ("TEST FAILED.")) << std::endl;

        delete[]t_call;
        delete[]t_put;
    }
}

void BlackScholesCUDA::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;
    if (outName.size() != 0) {
        std::fstream fs(outName.c_str(), std::ios_base::out);
        for (int i = 0; i < optNum; i += 1) {
            fs << std::fixed << std::setprecision(4) << i << ": " << call[i] << " "
               << put[i] << std::endl;
        }
        fs.close();
    }
}

void BlackScholesCUDA::finish()
{
    /* Cleanup */
    cudaDeviceReset();
    CUDA_CHECK_ERROR();

    t_all.stop();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_kernel.print_average_time("Kernel");
    showPrepTime && t_cpy.print_average_time("Memory Copyback");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

void BlackScholesCUDA::clean_mem()
{
    /* Free memory and cl variables */
    t_clean.start();

    delete[]call;
    delete[]put;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    cudaFree(d_call);
    CUDA_CHECK_ERROR();
    cudaFree(d_put);
    CUDA_CHECK_ERROR();
    cudaFree(d_stockPrice);
    CUDA_CHECK_ERROR();
    cudaFree(d_optionStrike);
    CUDA_CHECK_ERROR();
    cudaFree(d_optionYears);
    CUDA_CHECK_ERROR();

    t_clean.stop();
}

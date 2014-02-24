#include <iostream>
#include <iomanip>
#include <fstream>

#include "MonteCarlo.hpp"

extern "C" {

#include "mt19937ar.h"

};



MonteCarloCUDA::MonteCarloCUDA(int _pathNum,
                               int _optNum,
                               float _riskFree, float _volatility)
{
    pathNum = _pathNum;         /// Number of paths (random numbers)
    optNum = _optNum;           /// Number of options 
    riskFree = _riskFree;       /// Risk rate
    volatility = _volatility;   /// Volatility coef

    call = NULL;
    random = NULL;
    confidence = NULL;
    stockPrice = NULL;
    optionStrike = NULL;
    optionYears = NULL;

    d_call = NULL;
    d_random = NULL;
    d_confidence = NULL;
    d_stockPrice = NULL;
    d_optionStrike = NULL;
    d_optionYears = NULL;
}

MonteCarloCUDA::~MonteCarloCUDA()
{
    /* nothing */
}

void MonteCarloCUDA::init()
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

void MonteCarloCUDA::prep_memory()
{
    /* Initialization */
    t_mem.start();

    srand(1);

    /* Allocate host memory */
    random = new float[pathNum];
    call = new float[optNum];
    confidence = new float[optNum];
    stockPrice = new float[optNum];
    optionStrike = new float[optNum];
    optionYears = new float[optNum];
    ERROR_HANDLER((optionYears != NULL || optionStrike != NULL
                   || stockPrice != NULL || confidence != NULL
                   || call != NULL
                   || random != NULL),
                  "Error in allocation memory for parameters");

    /* Initialize variables */
    for (int i = 0; i < optNum; i++) {
        random[i] = 0.0f;
        call[i] = 0.0f;
        confidence[i] = 0.0f;
        stockPrice[i] = rand_float(10.0f, 100.0f);
        optionStrike[i] = rand_float(1.0f, 100.0f);
        optionYears[i] = rand_float(0.25f, 5.0f);
    }

    cudaMalloc((void**)&d_random, pathNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_call, optNum * sizeof(float));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_confidence, optNum * sizeof(float));
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

void MonteCarloCUDA::copyback()
{
    t_cpy.start();
    cudaMemcpy(call, d_call, optNum * sizeof(float),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaMemcpy(confidence, d_confidence, optNum * sizeof(float),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    cudaMemcpy(random, d_random, pathNum * sizeof(float),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    t_cpy.stop();
}

void MonteCarloCUDA::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    if (outName.size() != 0) {
        std::fstream fs(outName.c_str(), std::ios_base::out);

        for (int i = 0; i < optNum; i += 1) {
            fs << std::fixed << std::setprecision(10) << i << ": " << stockPrice[i]
               << " " << optionStrike[i] << " " << optionYears[i] << " "
               << call[i] << " " << confidence[i] << std::endl;
        }

        fs.close();
    }

}

void MonteCarloCUDA::compare_to_cpu()
{
    float *t_call;
    float *t_conf;
    float sum_delta_c = 0.0f;
    float sum_delta_p = 0.0f;
    float max_delta_c = 0.0f;
    float max_delta_p = 0.0f;
    float sum_call = 0.0f;
    float sum_conf = 0.0f;
    float d_c, d_p, L1c, L1p;
    StopWatch t_cpu;

    if (compResult) {
        t_call = new float[optNum * sizeof(float)];
        t_conf = new float[optNum * sizeof(float)];

        verbose
            && std::cerr << "Comparing to CPU (Single thread) results .." <<
            std::endl;
        t_cpu.start();
        montecarlo_gold(t_call, t_conf, stockPrice, optionStrike,
                        optionYears, riskFree, volatility, random, pathNum,
                        optNum);
        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int i = 0; i < optNum; ++i) {
            sum_call += fabs(t_call[i]);
            sum_conf += fabs(t_conf[i]);
            d_c = fabs(t_call[i] - call[i]);
            d_p = fabs(t_conf[i] - confidence[i]);
            sum_delta_c += d_c;
            sum_delta_p += d_p;
            max_delta_c = max(max_delta_c, d_c);
            max_delta_p = max(max_delta_p, d_p);
        }

        L1c = sum_delta_c / sum_call;
        L1p = sum_delta_p / sum_conf;
        verbose && std::cerr << std::fixed << std::setprecision(20)
                             << "# Succeed if precision error below 1e-2" << std::endl
                             << "Relative L1 norm (call)    = " << L1c << std::endl
                             << "Relative L1 norm (conf)    = " << L1p << std::endl
                             << "Max absolute error (call)  = " << max_delta_c << std::endl
                             << "Max absolute error (conf)  = " << max_delta_p << std::endl;
        std::cerr << ((L1c < 1e-2 && L1p < 1e-2) ? ("TEST PASSED.")
                      : ("TEST FAILED.")) << std::endl;

        delete[]t_call;
        delete[]t_conf;
    }
}

void MonteCarloCUDA::clean_mem()
{
    /* Cleanup and Output */
    t_clean.start();

    delete[]random;
    delete[]call;
    delete[]confidence;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    cudaFree(d_random);
    CUDA_CHECK_ERROR();
    cudaFree(d_call);
    CUDA_CHECK_ERROR();
    cudaFree(d_confidence);
    CUDA_CHECK_ERROR();
    cudaFree(d_stockPrice);
    CUDA_CHECK_ERROR();
    cudaFree(d_optionStrike);
    CUDA_CHECK_ERROR();
    cudaFree(d_optionYears);
    CUDA_CHECK_ERROR();

    t_clean.stop();
}

void MonteCarloCUDA::finish()
{
    /* Cleanup */
    cudaDeviceReset();
    CUDA_CHECK_ERROR();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_gr.print_average_time("Kernel_GenRand");
    t_bm.print_average_time("Kernel_BoxMuller");
    t_kernel.print_average_time("Kernel_MonteCarlo");
    showPrepTime && t_cpy.print_average_time("Memory Copyback");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;

    t_all.stop();
}

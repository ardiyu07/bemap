#include <iostream>
#include <iomanip>

#include "MonteCarlo.hpp"

extern "C" {

#include "mt19937ar.h"

};

MonteCarlo::MonteCarlo(int _pathNum,
                       int _optNum, float _riskFree, float _volatility)
{
    pathNum = _pathNum;
    optNum = _optNum;
    riskFree = _riskFree;
    volatility = _volatility;
}

MonteCarlo::~MonteCarlo()
{
    /* nothing */
}

void MonteCarlo::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    // nothing

    t_init.stop();

}

void MonteCarlo::prep_memory()
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
        call[i] = 0.0f;
        confidence[i] = 0.0f;
        stockPrice[i] = rand_float(10.0f, 100.0f);
        optionStrike[i] = rand_float(1.0f, 100.0f);
        optionYears[i] = rand_float(0.25f, 5.0f);
    }

    t_mem.stop();
}

void MonteCarlo::execute()
{

    t_gr.start();
    for (int i = 0; i < pathNum; i++) {
        random[i] = genrand_real2();
    }
    t_gr.stop();

    t_bm.start();
    boxmuller_calculation(random, pathNum);

    t_bm.stop();

    t_kernel.start();
    montecarlo_gold(call, confidence, stockPrice, optionStrike,
                    optionYears, riskFree, volatility, random, pathNum,
                    optNum);
    t_kernel.stop();

}

void MonteCarlo::output(void *param)
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

void MonteCarlo::clean_mem()
{
    /* Cleanup and Output */
    t_clean.start();

    delete[]random;
    delete[]call;
    delete[]confidence;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    t_clean.stop();
}

void MonteCarlo::finish()
{
    t_all.stop();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparation");
    t_gr.print_average_time("Kernel (Random Numbers Generation)");
    t_bm.print_average_time("Kernel (Box Muller)");
    t_kernel.print_average_time("Kernel (Monte Carlo)");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

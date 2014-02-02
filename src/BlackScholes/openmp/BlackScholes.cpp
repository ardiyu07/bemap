#include <iostream>
#include <iomanip>

#include "BlackScholes.hpp"

BlackScholes::BlackScholes(int _optNum, float _riskFree, float _volatility)
{
    optNum = _optNum;
    riskFree = _riskFree;
    volatility = _volatility;
}

BlackScholes::~BlackScholes()
{
    /* nothing */
}

void BlackScholes::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    // nothing

    t_init.stop();

}

void BlackScholes::prep_memory()
{
    /* Initialization */
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

    t_mem.stop();
}

void BlackScholes::execute()
{
    /* Kernel Execution */
    t_kernel.start();
    black_scholes_gold(call, put, stockPrice, optionStrike, optionYears,
                       riskFree, volatility, optNum);
    t_kernel.stop();
}

void BlackScholes::output(void *param)
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

void BlackScholes::clean_mem()
{
    /* Cleanup and Output */
    t_clean.start();

    delete[]call;
    delete[]put;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    t_clean.stop();
}

void BlackScholes::finish()
{
    t_all.stop();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparation");
    t_kernel.print_average_time("Kernel : black_scholes");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

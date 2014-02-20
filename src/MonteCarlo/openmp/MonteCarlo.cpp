/**
 * Copyright (c) 2012, Fixstars Corp.
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are 
 * met:
 *     * Redistributions of source code must retain the above copyright and 
 *       patent notices, this list of conditions and the following 
 *       disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in 
 *       the documentation and/or other materials provided with the 
 *       distribution.
 *     * Neither the name of Fixstars Corp. nor the names of its 
 *       contributors may be used to endorse or promote products derived 
 *       from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS 
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED 
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT 
 * HOLDER BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR 
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */

/**
 * @file   MonteCarlo.cpp
 * @author Yuri Ardila <y_ardila@fixstars.com>
 * @date   Tue Oct 30 16:01:19 JST 2012
 * 
 * @brief  
 *    MonteCarlo
 *    Reference Single Thread C++
 *
 */

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

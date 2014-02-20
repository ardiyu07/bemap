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
 * @file   MonteCarlo.hpp
 * @author Yuri Ardila <y_ardila@fixstars.com>
 * @date   Tue Oct 30 16:01:19 JST 2012
 * 
 * @brief  
 *    MonteCarlo
 *    Reference Single Thread C++
 *
 */

#ifndef __BLACKSCHOLES_HPP
#define __BLACKSCHOLES_HPP

#pragma once

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MonteCarlo_gold.hpp"

/* user parameters */
extern int verbose;
extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class MonteCarlo {

public:
    MonteCarlo(int _pathNum,
               int _optNum, float _riskFree, float _volatility);

    ~MonteCarlo();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

    /* MonteCarlo variables */
private:

    float *call;
    float *random;
    float *confidence;
    float *stockPrice;
    float *optionStrike;
    float *optionYears;

    int pathNum;
    int optNum;
    float riskFree;
    float volatility;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_gr, t_bm, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

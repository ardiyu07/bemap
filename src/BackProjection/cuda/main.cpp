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
 * @file   main.cpp
 * @author Yuri Ardila <y_ardila@fixstars.com>
 * @date   Tue Oct 30 16:01:19 JST 2012
 * 
 * @brief  
 *    BackProjection
 *    CUDA Implementation
 *
 */

#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "BackProjection.hpp"

#define N_ITER 5

#define DEFAULT_INPUT "../common/data/dla.ascii.pgm"

int nThreads = 64;
int nBlocks = -1;

std::string outputFilename = "";

/**************************/
/* @brief User parameters */
/**************************/
int verbose      = false;
int showPrepTime = false;
int isPGM        = false;

int rows = 1024;
int columns = 1024;

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "output",          required_argument,      NULL,              'o' },
    { "rows",            required_argument,      NULL,              'r' },
    { "columns",         required_argument,      NULL,              'c' },
    { "prep-time",       no_argument,            &showPrepTime,     true},
    { NULL,              0,                      NULL,               0  }
};

/***************************/
/* @name help              */
/* @brief Print help       */
/* @param filename argv[0] */
/***************************/
void help(const std::string & filename)
{
    std::cout
        << filename
        << " [--verbose|-v] [--help|-h] [--output|-o FILENAME]" << std::endl
        << "     [--kernel|-k NUMBER] [--prep-time]" << std::endl
        << "     FILENAME"
        << std::endl << std::endl
        << "* Options *" << std::endl
        << " --verbose                  Be verbose"<< std::endl
        << " --help                     Print this message"<< std::endl
        << " --rows=NUMBER              Number of rows in the data array -- default = 1024" << std::endl
        << " --columns=NUMBER      		Number of columns in the data array -- default = 1024" << std::endl
        << " --output=NAME              Write results to this file"<<std::endl
        << " --prep-time                Show initialization, memory preparation and copyback time"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -r 512 -c 512 -v" << std::endl
        << std::endl;

    exit(0);
}

/**********************************/
/* @name option                   */
/* @brief Process user parameters */
/* @param ac argc                 */
/*        av argv                 */
/**********************************/
void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:r:c:", longopts, NULL)) != -1) {
        switch (opt) {

        case '?':
            ERROR_HANDLER(0,
                          "Invalid option '" +
                          std::string(av[optind - 1]) + "'");
            break;

        case ':':
            ERROR_HANDLER(0,
                          "Missing argument of option '" +
                          std::string(av[optind - 1]) + "'");
            break;

        case 'v':
            verbose = true;
            break;

        case 'h':
            help(av[0]);
            break;

        case 'o':
            outputFilename = std::string(optarg);
            break;

        case 'r':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; rows = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        case 'c':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; columns = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;


        case 0:
            break;

        default:
            ERROR_HANDLER(0, "Error parsing arguments");
        }
    }
}

/*********************/
/* @name main        */
/* @brief main       */
/* @param argc, argv */
/*********************/

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "BACKPROJECTION, CUDA Implementation"
                         << std::endl << std::endl
						 << "Width                   = " << rows << std::endl 
						 << "Height                  = " << columns << std::endl 
						 << "Show prep time          = " << ((showPrepTime) ? ("True") : ("False")) << std::endl 
						 << "Executing .. " << std::endl;

    BackProjectionCUDA BackProjectionCUDA(rows, columns);

    BackProjectionCUDA.init();
    for (int i = 0; i < N_ITER; i++) {
        BackProjectionCUDA.prep_memory();
        BackProjectionCUDA.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        if (outputFilename.size() == 0)
            outputFilename = "BackProjection_cuda";
        verbose
            && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        BackProjectionCUDA.output(reinterpret_cast < void *>(&output));

        BackProjectionCUDA.clean_mem();
    }
    BackProjectionCUDA.finish();

    return 0;
}

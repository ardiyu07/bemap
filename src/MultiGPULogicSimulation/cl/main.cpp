#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>

#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MultiGPULogicSimulation.hpp"

#define N_ITER 5

/* User parameters */
int verbose      = false;
int showPrepTime = false;

/* MultiGPULogicSimulation parameters */
float alpha = 0.02f;
float beta  = 0.30f;
int partNum = 1 << 20;

std::string outputFilename;

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "output",          required_argument,      NULL,              'o' },
    { "alpha",           required_argument,      NULL,              'A' },
    { "beta",        	 required_argument,      NULL,              'B' },
    { "partnum",         required_argument,      NULL,              'N' },
    { "prep-time",       no_argument,            &showPrepTime,     true},
    { NULL,              0,                      NULL,               0  }
};

void help(const std::string & filename)
{
    std::cout
        << filename
        << " [--verbose|-v] [--help|-h] [--output|-o FILENAME]" << std::endl
        << "     [--alpha|-A NUMBER] [--beta|-B NUMBER] [--partnum|-N INT]" << std::endl
        << "     [--prep-time]" << std::endl
        << std::endl
        << "* Options *" << std::endl
        << " --verbose             Be verbose"<< std::endl
        << " --help                Print this message"<<std::endl
        << " --output=NAME         Write to this file"<<std::endl
        << " --alpha=NUMBER        Alpha for dummy calculation -- default - 0.02"<< std::endl
        << " --beta=NUMBER         Beta for dummy calculation -- default = 0.30"<< std::endl
        << " --num=INT             Number of elements -- default = 1M Element"<< std::endl
        << " --prep-time           Show initialization, memory preparation and copyback time"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -v" << std::endl
        << filename << " [OPTS...] -v -A 1.50" << std::endl
        << std::endl;
    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:A:B:N:", longopts, NULL)) != -1) {
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

        /* Alpha */
        case 'A':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a;
            alpha = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        /* Beta */
        case 'B':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a;
            beta = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        /* partNum */
        case 'N':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a;
            partNum = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        case 0:
            break;

        default:
            ERROR_HANDLER(0, "Error parsing arguments");
        }
    }
}

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "MultiGPULogicSimulation, CPU Single Thread Implementation"
                         << std::endl << std::endl
                         << "Alpha          = " << alpha << std::endl
                         << "Beta           = " << beta << std::endl
                         << "N Elements     = " << partNum << std::endl
                         << "Show prep time = " << ((showPrepTime)?("True"):("False")) << std::endl
                         << "Executing .. " << std::endl;

    MultiGPULogicSimulation MultiGPULogicSimulation(alpha, beta, partNum);

    MultiGPULogicSimulation.init();
    for (int i = 0; i < N_ITER; i++) {
        MultiGPULogicSimulation.prep_memory();
        MultiGPULogicSimulation.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        MultiGPULogicSimulation.output(reinterpret_cast < void *>(&output));

        MultiGPULogicSimulation.clean_mem();
    }
    MultiGPULogicSimulation.finish();

    return 0;
}

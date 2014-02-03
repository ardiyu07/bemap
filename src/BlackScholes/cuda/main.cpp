#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "BlackScholes.hpp"

#define N_ITER 5

int nThreads = 256;
int nBlocks = -1;

/* user parameters */
int verbose = false;
int showPrepTime = false;
int compResult = false;

/* BlackScholes parameters */
int optNum = 10 * 1024 * 1024;
float riskFree = 0.02f;
float volatility = 0.30f;

std::string outputFilename = "";

/* Options long names */
static struct option longopts[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"help", no_argument, NULL, 'h'},
    {"output", required_argument, NULL, 'o'},
    {"optnum", required_argument, NULL, 'O'},
    {"riskfree", required_argument, NULL, 'R'},
    {"volatility", required_argument, NULL, 'V'},
    {"prep-time", no_argument, &showPrepTime, true},
    {"comp-result", no_argument, &compResult, true},
    {NULL, 0, NULL, 0}
};

void help(const std::string & filename)
{
    std::cout << filename
              << " [--verbose|-v] [--help|-h] [--output|-o FILENAME]" << std::endl
              <<
        "     [--optnum|-O NUMBER] [--riskfree|-R NUMBER] [--volatility|-V NUMBER]"
              << std::endl << "     [--prep-time] [--comp-result]" << std::endl << std::endl <<
        "* Options *" << std::endl << " --verbose             Be verbose" <<
        std::endl << " --help                Print this message" << std::endl <<
        " --output=NAME         Write results to this file" << std::endl <<
        " --optnum=NUMBER       Number of elements in the data array -- default = 50 * 1024 * 1024"
              << std::endl <<
        " --riskfree=NUMBER     The annualized risk-free interest rate, continuously compounded -- default = 0.02"
              << std::endl <<
        " --volatility=NUMBER   The volatility of stock's returns -- default = 0.30"
              << std::endl <<
        " --prep-time           Show initialization, memory preparation and copyback time"
              << std::endl << " --comp-result         Compare GPU and CPU results" <<
        std::endl << std::endl << " * Examples *" << std::endl << filename <<
        " [OPTS...] -v -k 0" << std::endl << filename <<
        " [OPTS...] -v --workitems=128" << std::endl << std::endl;

    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:O:R:V:", longopts, NULL)) != -1) {
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

            /* Verbose */
        case 'v':
            verbose = true;
            break;

            /* Help */
        case 'h':
            help(std::string(av[0]));
            break;

            /* Output to file */
        case 'o':
            outputFilename = std::string(optarg);
            break;

            /* optNum */
        case 'O':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a;
            optNum = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        /* riskFree */
        case 'R':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a;
            riskFree = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        /* volatility */
        case 'V':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a;
            volatility = a;
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
        }
        break;

        case 0:
            break;

        default:
            ERROR_HANDLER(0, "Error: parsing arguments");
        }
    }
}

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "BLACK SCHOLES, CUDA Implementation"
                         << std::endl << std::endl
                         << "Number of options       = " << optNum << std::endl
                         << "Riskfree rate           = " << riskFree << std::endl
                         << "Volatility              = " << volatility << std::endl
                         << "Show prep time          = " << ((showPrepTime) ? ("True")
                                                             : ("False")) << std::endl <<
        "Compare to CPU results  = " << ((compResult) ? ("True")
                                         : ("False")) << std::endl <<
        "Executing .. " << std::endl;


    BlackScholesCUDA BlackScholesCUDA(optNum, riskFree, volatility);

    BlackScholesCUDA.init();
    for (int i = 0; i < N_ITER; i++) {
        BlackScholesCUDA.prep_memory();
        BlackScholesCUDA.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;
        BlackScholesCUDA.copyback();

        BlackScholesCUDA.compare_to_cpu();

        outParam output;
        output.outputFilename = outputFilename;
        BlackScholesCUDA.output(reinterpret_cast < void *>(&output));

        BlackScholesCUDA.clean_mem();
    }
    BlackScholesCUDA.finish();

    return 0;
}

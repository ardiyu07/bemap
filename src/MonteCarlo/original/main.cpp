#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>

#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MonteCarlo.hpp"

#define N_ITER 5

/* User parameters */
int verbose        = false;
int showPrepTime   = false;

/* MonteCarlo parameters */
int   pathNum      = 8 * 1024 * 1024;
int   optNum       = 128;
float riskFree     = 0.02f;
float volatility   = 0.30f;

std::string outputFilename = "MonteCarlo_ref_out.dat";

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "output",          required_argument,      NULL,              'o' },
    { "pathnum",         required_argument,      NULL,              'P' },
    { "optnum",          required_argument,      NULL,              'O' },
    { "riskfree",        required_argument,      NULL,              'R' },
    { "volatility",      required_argument,      NULL,              'V' },
    { "prep-time",       no_argument,            &showPrepTime,     true},
    { NULL,              0,                      NULL,               0  }
};

void help(const std::string &  filename)
{
    std::cout
        << filename
        << " [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--pathnum|-P NUMBER]" << std::endl
        << "     [--optnum|-O NUMBER] [--riskfree|-R NUMBER] [--volatility|-V NUMBER]" << std::endl
        << "     [--prep-time]" << std::endl
        << std::endl
        << "* Options *" << std::endl
        << " --verbose             Be verbose"<< std::endl
        << " --help                Print this message"<<std::endl
        << " --output=NAME         Write to this file"<<std::endl
        << " --pathnum=NUMBER      Number of paths -- default = 8 * 1024 * 1024"<< std::endl
        << " --optnum=NUMBER       Number of elements in the data array -- default = 128"<< std::endl
        << " --riskfree=NUMBER     The annualized risk-free interest rate, continuously compounded -- default = 0.02"<< std::endl
        << " --volatility=NUMBER   The volatility of stock's returns -- default = 0.30"<< std::endl
        << " --prep-time           Show initialization, memory preparation and copyback time"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -v" << std::endl
        << filename << " [OPTS...] -v -O 4194304" << std::endl
        << std::endl;
    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:O:P:R:V:", longopts, NULL)) != -1) {
        switch (opt) {

        case '?' :
            ERROR_HANDLER(0, "Invalid option '" + std::string(av[optind-1]) + "'" );
            break;

        case ':' :
            ERROR_HANDLER(0, "Missing argument of option '" + std::string(av[optind-1]) + "'");
            break;

        case 'v' :
            verbose = true;
            break;

        case 'h' :
            help(av[0]);
            break;

        case 'o' :
            outputFilename = std::string(optarg);
            break;

            /* optNum */
        case 'O':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; optNum = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        /* optNum */
        case 'P':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; pathNum = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        /* riskFree */
        case 'R':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a; riskFree = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        /* volatility */
        case 'V':
        {
            std::istringstream iss(optarg);
            float a = -1;
            iss >> a; volatility = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        case 0:
            break;

        default :
            ERROR_HANDLER(0, "Error parsing arguments");
        }
    }
}

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "MONTECARLO, CPU Single Thread Implementation"
                         << std::endl << std::endl
                         << "Number of paths         = " << pathNum << std::endl
                         << "Number of options       = " << optNum << std::endl
                         << "Riskfree rate           = " << riskFree << std::endl
                         << "Volatility              = " << volatility << std::endl
                         << "Show prep time          = " << ((showPrepTime)?("True"):("False")) << std::endl
                         << "Executing .. " << std::endl;

    MonteCarlo MonteCarlo(pathNum, optNum, riskFree, volatility);

    MonteCarlo.init();
    for (int i = 0; i < N_ITER; i++) {
        MonteCarlo.prep_memory();
        MonteCarlo.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        MonteCarlo.output(reinterpret_cast<void*>(&output));

        MonteCarlo.clean_mem();
    }
    MonteCarlo.finish();

    return 0;
}

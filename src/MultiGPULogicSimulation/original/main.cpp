#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>

#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MultiGPULogicSimulation.hpp"

#define DEFAULT_INPUT "../common/data/Blif/alu2.blif"

#define N_ITER 1

using namespace std;

/* User parameters */
int verbose      = false;
int showPrepTime = false;

/* MultiGPULogicSimulation parameters */
std::string inputFile;

std::string outputFilename;

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "output",          required_argument,      NULL,              'o' },
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

        case 0:
            break;

        default:
            ERROR_HANDLER(0, "Error parsing arguments");
        }
    }

    ac -= optind;
    av += optind;

    if (ac == 0) {
        std::cout << "Executing with default input: " << DEFAULT_INPUT << std::endl;
        inputFile = DEFAULT_INPUT;
        return;
    }

    ERROR_HANDLER((outputFilename.size() == 0
                   || ac <= 1),
                  "Error: --output cannot be used with multiple files");

    inputFile = av[0];
    ac--;
    ERROR_HANDLER((ac == 0), "Error: Too many inputs");
}

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "MultiGPULogicSimulation, CPU Single Thread Implementation"
                         << std::endl << std::endl
                         << "Show prep time = " << ((showPrepTime)?("True"):("False")) << std::endl
                         << "Executing .. " << std::endl;

    MultiGPULogicSimulation MultiGPULogicSimulation(inputFile);

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

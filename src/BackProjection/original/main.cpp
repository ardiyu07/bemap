#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "BackProjection.hpp"

#define N_ITER 5

/* User parameters */
int verbose      = false;
int showPrepTime = false;
int isPGM        = false;

std::string outputFilename;

int rows = 1024;
int columns = 1024;

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "rows",            required_argument,      NULL,              'r' },
    { "columns",         required_argument,      NULL,              'c' },
    { "prep-time",       no_argument,            &showPrepTime,     true},
    { NULL,              0,                      NULL,               0  }
};

void help(const std::string & filename)
{
    std::cout
        << filename
        << " [--verbose|-v] [--help|-h]" << std::endl
        << "     [--kernel|-k NUMBER] [--rows|-r NUMBER] [--columns|-c NUMBER]" << std::endl
        << "     [--prep-time]" << std::endl
        << std::endl
        << "* Options *" << std::endl
        << " --verbose             Be verbose"<< std::endl
        << " --help                Print this message"<< std::endl
        << " --rows=NUMBER         Number of rows in the data array -- default = 1024" << std::endl
        << " --columns=NUMBER      Number of columns in the data array -- default = 1024" << std::endl
        << " --prep-time           Show initialization, memory preparation and copy_back time"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -v -r 512 -c 512" << std::endl
        << filename << " [OPTS...] -v --workitems=128" << std::endl
        << std::endl;

    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vhr:c:", longopts, NULL)) != -1) {
        switch (opt) {

        case '?' :
            ERROR_HANDLER(0, "Invalid option '" + std::string(av[optind-1]) + "'");
            break;

        case ':' :
            ERROR_HANDLER(0, "Missing argument of option '" + std::string(av[optind-1]) + "'");
            break;

            /* Verbose */
        case 'v' :
            verbose = true;
            break;

            /* Help */
        case 'h' :
            help(std::string(av[0]));
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

        case 0 :
            break;

        default :
            ERROR_HANDLER(0, "Error: parsing arguments");
        }
    }
}

int main(int argc, char **argv)
{
    /* Parse user input */
    option(argc, argv);

    verbose && std::cerr << "BACKPROJECTION, CPU Single Thread Implementation"
                         << std::endl << std::endl
                         << "Number of rows          = " << rows << std::endl
                         << "Number of columns       = " << columns << std::endl
                         << "Show prep time          = " << ((showPrepTime)?("True"):("False")) << std::endl 
                         << "Executing .. " << std::endl;

    BackProjection BackProjection(rows, columns);

    BackProjection.init();
    for (int i = 0; i < N_ITER; i++) {
        BackProjection.prep_memory();
        BackProjection.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        if (outputFilename.size() == 0)
            outputFilename = "BackProjection_ref_out";        /* Produce [filename]_out */
        verbose
            && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        BackProjection.output(reinterpret_cast < void *>(&output));

        BackProjection.clean_mem();
    }
    BackProjection.finish();

    return 0;
}

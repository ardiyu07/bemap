#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "GrayScale.hpp"

#define N_ITER 5

#define DEFAULT_INPUT "../common/data/newton.ppm"

/* User parameters */
int verbose      = false;
int showPrepTime = false;
int isPGM        = false;

std::string outputFilename;
std::string inputFile;
std::string realName;
std::string prefix;

int height;
int width;

/* Options long names */
static struct option longopts[] = {
    { "verbose",                  no_argument,            NULL,              'v' },
    { "help",                     no_argument,            NULL,              'h' },
    { "output",                   required_argument,      NULL,              'o' },
    { "prep-time",                no_argument,            &showPrepTime,     true},
    { NULL,                       0,                      NULL,               0  }
};

void help(const std::string & filename)
{
    std::cout
        << filename
        << " [--verbose|-v] [--help|-h] [--output|-o FILENAME] [--prep-time]" << std::endl
        << "     FILENAME"
        << std::endl << std::endl
        << "* Options *" << std::endl
        << " --verbose                  Be verbose"<< std::endl
        << " --help                     Print this message"<< std::endl
        << " --output=NAME              Write results to this file"<<std::endl
        << " --prep-time                Show initialization, memory preparation and copyback time"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -v test_data.ppm" << std::endl
        << filename << " [OPTS...] --output=test_output.ppm test_data.ppm" << std::endl
        << std::endl;

    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:", longopts, NULL)) != -1) {
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

        case 0:
            break;

        default :
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

    ERROR_HANDLER((ac!=0), "No file input specified");
    ERROR_HANDLER((outputFilename.size() == 0 || ac <= 1), "--output cannot be used with multiple files");

    inputFile = av[0];
    ac--;
    ERROR_HANDLER((ac==0), "Error: Too many inputs");
}

int main(int argc, char **argv)
{
    imgStream inp;

    /* Parse user input */
    option(argc, argv);
    realName = extract_filename(inputFile, prefix);
    read_pgpm(inputFile, inp, isPGM);
    ERROR_HANDLER((!isPGM), "Error: The input is already a gray-scaled image");

    verbose && std::cerr << "GRAYSCALE, CPU Single Thread Implementation"
                         << std::endl << std::endl
                         << "Image name              = " << inputFile << std::endl
                         << "Width                   = " << width << std::endl
                         << "Height                  = " << height << std::endl
                         << "Show prep time          = " << ((showPrepTime)?("True"):("False")) << std::endl
                         << "Executing .. " << std::endl;

    GrayScale GrayScale(inp);

    for (int i = 0; i < N_ITER; i++) {
        GrayScale.init();
        GrayScale.prep_memory();
        GrayScale.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        if (outputFilename.size() == 0) outputFilename = "GrayScale_ref_" + realName + "_out" + prefix; /* Produce [filename]_out */
        verbose && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        GrayScale.output(reinterpret_cast<void*>(&output));

        GrayScale.finish();
    }
    return 0;
}

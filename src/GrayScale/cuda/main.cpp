#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "GrayScale.hpp"

#define N_ITER 5

#define DEFAULT_INPUT "../common/data/newton.ppm"



int nThreads = 256;
int nBlocks = -1;

/* User parameters */
int verbose = false;
int showPrepTime = false;
int showDevInfo = false;
int compResult = false;
int isPGM = false;
int choose = false;

std::string inputFile;
std::string realName;
std::string prefix;
std::string outputFilename = "";

/* Options long names */
static struct option longopts[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"help", no_argument, NULL, 'h'},
    {"output", required_argument, NULL, 'o'},
    {"prep-time", no_argument, &showPrepTime, true},
    {"comp-result", no_argument, &compResult, true},
    {NULL, 0, NULL, 0}
};

void help(const std::string & filename)
{
    std::cout << filename
              << " [--verbose|-v] [--help|-h] [--output|-o FILENAME]" << std::endl
              << "     [--prep-time] [--comp-result]" << std::endl
              << "     FILENAME"
              << std::endl << std::endl
              << "* Options *" << std::endl
              << " --verbose                  Be verbose" << std::endl
              << " --help                     Print this message" << std::endl
              << " --output=NAME              Write results to this file" << std::endl
              <<
        " --prep-time                Show initialization, memory preparation and copyback time"
              << std::endl <<
        " --comp-result              Compare native and OpenCL results" <<
        std::endl << std::endl << " * Examples *" << std::endl << filename <<
        " [OPTS...] -v -w 256 test_data.ppm" << std::endl << filename <<
        " [OPTS...] --output=test_output.ppm test_data.ppm" << std::endl <<
        std::endl;

    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:", longopts, NULL)) != -1) {
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

        case 0:
            break;

        default:
            ERROR_HANDLER(0, "Error: parsing arguments");
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
    imgStream inp;

    /* Parse user input */
    option(argc, argv);

    /* Read image */
    realName = extract_filename(inputFile, prefix);
    read_pgpm(inputFile, inp, isPGM);
    ERROR_HANDLER((!isPGM),
                  "Error: The input is already a gray-scaled image");

    verbose && std::cerr << "GRAYSCALE, CUDA Implementation"
                         << std::endl
                         << "Image name              = " << inputFile << std::endl
                         << "Width                   = " << inp.width << std::endl
                         << "Height                  = " << inp.height << std::endl
                         << "Show prep time          = " << ((showPrepTime) ? ("True")
                                                             : ("False")) << std::endl <<
        "Compare results         = " << ((compResult) ? ("True")
                                         : ("False")) << std::endl <<
        "Executing .. " << std::endl;

    GrayScaleCUDA GrayScaleCUDA(inp);

    GrayScaleCUDA.init();
    for (int i = 0; i < N_ITER; i++) {
        GrayScaleCUDA.prep_memory();
        GrayScaleCUDA.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        GrayScaleCUDA.copyback();
        GrayScaleCUDA.compare_to_cpu();

        if (outputFilename.size() == 0)
            outputFilename = "GrayScale_cuda_" + realName + "_out" + prefix;    /* Produce [filename]_out */
        verbose
            && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        GrayScaleCUDA.output(reinterpret_cast < void *>(&output));

        GrayScaleCUDA.clean_mem();
    }
    GrayScaleCUDA.finish();

    return 0;
}

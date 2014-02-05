#include <iostream>
#include <string>
#include <cmath>
#include <iomanip>

#include "GaussianFilter.hpp"

#define N_ITER 5

#define DEFAULT_INPUT "../common/data/dla.ascii.pgm"

int nThreads = 256;
int nBlocks = -1;

kernelVersion kernelVer = SCALAR;
std::string outputFilename = "";

/* User parameters */
int verbose = false;
int showPrepTime = false;
int showDevInfo = false;
int compResult = false;
int isPGM = false;
int isFast = false;
int choose = false;

std::string inputFile;
std::string realName;
std::string prefix;

int width;
int height;

/* Options long names */
static struct option longopts[] = {
    {"verbose", no_argument, NULL, 'v'},
    {"help", no_argument, NULL, 'h'},
    {"output", required_argument, NULL, 'o'},
    {"kernel", required_argument, NULL, 'k'},
    {"prep-time", no_argument, &showPrepTime, true},
    {"comp-result", no_argument, &compResult, true},

    {NULL, 0, NULL, 0}
};

void help(const std::string & filename)
{
    std::cout << filename
              << " [--verbose|-v] [--help|-h] [--output|-o FILENAME]" << std::endl
              << "     [--kernel|-k NUMBER] [--prep-time] [--comp-result]" <<
        std::endl << "     FILENAME" << std::endl << std::endl << "* Options *" << std::endl <<
        " --verbose                  Be verbose" << std::endl <<
        " --help                     Print this message" << std::endl <<
        " --output=NAME              Write results to this file" << std::endl <<
        " --kernel=KERNEL            Kernel mode (0, 1, 2) -- default = 0"
              << std::endl << "                                      [0] Scalar" <<
        std::endl <<
        "                                      [1] Scalar Fast no Shared Memory (Using convolution separable matrix)"
              << std::endl <<
        "                                      [2] Scalar Fast with Shared Memory (Using convolution separable matrix)"
              << std::endl <<
        " --prep-time                Show initialization, memory preparation and copyback time"
              << std::endl <<
        " --comp-result              Compare native and OpenCL results" <<
        std::endl << std::endl << " * Examples *" << std::endl << filename <<
        " [OPTS...] -v -w 256 test_data.pgm" << std::endl << filename <<
        " [OPTS...] --output=test_output.ppm test_data.ppm" << std::endl <<
        std::endl;

    exit(0);
}

void option(int ac, char **av)
{
    if (ac == 1) std::cout << av[0] << ": Execute with default parameter(s)..\n(--help for program usage)\n\n";
    int opt;
    while ((opt = getopt_long(ac, av, "vho:k:", longopts, NULL)) != -1) {
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

            /* Kernel mode */
        case 'k':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a;
            kernelVer = kernelVersion(a);
            ERROR_HANDLER((!iss.fail()),
                          "Invalid argument '" + std::string(optarg) +
                          "'");
            ERROR_HANDLER((kernelVer >= 0
                           && kernelVer <= 5),
                          "Invalid kernel mode: '" +
                          std::string(optarg) + "'");
        }
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

    verbose && std::cerr << "GAUSSIAN BLUR, CUDA Implementation"
                         << std::endl << std::endl
                         << "Image name              = " << inputFile << std::endl
                         << "Type of image           = " << ((isPGM) ? ("BW") : ("RGB")) <<
        std::endl << "Width                   = " << width << std::endl <<
        "Height                  = " << height << std::endl <<
        "Kernel mode             = " << kernelStr[kernelVer] << std::endl <<
        "Show prep time          = " << ((showPrepTime) ? ("True")
                                         : ("False")) << std::endl <<
        "Compare results         = " << ((compResult) ? ("True")
                                         : ("False")) << std::endl <<
        "Executing .. " << std::endl;

    GaussianFilterCUDA GaussianFilterCUDA(inp);

    GaussianFilterCUDA.init();
    for (int i = 0; i < N_ITER; i++) {
        GaussianFilterCUDA.prep_memory();
        GaussianFilterCUDA.execute();
        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        GaussianFilterCUDA.copyback();
        GaussianFilterCUDA.compare_to_cpu();

        if (outputFilename.size() == 0)
            outputFilename = "GaussianFilter_cuda_" + realName + "_out" + prefix;       /* Produce [filename]_out */
        verbose
            && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        GaussianFilterCUDA.output(reinterpret_cast < void *>(&output));

        GaussianFilterCUDA.clean_mem();
    }
    GaussianFilterCUDA.finish();

    return 0;
}

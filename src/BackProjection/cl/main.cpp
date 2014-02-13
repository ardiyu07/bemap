#include <iostream>

#include "BackProjection.hpp"

#define N_ITER 5

/* cl variables */
std::string sourceName("BackProjection.cl");
std::string compileOptions;

/* Constants */
std::string    platformId = "";
cl_device_type deviceType = CL_DEVICE_TYPE_CPU;
kernelVersion  kernelVer  = SCALAR;
std::string outputFilename     = "";
size_t globalWorkSize[3] = { -1, -1, -1};
size_t localWorkSize[3] = { 256, -1, -1};
size_t nLocals = 64;
size_t nGlobals = -1;

/**************************/
/* @brief User parameters */
/**************************/
int  verbose        = false;
int  showPrepTime   = false;
int  showDevInfo    = false;
int  choose         = false;
int  naive          = false;

int rows = 1024;
int columns = 1024;

/* Options long names */
static struct option longopts[] = {
    { "verbose",         no_argument,            NULL,              'v' },
    { "help",            no_argument,            NULL,              'h' },
    { "kernel",          required_argument,      NULL,              'k' },
    { "rows",            required_argument,      NULL,              'r' },
    { "columns",         required_argument,      NULL,              'c' },
    { "workitems",       required_argument,      NULL,              'w' },
    { "use-gpu",         no_argument,            NULL,              'g' },
    { "choose-dev",      no_argument,            NULL,              'd' },
    { "choose-plat",     required_argument,      NULL,              'p' },
    { "dev-info",        no_argument,            &showDevInfo,      true},
    { "prep-time",       no_argument,            &showPrepTime,     true},
    { "naive",           no_argument,            NULL,              'n'},
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
        << " [--verbose|-v] [--help|-h] [--naive|-n]" << std::endl
        << "     [--kernel|-k NUMBER] [--rows|-r NUMBER] [--columns|-c NUMBER]" << std::endl
        << "     [--workitems|-w NUMBER]" << std::endl
        << "     [--use-gpu|-g] [--choose-dev|-d] [--choose-plat|-p DEV]" << std::endl
        << "     [--dev-info] [--prep-time] [--comp-result]" << std::endl
        << std::endl
        << "* Options *" << std::endl
        << " --verbose             Be verbose"<< std::endl
        << " --help                Print this message"<< std::endl
        << " --naive               Sequential execution"<< std::endl
        << " --kernel=KERNEL       Kernel mode (0, 1) -- default = 0"<< std::endl
        << "                                 [0] Scalar" << std::endl
        << "                                 [1] SIMD = Single Instruction Multiple Data" << std::endl
        << " --rows=NUMBER         Number of rows in the data array -- default = 1024" << std::endl
        << " --columns=NUMBER      Number of columns in the data array -- default = 1024" << std::endl
        << " --workitems=NUMBER    Number of (local) workitems for Scalar mode"<<std::endl
        << " --use-gpu             Use GPU as the CL device"<<std::endl
        << " --choose-dev          Choose which OpenCL device to use"<<std::endl
        << " --choose-plat=DEV     Choose which OpenCL platform to use (0, 1, 2)"<<std::endl
        << "                                  [0] Advanced Micro Devices, Inc." << std::endl
        << "                                  [1] NVIDIA Corporation" << std::endl
        << "                                  [2] Intel(R) Corporation" << std::endl
        << "                                  default: Any CPU device"<< std::endl
        << " --dev-info            Show Device Info"<<std::endl
        << " --prep-time           Show initialization, memory preparation and copy_back time"<<std::endl
        << " --comp-result         Compare GPU and CPU results"<<std::endl
        << std::endl
        << " * Examples *" << std::endl
        << filename << " [OPTS...] -v -k 0" << std::endl
        << filename << " [OPTS...] -v --workitems=128" << std::endl
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
    while ((opt = getopt_long(ac, av, "vhk:r:c:w:gdp:nT:", longopts, NULL)) != -1) {
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

            /* Kernel mode */
        case 'n' : 
        {
            naive = true;
        }
        break;

        case 'k':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; kernelVer = kernelVersion(a);
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
            ERROR_HANDLER((kernelVer >= 0 && kernelVer <= 3),
                          "Invalid kernel mode: '" + std::string(optarg) + "'");
        }
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

        /* Number of workitems */
        case 'w':
        {
            std::istringstream iss(optarg);
            int a = -1;
            iss >> a; nLocals = a;
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
        }
        break;

        /* Use GPU if invoked */
        case 'g' :
            deviceType = CL_DEVICE_TYPE_GPU;
            break;

            /* Choose which OpenCL device to use */
        case 'd' : 
        {
            choose = true;
            showDevInfo = true;
        }
        break;

        /* Choose which OpenCL platform to use */
        case 'p' :
        {
            std::istringstream iss(optarg);
            platforms platformNum;
            int a = -1;
            iss >> a; platformNum = platforms(a);
            ERROR_HANDLER((!iss.fail()), "Invalid argument '" + std::string(optarg) + "'");
            ERROR_HANDLER((platformNum >= 0 && platformNum <= 2),
                          "Invalid platform: '" + std::string(optarg) + "'");
            platformId = platform[platformNum];
            if (platformNum == NVIDIA) deviceType = CL_DEVICE_TYPE_GPU;
        }
        break;

        case 0 :
            break;

        default :
            ERROR_HANDLER(0, "Error: parsing arguments");
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

    verbose && std::cerr << "BACKPROJECTION, OpenCL Implementation with " << ((deviceType==CL_DEVICE_TYPE_GPU)?("GPU"):("CPU"))
                         << std::endl << std::endl
                         << "Number of workitems     = " << nLocals << std::endl
                         << "Number of rows          = " << rows << std::endl
                         << "Number of columns       = " << columns << std::endl
                         << "Show prep time          = " << ((showPrepTime)?("True"):("False")) << std::endl 
                         << "Show device info        = " << ((showDevInfo)?("True"):("False")) << std::endl
                         << "Executing .. " << std::endl;

    BackProjectionCL BackProjectionCL(rows, columns, platformId, deviceType, kernelVer);

    BackProjectionCL.init();
    for (int i = 0; i < N_ITER; i++) {
        BackProjectionCL.prep_memory();
        BackProjectionCL.execute();

        std::cerr << "Iteration " << i+1 << "/" << N_ITER << ": DONE." << std::endl;

        if (outputFilename.size() == 0)
            outputFilename = "BackProjection_ocl";          /* Produce [filename]_out */
        verbose
            && std::cerr << "Producing output to: " << outputFilename << std::endl;

        outParam output;
        output.outputFilename = outputFilename;
        BackProjectionCL.output(reinterpret_cast <void *>(&output));

        BackProjectionCL.clean_mem();
    }
    BackProjectionCL.finish();

    return 0;
}

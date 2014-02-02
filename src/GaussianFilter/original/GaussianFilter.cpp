#include <iostream>
#include <iomanip>

#include "GaussianFilter.hpp"

GaussianFilter::GaussianFilter(imgStream _inp)
{
    inp = _inp;
    nElem = inp.height * inp.width;
    width = inp.width;
    height = inp.height;
}

GaussianFilter::~GaussianFilter()
{
    /* nothing */
}

void GaussianFilter::init()
{
    t_all.start();

    /* Initialization */
    t_init.start();

    /* nothing */

    t_init.stop();
}

void GaussianFilter::prep_memory()
{
    t_mem.start();

    /* Allocate memory for the output */
    out.data_r = new pixel_uc[width * height];
    if (!isPGM) {
        out.data_g = new pixel_uc[width * height];
        out.data_b = new pixel_uc[width * height];
    }
    out.height = height;
    out.width = width;

    t_mem.stop();
}

void GaussianFilter::execute()
{
    /* Kernel Execution */
    t_kernel.start();

    if (kernelVer == FAST)
        (isPGM) ? gaussian_fast_gold_bw(inp, out, nElem, height, width) : 
            gaussian_fast_gold_rgb(inp, out, nElem, height, width);
    else
        (isPGM) ? gaussian_gold_bw(inp, out, nElem, height, width) : 
            gaussian_gold_rgb(inp, out, nElem, height, width);

    t_kernel.stop();
}

void GaussianFilter::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GaussianFilter::clean_mem()
{
    /* Cleanup and Output */
    t_clean.start();

    delete[]out.data_r;
    if (!isPGM) {
        delete[]out.data_g;
        delete[]out.data_b;
    }

    t_clean.stop();

}

void GaussianFilter::finish()
{
    t_all.stop();

    delete[]inp.data_r;
    if (!isPGM) {
        delete[]inp.data_g;
        delete[]inp.data_b;
    }

    std::string kernelName =
        "Kernel : " + std::string(kernelStr[kernelVer]);
    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_kernel.print_average_time(kernelName.c_str());
    showPrepTime && t_clean.print_average_time("Cleanup and Output");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

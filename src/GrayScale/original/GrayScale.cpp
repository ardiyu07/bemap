#include <iostream>
#include <iomanip>

#include "GrayScale.hpp"

GrayScale::GrayScale(imgStream _inp)
{
    inp = _inp;
    nElem = inp.height * inp.width;
    width = inp.width;
    height = inp.height;
}

GrayScale::~GrayScale()
{
    t_all.stop();

    /* Free Memory */
    delete[]inp.data_r;
    delete[]inp.data_g;
    delete[]inp.data_b;
    delete[]out.data_r;

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_kernel.print_average_time("Kernel");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

void GrayScale::init()
{
    t_all.start();

    t_init.start();
    // nothing
    t_init.stop();
}

void GrayScale::prep_memory()
{
    t_mem.start();

    /* Allocate memory for the output */
    out.data_r = new pixel_uc[width * height];
    out.data_g = NULL;
    out.data_b = NULL;
    out.height = height;
    out.width = width;

    t_mem.stop();
}

void GrayScale::execute()
{
    /* Kernel Execution */
    t_kernel.start();
    grayscale_gold(out, inp, height, width);
    isPGM = true;
    t_kernel.stop();
}

void GrayScale::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GrayScale::finish()
{
    t_clean.start();
    // nothing
    t_clean.stop();
}

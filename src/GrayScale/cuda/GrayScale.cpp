#include <iostream>
#include <iomanip>

#include "GrayScale.hpp"

GrayScaleCUDA::GrayScaleCUDA(imgStream _inp)
{
    inp = _inp;
    nElem = inp.height * inp.width;
    width = inp.width;
    height = inp.height;

    d_inp_r = NULL;
    d_inp_g = NULL;
    d_inp_b = NULL;
    d_out = NULL;
}

GrayScaleCUDA::~GrayScaleCUDA()
{
    /* nothing  */
}

void GrayScaleCUDA::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    cudaDeviceReset();
    CUDA_CHECK_ERROR();
    cudaSetDevice(0);
    CUDA_CHECK_ERROR();

    t_init.stop();

}

void GrayScaleCUDA::prep_memory()
{
    t_mem.start();

    /* Allocate memory for the output */
    out.data_r = new pixel_uc[width * height];
    out.data_g = NULL;
    out.data_b = NULL;
    out.height = height;
    out.width = width;
    nElem = height * width;

    cudaMalloc((void**)&d_inp_r, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_inp_g, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_inp_b, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)&d_out, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();

    cudaMemcpy(d_inp_r, inp.data_r, nElem * sizeof(pixel_uc),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(d_inp_g, inp.data_g, nElem * sizeof(pixel_uc),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();
    cudaMemcpy(d_inp_b, inp.data_b, nElem * sizeof(pixel_uc),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    t_mem.stop();
}

void GrayScaleCUDA::copyback()
{
    t_cpy.start();
    cudaMemcpy(out.data_r, d_out, nElem * sizeof(pixel_uc),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();
    isPGM = true;
    t_cpy.stop();
}

void GrayScaleCUDA::compare_to_cpu()
{
    imgStream t_out;
    float err_rate;
    float sum_delta = 0.0f;
    float img_size = (float) nElem;
    StopWatch t_cpu;

    /* Compare result to native code */
    if (compResult) {
        verbose && std::cerr << "Comparing to native code results .." << std::endl;

        /* Allocate memory for the output */
        t_out.data_r = new pixel_uc[width * height];
        t_out.data_g = NULL;
        t_out.data_b = NULL;
        t_out.height = height;
        t_out.width = width;

        t_cpu.start();
        grayscale_gold(t_out, inp, height, width);
        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int j = 0; j < nElem; ++j) {
            sum_delta += abs(out.data_r[j] - t_out.data_r[j]);
        }

        err_rate = sum_delta / img_size;
        std::cerr << std::fixed << std::setprecision(4)
                  << "Average error rate per pixel = " << err_rate << " bits" <<
            std::endl;

        /* Output image */
        out_pgpm(realName + "_out_singlethread.pgm", t_out, isPGM);

        delete[]t_out.data_r;
    }
}

void GrayScaleCUDA::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GrayScaleCUDA::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    delete[]out.data_r;

    cudaFree(d_inp_r);
    CUDA_CHECK_ERROR();
    cudaFree(d_inp_g);
    CUDA_CHECK_ERROR();
    cudaFree(d_inp_b);
    CUDA_CHECK_ERROR();
    cudaFree(d_out);
    CUDA_CHECK_ERROR();

    t_clean.stop();
}

void GrayScaleCUDA::finish()
{
    /* Free Memory */
    delete[]inp.data_r;
    delete[]inp.data_g;
    delete[]inp.data_b;

    t_all.stop();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_kernel.print_average_time("Kernel");
    showPrepTime && t_cpy.print_average_time("Memory Copyback"),
        showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

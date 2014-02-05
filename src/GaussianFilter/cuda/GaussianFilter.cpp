#include <iostream>
#include <iomanip>

#include "GaussianFilter.hpp"

GaussianFilterCUDA::GaussianFilterCUDA(imgStream _inp)
{
    inp = _inp;
    nElem = inp.height * inp.width;
    width = inp.width;
    height = inp.height;

    if (kernelVer == SCALAR_FAST || kernelVer == SCALAR_FAST_SHM) {
        isFast = true;
    }

    d_inp_r = NULL;
    d_inp_g = NULL;
    d_inp_b = NULL;
    d_out_r = NULL;
    d_out_g = NULL;
    d_out_b = NULL;

    d_buf_r = NULL;
    d_buf_g = NULL;
    d_buf_b = NULL;
}

GaussianFilterCUDA::~GaussianFilterCUDA()
{
    /* nothing */
}

void GaussianFilterCUDA::init()
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

void GaussianFilterCUDA::prep_memory()
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

    cudaMalloc((void**)(void**)&d_inp_r, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();
    cudaMalloc((void**)(void**)&d_out_r, nElem * sizeof(pixel_uc));
    CUDA_CHECK_ERROR();
    cudaMemcpy(d_inp_r, inp.data_r, nElem * sizeof(pixel_uc),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR();

    if (!isPGM) {

        cudaMalloc((void**)(void**)&d_inp_g, nElem * sizeof(pixel_uc));
        CUDA_CHECK_ERROR();
        cudaMalloc((void**)(void**)&d_out_g, nElem * sizeof(pixel_uc));
        CUDA_CHECK_ERROR();
        cudaMemcpy(d_inp_g, inp.data_g, nElem * sizeof(pixel_uc),
                   cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();

        cudaMalloc((void**)(void**)&d_inp_b, nElem * sizeof(pixel_uc));
        CUDA_CHECK_ERROR();
        cudaMalloc((void**)(void**)&d_out_b, nElem * sizeof(pixel_uc));
        CUDA_CHECK_ERROR();
        cudaMemcpy(d_inp_b, inp.data_b, nElem * sizeof(pixel_uc),
                   cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR();
    }

    /* Buffer for fast kernel */
    if (isFast) {
        cudaMalloc((void**)(void**)&d_buf_r, nElem * sizeof(pixel_uc));
        CUDA_CHECK_ERROR();
        if (!isPGM) {
            cudaMalloc((void**)(void**)&d_buf_g, nElem * sizeof(pixel_uc));
            CUDA_CHECK_ERROR();
            cudaMalloc((void**)(void**)&d_buf_b, nElem * sizeof(pixel_uc));
            CUDA_CHECK_ERROR();
        }
    }
    t_mem.stop();
}

void GaussianFilterCUDA::copyback()
{
    t_cpy.start();

    cudaMemcpy(out.data_r, d_out_r, nElem * sizeof(pixel_uc),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR();

    if (!isPGM) {
        cudaMemcpy(out.data_g, d_out_g, nElem * sizeof(pixel_uc),
                   cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
        cudaMemcpy(out.data_b, d_out_b, nElem * sizeof(pixel_uc),
                   cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR();
    }

    t_cpy.stop();
}

void GaussianFilterCUDA::compare_to_cpu()
{
    imgStream t_out;
    int t_isPGM;
    float err_rate;
    float sum_delta = 0.0f;
    float img_size = (float) nElem;
    int comp = false;
    StopWatch t_cpu;

    /* Compare result to native code */
    if (compResult) {
        verbose && std::cerr << "Comparing to native code results .." << std::endl;

        /* Allocate memory for the output */
        t_out.data_r = new pixel_uc[width * height];
        if (!isPGM) {
            t_out.data_g = new pixel_uc[width * height];
            t_out.data_b = new pixel_uc[width * height];
        }
        t_out.height = height;
        t_out.width = width;

        t_cpu.start();

        if (isFast)
            (isPGM) ? gaussian_fast_gold_bw(inp, t_out, nElem, height, width) : 
                gaussian_fast_gold_rgb(inp, t_out, nElem, height, width);
        else
            (isPGM) ? gaussian_gold_bw(inp, t_out, nElem, height, width) : 
                gaussian_gold_rgb(inp, t_out, nElem, height, width);

        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int j = 0; j < nElem; ++j) {
            sum_delta += fabs((float) (out.data_r[j] - t_out.data_r[j]));
            if (!isPGM) {
                sum_delta +=
                    fabs((float) (out.data_g[j] - t_out.data_g[j]));
                sum_delta +=
                    fabs((float) (out.data_b[j] - t_out.data_b[j]));
            }
        }

        err_rate = sum_delta / img_size;
        std::cerr << std::fixed << std::setprecision(4)
                  << "Average error rate per pixel = " << err_rate << " bits" <<
            std::endl;

        /* Output image */
        out_pgpm(realName + "_out" +
                 ((isPGM) ? ("_singlethread.pgm") : ("_singlethread.ppm")),
                 t_out, isPGM);

        delete[]t_out.data_r;
        if (!isPGM) {
            delete[]t_out.data_g;
            delete[]t_out.data_b;
        }
    }
}

void GaussianFilterCUDA::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GaussianFilterCUDA::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    delete[]out.data_r;
    if (!isPGM) {
        delete[]out.data_g;
        delete[]out.data_b;
    }

    cudaFree(d_inp_r);
    CUDA_CHECK_ERROR();
    cudaFree(d_out_r);
    CUDA_CHECK_ERROR();

    if (!isPGM) {
        cudaFree(d_inp_g);
        CUDA_CHECK_ERROR();
        cudaFree(d_out_g);
        CUDA_CHECK_ERROR();

        cudaFree(d_inp_b);
        CUDA_CHECK_ERROR();
        cudaFree(d_out_b);
        CUDA_CHECK_ERROR();
    }
    if (isFast) {
        cudaFree(d_buf_r);
        CUDA_CHECK_ERROR();
        if (!isPGM) {
            cudaFree(d_buf_g);
            CUDA_CHECK_ERROR();
            cudaFree(d_buf_b);
            CUDA_CHECK_ERROR();
        }
    }

    t_clean.stop();
}

void GaussianFilterCUDA::finish()
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
    showPrepTime && t_cpy.print_average_time("Memory Copyback"),
        showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

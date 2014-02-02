#include <iostream>
#include <iomanip>

#include "GaussianFilter.hpp"
#include "GaussianFilter_common.hpp"

GaussianFilterCL::GaussianFilterCL(imgStream _inp,
                                   std::string _platformId,
                                   cl_device_type _deviceType,
                                   kernelVersion _kernelVer)
{
    inp = _inp;
    nElem = inp.height * inp.width;
    width = inp.width;
    height = inp.height;

    platformId = _platformId;
    deviceType = _deviceType;
    kernelVer = _kernelVer;

    if (kernelVer == SCALAR_FAST || kernelVer == SIMD_FAST
        || kernelVer == SCALAR_FAST_SHM) {
        isFast = true;
    }

    context = NULL;
    commandQueue = NULL;
    program = NULL;

    kernel_r = NULL;
    kernel_g = NULL;
    kernel_b = NULL;
    kernel_col_r = NULL;
    kernel_col_g = NULL;
    kernel_col_b = NULL;

    memobj_inp_r = NULL;
    memobj_inp_g = NULL;
    memobj_inp_b = NULL;
    memobj_out_r = NULL;
    memobj_out_g = NULL;
    memobj_out_b = NULL;

    memobj_buf_r = NULL;
    memobj_buf_g = NULL;
    memobj_buf_b = NULL;
}

GaussianFilterCL::~GaussianFilterCL()
{
    /* nothign  */
}

void GaussianFilterCL::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    /* set CL compiler options */
    std::string compileOptions = "-cl-fast-relaxed-math -Werror \
                    -I ./                                \
                   ";

    context = create_context(deviceType, platformId, showDevInfo, choose);
    CL_ERROR_HANDLER((context != NULL), "Failed to create CL context");
    commandQueue = create_command_queue(context, &device, 0, showDevInfo);
    CL_ERROR_HANDLER((commandQueue != NULL),
                     "Failed to create CL command queue");
    program =
        create_program(context, device, sourceName.c_str(),
                       compileOptions.c_str());
    CL_ERROR_HANDLER((program != NULL), "Failed to create CL program");

    kernel_r = clCreateKernel(program, kernelStr[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel_r != NULL), "Failed to create CL kernel");
    kernel_g = clCreateKernel(program, kernelStr[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel_g != NULL), "Failed to create CL kernel");
    kernel_b = clCreateKernel(program, kernelStr[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel_b != NULL), "Failed to create CL kernel");

    /* We have anoter step for fast kernels -- Column step */
    if (isFast) {
        kernel_col_r =
            clCreateKernel(program, kernelStr[kernelVer + 5], NULL);
        CL_ERROR_HANDLER((kernel_col_r != NULL),
                         "Failed to create CL kernel");
        kernel_col_g =
            clCreateKernel(program, kernelStr[kernelVer + 5], NULL);
        CL_ERROR_HANDLER((kernel_col_g != NULL),
                         "Failed to create CL kernel");
        kernel_col_b =
            clCreateKernel(program, kernelStr[kernelVer + 5], NULL);
        CL_ERROR_HANDLER((kernel_col_b != NULL),
                         "Failed to create CL kernel");
    }

    t_init.stop();
}

void GaussianFilterCL::prep_memory()
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

    memobj_inp_r =
        clCreateBuffer(context, CL_MEM_READ_ONLY, nElem * sizeof(pixel_uc),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    memobj_out_r =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       nElem * sizeof(pixel_uc), NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    errNum =
        clEnqueueWriteBuffer(commandQueue, memobj_inp_r, CL_TRUE, 0,
                             nElem * sizeof(pixel_uc), inp.data_r, 0, NULL,
                             NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    if (!isPGM) {
        memobj_inp_g =
            clCreateBuffer(context, CL_MEM_READ_ONLY,
                           nElem * sizeof(pixel_uc), NULL, &errNum);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clCreateBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        memobj_out_g =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           nElem * sizeof(pixel_uc), NULL, &errNum);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clCreateBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        memobj_inp_b =
            clCreateBuffer(context, CL_MEM_READ_ONLY,
                           nElem * sizeof(pixel_uc), NULL, &errNum);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clCreateBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        memobj_out_b =
            clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                           nElem * sizeof(pixel_uc), NULL, &errNum);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clCreateBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum |=
            clEnqueueWriteBuffer(commandQueue, memobj_inp_g, CL_TRUE, 0,
                                 nElem * sizeof(pixel_uc), inp.data_g, 0,
                                 NULL, NULL);
        errNum |=
            clEnqueueWriteBuffer(commandQueue, memobj_inp_b, CL_TRUE, 0,
                                 nElem * sizeof(pixel_uc), inp.data_b, 0,
                                 NULL, NULL);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueWriteBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
    }

    /* Buffer for fast kernel */
    if (isFast) {
        memobj_buf_r =
            clCreateBuffer(context, CL_MEM_READ_WRITE,
                           nElem * sizeof(pixel_uc), NULL, &errNum);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clCreateBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        if (!isPGM) {
            memobj_buf_g =
                clCreateBuffer(context, CL_MEM_READ_WRITE,
                               nElem * sizeof(pixel_uc), NULL, &errNum);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clCreateBuffer not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
            memobj_buf_b =
                clCreateBuffer(context, CL_MEM_READ_WRITE,
                               nElem * sizeof(pixel_uc), NULL, &errNum);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clCreateBuffer not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
        }
    }

    t_mem.stop();
}

void GaussianFilterCL::execute()
{
    /* number of work items */
    int dim = 1;                // dimension

    switch (kernelVer) {
    case SCALAR:
    case SCALAR_FAST:
    {
        localWorkSize[0] = nLocals;
        globalWorkSize[0] =
            ((nElem + (nLocals - 1)) / nLocals) * nLocals;
    }
    break;
    case SCALAR_FAST_SHM:
    {
        localWorkSize[0] = ROWS_BLOCKDIM_X;
        localWorkSize[1] = ROWS_BLOCKDIM_Y;
        globalWorkSize[0] =
            (int) ((width + ROWS_BLOCKDIM_X -
                    1) / ROWS_BLOCKDIM_X) * ROWS_BLOCKDIM_X;
        globalWorkSize[1] =
            (int) ((height + ROWS_BLOCKDIM_Y -
                    1) / ROWS_BLOCKDIM_Y) * ROWS_BLOCKDIM_Y;
        dim = 2;
    }
    break;
    case SIMD:
    case SIMD_FAST:
    {
        localWorkSize[0] = 1;
        globalWorkSize[0] = height;
    }
    break;
    case STSD:
    case STAD:
    {
        localWorkSize[0] = 1;
        globalWorkSize[0] = 1;
    }
    break;
    }

    /* Kernel arguments */
    /* For the fast kernels, we pass the output to a temporary buffer first */
    errNum =
        clSetKernelArg(kernel_r, 0, sizeof(cl_mem),
                       (void *) &memobj_inp_r);
    errNum |=
        clSetKernelArg(kernel_r, 1, sizeof(cl_mem),
                       ((isFast) ? ((void *) &memobj_buf_r)
                        : ((void *) &memobj_out_r)));
    errNum |= clSetKernelArg(kernel_r, 2, sizeof(int), (void *) &width);
    errNum |= clSetKernelArg(kernel_r, 3, sizeof(int), (void *) &height);

    if (!isPGM) {
        errNum |=
            clSetKernelArg(kernel_g, 0, sizeof(cl_mem),
                           (void *) &memobj_inp_g);
        errNum |=
            clSetKernelArg(kernel_g, 1, sizeof(cl_mem),
                           ((isFast) ? ((void *) &memobj_buf_g)
                            : ((void *) &memobj_out_g)));
        errNum |=
            clSetKernelArg(kernel_g, 2, sizeof(int), (void *) &width);
        errNum |=
            clSetKernelArg(kernel_g, 3, sizeof(int), (void *) &height);
        errNum |=
            clSetKernelArg(kernel_b, 0, sizeof(cl_mem),
                           (void *) &memobj_inp_b);
        errNum |=
            clSetKernelArg(kernel_b, 1, sizeof(cl_mem),
                           ((isFast) ? ((void *) &memobj_buf_b)
                            : ((void *) &memobj_out_b)));
        errNum |=
            clSetKernelArg(kernel_b, 2, sizeof(int), (void *) &width);
        errNum |=
            clSetKernelArg(kernel_b, 3, sizeof(int), (void *) &height);
    }
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clSetKernelArg not CL_SUCCESS: " +
                     std::string(error_string(errNum)));


    /* We have anoter step for fast kernels -- Column step */
    if (isFast) {
        errNum =
            clSetKernelArg(kernel_col_r, 0, sizeof(cl_mem),
                           (void *) &memobj_buf_r);
        errNum |=
            clSetKernelArg(kernel_col_r, 1, sizeof(cl_mem),
                           (void *) &memobj_out_r);
        errNum |=
            clSetKernelArg(kernel_col_r, 2, sizeof(int), (void *) &width);
        errNum |=
            clSetKernelArg(kernel_col_r, 3, sizeof(int), (void *) &height);
        if (!isPGM) {
            errNum |=
                clSetKernelArg(kernel_col_g, 0, sizeof(cl_mem),
                               (void *) &memobj_buf_g);
            errNum |=
                clSetKernelArg(kernel_col_g, 1, sizeof(cl_mem),
                               (void *) &memobj_out_g);
            errNum |=
                clSetKernelArg(kernel_col_g, 2, sizeof(int),
                               (void *) &width);
            errNum |=
                clSetKernelArg(kernel_col_g, 3, sizeof(int),
                               (void *) &height);
            errNum |=
                clSetKernelArg(kernel_col_b, 0, sizeof(cl_mem),
                               (void *) &memobj_buf_b);
            errNum |=
                clSetKernelArg(kernel_col_b, 1, sizeof(cl_mem),
                               (void *) &memobj_out_b);
            errNum |=
                clSetKernelArg(kernel_col_b, 2, sizeof(int),
                               (void *) &width);
            errNum |=
                clSetKernelArg(kernel_col_b, 3, sizeof(int),
                               (void *) &height);
        }
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clSetKernelArg not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
    }

    /* For BW channeled image */
    if (isPGM) {
        /* Execute Kernel */
        errNum =
            clEnqueueNDRangeKernel(commandQueue, kernel_r, dim, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL,
                                   &perfEvent);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum = clWaitForEvents(1, &perfEvent);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clWaitForEvents not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
    }

    /* For RGB Channeled image */
    else {
        /* Execute Kernels */
        errNum =
            clEnqueueNDRangeKernel(commandQueue, kernel_r, dim, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL,
                                   NULL);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum =
            clEnqueueNDRangeKernel(commandQueue, kernel_g, dim, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL,
                                   NULL);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum =
            clEnqueueNDRangeKernel(commandQueue, kernel_b, dim, NULL,
                                   globalWorkSize, localWorkSize, 0, NULL,
                                   &perfEvent);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum = clWaitForEvents(1, &perfEvent);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clWaitForEvents not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
    }

    float tmp = 0.0f;
    start = end = 0;
    clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &end, NULL);
    tmp += (end - start) / 1000000.0f;

    errNum = clReleaseEvent(perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clReleaseEvent not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* std::cout << tmp; */

    /* We have anoter step for fast kernels -- Column step */
    if (isFast) {
        if (dim == 2) {
            localWorkSize[0] = COLUMNS_BLOCKDIM_X;
            localWorkSize[1] = COLUMNS_BLOCKDIM_Y;
            globalWorkSize[0] =
                (int) ((width + COLUMNS_BLOCKDIM_X -
                        1) / COLUMNS_BLOCKDIM_X) * COLUMNS_BLOCKDIM_X;
            globalWorkSize[1] =
                (int) ((height + COLUMNS_BLOCKMEM_Y -
                        1) / COLUMNS_BLOCKMEM_Y) * COLUMNS_BLOCKDIM_Y;
            /* globalWorkSize[0] = (int)((width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0]; */
            /* globalWorkSize[1] = (int)((height / COLUMNS_RESULT_STEPS + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1]; */
        }
        /* For BW channeled image */
        if (isPGM) {
            errNum =
                clEnqueueNDRangeKernel(commandQueue, kernel_col_r, dim,
                                       NULL, globalWorkSize, localWorkSize,
                                       0, NULL, &perfEvent);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
            errNum = clWaitForEvents(1, &perfEvent);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clWaitForPerfEvents not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
        }
        /* For RGB Channeled image */
        else {
            errNum =
                clEnqueueNDRangeKernel(commandQueue, kernel_col_r, dim,
                                       NULL, globalWorkSize, localWorkSize,
                                       0, NULL, NULL);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
            errNum |=
                clEnqueueNDRangeKernel(commandQueue, kernel_col_g, dim,
                                       NULL, globalWorkSize, localWorkSize,
                                       0, NULL, NULL);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
            errNum |=
                clEnqueueNDRangeKernel(commandQueue, kernel_col_b, dim,
                                       NULL, globalWorkSize, localWorkSize,
                                       0, NULL, &perfEvent);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
            errNum = clWaitForEvents(1, &perfEvent);
            CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                             "clWaitForPerfEvents not CL_SUCCESS: " +
                             std::string(error_string(errNum)));
        }
        start = end = 0;
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end, NULL);
        tmp += (end - start) / 1000000.0f;

        errNum = clReleaseEvent(perfEvent);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clReleaseEvent not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        /* std::cout << " " << (end-start)/1000000.0f; */
    }
    /* std::cout << std::endl; */
    t_kernel += tmp;
}

void GaussianFilterCL::copyback()
{
    t_cpy.start();
    errNum =
        clEnqueueReadBuffer(commandQueue, memobj_out_r, CL_TRUE, 0,
                            nElem * sizeof(pixel_uc), out.data_r, 0, NULL,
                            NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueReadBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    if (!isPGM) {
        errNum =
            clEnqueueReadBuffer(commandQueue, memobj_out_g, CL_TRUE, 0,
                                nElem * sizeof(pixel_uc), out.data_g, 0,
                                NULL, NULL);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueReadBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));
        errNum =
            clEnqueueReadBuffer(commandQueue, memobj_out_b, CL_TRUE, 0,
                                nElem * sizeof(pixel_uc), out.data_b, 0,
                                NULL, NULL);
        CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                         "clEnqueueReadBuffer not CL_SUCCESS: " +
                         std::string(error_string(errNum)));

    }
    t_cpy.stop();
}

void GaussianFilterCL::compare_to_cpu()
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
            sum_delta += abs(out.data_r[j] - t_out.data_r[j]);
            if (!isPGM) {
                sum_delta += abs(out.data_g[j] - t_out.data_g[j]);
                sum_delta += abs(out.data_b[j] - t_out.data_b[j]);
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

void GaussianFilterCL::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GaussianFilterCL::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    delete[]out.data_r;
    if (!isPGM) {
        delete[]out.data_g;
        delete[]out.data_b;
    }

    errNum = clReleaseMemObject(memobj_inp_r);
    errNum |= clReleaseMemObject(memobj_out_r);
    if (!isPGM) {
        errNum |= clReleaseMemObject(memobj_inp_g);
        errNum |= clReleaseMemObject(memobj_out_g);
        errNum |= clReleaseMemObject(memobj_inp_b);
        errNum |= clReleaseMemObject(memobj_out_b);
    }
    if (isFast) {
        errNum |= clReleaseMemObject(memobj_buf_r);
        if (!isPGM) {
            errNum |= clReleaseMemObject(memobj_buf_g);
            errNum |= clReleaseMemObject(memobj_buf_b);
        }
    }
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clReleaseMemObject not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_clean.stop();

}

void GaussianFilterCL::finish()
{
    t_all.stop();

    delete[]inp.data_r;
    if (!isPGM) {
        delete[]inp.data_g;
        delete[]inp.data_b;
    }

    errNum = clReleaseKernel(kernel_r);
    errNum |= clReleaseKernel(kernel_g);
    errNum |= clReleaseKernel(kernel_b);
    if (isFast) {
        errNum |= clReleaseKernel(kernel_col_r);
        errNum |= clReleaseKernel(kernel_col_g);
        errNum |= clReleaseKernel(kernel_col_b);
    }
    errNum |= clReleaseProgram(program);
    errNum |= clReleaseCommandQueue(commandQueue);
    errNum |= clReleaseContext(context);

    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clRelease not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

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

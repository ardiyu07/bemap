#include <iostream>
#include <iomanip>

#include "GrayScale.hpp"



GrayScaleCL::GrayScaleCL(imgStream _inp,
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

    context = NULL;
    commandQueue = NULL;
    program = NULL;
    kernel = NULL;

    d_inp_r = NULL;
    d_inp_g = NULL;
    d_inp_b = NULL;
    d_out = NULL;
}

GrayScaleCL::~GrayScaleCL()
{
    /* nothing */
}

void GrayScaleCL::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    context = create_context(deviceType, platformId, showDevInfo, choose);
    CL_ERROR_HANDLER((context != NULL), "Failed to create CL context");
    commandQueue = create_command_queue(context, &device, 0, showDevInfo);
    CL_ERROR_HANDLER((commandQueue != NULL),
                     "Failed to create CL command queue");
    program = create_program(context, device, sourceName.c_str());
    CL_ERROR_HANDLER((program != NULL), "Failed to create CL program");
    kernel = clCreateKernel(program, kernelStr[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel != NULL), "Failed to create CL kernel");

    t_init.stop();

}

void GrayScaleCL::prep_memory()
{
    t_mem.start();

    /* Allocate memory for the output */
    out.data_r = new pixel_uc[width * height];
    out.data_g = NULL;
    out.data_b = NULL;
    out.height = height;
    out.width = width;
    nElem = height * width;

    d_inp_r =
        clCreateBuffer(context, CL_MEM_READ_ONLY, nElem * sizeof(pixel_uc),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    d_inp_g =
        clCreateBuffer(context, CL_MEM_READ_ONLY, nElem * sizeof(pixel_uc),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    d_inp_b =
        clCreateBuffer(context, CL_MEM_READ_ONLY, nElem * sizeof(pixel_uc),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    d_out =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       nElem * sizeof(pixel_uc), NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Write input buffers to device */
    errNum = clEnqueueWriteBuffer
        (commandQueue, d_inp_r, CL_FALSE, 0, nElem * sizeof(pixel_uc),
         inp.data_r, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    errNum =
        clEnqueueWriteBuffer(commandQueue, d_inp_g, CL_FALSE, 0,
                             nElem * sizeof(pixel_uc), inp.data_g, 0, NULL,
                             NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    errNum =
        clEnqueueWriteBuffer(commandQueue, d_inp_b, CL_FALSE, 0,
                             nElem * sizeof(pixel_uc), inp.data_b, 0, NULL,
                             NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_mem.stop();
}

void GrayScaleCL::execute()
{
    /* work itemsの個数 */
    switch (kernelVer) {
    case SCALAR:
        {
            localWorkSize[0] = nLocals;
            globalWorkSize[0] =
                ((nElem + (nLocals - 1)) / nLocals) * nLocals;
        }
        break;
    case SIMD:
        {
            localWorkSize[0] = 1;
            globalWorkSize[0] = height;
            break;
        }
    }

    /* Kernel arguments */
    /* For the fast kernels, we pass the output to a temporary buffer first */
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &d_out);
    errNum |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &d_inp_r);
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &d_inp_g);
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *) &d_inp_b);
    errNum |= clSetKernelArg(kernel, 4, sizeof(int), (void *) &width);
    errNum |= clSetKernelArg(kernel, 5, sizeof(int), (void *) &height);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clSetKernelArg not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    errNum =
        clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,
                               globalWorkSize, localWorkSize, 0, NULL,
                               &perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueNDRangeKernel not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    errNum = clWaitForEvents(1, &perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clWaitForEvents not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Get the execution time */
    start = end = 0;
    clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START,
                            sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END,
                            sizeof(cl_ulong), &end, NULL);
    t_kernel += (end - start) / 1000000.0f;

    errNum = clReleaseEvent(perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clReleaseEvent not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
}

void GrayScaleCL::copyback()
{
    t_cpy.start();
    errNum =
        clEnqueueReadBuffer(commandQueue, d_out, CL_TRUE, 0,
                            nElem * sizeof(pixel_uc), out.data_r, 0, NULL,
                            NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueReadBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    isPGM = true;
    t_cpy.stop();
}

void GrayScaleCL::compare_to_cpu()
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

void GrayScaleCL::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    out_pgpm(outName, out, isPGM);
}

void GrayScaleCL::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    delete[]out.data_r;

    errNum = clReleaseMemObject(d_inp_r);
    errNum |= clReleaseMemObject(d_inp_g);
    errNum |= clReleaseMemObject(d_inp_b);
    errNum |= clReleaseMemObject(d_out);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clRelease not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_clean.stop();

}

void GrayScaleCL::finish()
{
    /* Free Memory */
    delete[]inp.data_r;
    delete[]inp.data_g;
    delete[]inp.data_b;

    errNum = clReleaseKernel(kernel);
    errNum |= clReleaseProgram(program);
    errNum |= clReleaseCommandQueue(commandQueue);
    errNum |= clReleaseContext(context);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clRelease not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_all.stop();

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

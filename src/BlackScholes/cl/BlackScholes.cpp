#include <iostream>
#include <iomanip>
#include <algorithm>

#include "BlackScholes.hpp"

BlackScholesCL::BlackScholesCL(int _optNum,
                               float _riskFree,
                               float _volatility,
                               int _width,
                               int _height,
                               std::string _platformId,
                               cl_device_type _deviceType,
                               kernelVersion _kernelVer)
{
    optNum = _optNum;
    riskFree = _riskFree;
    volatility = _volatility;
    width = _width;
    height = _height;
    platformId = _platformId;
    deviceType = _deviceType;
    kernelVer = _kernelVer;

    call = NULL;
    put = NULL;
    stockPrice = NULL;
    optionStrike = NULL;
    optionYears = NULL;

    context = NULL;
    commandQueue = NULL;
    kernel = NULL;
    program = NULL;

    memobj_call = NULL;
    memobj_put = NULL;
    memobj_stockPrice = NULL;
    memobj_optionStrike = NULL;
    memobj_optionYears = NULL;
}

BlackScholesCL::~BlackScholesCL()
{
    /* nothing */
}

void BlackScholesCL::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    /* set CL compiler options */
    compileOptions = "-cl-fast-relaxed-math -Werror \
                      -I ./                         \
                     ";

    context = create_context(deviceType, platformId, showDevInfo, choose, deviceId);
    CL_ERROR_HANDLER((context != NULL), "Failed to create CL context");
    commandQueue = create_command_queue(context, &device, deviceId, showDevInfo);
    CL_ERROR_HANDLER((commandQueue != NULL),
                     "Failed to create CL command queue");
    program = create_program(context, device, sourceName.c_str(),
                             compileOptions.c_str());

    CL_ERROR_HANDLER((program != NULL), "Failed to create CL program");
    kernel = clCreateKernel(program, kernelStr[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel != NULL), "Failed to create CL kernel");

    t_init.stop();
}

void BlackScholesCL::prep_memory()
{
    /* memobjの準備 */
    t_mem.start();

    srand(time(NULL));

    /* Allocate host memory */
    call = new float[optNum];
    put = new float[optNum];
    stockPrice = new float[optNum];
    optionStrike = new float[optNum];
    optionYears = new float[optNum];
    ERROR_HANDLER((optionYears != NULL || optionStrike != NULL
                   || stockPrice != NULL || put != NULL
                   || call != NULL),
                  "Error in allocation memory for parameters");

    /* Initialize variables */
    for (int i = 0; i < optNum; i++) {
        call[i] = 0.0f;
        put[i] = 0.0f;
        stockPrice[i] = rand_float(10.0f, 100.0f);
        optionStrike[i] = rand_float(1.0f, 100.0f);
        optionYears[i] = rand_float(0.25f, 5.0f);
    }

    memobj_call =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_put =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_stockPrice =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_optionStrike =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_optionYears =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Copy elements to device memory */
    errNum = clEnqueueWriteBuffer
        (commandQueue, memobj_stockPrice, CL_TRUE, 0,
         optNum * sizeof(float), stockPrice, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    errNum = clEnqueueWriteBuffer
        (commandQueue, memobj_optionStrike, CL_TRUE, 0,
         optNum * sizeof(float), optionStrike, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    errNum = clEnqueueWriteBuffer
        (commandQueue, memobj_optionYears, CL_TRUE, 0,
         optNum * sizeof(float), optionYears, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_mem.stop();
}

void BlackScholesCL::execute()
{
    /* Kernel arguments */
    errNum =
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &memobj_call);
    errNum |=
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &memobj_put);
    errNum |=
        clSetKernelArg(kernel, 2, sizeof(cl_mem),
                       (void *) &memobj_stockPrice);
    errNum |=
        clSetKernelArg(kernel, 3, sizeof(cl_mem),
                       (void *) &memobj_optionStrike);
    errNum |=
        clSetKernelArg(kernel, 4, sizeof(cl_mem),
                       (void *) &memobj_optionYears);
    errNum |= clSetKernelArg(kernel, 5, sizeof(float), (void *) &riskFree);
    errNum |=
        clSetKernelArg(kernel, 6, sizeof(float), (void *) &volatility);
    errNum |= clSetKernelArg(kernel, 7, sizeof(int), (void *) &optNum);
    errNum |= clSetKernelArg(kernel, 8, sizeof(int), (void *) &height);
    errNum |= clSetKernelArg(kernel, 9, sizeof(int), (void *) &width);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clSetKernelArg not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Number of workitems */
    switch (kernelVer) {
    case SCALAR:
        {
            localWorkSize[0] = nLocals;
            globalWorkSize[0] =
                ((optNum + (nLocals - 1)) / nLocals) * nLocals;
        }
        break;
    case SIMD:
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

    /* Enqueue kernel */
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

void BlackScholesCL::copyback()
{
    t_cpy.start();
    errNum =
        clEnqueueReadBuffer(commandQueue, memobj_call, CL_TRUE, 0,
                            optNum * sizeof(float), call, 0, NULL, NULL);
    errNum |=
        clEnqueueReadBuffer(commandQueue, memobj_put, CL_TRUE, 0,
                            optNum * sizeof(float), put, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueReadBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
    t_cpy.stop();
}

void BlackScholesCL::compare_to_cpu()
{
    float *t_call;
    float *t_put;
    float sum_delta_c = 0.0f;
    float sum_delta_p = 0.0f;
    float max_delta_c = 0.0f;
    float max_delta_p = 0.0f;
    float sum_call = 0.0f;
    float sum_put = 0.0f;
    float d_c, d_p, L1c, L1p;
    StopWatch t_cpu;

    if (compResult) {
        t_call = new float[optNum * sizeof(float)];
        t_put = new float[optNum * sizeof(float)];

        verbose && std::cerr << "Comparing to CPU (Single thread) results .." << std::endl;
        t_cpu.start();
        black_scholes_gold(t_call, t_put, stockPrice, optionStrike,
                           optionYears, riskFree, volatility, optNum);
        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int i = 0; i < optNum; ++i) {
            sum_call += fabs(t_call[i]);
            sum_put += fabs(t_put[i]);
            d_c = fabs(t_call[i] - call[i]);
            d_p = fabs(t_put[i] - put[i]);
            sum_delta_c += d_c;
            sum_delta_p += d_p;
            max_delta_c = max(max_delta_c, d_c);
            max_delta_p = max(max_delta_p, d_p);
        }

        L1c = sum_delta_c / sum_call;
        L1p = sum_delta_p / sum_put;
        verbose && std::cerr << std::fixed << std::setprecision(20)
                             << "Relative L1 norm (call)    = " << L1c << std::endl
                             << "Relative L1 norm (put)     = " << L1p << std::endl
                             << "Max absolute error (call)  = " << max_delta_c << std::endl
                             << "Max absolute error (put)   = " << max_delta_p << std::endl;
        std::cerr << ((L1c < 1e-6 && L1p < 1e-6) ? ("TEST PASSED.") : ("TEST FAILED.")) << std::endl;

        delete [] t_call;
        delete [] t_put;
    }
}

void BlackScholesCL::output(void *param)
{
    outParam *Param = reinterpret_cast <outParam*> (param);
    std::string outName = Param->outputFilename;
    if (outName.size() != 0) {
        std::fstream fs(outName.c_str(), std::ios_base::out);
        for (int i = 0; i < optNum; i += 1) {
            fs << std::fixed << std::setprecision(4) << i << ": " << call[i] << " "
               << put[i] << std::endl;
        }
        fs.close();
    }
}

void BlackScholesCL::finish()
{
    errNum |= clReleaseKernel(kernel);
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
    showPrepTime && t_cpy.print_average_time("Memory Copyback");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;

    t_all.stop();
}

void BlackScholesCL::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    /* Free memory and cl variables */
    delete[]call;
    delete[]put;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    errNum = clReleaseMemObject(memobj_call);
    errNum |= clReleaseMemObject(memobj_put);
    errNum |= clReleaseMemObject(memobj_stockPrice);
    errNum |= clReleaseMemObject(memobj_optionStrike);
    errNum |= clReleaseMemObject(memobj_optionYears);

    t_clean.stop();
}

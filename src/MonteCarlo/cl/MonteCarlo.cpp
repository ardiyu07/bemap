#include <iostream>
#include <iomanip>
#include <algorithm>

#include "MonteCarlo.hpp"

extern "C" {

#include "mt19937ar.h"

};



MonteCarloCL::MonteCarloCL(int _pathNum,
                           int _optNum,
                           float _riskFree,
                           float _volatility,
                           int _width,
                           int _height,
                           std::string _platformId,
                           cl_device_type _deviceType,
                           kernelVersion _kernelVer)
{
    pathNum = _pathNum;         /// Number of paths (random numbers)
    optNum = _optNum;           /// Number of options 
    riskFree = _riskFree;       /// Risk rate
    volatility = _volatility;   /// Volatility coef
    width = _width;             /// Number of elements per thread (SIMD mode only) for BoxMuller
    height = _height;           /// Number of SIMD threads (SIMD mode only, else is ignored)
    platformId = _platformId;
    deviceType = _deviceType;
    kernelVer = _kernelVer;

    call = NULL;
    random = NULL;
    confidence = NULL;
    stockPrice = NULL;
    optionStrike = NULL;
    optionYears = NULL;

    context = NULL;
    commandQueue = NULL;
    kernel_gr = NULL;           /// genrands
    kernel_bm = NULL;           /// boxmuller
    kernel_mc = NULL;           /// montecarlo
    program = NULL;

    memobj_call = NULL;
    memobj_random = NULL;
    memobj_confidence = NULL;
    memobj_stockPrice = NULL;
    memobj_optionStrike = NULL;
    memobj_optionYears = NULL;
}

MonteCarloCL::~MonteCarloCL()
{
    /* nothing */
}

void MonteCarloCL::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    /// Fast Relaxed Math, faster but more coarse math
    compileOptions = "-cl-fast-relaxed-math -Werror \
                    -I ./                         \
                   ";

    context = create_context(deviceType, platformId, showDevInfo, choose, deviceId);
    CL_ERROR_HANDLER((context != NULL), "Failed to create CL context");
    commandQueue = create_command_queue(context, &device, deviceId, showDevInfo);
    CL_ERROR_HANDLER((commandQueue != NULL),
                     "Failed to create CL command queue");
    program =
        create_program(context, device, sourceName.c_str(),
                       compileOptions.c_str());
    CL_ERROR_HANDLER((program != NULL), "Failed to create CL program");
    kernel_bm = clCreateKernel(program, kernelStrBM[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel_bm != NULL), "Failed to create CL kernel");
    kernel_mc = clCreateKernel(program, kernelStrMC[kernelVer], NULL);
    CL_ERROR_HANDLER((kernel_mc != NULL), "Failed to create CL kernel");

    t_init.stop();

}

void MonteCarloCL::prep_memory()
{
    /* Initialization */
    t_mem.start();

    srand(1);

    /* Allocate host memory */
    random = new float[pathNum];
    call = new float[optNum];
    confidence = new float[optNum];
    stockPrice = new float[optNum];
    optionStrike = new float[optNum];
    optionYears = new float[optNum];
    ERROR_HANDLER((optionYears != NULL || optionStrike != NULL
                   || stockPrice != NULL || confidence != NULL
                   || call != NULL
                   || random != NULL),
                  "Error in allocation memory for parameters");

    /* Initialize variables */
    for (int i = 0; i < optNum; i++) {
        random[i] = 0.0f;
        call[i] = 0.0f;
        confidence[i] = 0.0f;
        stockPrice[i] = rand_float(10.0f, 100.0f);
        optionStrike[i] = rand_float(1.0f, 100.0f);
        optionYears[i] = rand_float(0.25f, 5.0f);
    }

    memobj_random =
        clCreateBuffer(context, CL_MEM_READ_ONLY, pathNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_call =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_confidence =
        clCreateBuffer(context, CL_MEM_READ_WRITE, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_stockPrice =
        clCreateBuffer(context, CL_MEM_READ_ONLY, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_optionStrike =
        clCreateBuffer(context, CL_MEM_READ_ONLY, optNum * sizeof(float),
                       NULL, &errNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clCreateBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    memobj_optionYears =
        clCreateBuffer(context, CL_MEM_READ_ONLY, optNum * sizeof(float),
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

void MonteCarloCL::execute()
{

    /**
     * TODO:
     * 1. Execute GenRand on CPU? -> only takes 47ms
     * 2. Execute BoxMuller OpenCL
     * 3. Execute MonteCarlo OpenCL
     * 
     */

    float time_kernel = 0.0f;

    /// GenRand

    t_gr.start();
    for (int i = 0; i < pathNum; i++) {
        random[i] = genrand_real2();
    }

    t_gr.stop();

    /// BoxMuller

    t_bm.start();

#if 0

    boxmuller_calculation(random, pathNum);

    errNum = clEnqueueWriteBuffer
        (commandQueue, memobj_random, CL_TRUE, 0, pathNum * sizeof(float),
         random, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));


#else

    errNum = clEnqueueWriteBuffer
        (commandQueue, memobj_random, CL_TRUE, 0, pathNum * sizeof(float),
         random, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueWriteBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));


    /* Kernel arguments */
    errNum =
        clSetKernelArg(kernel_bm, 0, sizeof(cl_mem),
                       (void *) &memobj_random);
    errNum |= clSetKernelArg(kernel_bm, 1, sizeof(int), (void *) &pathNum);
    errNum |= clSetKernelArg(kernel_bm, 2, sizeof(int), (void *) &width);
    errNum |= clSetKernelArg(kernel_bm, 3, sizeof(int), (void *) &height);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clSetKernelArg not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Number of workitems */
    switch (kernelVer) {
    case SCALAR:
    case SHM:
    case SIMD:
		{
			localWorkSize[0] = nLocals;
			globalWorkSize[0] =
				((pathNum / 2 + 1 + (nLocals - 1)) / nLocals) * nLocals;
		}
		break;
		/* case SIMD: */
		/*   { */
		/*     localWorkSize[0] = 1; */
		/*     globalWorkSize[0] = height; */
		/*   } */
		break;
    }

    /* Enqueue kernel */
    errNum =
        clEnqueueNDRangeKernel(commandQueue, kernel_bm, 1, NULL,
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
    time_kernel += (end - start) / 1000000.0f;

    errNum = clReleaseEvent(perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clReleaseEvent not CL_SUCCESS: " +
                     std::string(error_string(errNum)));


#endif

    t_bm.stop();

    /// MonteCarlo European Call Simulation

    /* Kernel arguments */
    errNum =
        clSetKernelArg(kernel_mc, 0, sizeof(cl_mem),
                       (void *) &memobj_call);
    errNum |=
        clSetKernelArg(kernel_mc, 1, sizeof(cl_mem),
                       (void *) &memobj_random);
    errNum |=
        clSetKernelArg(kernel_mc, 2, sizeof(cl_mem),
                       (void *) &memobj_confidence);
    errNum |=
        clSetKernelArg(kernel_mc, 3, sizeof(cl_mem),
                       (void *) &memobj_stockPrice);
    errNum |=
        clSetKernelArg(kernel_mc, 4, sizeof(cl_mem),
                       (void *) &memobj_optionStrike);
    errNum |=
        clSetKernelArg(kernel_mc, 5, sizeof(cl_mem),
                       (void *) &memobj_optionYears);
    errNum |=
        clSetKernelArg(kernel_mc, 6, sizeof(float), (void *) &riskFree);
    errNum |=
        clSetKernelArg(kernel_mc, 7, sizeof(float), (void *) &volatility);
    errNum |= clSetKernelArg(kernel_mc, 8, sizeof(int), (void *) &optNum);
    errNum |= clSetKernelArg(kernel_mc, 9, sizeof(int), (void *) &pathNum);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clSetKernelArg not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    /* Number of workitems */
    switch (kernelVer) {
    case SHM:
		{
			localWorkSize[0] = nLocals;
			globalWorkSize[0] = optNum * nLocals;
		}
		break;
    case SCALAR:
    case SIMD:
		{
			localWorkSize[0] = 1;
			globalWorkSize[0] = optNum;
		}
		break;
    }

    /* Enqueue kernel */
    errNum =
        clEnqueueNDRangeKernel(commandQueue, kernel_mc, 1, NULL,
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
    time_kernel += (end - start) / 1000000.0f;
    t_kernel += time_kernel;

    errNum = clReleaseEvent(perfEvent);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clReleaseEvent not CL_SUCCESS: " +
                     std::string(error_string(errNum)));
}

void MonteCarloCL::copyback()
{
    t_cpy.start();
    errNum =
        clEnqueueReadBuffer(commandQueue, memobj_call, CL_TRUE, 0,
                            optNum * sizeof(float), call, 0, NULL, NULL);
    errNum |=
        clEnqueueReadBuffer(commandQueue, memobj_confidence, CL_TRUE, 0,
                            optNum * sizeof(float), confidence, 0, NULL,
                            NULL);
    errNum |=
        clEnqueueReadBuffer(commandQueue, memobj_random, CL_TRUE, 0,
                            pathNum * sizeof(float), random, 0, NULL,
                            NULL);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clEnqueueReadBuffer not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    t_cpy.stop();
}

void MonteCarloCL::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    if (outName.size() != 0) {
        std::fstream fs(outName.c_str(), std::ios_base::out);

        for (int i = 0; i < optNum; i += 1) {
            fs << std::fixed << std::setprecision(10) << i << ": " << stockPrice[i]
               << " " << optionStrike[i] << " " << optionYears[i] << " "
               << call[i] << " " << confidence[i] << std::endl;
        }

        fs.close();
    }

}

void MonteCarloCL::compare_to_cpu()
{
    float *t_call;
    float *t_conf;
    float sum_delta_c = 0.0f;
    float sum_delta_p = 0.0f;
    float max_delta_c = 0.0f;
    float max_delta_p = 0.0f;
    float sum_call = 0.0f;
    float sum_conf = 0.0f;
    float d_c, d_p, L1c, L1p;
    StopWatch t_cpu;

    if (compResult) {
        t_call = new float[optNum * sizeof(float)];
        t_conf = new float[optNum * sizeof(float)];

        verbose
            && std::cerr << "Comparing to CPU (Single thread) results .." <<
            std::endl;
        t_cpu.start();
        montecarlo_gold(t_call, t_conf, stockPrice, optionStrike,
                        optionYears, riskFree, volatility, random, pathNum,
                        optNum);
        t_cpu.stop();
        verbose && t_cpu.print_total_time("Native code kernel");

        for (int i = 0; i < optNum; ++i) {
            sum_call += fabs(t_call[i]);
            sum_conf += fabs(t_conf[i]);
            d_c = fabs(t_call[i] - call[i]);
            d_p = fabs(t_conf[i] - confidence[i]);
            sum_delta_c += d_c;
            sum_delta_p += d_p;
            max_delta_c = max(max_delta_c, d_c);
            max_delta_p = max(max_delta_p, d_p);
        }

        L1c = sum_delta_c / sum_call;
        L1p = sum_delta_p / sum_conf;
        verbose && std::cerr << std::fixed << std::setprecision(20)
                             << "# Succeed if precision error below 1e-2" << std::endl
                             << "Relative L1 norm (call)    = " << L1c << std::endl
                             << "Relative L1 norm (conf)    = " << L1p << std::endl
                             << "Max absolute error (call)  = " << max_delta_c << std::endl
                             << "Max absolute error (conf)  = " << max_delta_p << std::endl;
        std::cerr << ((L1c < 1e-2 && L1p < 1e-2) ? ("TEST PASSED.")
                      : ("TEST FAILED.")) << std::endl;

        delete[]t_call;
        delete[]t_conf;
    }
}

void MonteCarloCL::clean_mem()
{
    /* Cleanup and Output */
    t_clean.start();

    delete[]random;
    delete[]call;
    delete[]confidence;
    delete[]stockPrice;
    delete[]optionStrike;
    delete[]optionYears;

    errNum = clReleaseMemObject(memobj_call);
    errNum |= clReleaseMemObject(memobj_random);
    errNum |= clReleaseMemObject(memobj_confidence);
    errNum |= clReleaseMemObject(memobj_stockPrice);
    errNum |= clReleaseMemObject(memobj_optionStrike);
    errNum |= clReleaseMemObject(memobj_optionYears);

    t_clean.stop();
}

void MonteCarloCL::finish()
{
    /* errNum |= clReleaseKernel( kernel_gr ); */
    errNum |= clReleaseKernel(kernel_bm);
    errNum |= clReleaseKernel(kernel_mc);
    errNum |= clReleaseProgram(program);
    errNum |= clReleaseCommandQueue(commandQueue);
    errNum |= clReleaseContext(context);
    CL_ERROR_HANDLER((errNum == CL_SUCCESS),
                     "clRelease not CL_SUCCESS: " +
                     std::string(error_string(errNum)));

    std::string kernelName1 =
        "Kernel_BoxMuller  : " + std::string(kernelStrBM[kernelVer]);
    std::string kernelName2 =
        "Kernel_MonteCarlo : " + std::string(kernelStrMC[kernelVer]);
    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_gr.print_average_time("Kernel_GenRand");
    t_bm.print_average_time(kernelName1.c_str());
    t_kernel.print_average_time(kernelName2.c_str());
    showPrepTime && t_cpy.print_average_time("Memory Copyback");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;

    t_all.stop();
}


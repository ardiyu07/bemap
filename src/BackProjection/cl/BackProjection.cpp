#include <iostream>
#include "BackProjection.hpp"

BackProjectionCL::BackProjectionCL(int _rows,
                                   int _columns,
                                   std::string _platformId,
                                   cl_device_type _deviceType,
                                   kernelVersion _kernelVer)
{
    platformId = _platformId;
    deviceType = _deviceType;
    kernelVer = _kernelVer;

    device = NULL;
    context = NULL;
    command_queue = NULL;
    command_queue2 = NULL;
    program = NULL;

    k_BackProjection = NULL;
    k_findmax = NULL;
    k_decreaseproj = NULL;
    k_init = NULL;

    r = _rows;
    c = _columns;
}

BackProjectionCL::~BackProjectionCL()
{
    /* nothing */
}

void BackProjectionCL::create_prob()
{
    int i, j;

    input = (unsigned char *)malloc(r * c * sizeof(unsigned char));

    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            input[ i * c + j ] = FALSE;
        }
    }
    create_image(r, c, input, r < c ? r : c, r < c ? (F_TYPE) r / (F_TYPE) 4.0: (F_TYPE) c / (F_TYPE) 4.0);

    std::string inName = "BackProjection_ocl_in.dat";
    std::string bmpName = "BackProjection_ocl_in.bmp";
    printimage(r, c, input, inName.c_str());
    write_bmp(bmpName.c_str(), input, r, c);

    rproj = (int *)malloc(r * sizeof(int));
    cproj = (int *)malloc(c * sizeof(int));
    uproj = (int *)malloc((r + c - 1) * sizeof(int));
    dproj = (int *)malloc((r + c - 1) * sizeof(int));

    rband = (int *)malloc(r * c * sizeof(int));
    cband = (int *)malloc(c * sizeof(int));
    uband = (int *)malloc((r + c - 1) * sizeof(int));
    dband = (int *)malloc((r + c - 1) * sizeof(int));

    image = (F_TYPE *)malloc(r * c * sizeof(F_TYPE));

    makeband(r, c, rband, cband, uband, dband);
    create_input(r, c, input, rproj, cproj, uproj, dproj, uband, dband);
}

void BackProjectionCL::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    /* ----- Initialization for device ----- */
    compileOptions = "-cl-fast-relaxed-math -Werror \
                    -I ./                         \
                   ";

    context = create_context(deviceType, platformId, showDevInfo, choose, deviceId);
    CL_ERROR_HANDLER((context!=NULL), "Failed to create CL context");
    command_queue = create_command_queue(context, &device, deviceId, showDevInfo);
    CL_ERROR_HANDLER((command_queue!=NULL), "Failed to create CL command queue");
    command_queue2 = create_command_queue(context, &device, deviceId, showDevInfo);
    CL_ERROR_HANDLER((command_queue!=NULL), "Failed to create CL command queue");
    program = create_program(context, device, sourceName.c_str(), compileOptions.c_str());
    CL_ERROR_HANDLER((program!=NULL), "Failed to create CL program");

    switch (kernelVer) {
    case SCALAR:
		{
			k_init = clCreateKernel(program, "dev_init", NULL);
			CL_ERROR_HANDLER((k_init!=NULL), "Failed to create CL kernel");
			k_BackProjection = clCreateKernel(program, "dev_backprojection_scalar", NULL);
			CL_ERROR_HANDLER((k_BackProjection!=NULL), "Failed to create CL kernel");     
			k_findmax = clCreateKernel(program, "dev_findmax", NULL);
			CL_ERROR_HANDLER((k_findmax!=NULL), "Failed to create CL kernel");
			k_decreaseproj = clCreateKernel(program, "dev_decreaseproj_gpu", NULL);
			CL_ERROR_HANDLER((k_decreaseproj!=NULL), "Failed to create CL kernel");
			break;
		}
    case SIMD:
		{
			k_init = clCreateKernel(program, "dev_init_simd", NULL);
			CL_ERROR_HANDLER((k_init!=NULL), "Failed to create CL kernel");
			k_BackProjection = clCreateKernel(program, "dev_backprojection_simd4", NULL);
			CL_ERROR_HANDLER((k_BackProjection!=NULL), "Failed to create CL kernel");     
			k_findmax = clCreateKernel(program, "dev_findmax_simd4", NULL);
			CL_ERROR_HANDLER((k_findmax!=NULL), "Failed to create CL kernel");
			k_decreaseproj = clCreateKernel(program, "dev_decreaseproj", NULL);
			CL_ERROR_HANDLER((k_decreaseproj!=NULL), "Failed to create CL kernel");
			break;
		}
    }

    t_init.stop();      
}

void BackProjectionCL::prep_memory()
{
    t_mem.start();

    int i, j;

    /* ----- Malloc for Host ----- */

    /* Create back projection problem */
    create_prob();

    guess = (unsigned char *)malloc(r * c * sizeof(unsigned char));

    rscore = (F_TYPE *)malloc(r * sizeof(F_TYPE));
    cscore = (F_TYPE *)malloc(c * sizeof(F_TYPE));
    uscore = (F_TYPE *)malloc((r + c - 1) * sizeof(F_TYPE));
    dscore = (F_TYPE *)malloc((r + c - 1) * sizeof(F_TYPE));

    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            guess[ i * c + j ] = FALSE;
        }
    }

    for (i = 0; i < r; i++) {
        rscore[i] = (F_TYPE) rproj[i] / (F_TYPE) rband[i];
    }
    for (i = 0; i < c; i++) {
        cscore[i] = (F_TYPE) cproj[i] / (F_TYPE) cband[i];
    }
    for (i = 0; i < r + c - 1; i++) {
        uscore[i] = (F_TYPE) uproj[i] / (F_TYPE) uband[i];
        dscore[i] = (F_TYPE) dproj[i] / (F_TYPE) dband[i];
    }

    /* ----- Malloc for Device ----- */
    memobj_guess = clCreateBuffer(context, CL_MEM_READ_WRITE, r * c * sizeof(unsigned char), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_zero = clCreateBuffer(context, CL_MEM_READ_WRITE, r * c * sizeof(unsigned char), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_maxId = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    memobj_ori =clCreateBuffer(context, CL_MEM_READ_WRITE, r * c * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_ori_id = clCreateBuffer(context, CL_MEM_READ_WRITE, r * c * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    memobj_image = clCreateBuffer(context, CL_MEM_READ_WRITE, (((r * c / nLocals) + nLocals - 1) / nLocals) * nLocals * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_image2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (((r * c / nLocals) + nLocals - 1) / nLocals) * nLocals * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_index = clCreateBuffer(context, CL_MEM_READ_WRITE, (((r * c / nLocals) + nLocals - 1) / nLocals) * nLocals * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_index2 = clCreateBuffer(context, CL_MEM_READ_WRITE, (((r * c / nLocals) + nLocals - 1) / nLocals) * nLocals * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    memobj_rscore = clCreateBuffer(context, CL_MEM_READ_WRITE, r * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_cscore = clCreateBuffer(context, CL_MEM_READ_WRITE, c * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_uscore = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_dscore = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(F_TYPE), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    memobj_rproj = clCreateBuffer(context, CL_MEM_READ_WRITE, r * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_cproj = clCreateBuffer(context, CL_MEM_READ_WRITE, c * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_uproj = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_dproj = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_rband = clCreateBuffer(context, CL_MEM_READ_WRITE, r * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_cband = clCreateBuffer(context, CL_MEM_READ_WRITE, c * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_uband = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    memobj_dband = clCreateBuffer(context, CL_MEM_READ_WRITE, (r + c -1) * sizeof(int), NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clCreateBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    maxId_p = (int*)clEnqueueMapBuffer(command_queue, memobj_maxId, CL_TRUE, CL_MAP_READ, 0, sizeof(int), 0, NULL, NULL, &errNum);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueMapBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
  
    errNum = clEnqueueWriteBuffer(command_queue, memobj_guess, CL_TRUE, 0, r * c * sizeof(unsigned char), guess, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_rscore, CL_TRUE, 0, r * sizeof(F_TYPE), rscore, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_cscore, CL_TRUE, 0, c * sizeof(F_TYPE), cscore, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_uscore, CL_TRUE, 0, (r + c - 1) * sizeof(F_TYPE), uscore, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_dscore, CL_TRUE, 0, (r + c - 1) * sizeof(F_TYPE), dscore, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_rproj, CL_TRUE, 0, r * sizeof(int), rproj, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_rband, CL_TRUE, 0, r * sizeof(int), rband, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_cproj, CL_TRUE, 0, c * sizeof(int), cproj, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_cband, CL_TRUE, 0, c * sizeof(int), cband, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_uproj, CL_TRUE, 0, (r + c - 1) * sizeof(int), uproj, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_uband, CL_TRUE, 0, (r + c - 1) * sizeof(int), uband, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_dproj, CL_TRUE, 0, (r + c - 1) * sizeof(int), dproj, 0, NULL, NULL);
    errNum |= clEnqueueWriteBuffer(command_queue, memobj_dband, CL_TRUE, 0, (r + c - 1) * sizeof(int), dband, 0, NULL, NULL);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueWriteBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));

    t_mem.stop();
}

void BackProjectionCL::execute()
{
    int i, j;
    int maxi, maxj;
    int num;
    F_TYPE max;
    cl_mem swap, image_p, image2_p, index_p, index2_p;
    cl_event *events;
    int eventNum;
    cl_command_queue command_queue_p;


    t_kernel.start();
    errNum = clSetKernelArg(k_BackProjection, 0, sizeof(int), (void *)&r);
    errNum |= clSetKernelArg(k_BackProjection, 1, sizeof(int), (void *)&c);
    errNum |= clSetKernelArg(k_BackProjection, 3, sizeof(cl_mem), (void *)&memobj_guess);
    errNum |= clSetKernelArg(k_BackProjection, 4, sizeof(cl_mem), (void *)&memobj_rscore);
    errNum |= clSetKernelArg(k_BackProjection, 5, sizeof(cl_mem), (void *)&memobj_cscore);
    errNum |= clSetKernelArg(k_BackProjection, 6, sizeof(cl_mem), (void *)&memobj_uscore);
    errNum |= clSetKernelArg(k_BackProjection, 7, sizeof(cl_mem), (void *)&memobj_dscore);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

    errNum = clSetKernelArg(k_decreaseproj, 1, sizeof(int), (void *)&c);
    errNum |= clSetKernelArg(k_decreaseproj, 2, sizeof(cl_mem), (void *)&memobj_guess);
    errNum |= clSetKernelArg(k_decreaseproj, 3, sizeof(cl_mem), (void *)&memobj_rscore);
    errNum |= clSetKernelArg(k_decreaseproj, 4, sizeof(cl_mem), (void *)&memobj_cscore);
    errNum |= clSetKernelArg(k_decreaseproj, 5, sizeof(cl_mem), (void *)&memobj_uscore);
    errNum |= clSetKernelArg(k_decreaseproj, 6, sizeof(cl_mem), (void *)&memobj_dscore);
    errNum |= clSetKernelArg(k_decreaseproj, 7, sizeof(cl_mem), (void *)&memobj_rproj);
    errNum |= clSetKernelArg(k_decreaseproj, 8, sizeof(cl_mem), (void *)&memobj_rband);
    errNum |= clSetKernelArg(k_decreaseproj, 9, sizeof(cl_mem), (void *)&memobj_cproj);
    errNum |= clSetKernelArg(k_decreaseproj, 10, sizeof(cl_mem), (void *)&memobj_cband);
    errNum |= clSetKernelArg(k_decreaseproj, 11, sizeof(cl_mem), (void *)&memobj_uproj);
    errNum |= clSetKernelArg(k_decreaseproj, 12, sizeof(cl_mem), (void *)&memobj_uband);
    errNum |= clSetKernelArg(k_decreaseproj, 13, sizeof(cl_mem), (void *)&memobj_dproj);
    errNum |= clSetKernelArg(k_decreaseproj, 14, sizeof(cl_mem), (void *)&memobj_dband);
    errNum |= clSetKernelArg(k_decreaseproj, 15, sizeof(cl_mem), (void *)&memobj_maxId);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

    switch (kernelVer) {
    case SCALAR:
		{
			BackProjection_nglobals[0] = (c + nLocals - 1) / nLocals * nLocals;
			BackProjection_nglobals[1] = (r + nLocals - 1) / nLocals;
			BackProjection_nlocals[0] = nLocals; 
			BackProjection_nlocals[1] = 1;
			errNum = clSetKernelArg(k_BackProjection, 2, sizeof(cl_mem), (void *)&memobj_image); 
			errNum |= clSetKernelArg(k_BackProjection, 8, sizeof(cl_mem), (void *)&memobj_index);
			errNum |= clSetKernelArg(k_BackProjection, 9, nLocals * sizeof(F_TYPE), NULL);
			errNum |= clSetKernelArg(k_BackProjection, 10, nLocals * sizeof(int), NULL);
			errNum |= clSetKernelArg(k_BackProjection, 11, nLocals * sizeof(F_TYPE), NULL);
			errNum |= clSetKernelArg(k_BackProjection, 12, 2 * nLocals * sizeof(F_TYPE), NULL);
			errNum |= clSetKernelArg(k_BackProjection, 13, 2 * nLocals * sizeof(F_TYPE), NULL);
			CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));
			break;
		}
    case SIMD:
		{
			BackProjection_nglobals[0] = 1;
			BackProjection_nglobals[1] = 4;
			BackProjection_nlocals[0] = 1; 
			BackProjection_nlocals[1] = 1;
			init_nglobals = r;
			init_nlocals = 1;
			errNum = clSetKernelArg(k_init, 0, sizeof(int), (void *)&r);
			errNum |= clSetKernelArg(k_init, 1, sizeof(int), (void *)&c);
			errNum |= clSetKernelArg(k_init, 2, sizeof(cl_mem), (void *)&memobj_ori); 
			errNum |= clSetKernelArg(k_init, 3, sizeof(cl_mem), (void *)&memobj_guess);
			errNum |= clSetKernelArg(k_init, 4, sizeof(cl_mem), (void *)&memobj_rscore);
			errNum |= clSetKernelArg(k_init, 5, sizeof(cl_mem), (void *)&memobj_cscore);
			errNum |= clSetKernelArg(k_init, 6, sizeof(cl_mem), (void *)&memobj_uscore);
			errNum |= clSetKernelArg(k_init, 7, sizeof(cl_mem), (void *)&memobj_dscore);
			errNum |= clSetKernelArg(k_init, 8, sizeof(cl_mem), (void *)&memobj_ori_id);
			CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

			errNum |= clSetKernelArg(k_BackProjection, 2, sizeof(cl_mem), (void *)&memobj_ori); 
			errNum |= clSetKernelArg(k_BackProjection, 8, sizeof(cl_mem), (void *)&memobj_maxId);
			CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));
			break;
		}
    }

    decreaseproj_nglobals = 1;
    decreaseproj_nlocals = 1;

    if (naive) {
        BackProjection_gold(r, c, guess, rproj, rband, cproj, cband, uproj, uband, dproj, dband, maxId_p);
    }

    else if (kernelVer == SCALAR) {

        init_nglobals = (((r * c / nLocals) + nLocals - 1) / nLocals) * nLocals;
        init_nlocals = nLocals;

        errNum = clSetKernelArg(k_init, 0, sizeof(cl_mem), (void *)&memobj_image); 
        errNum |= clSetKernelArg(k_init, 1, sizeof(cl_mem), (void *)&memobj_index);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

        errNum = clEnqueueNDRangeKernel(command_queue, k_init, 1, NULL, &init_nglobals, &init_nlocals, 0, NULL, NULL);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));  

        errNum = clSetKernelArg(k_init, 0, sizeof(cl_mem), (void *)&memobj_image2); 
        errNum |= clSetKernelArg(k_init, 1, sizeof(cl_mem), (void *)&memobj_index2);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

        errNum = clEnqueueNDRangeKernel(command_queue, k_init, 1, NULL, &init_nglobals, &init_nlocals, 0, NULL, NULL);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));  

        if(0 || deviceType != CL_DEVICE_TYPE_GPU) {
            command_queue_p = command_queue;
            events = NULL;
            eventNum = 0;
        }
        else {
            command_queue_p = command_queue2;
            events = (cl_event *)malloc(sizeof(cl_event));
            eventNum = 1;
        }
        for (i = 0;; i++)
        {
            errNum = clEnqueueNDRangeKernel(command_queue, k_BackProjection, 2, NULL, BackProjection_nglobals, BackProjection_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            image_p = memobj_image;
            image2_p = memobj_image2;
            index_p = memobj_index;
            index2_p = memobj_index2;

            if (i != 0) {
                errNum = clEnqueueReadBuffer(command_queue_p, memobj_maxId, CL_TRUE, 0, sizeof(int), maxId_p, eventNum, events, NULL);
                CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueReadBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
                if (deviceType == CL_DEVICE_TYPE_GPU) 
                    clReleaseEvent(*events);
                if (*maxId_p < 0)
                {
                    break;
                } else {
                    guess[*maxId_p] = TRUE;
                }
            }
            for (findmax_nglobals = (((BackProjection_nglobals[0] * BackProjection_nglobals[1] / BackProjection_nlocals[0]) + nLocals -1) / nLocals) * nLocals,
                     findmax_nlocals = nLocals;
                 findmax_nglobals > nLocals;
                 findmax_nglobals = (((findmax_nglobals / nLocals) + nLocals - 1) / nLocals) * nLocals) {
                errNum = clSetKernelArg(k_findmax, 0, sizeof(cl_mem), (void *)&image_p);
                errNum |= clSetKernelArg(k_findmax, 1, sizeof(cl_mem), (void *)&index_p);
                errNum |= clSetKernelArg(k_findmax, 2, sizeof(cl_mem), (void *)&image2_p); 
                errNum |= clSetKernelArg(k_findmax, 3, sizeof(cl_mem), (void *)&index2_p);
                errNum |= clSetKernelArg(k_findmax, 4, findmax_nlocals * sizeof(F_TYPE), NULL);
                errNum |= clSetKernelArg(k_findmax, 5, findmax_nlocals * sizeof(int), NULL);
                CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

                errNum = clEnqueueNDRangeKernel(command_queue, k_findmax, 1, NULL, &findmax_nglobals, &findmax_nlocals, 0, NULL, NULL);
                CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

                swap = image_p;
                image_p = image2_p;
                image2_p = swap;
                swap = index_p;
                index_p = index2_p;
                index2_p = swap;
            }

            errNum = clSetKernelArg(k_decreaseproj, 0, sizeof(cl_mem), (void *)&index_p);
            errNum |= clSetKernelArg(k_decreaseproj, 16, sizeof(cl_mem), (void *)&image_p);
            errNum |= clSetKernelArg(k_decreaseproj, 17, findmax_nlocals * sizeof(F_TYPE), NULL);
            errNum |= clSetKernelArg(k_decreaseproj, 18, findmax_nglobals * sizeof(F_TYPE), NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));
            errNum = clEnqueueNDRangeKernel(command_queue, k_decreaseproj, 1, NULL, &findmax_nglobals, &findmax_nglobals, 0, NULL, events);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));
        }
        errNum = clEnqueueReadBuffer(command_queue, memobj_guess, CL_TRUE, 0, r * c * sizeof(unsigned char), guess, 0, NULL, NULL);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueReadBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
    }

    else if (kernelVer == SIMD) {
        errNum = clEnqueueNDRangeKernel(command_queue, k_init, 1, NULL, &init_nglobals, &init_nlocals, 0, NULL, NULL);
        CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));
        for (i = 0; ; i++) {
            //if (i % 1 == 0) printf("iteration %d \n", i);
            if (i!=0) errNum = clEnqueueNDRangeKernel(command_queue, k_BackProjection, 2, NULL, BackProjection_nglobals, BackProjection_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            num = c;
            findmax_nglobals = r;
            findmax_nlocals = 1;

            errNum = clSetKernelArg(k_findmax, 0, sizeof(cl_mem), (void *)&memobj_ori);
            errNum |= clSetKernelArg(k_findmax, 1, sizeof(cl_mem), (void *)&memobj_ori_id);
            errNum |= clSetKernelArg(k_findmax, 2, sizeof(cl_mem), (void *)&memobj_image); 
            errNum |= clSetKernelArg(k_findmax, 3, sizeof(cl_mem), (void *)&memobj_index);
            errNum |= clSetKernelArg(k_findmax, 4, sizeof(int), (void *)&num);
            errNum |= clSetKernelArg(k_findmax, 5, sizeof(cl_mem), (void *)&memobj_rscore);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clEnqueueNDRangeKernel(command_queue, k_findmax, 1, NULL, &findmax_nglobals, &findmax_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            num = r;
            findmax_nglobals = 1;
            findmax_nlocals = 1;

            errNum = clSetKernelArg(k_findmax, 0, sizeof(cl_mem), (void *)&memobj_image);
            errNum |= clSetKernelArg(k_findmax, 1, sizeof(cl_mem), (void *)&memobj_index);
            errNum |= clSetKernelArg(k_findmax, 2, sizeof(cl_mem), (void *)&memobj_image2); 
            errNum |= clSetKernelArg(k_findmax, 3, sizeof(cl_mem), (void *)&memobj_index2);
            errNum |= clSetKernelArg(k_findmax, 4, sizeof(int), (void *)&num);
            errNum |= clSetKernelArg(k_findmax, 5, sizeof(cl_mem), (void *)&memobj_rscore);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clEnqueueNDRangeKernel(command_queue, k_findmax, 1, NULL, &findmax_nglobals, &findmax_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clSetKernelArg(k_decreaseproj, 0, sizeof(cl_mem), (void *)&memobj_index2);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));
            errNum = clEnqueueTask(command_queue, k_decreaseproj, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clEnqueueReadBuffer(command_queue, memobj_maxId, CL_TRUE, 0, sizeof(int), maxId_p, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueReadBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
            if (*maxId_p < 0) {
                //printf("[iteration %d] last max image : %d\n", i, *maxId_p);
                break;
            } else {
                guess[*maxId_p] = TRUE;
            }
        }
    }

    else if (kernelVer == SIMD) {
        for (i = 0; ; i++) {
            errNum = clEnqueueNDRangeKernel(command_queue, k_BackProjection, 1, NULL, BackProjection_nglobals, BackProjection_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            num = r;
            findmax_nglobals = 1;
            findmax_nlocals = 1;

            errNum = clSetKernelArg(k_findmax, 0, sizeof(cl_mem), (void *)&memobj_image);
            errNum |= clSetKernelArg(k_findmax, 1, sizeof(cl_mem), (void *)&memobj_index);
            errNum |= clSetKernelArg(k_findmax, 2, sizeof(cl_mem), (void *)&memobj_image2); 
            errNum |= clSetKernelArg(k_findmax, 3, sizeof(cl_mem), (void *)&memobj_index2);
            errNum |= clSetKernelArg(k_findmax, 4, sizeof(int), (void *)&num);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clEnqueueNDRangeKernel(command_queue, k_findmax, 1, NULL, &findmax_nglobals, &findmax_nlocals, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clSetKernelArg(k_decreaseproj, 0, sizeof(cl_mem), (void *)&memobj_index2);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clSetKernelArg not CL_SUCCESS: " + std::string(error_string(errNum)));
            errNum = clEnqueueTask(command_queue, k_decreaseproj, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueNDRangeKernel not CL_SUCCESS: " + std::string(error_string(errNum)));

            errNum = clEnqueueReadBuffer(command_queue, memobj_maxId, CL_TRUE, 0, sizeof(int), maxId_p, 0, NULL, NULL);
            CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clEnqueueReadBuffer not CL_SUCCESS: " + std::string(error_string(errNum)));
            if (*maxId_p < 0) {
                break;
            } else {
                guess[*maxId_p] = TRUE;
            }
        }
    }
	
    t_kernel.stop();
}

void BackProjectionCL::clean_mem()
{
    /* Free Memory */
    t_clean.start();

    free(guess);
    free(rscore);
    free(cscore);
    free(uscore);
    free(dscore);

    release_prob();
        
    errNum |= clEnqueueUnmapMemObject(command_queue, memobj_maxId, maxId_p, 0, NULL, NULL);  

    errNum |= clReleaseMemObject(memobj_guess);
    errNum |= clReleaseMemObject(memobj_maxId);

    errNum |= clReleaseMemObject(memobj_ori);
    errNum |= clReleaseMemObject(memobj_ori_id);

    errNum = clReleaseMemObject(memobj_image);
    errNum |= clReleaseMemObject(memobj_image2);
    errNum |= clReleaseMemObject(memobj_index);
    errNum |= clReleaseMemObject(memobj_index2);

    errNum |= clReleaseMemObject(memobj_rscore);
    errNum |= clReleaseMemObject(memobj_cscore);
    errNum |= clReleaseMemObject(memobj_uscore);
    errNum |= clReleaseMemObject(memobj_dscore);

    errNum |= clReleaseMemObject(memobj_rproj);
    errNum |= clReleaseMemObject(memobj_cproj);
    errNum |= clReleaseMemObject(memobj_uproj);
    errNum |= clReleaseMemObject(memobj_dproj);

    errNum |= clReleaseMemObject(memobj_rband);
    errNum |= clReleaseMemObject(memobj_cband);
    errNum |= clReleaseMemObject(memobj_uband);
    errNum |= clReleaseMemObject(memobj_dband);


    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clRelease not CL_SUCCESS: " + std::string(error_string(errNum)));

    t_clean.stop();
}

void BackProjectionCL::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    std::string datName = outName + "_out.dat";
    std::string bmpName = outName + "_out.bmp";

    printimage(r, c, guess, datName.c_str());
    write_bmp(bmpName.c_str(), guess, r, c);
}


void BackProjectionCL::release_prob()
{
    free(input);
    free(rproj);
    free(cproj);
    free(uproj);
    free(dproj);
    free(rband);
    free(cband);
    free(uband);
    free(dband);
    free(image);
}

void BackProjectionCL::finish()
{
    errNum = clReleaseKernel(k_init);
    errNum |= clReleaseKernel(k_BackProjection);
    errNum |= clReleaseKernel(k_findmax);
    errNum |= clReleaseKernel(k_decreaseproj);
    errNum |= clReleaseProgram(program);
    errNum |= clFlush(command_queue);
    errNum |= clFinish(command_queue);
    errNum |= clReleaseContext(context);
    CL_ERROR_HANDLER((errNum==CL_SUCCESS), "clRelease not CL_SUCCESS: " + std::string(error_string(errNum)));

	t_all.stop();

    std::string kernelName = "Kernel : " + std::string(kernelStr[kernelVer]);
    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparing");
    t_kernel.print_average_time(kernelName.c_str());
    showPrepTime && t_clean.print_average_time("Cleanup");
    t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;

}

#include <iostream>
#include <iomanip>

#include "BackProjection.hpp"

BackProjectionCUDA::BackProjectionCUDA(int _rows,
									   int _columns)
{
    r = _rows;
    c = _columns;
}

BackProjectionCUDA::~BackProjectionCUDA()
{
    /* nothing */
}

void BackProjectionCUDA::create_prob()
{
    int i, j;

	cudaMallocHost((void**)&input, r * c * sizeof(unsigned char));
	CUDA_CHECK_ERROR();

    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            input[ i * c + j ] = FALSE;
        }
    }
    create_image(r, c, input, r < c ? r : c, r < c ? (F_TYPE) r / (F_TYPE) 4.0: (F_TYPE) c / (F_TYPE) 4.0);

    std::string inName = "BackProjection_cuda_in.dat";
    std::string bmpName = "BackProjection_cuda_in.bmp";
    printimage(r, c, input, inName.c_str());
    write_bmp(bmpName.c_str(), input, r, c);

	cudaMallocHost((void**)&rproj, r * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&cproj, c * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&uproj, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&dproj, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();

    cudaMallocHost((void**)&rband, r * c * sizeof(int));
	CUDA_CHECK_ERROR();
    cudaMallocHost((void**)&cband, c * sizeof(int));
	CUDA_CHECK_ERROR();
    cudaMallocHost((void**)&uband, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&dband, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();

	cudaMallocHost((void**)&image, r * c * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&maxId_p, sizeof(int));
	CUDA_CHECK_ERROR();

    makeband(r, c, rband, cband, uband, dband);
    create_input(r, c, input, rproj, cproj, uproj, dproj, uband, dband);
}

void BackProjectionCUDA::init()
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

void BackProjectionCUDA::prep_memory()
{
	int i, j;

    t_mem.start();

    /* Create back projection problem */
    create_prob();

	cudaMallocHost((void**)&guess, r * c * sizeof(unsigned char));
	CUDA_CHECK_ERROR();

	cudaMallocHost((void**)&rscore, r * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&cscore, c * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&uscore, (r + c - 1) * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMallocHost((void**)&dscore, (r + c - 1) * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();

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

	cudaMalloc((void**)&d_guess, r * c * sizeof(unsigned char));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_zero, r * c * sizeof(unsigned char));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_maxId, sizeof(unsigned int));
	CUDA_CHECK_ERROR();

	cudaMalloc((void**)&d_ori, r * c * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_ori_id, r * c * sizeof(int));
	CUDA_CHECK_ERROR();

	cudaMalloc((void**)&d_image, (((r * c / nThreads) + nThreads - 1) / nThreads) * nThreads * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_image2, (((r * c / nThreads) + nThreads - 1) / nThreads) * nThreads * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_index, (((r * c / nThreads) + nThreads - 1) / nThreads) * nThreads * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_index2, (((r * c / nThreads) + nThreads - 1) / nThreads) * nThreads * sizeof(int));
	CUDA_CHECK_ERROR();

	cudaMalloc((void**)&d_rscore, r * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_cscore, c * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_uscore, (r + c - 1) * sizeof(F_TYPE));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_dscore, (r + c - 1) *  sizeof(F_TYPE));
	CUDA_CHECK_ERROR();

	cudaMalloc((void**)&d_rproj, r * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_cproj, c * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_uproj, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_dproj, (r + c - 1) *  sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_rband, r * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_cband, c * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_uband, (r + c - 1) * sizeof(int));
	CUDA_CHECK_ERROR();
	cudaMalloc((void**)&d_dband, (r + c - 1) *  sizeof(int));
	CUDA_CHECK_ERROR();
	
	cudaMemcpy(d_guess, guess, r * c * sizeof(unsigned char), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_rscore, rscore, r * sizeof(F_TYPE), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_cscore, cscore, c * sizeof(F_TYPE), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_uscore, uscore, (r + c - 1) * sizeof(F_TYPE), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_dscore, dscore, (r + c - 1) * sizeof(F_TYPE), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_rproj, rproj, r * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_rband, rband, r * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_cproj, cproj, c * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_cband, cband, c * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_uproj, uproj, (r + c - 1) * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_uband, uband, (r + c - 1) * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_dproj, dproj, (r + c - 1) * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();
	cudaMemcpy(d_dband, dband, (r + c - 1) * sizeof(int), cudaMemcpyHostToDevice);
	CUDA_CHECK_ERROR();

    t_mem.stop();
}

void BackProjectionCUDA::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    std::string datName = outName + "_out.dat";
    std::string bmpName = outName + "_out.bmp";

    printimage(r, c, guess, datName.c_str());
    write_bmp(bmpName.c_str(), guess, r, c);
}

void BackProjectionCUDA::release_prob()
{
    cudaFreeHost(input);
	CUDA_CHECK_ERROR();
    cudaFreeHost(rproj);
	CUDA_CHECK_ERROR();
    cudaFreeHost(cproj);
	CUDA_CHECK_ERROR();
    cudaFreeHost(uproj);
	CUDA_CHECK_ERROR();
    cudaFreeHost(dproj);
	CUDA_CHECK_ERROR();
    cudaFreeHost(rband);
	CUDA_CHECK_ERROR();
    cudaFreeHost(cband);
	CUDA_CHECK_ERROR();
    cudaFreeHost(uband);
	CUDA_CHECK_ERROR();
    cudaFreeHost(dband);
	CUDA_CHECK_ERROR();
    cudaFreeHost(image);
	CUDA_CHECK_ERROR();
}

void BackProjectionCUDA::clean_mem()
{
    t_clean.start();

    cudaFreeHost(guess);
	CUDA_CHECK_ERROR();
    cudaFreeHost(rscore);
	CUDA_CHECK_ERROR();
    cudaFreeHost(cscore);
	CUDA_CHECK_ERROR();
    cudaFreeHost(uscore);
	CUDA_CHECK_ERROR();
    cudaFreeHost(dscore);
	CUDA_CHECK_ERROR();

    release_prob();

	cudaFree(d_guess);	
	CUDA_CHECK_ERROR();
	cudaFree(d_maxId);	
	CUDA_CHECK_ERROR();

	cudaFree(d_ori);	
	CUDA_CHECK_ERROR();
	cudaFree(d_ori_id);	
	CUDA_CHECK_ERROR();

	cudaFree(d_image);	
	CUDA_CHECK_ERROR();
	cudaFree(d_image2);	
	CUDA_CHECK_ERROR();
	cudaFree(d_index);	
	CUDA_CHECK_ERROR();
	cudaFree(d_index2);	
	CUDA_CHECK_ERROR();

	cudaFree(d_rscore);	
	CUDA_CHECK_ERROR();
	cudaFree(d_cscore);	
	CUDA_CHECK_ERROR();
	cudaFree(d_uscore);	
	CUDA_CHECK_ERROR();
	cudaFree(d_dscore);	
	CUDA_CHECK_ERROR();

	cudaFree(d_rproj);	
	CUDA_CHECK_ERROR();
	cudaFree(d_cproj);	
	CUDA_CHECK_ERROR();
	cudaFree(d_uproj);	
	CUDA_CHECK_ERROR();
	cudaFree(d_dproj);	
	CUDA_CHECK_ERROR();

	cudaFree(d_rband);	
	CUDA_CHECK_ERROR();
	cudaFree(d_cband);	
	CUDA_CHECK_ERROR();
	cudaFree(d_uband);	
	CUDA_CHECK_ERROR();
	cudaFree(d_dband);	
	CUDA_CHECK_ERROR();

    t_clean.stop();
}

void BackProjectionCUDA::finish()
{
    t_all.stop();

	cudaDeviceReset();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparing");
    t_kernel.print_average_time("Kernel: BackProjection_scalar");
    showPrepTime && t_clean.print_average_time("Cleanup");
    t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;	
}

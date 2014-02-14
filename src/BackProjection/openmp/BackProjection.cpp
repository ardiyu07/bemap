#include <iostream>
#include <iomanip>
#include "BackProjection.hpp"

BackProjection::BackProjection(int _rows, int _columns)
{
    r = _rows;
    c = _columns;
}

BackProjection::~BackProjection()
{
    /* nothing */
}

void BackProjection::create_prob()
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

    std::string inName = "BackProjection_ref_in.dat";
    std::string bmpName = "BackProjection_ref_in.bmp";
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

    makeband(r, c, rband, cband, uband, dband);
    create_input(r, c, input, rproj, cproj, uproj, dproj, uband, dband);
}

void BackProjection::init()
{
    t_all.start();

    /* Initialization phase */
    t_init.start();

    /* Nothing */

    t_init.stop();      
}

void BackProjection::prep_memory()
{
    t_mem.start();

    /* Create back projection problem */
    create_prob();

    int i, j;

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

    t_mem.stop();
}

void BackProjection::execute()
{
    /* Kernel Execution */
    t_kernel.start();

    BackProjection_gold(r, c, guess, rproj, rband, cproj, cband, uproj, uband, dproj, dband, &maxId_p);

    t_kernel.stop();

}

void BackProjection::output(void *param)
{
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    std::string datName = outName + ".dat";
    std::string bmpName = outName + ".bmp";

    printimage(r, c, guess, datName.c_str());
    write_bmp(bmpName.c_str(), guess, r, c);
}

void BackProjection::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    free(guess);
    free(rscore);
    free(cscore);
    free(uscore);
    free(dscore);

    release_prob();

    t_clean.stop();
}

void BackProjection::release_prob()
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
}

void BackProjection::finish()
{
    t_all.stop();

    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Transfer");
    t_kernel.print_average_time("Kernel: BackProjection");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

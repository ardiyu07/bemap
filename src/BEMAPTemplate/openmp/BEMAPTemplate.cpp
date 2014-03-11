#include <iostream>
#include <iomanip>

#include "BEMAPTemplate.hpp"

BEMAPTemplate::BEMAPTemplate(float _alpha, float _beta, int _partNum)
{
    /* Constructor */
    alpha = _alpha;
    beta = _beta;
    partNum = _partNum;
}

BEMAPTemplate::~BEMAPTemplate()
{
    /* Destructor */
}

void BEMAPTemplate::init()
{
    /* Start global timer */
    t_all.start();

    t_init.start();

    /* Do initialization */

    t_init.stop();
}

void BEMAPTemplate::prep_memory()
{
    t_mem.start();

    /* Initialize random number seed  */
    srand(time(NULL));

    /* Allocate host memory */
    memIn = new float[partNum];
    memOut = new float[partNum];
    ERROR_HANDLER((memIn != NULL || memOut != NULL),
                  "Error in allocation memory for parameters");

    /* Initialize variables */
    for (int i = 0; i < partNum; i++) {
        memIn[i] = 0.0f;
    }

    t_mem.stop();
}

void BEMAPTemplate::execute()
{
    /* Kernel Execution */
    t_kernel.start();
    bemap_template_gold(memOut, memIn, alpha, beta, partNum);
    t_kernel.stop();
}

void BEMAPTemplate::output(void *param)
{
    /* Output */
    outParam *Param = reinterpret_cast < outParam * >(param);
    std::string outName = Param->outputFilename;

    if (outName.size() != 0) {
        std::fstream fs(outName.c_str(), std::ios_base::out);
        for (int i = 0; i < partNum; i += 1) {
            fs << std::fixed << std::setprecision(4) << i << ": " << memIn[i] << " "
               << memOut[i] << std::endl;
        }
        fs.close();
    }
}

void BEMAPTemplate::clean_mem()
{
    /* Cleanup */
    t_clean.start();

    delete [] memIn;
    delete [] memOut;

    t_clean.stop();
}

void BEMAPTemplate::finish()
{
    /* Stop global timer */
    t_all.stop();

    /* Output time */
    showPrepTime && t_init.print_average_time("Initialization");
    showPrepTime && t_mem.print_average_time("Memory Preparation");
    t_kernel.print_average_time("Kernel : bemap_template");
    showPrepTime && t_clean.print_average_time("Cleanup");
    showPrepTime && t_all.print_total_time("Total Execution Time");
    std::cerr << std::endl;
}

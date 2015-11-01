#ifndef __MULTIGPULOGICSIMULATION_HPP
#define __MULTIGPULOGICSIMULATION_HPP

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "MultiGPULogicSimulation_gold.hpp"
#include "module.hpp"

/* user parameters */
extern int verbose;
extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class MultiGPULogicSimulation {

public:
    MultiGPULogicSimulation(std::string _inputFile);
    ~MultiGPULogicSimulation();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

    /* bemap_template variables */
private:

    std::string inputFile;

    Module module;  

private:

    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_clean;
    StopWatch t_all;

};

#endif
#ifndef __BEMAPTEMPLATE_HPP
#define __BEMAPTEMPLATE_HPP

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "BEMAPTemplate_gold.hpp"

/* user parameters */
extern int verbose;
extern int showPrepTime;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class BEMAPTemplate {

public:
    BEMAPTemplate(float _alpha, float _beta, int _partNum);
    ~BEMAPTemplate();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

    /* bemap_template variables */
private:

    float *memIn;
    float *memOut;

    float alpha;
    float beta;

    int partNum;

private:

    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

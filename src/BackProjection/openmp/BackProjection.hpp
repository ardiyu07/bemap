#ifndef __BACKPROJECTION_HPP
#define __BACKPROJECTION_HPP

/* include utilities */
#include <c_util_img.hpp>
#include <c_util_stopwatch.hpp>
#include <c_util_getopt.hpp>

#include "BackProjection_gold.hpp"

extern int verbose;
extern int showPrepTime;
extern int isPGM;

typedef struct outParam {
    std::string outputFilename;
} outParam;

class BackProjection {
public:

    BackProjection(int _rows, int _columns);
    ~BackProjection();

public:

    void init();
    void prep_memory();
    void execute();
    void output(void *param);
    void clean_mem();
    void finish();

private:
    void create_prob();
    void release_prob();

    /* BackProjection variables */
private:

    /* tmp? variable */
    unsigned char *guess;
    int *rproj, *cproj, *uproj, *dproj, *rband, *cband, *uband, *dband;
    F_TYPE *rscore, *cscore, *uscore, *dscore;
    unsigned char *input;
    int maxId_p;
    int r, c;

private:
    /* time measurement */
    StopWatch t_init, t_mem, t_kernel, t_clean;
    StopWatch t_all;

};

#endif

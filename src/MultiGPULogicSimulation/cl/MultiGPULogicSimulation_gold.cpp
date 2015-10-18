#include "MultiGPULogicSimulation_gold.hpp"

void bemap_template_gold(float *memOut, float *memIn, float alpha, float beta, int partNum)
{
    for (int i = 0; i < partNum; i++) {
        memOut[i] = alpha * memIn[i] + beta;
    }
}

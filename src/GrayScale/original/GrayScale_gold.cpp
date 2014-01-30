#include "GrayScale_gold.hpp"

#define R_RATE 0.298912f
#define G_RATE 0.586611f
#define B_RATE 0.114478f
void grayscale_gold(imgStream & t_out, const imgStream & inp, const int h,
                    const int w)
{
    float rr, gg, bb;
    float v;
    int i, j;
    int idx;
    for (i = 0; i < h; ++i) {
        for (j = 0; j < w; ++j) {
            idx = i * w + j;
            rr = (float) inp.data_r[idx];
            gg = (float) inp.data_g[idx];
            bb = (float) inp.data_b[idx];
            v = R_RATE * rr + G_RATE * gg + B_RATE * bb;
            t_out.data_r[idx] = (pixel_uc) v;
        }
    }
}

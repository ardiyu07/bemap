#include <iostream>
#include <cmath>
#include <c_util_img.hpp>

#define FILTER_SIZE 11
#define IDX( i, j, k )  ( ( i * width + j ) + k )

void gaussian_fast_gold_bw(imgStream & inp, imgStream & t_out, int s, int h,
                           int w);
void gaussian_fast_gold_rgb(imgStream & inp, imgStream & t_out, int s, int h,
                            int w);
void gaussian_gold_bw(imgStream & inp, imgStream & t_out, int s, int h, int w);
void gaussian_gold_rgb(imgStream & inp, imgStream & t_out, int s, int h, int w);

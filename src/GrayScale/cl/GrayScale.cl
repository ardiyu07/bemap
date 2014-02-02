/* Enable uchar byte addressing */
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

/* enumerators */
#define clGlobIdx get_global_id(0)
#define clGlobIdy get_global_id(1)
#define clLocIdx  get_local_id(0)
#define clLocIdy  get_local_id(1)
#define clGrpIdx  get_group_id(0)
#define clGrpIdy  get_group_id(1)

typedef uchar pixel_uc;
typedef float number;

__constant number R_RATE = 0.298912f;
__constant number G_RATE = 0.586611f;
__constant number B_RATE = 0.114478f;

__kernel
void grayscale_scalar (__global pixel_uc* out, 
                       __global pixel_uc* inp_r,
                       __global pixel_uc* inp_g, 
                       __global pixel_uc* inp_b,
                       int w,
                       int h)
{
    int tid = clGlobIdx;
    int i = tid / w;
    int j = tid % w;
    int idx;
    number rr,gg,bb;
    number v;

    if (i < h && j < w) {
        idx = i*w+j;
        rr  = (number)inp_r[idx];
        gg  = (number)inp_g[idx];
        bb  = (number)inp_b[idx];
        v = R_RATE*rr+G_RATE*gg+B_RATE*bb;
  
        out[idx] = (pixel_uc)v;  
    }
}

__constant float8 R_RATE8 = (float8)(0.298912f);
__constant float8 G_RATE8 = (float8)(0.586611f);
__constant float8 B_RATE8 = (float8)(0.114478f);

__kernel
void grayscale_simd (__global pixel_uc* out, 
                     __global pixel_uc* inp_r,
                     __global pixel_uc* inp_g, 
                     __global pixel_uc* inp_b,
                     int w,
                     int h)
{
    int i = clGlobIdx;
    int j = 0;
    int idx;

    if (i >= h) return;

    for (j = 0; j < w - 7; j+=8) {
        float8 rr,gg,bb;
        float8 v;
        idx = i*w+j;
        rr = convert_float8(vload8(0, &inp_r[idx]));
        gg = convert_float8(vload8(0, &inp_g[idx]));
        bb = convert_float8(vload8(0, &inp_b[idx]));
        v = R_RATE8*rr+G_RATE8*gg+B_RATE8*bb;    
        vstore8(convert_uchar8_rtn(v), 0, &out[idx]);
    }

    for (; j < w; ++j) {
        number rr,gg,bb;
        number v;
        idx = i*w+j;
        rr  = (number)inp_r[idx];
        gg  = (number)inp_g[idx];
        bb  = (number)inp_b[idx];
        v = R_RATE*rr+G_RATE*gg+B_RATE*bb;  
        out[idx] = (pixel_uc)v;  
    }
}

#include "GaussianFilter_gold.hpp"
    
/* arbitrary circularly symmetric filter -> AT * A */
static const float filter[121] =        
{ 
    1.6287563414420342e-8f, 2.712122801646884e-7f, 0.00000241728874041747f, 0.000011532239604305994f, 0.0000294486054814111f, 0.00004025147128665112f, 0.0000294486054814111f, 0.000011532239604305994f, 0.00000241728874041747f, 2.712122801646884e-7f, 1.6287563414420342e-8f, 
    2.712122801646884e-7f, 0.00000451608991723132f, 0.00004025147128665112f, 0.00019202902969023768f, 0.0004903633058590361f, 0.0006702471714075378f, 0.0004903633058590361f, 0.00019202902969023768f, 0.00004025147128665112f, 0.00000451608991723132f, 2.712122801646884e-7f, 
    0.00000241728874041747f, 0.00004025147128665112f, 0.000358757458428415f, 0.0017115361112027578f, 0.004370560570661311f, 0.005973848012178047f, 0.004370560570661311f, 0.0017115361112027578f, 0.000358757458428415f, 0.00004025147128665112f, 0.00000241728874041747f, 
    0.000011532239604305994f, 0.00019202902969023768f, 0.0017115361112027578f, 0.008165282117850574f, 0.02085077833825263f, 0.02849963493572827f, 0.02085077833825263f, 0.008165282117850574f, 0.0017115361112027578f, 0.00019202902969023768f, 0.000011532239604305994f, 
    0.0000294486054814111f, 0.0004903633058590361f, 0.004370560570661311f, 0.02085077833825263f, 0.05324432775696791f, 0.07277636733051646f, 0.05324432775696791f, 0.02085077833825263f, 0.004370560570661311f, 0.0004903633058590361f, 0.0000294486054814111f, 
    0.00004025147128665112f, 0.0006702471714075378f, 0.005973848012178047f, 0.02849963493572827f, 0.07277636733051646f, 0.09947350008815053f, 0.07277636733051646f, 0.02849963493572827f, 0.005973848012178047f, 0.0006702471714075378f, 0.00004025147128665112f, 
    0.0000294486054814111f, 0.0004903633058590361f, 0.004370560570661311f, 0.02085077833825263f, 0.05324432775696791f, 0.07277636733051646f, 0.05324432775696791f, 0.02085077833825263f, 0.004370560570661311f, 0.0004903633058590361f, 0.0000294486054814111f, 
    0.000011532239604305994f, 0.00019202902969023768f, 0.0017115361112027578f, 0.008165282117850574f, 0.02085077833825263f, 0.02849963493572827f, 0.02085077833825263f, 0.008165282117850574f, 0.0017115361112027578f, 0.00019202902969023768f, 0.000011532239604305994f, 
    0.00000241728874041747f, 0.00004025147128665112f, 0.000358757458428415f, 0.0017115361112027578f, 0.004370560570661311f, 0.005973848012178047f, 0.004370560570661311f, 0.0017115361112027578f, 0.000358757458428415f, 0.00004025147128665112f, 0.00000241728874041747f, 
    2.712122801646884e-7f, 0.00000451608991723132f, 0.00004025147128665112f, 0.00019202902969023768f, 0.0004903633058590361f, 0.0006702471714075378f, 0.0004903633058590361f, 0.00019202902969023768f, 0.00004025147128665112f, 0.00000451608991723132f, 2.712122801646884e-7f, 
    1.6287563414420342e-8f, 2.712122801646884e-7f, 0.00000241728874041747f, 0.000011532239604305994f, 0.0000294486054814111f, 0.00004025147128665112f, 0.0000294486054814111f, 0.000011532239604305994f, 0.00000241728874041747f, 2.712122801646884e-7f, 1.6287563414420342e-8f
};

static const float filter_src[11] = 
{ 
    0.0001276227386260785f, 0.0021251093894741795f, 0.018940893812817154f, 0.09036195060892929f, 0.23074732448496107f, 0.31539419793038453f, 0.23074732448496107f, 0.09036195060892929f, 0.018940893812817154f, 0.0021251093894741795f, 0.0001276227386260785f
};

void gaussian_fast_gold_bw(imgStream & inp, imgStream & t_out, int s,
                           int h, int w)
{
    float px;
    int i, j;
    int height = inp.height;
    int width = inp.width;
    int fil_half = FILTER_SIZE / 2;
     
    /* Execute to row first */ 
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            px = 0.0f;
            int fil_idx = 0;
            int jj;
            for (jj = j - fil_half; jj <= j + fil_half; jj++)
            {
                int jjj;
                if (jj < 0)
                    jjj = 0;
                else if (jj >= w)
                    jjj = w - 1;
                else
                    jjj = jj;

                px += (float) inp.data_r[(i * width + jjj)] *
                    filter_src[fil_idx];
                fil_idx++;
            } 

            if (px < 0.0f)
                px = 0.0f;
            else if (px > 255.0f)
                px = 255.0f;

            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) px;
        }
    }
     
    /* Execute to column */ 
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            px = 0.0f;
            int fil_idx = 0;
            int ii;
            for (ii = i - fil_half; ii <= i + fil_half; ii++)
            {
                int iii;
                iii = max(0, min(h - 1, ii));
                if (ii < 0)
                    iii = 0;
                else if (ii >= h)
                    iii = h - 1;
                else
                    iii = ii;

                px += (float) t_out.data_r[(iii * width + j)] *
                    filter_src[fil_idx];
                fil_idx++;
            } 

            if (px < 0.0f)
                px = 0.0f;
            else if (px > 255.0f)
                px = 255.0f;

            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) px;
        }
    }
}

void gaussian_fast_gold_rgb(imgStream & inp, imgStream & t_out, int s,
                            int h, int w) 
{
    float rr, gg, bb;
    int i, j;
    int height = inp.height;
    int width = inp.width;
    int fil_half = FILTER_SIZE / 2;
     
    /* Execute to row first */ 
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            rr = 0;
            gg = 0;
            bb = 0;
            int fil_idx = 0;
            int jj;
            for (jj = j - fil_half; jj <= j + fil_half; jj++)
            {
                int jjj;
                if (jj < 0)
                    jjj = 0;
                else if (jj >= w)
                    jjj = w - 1;
                else
                    jjj = jj;

                rr +=
                    (float) inp.data_r[(i * width + jjj)] *
                    filter_src[fil_idx];
                gg +=
                    (float) inp.data_g[(i * width + jjj)] *
                    filter_src[fil_idx];
                bb +=
                    (float) inp.data_b[(i * width + jjj)] *
                    filter_src[fil_idx];
                fil_idx++;
            } 

            if (rr < 0.0f)
                rr = 0.0f;
            else if (rr > 255.0f)
                rr = 255.0f;

            if (gg < 0.0f)
                gg = 0.0f;
            else if (gg > 255.0f)
                gg = 255.0f;

            if (bb < 0.0f)
                bb = 0.0f;
            else if (bb > 255.0f)
                bb = 255.0f;

            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) rr;
            t_out.data_g[IDX(i, j, 0)] = (pixel_uc) gg;
            t_out.data_b[IDX(i, j, 0)] = (pixel_uc) bb;
        }
    }
     
    /* Execute to column */ 
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            rr = 0;
            gg = 0;
            bb = 0;
            int fil_idx = 0;
            int ii;
            for (ii = i - fil_half; ii <= i + fil_half; ii++)
            {
                int iii;
                if (ii < 0)
                    iii = 0;
                else if (ii >= h)
                    iii = h - 1;
                else
                    iii = ii;
                
                rr +=
                    (float) t_out.data_r[(iii * width + j)] *
                    filter_src[fil_idx];
                gg +=
                    (float) t_out.data_g[(iii * width + j)] *
                    filter_src[fil_idx];
                bb +=
                    (float) t_out.data_b[(iii * width + j)] *
                    filter_src[fil_idx];
                fil_idx++;
            } 

            if (rr < 0.0f)
                rr = 0.0f;
            else if (rr > 255.0f)
                rr = 255.0f;

            if (gg < 0.0f)
                gg = 0.0f;
            else if (gg > 255.0f)
                gg = 255.0f;

            if (bb < 0.0f)
                bb = 0.0f;
            else if (bb > 255.0f)
                bb = 255.0f;

            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) rr;
            t_out.data_g[IDX(i, j, 0)] = (pixel_uc) gg;
            t_out.data_b[IDX(i, j, 0)] = (pixel_uc) bb;
        }
    }
}

void gaussian_gold_bw(imgStream & inp, imgStream & t_out, int s, int h,
                      int w) 
{
    float px;
    int i, j;
    int height = inp.height;
    int width = inp.width;
    int fil_half = FILTER_SIZE / 2;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            px = 0.0f;
            int fil_idx = 0;
            int ii, jj;
            for (ii = i - fil_half; ii <= i + fil_half; ii++) 
            {
                for (jj = j - fil_half; jj <= j + fil_half; jj++)
                {
                    int iii;
                    if (ii < 0)
                        iii = 0;
                    else if (ii >= h)
                        iii = h - 1;
                    else
                        iii = ii;

                    int jjj;
                    if (jj < 0)
                        jjj = 0;
                    else if (jj >= w)
                        jjj = w - 1;
                    else
                        jjj = jj;

                    px += (float) inp.data_r[(iii * width + jjj)] *
                        filter[fil_idx];
                    fil_idx++;
                }
            }
            if (px < 0.0f)
                px = 0.0f;
            else if (px > 255.0f)
                px = 255.0f;
                
            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) px;
        }
    }
}

void gaussian_gold_rgb(imgStream & inp, imgStream & t_out, int s, int h,
                       int w) 
{
    float rr, gg, bb;
    int i, j;
    int height = inp.height;
    int width = inp.width;
    int fil_half = FILTER_SIZE / 2;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            rr = 0;
            gg = 0;
            bb = 0;
            int fil_idx = 0;
            int ii, jj;
            for (ii = i - fil_half; ii <= i + fil_half; ii++) 
            {
                for (jj = j - fil_half; jj <= j + fil_half; jj++)
                {
                    int iii;
                    if (ii < 0)
                        iii = 0;
                    
                    else if (ii >= h)
                        iii = h - 1;
                    
                    else
                        iii = ii;
                    int jjj;
                    if (jj < 0)
                        jjj = 0;
                    
                    else if (jj >= w)
                        jjj = w - 1;
                    
                    else
                        jjj = jj;
                    rr +=
                        (float) inp.data_r[(iii * width + jjj)] *
                        filter[fil_idx];
                    gg +=
                        (float) inp.data_g[(iii * width + jjj)] *
                        filter[fil_idx];
                    bb +=
                        (float) inp.data_b[(iii * width + jjj)] *
                        filter[fil_idx];
                    fil_idx++;
                }
            }
            if (rr < 0.0f)
                rr = 0.0f;
            else if (rr > 255.0f)
                rr = 255.0f;

            if (gg < 0.0f)
                gg = 0.0f;
            else if (gg > 255.0f)
                gg = 255.0f;

            if (bb < 0.0f)
                bb = 0.0f;
            else if (bb > 255.0f)
                bb = 255.0f;

            t_out.data_r[IDX(i, j, 0)] = (pixel_uc) rr;
            t_out.data_g[IDX(i, j, 0)] = (pixel_uc) gg;
            t_out.data_b[IDX(i, j, 0)] = (pixel_uc) bb;
        }
    }
}

#include <cstdio>
#include <cstring>
#include <cmath>
#include "BackProjection_gold.hpp"

void BackProjection_gold(int r, int c, unsigned char *guess, int *rproj, int *rband, int *cproj, int *cband, int *uproj, int *uband, int *dproj, int *dband, int *maxId_p)
{
	F_TYPE r_tmp, c_tmp, u_tmp, d_tmp;

	/* inf loop */
    while (true) {		

		int maxi = -1, maxj = -1;
		F_TYPE imgVal, maxVal = 0.0;

		/* find max pixel value */
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				imgVal = 0.0;
				if (guess[i * c + j] != TRUE) {
					r_tmp = (F_TYPE)rproj[i] / (F_TYPE)rband[i];
					c_tmp = (F_TYPE)cproj[j] / (F_TYPE)cband[j];
					u_tmp = (F_TYPE)uproj[i + j] / (F_TYPE)uband[i + j];
					d_tmp = (F_TYPE)dproj[i - j + c - 1] / (F_TYPE)dband[i - j + c - 1];					
					imgVal = r_tmp * c_tmp * u_tmp * d_tmp;					
				}				
				if (imgVal > maxVal) {
					maxVal = imgVal;
					maxi = i;
					maxj = j;					
				}
			}
		}

		/* break if all pixel values are the same */
        if (maxi == - 1 || maxj == - 1)
        {
            break;
        }

        *maxId_p = maxi * c + maxj;
        guess[maxi * c + maxj] = TRUE;

		/* decrease projection */
        decreaseproj(maxi, maxj, c, rproj, rband, cproj, cband, uproj, uband, dproj, dband);
    }
}

void makeband(int r, int c, int *rband, int *cband, int *uband, int *dband)
{
    int k, m, min;

    for (k = 0; k < r; k++)
    {
        rband[k] = c;
    }

    for (k = 0; k < c; k++)
    {
        cband[k] = r;
    }

    for (k = 0, m = r + c - 2; k < r && k < c ; k++, m--)
    {
        uband[k] = k + 1;
        uband[m] = k + 1;

        dband[k] = k + 1;
        dband[m] = k + 1;
    }

    min = k;
    for (k = k; k <= m; k++, m--)
    {
        uband[k] = min;
        uband[m] = min;

        dband[k] = min;
        dband[m] = min;
    }
}

void decreaseproj(int i, int j, int c, int *rproj, int *rband, int *cproj, int *cband, int *uproj, int *uband, int *dproj, int *dband)
{
    rproj[i]--;
    rband[i]--;

    cproj[j]--;
    cband[j]--;

    uproj[i + j]--;
    uband[i + j]--;

    dproj[i - j + c - 1]--;
    dband[i - j + c - 1]--;
}

void write_bmp(const char *filename, unsigned char *guess, int c, int r)
{
    int i, j;
    FILE *fp;
    int real_width;
    unsigned char *bmp_line_data;
    unsigned char header_buf[HEADERSIZE];
    unsigned int file_size;
    unsigned int offset_to_data;
    unsigned long info_header_size;
    unsigned int planes;
    unsigned int color;
    unsigned long compress;
    unsigned long data_size;
    long xppm;
    long yppm;

    if((fp = fopen(filename, "wb")) == NULL){
        fprintf(stderr, "Error: %s could not open.", filename);
        return;
    }

    real_width = r * 3 + r % 4;

    file_size = r * c + HEADERSIZE;
    offset_to_data = HEADERSIZE;
    info_header_size = INFOHEADERSIZE;
    planes = 1;
    color = 24;
    compress = 0;
    data_size = r * c;
    xppm = 1;
    yppm = 1;
  
    header_buf[0] = 'B';
    header_buf[1] = 'M';
    memcpy(header_buf + 2, &file_size, sizeof(file_size));
    header_buf[6] = 0;
    header_buf[7] = 0;
    header_buf[8] = 0;
    header_buf[9] = 0;
    memcpy(header_buf + 10, &offset_to_data, sizeof(file_size));
    header_buf[11] = 0;
    header_buf[12] = 0;
    header_buf[13] = 0;

    memcpy(header_buf + 14, &info_header_size, sizeof(info_header_size));
    header_buf[15] = 0;
    header_buf[16] = 0;
    header_buf[17] = 0;
    memcpy(header_buf + 18, &r, sizeof(r));
    memcpy(header_buf + 22, &c, sizeof(c));
    memcpy(header_buf + 26, &planes, sizeof(planes));
    memcpy(header_buf + 28, &color, sizeof(color));
    memcpy(header_buf + 30, &compress, sizeof(compress));
    memcpy(header_buf + 34, &data_size, sizeof(data_size));
    memcpy(header_buf + 38, &xppm, sizeof(xppm));
    memcpy(header_buf + 42, &yppm, sizeof(yppm));
    header_buf[46] = 0;
    header_buf[47] = 0;
    header_buf[48] = 0;
    header_buf[49] = 0;
    header_buf[50] = 0;
    header_buf[51] = 0;
    header_buf[52] = 0;
    header_buf[53] = 0;

    fwrite(header_buf, sizeof(unsigned char), HEADERSIZE, fp);
  
    if((bmp_line_data = (unsigned char *)malloc(sizeof(unsigned char)*real_width)) == NULL){
        fprintf(stderr, "Error: Allocation error.\n");
        fclose(fp);
        return;
    }

    for(i=0; i<c; i++){
        for(j=0; j<r; j++){
            bmp_line_data[j*3]=guess[(c - i - 1)*r + j] * COLOR;
            bmp_line_data[j*3 + 1]=guess[(c - i - 1)*r + j] * COLOR;
            bmp_line_data[j*3 + 2]=guess[(c - i - 1)*r + j] * COLOR;
        }
        for(j=r*3; j<real_width; j++){
            bmp_line_data[j] = 0;
        }
        fwrite(bmp_line_data, sizeof(unsigned char), real_width, fp);
    }

    free(bmp_line_data);

    fclose(fp);

    return;
}

void create_image(int r, int c, unsigned char *input, int num, F_TYPE radius)
{
    int i, j;
    int x, y;

    for (i = 0; i < num; i++)
    {
        x = (int)(radius * cos((F_TYPE)i * 2.0 * M_PI / (F_TYPE)num)) + r / 2 - 1;
        y = (int)(radius * sin((F_TYPE)i * 2.0 * M_PI / (F_TYPE)num)) + c / 2 - 1;

        input[x * c + y] = TRUE;
    }
}

void create_input(int r, int c, unsigned char *input, int *rproj, int *cproj, int *uproj, int *dproj, int *uband, int *dband)
{
    int i, j, k, l;
    int si, sj;
    int dif;

    for (i = 0; i < r; i++)
    {
        rproj[i] = 0;

        if (i < r) {
            for (j = 0; j < c; j++)
            {
                if (input[i * c + j] == TRUE)
                {
                    rproj[i]+=MAX_SCORE;
                }
            }
        }
    }

    for (i = 0; i < c; i++)
    {
        cproj[i] = 0;

        if (i < c) {
            for (j = 0; j < r; j++)
            {
                if (input[j * c + i] == TRUE)
                {
                    cproj[i]+=MAX_SCORE;
                }
            }
        }
    }

    for (i = 0; i < r + c - 1; i++)
    {
        uproj[i] = 0;

        si = i;
        sj = 0;
        if (si > r - 1)
        {
            si = r - 1;
            sj = i - r + 1;
        }

        for (k = si, l = sj; k >= 0 && l < c; k--, l++)
        {
            if (input[k * c + l] == TRUE)
            {
                uproj[i]+=MAX_SCORE;
            }
        }
    }

    for (i = 0; i < r + c - 1; i++)
    {
        dproj[i] = 0;
    }
    for (i = 0; i < r + c - 1; i++)
    {
        si = r - 1 - i;
        sj = 0;
        if (si < 0)
        {
            si = 0;
            sj = i - r + 1;
        }

        for (k = si, l = sj; k < r && l < c; k++, l++)
        {
            if (input[k * c + l] == TRUE)
            {
                dproj[i]+=MAX_SCORE;
            }
        }
    }
}

void printimage(int r, int c, unsigned char *input, const char *filename)
{
    int i, j;


    if (filename != NULL)
    {
        FILE *fp;


        fp = fopen(filename, "w");
        if (fp == NULL)
        {
            fprintf(stderr, "file not opened\n");

            exit(EXIT_FAILURE);
        }

        for (i = 0; i < r; i++)
        {
            for (j = 0; j < c; j++)
            {
                if (input[i * c + j] == TRUE)
                {
                    fprintf(fp, "o ");
                }
                else
                {
                    fprintf(fp, "+ ");
                }
            }
            fprintf(fp, "\n");
        }
        fprintf(fp, "\n");

        fclose(fp);
    }
    else
    {
        for (i = 0; i < r; i++)
        {
            for (j = 0; j < c; j++)
            {
                if (input[i * c + j] == TRUE)
                {
                    fprintf(stdout, "o ");
                }
                else
                {
                    fprintf(stdout, "+ ");
                }
            }
            fprintf(stdout, "\n");
        }
        fprintf(stdout, "\n");
    }
}

void printproj(int r, int c, int *rproj, int *cproj, int *uproj, int *dproj)
{
    int i;

    for (i = 0; i < r; i++)
    {
        fprintf(stdout, "%d ", rproj[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < c; i++)
    {
        fprintf(stdout, "%d ", cproj[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < r + c - 1; i++)
    {
        fprintf(stdout, "%d ", uproj[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < r + c - 1; i++)
    {
        fprintf(stdout, "%d ", dproj[i]);
    }
    fprintf(stdout, "\n");
}


void printband(int r, int c, int *rband, int *cband, int *uband, int *dband)
{
    int i;


    for (i = 0; i < r; i++)
    {
        fprintf(stdout, "%d ", rband[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < c; i++)
    {
        fprintf(stdout, "%d ", cband[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < r + c - 1; i++)
    {
        fprintf(stdout, "%d ", uband[i]);
    }
    fprintf(stdout, "\n");

    for (i = 0; i < r + c - 1; i++)
    {
        fprintf(stdout, "%d ", dband[i]);
    }
    fprintf(stdout, "\n");
}

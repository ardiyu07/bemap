#include <c_util.hpp>
#include "BackProjection_common.hpp"

#define FILEHEADERSIZE 14
#define INFOHEADERSIZE 40
#define HEADERSIZE (FILEHEADERSIZE+INFOHEADERSIZE)
#define COLOR 255

void BackProjection_gold(int r, int c, unsigned char *guess, int *rproj, int *rband, int *cproj, int *cband, int *uproj, int *uband, int *dproj, int *dband, int *maxId_p);
void makeband(int r, int c, int *rband, int *cband, int *uband, int *dband);
void decreaseproj(int i, int j, int c, int *rproj, int *rband, int *cproj, int *cband, int *uproj, int *uband, int *dproj, int *dband);
void write_bmp(const char *filename, unsigned char *guess, int r, int c);
void create_image(int r, int c, unsigned char *input, int num, F_TYPE radius);
void create_input(int r, int c, unsigned char *input, int *rproj, int *cproj, int *uproj, int *dproj, int *uband, int *dband);
void printimage(int r, int c, unsigned char *input, const char *filename);
void printproj(int r, int c, int *rproj, int *cproj, int *uproj, int *dproj);
void printband(int r, int c, int *rband, int *cband, int *uband, int *dband);

#define MAX(a, b) ((a>=b)?(a):(b))
#define MIN(a, b) ((a<=b)?(a):(b))

#define   ROWS_BLOCKDIM_X 128
#define   ROWS_BLOCKDIM_Y 1

#define COLUMNS_BLOCKDIM_X   16
#define COLUMNS_BLOCKDIM_Y   32
#define COLUMNS_RESULT_STEPS 4
#define COLUMNS_BLOCKMEM_Y   (COLUMNS_BLOCKDIM_Y * COLUMNS_RESULT_STEPS)

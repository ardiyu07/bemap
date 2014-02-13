#include "BackProjection_common.hpp"

__kernel 
void dev_init (
			   __global F_TYPE *image,
			   __global int *index
			   )
{
	image[get_global_id(0)] = 0;
}

__kernel 
void dev_init_simd (
					int r,
					int c,
					__global F_TYPE *image,
					__global unsigned char *guess,
					__global F_TYPE *rscore,
					__global F_TYPE *cscore,
					__global F_TYPE *uscore,
					__global F_TYPE *dscore,
					__global int *index
					)
{
	int i, j;
	int index_tmp;
	F_TYPE image_tmp, r_tmp;

	i = get_global_id(0);

	r_tmp = rscore[i];
	for (j = 0; j < c; j++) {
		image_tmp = r_tmp;
		index_tmp = i * c + j;
		if (r_tmp > 0.0) {
			image_tmp = image_tmp * cscore[j] * uscore[i + j] * dscore[i - j + c - 1];
		}
		image[i * c + j] = image_tmp;
		index[i * c + j] = index_tmp;
	}
}

__kernel 
void dev_backprojection_scalar (
								int r,
								int c,
								__global F_TYPE *image,
								__global unsigned char *guess,
								__global F_TYPE *rscore,
								__global F_TYPE *cscore,
								__global F_TYPE *uscore,
								__global F_TYPE *dscore,
								__global int *index,
								__local F_TYPE *l_image,
								__local int *l_index,
								__local F_TYPE *r_local,
								__local F_TYPE *u_local,
								__local F_TYPE *d_local
								)
{
	int lj, li, lsize, i, j, index_result;
	F_TYPE c_tmp, image_result;
	F_TYPE image_tmp;

	lj = get_local_id(0);
	lsize = get_local_size(0);

	i = get_global_id(1) * get_local_size(0);
	j = get_group_id(0) * get_local_size(0);

	image_result = 0.0;
	index_result = -1;

	if (j + lj < c)
		c_tmp = cscore[ j + lj ];
	if (i + lj < r)
		r_local[lj] = rscore[ i + lj ];
	if (i + j + lj < r + c - 1)
		u_local[lj] = uscore[ i + j + lj ];
	if (i + j + lj + lsize < r + c - 1)
		u_local[lj + lsize] = uscore[ i + j + lj + lsize];
	if (i - j + c - lsize + lj < r + c - 1)
		d_local[lj] = dscore[ i - j + c - lsize + lj ]; 
	if (i - j + c + lj < r + c - 1)
		d_local[lj + lsize] = dscore[ i - j + c + lj ];

	barrier(CLK_LOCAL_MEM_FENCE);

	j = get_global_id(0);

	if (j < c || c_tmp <= 0) {
		for (li = 0; li < lsize && i + li < r; li++) {
			image_tmp = r_local[li];
			if ( image_tmp <= 0.0 ) {
				continue;
			}
			if ( guess[ (i + li) * c + j ] == TRUE ) {
				continue;
			} else {
				image_tmp = image_tmp * c_tmp * u_local[lj + li] * d_local[lsize - (lj + 1) + li];
			}
			if (image_tmp <= 0.0) {
				continue;
			}
			if (image_result < image_tmp) {
				image_result = image_tmp;
				index_result = (i + li) * c + j;
			}
		}
	}

	l_image[lj] = image_result;
	l_index[lj] = index_result;
	barrier(CLK_LOCAL_MEM_FENCE);

	for ( i = get_local_size(0) / 2; i > 0; i /= 2 ) {
		if ( lj < i ) {
			if (l_image[ lj ] < l_image[ lj + i ]) {
				l_image[ lj ] = l_image[ lj + i ];
				l_index[ lj ] = l_index[ lj + i ];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if( lj == 0 ) {
		image[ get_group_id(1) * get_num_groups(1) + get_group_id(0)] = l_image[0];
		index[ get_group_id(1) * get_num_groups(1) + get_group_id(0)] = l_index[0];
	}
}

__kernel 
void dev_backprojection_simd (
							  int r,
							  int c,
							  __global F_TYPE *image,
							  __global unsigned char *guess,
							  __global F_TYPE *rscore,
							  __global F_TYPE *cscore,
							  __global F_TYPE *uscore,
							  __global F_TYPE *dscore,
							  __global int *maxId
							  )
{
	int k, l, maxi, maxj;
	F_TYPE r_tmp, c_tmp, u_tmp, d_tmp;
	F_TYPE image_tmp;

	maxi = maxId[0] / c;
	maxj = maxId[0] % c;

	if (get_global_id(1) == 0) {
		r_tmp = rscore[maxi];
		for (k = maxi, l = 0; l < c; l ++) {
			if (guess[k * c + l] == TRUE || r_tmp <= 0.0) {
				image_tmp = 0.0;
			} else {
				c_tmp = cscore[l];
				u_tmp = uscore[k + l];
				d_tmp = dscore[k - l + c - 1];
				image_tmp = r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp;
		}
	} else if (get_global_id(1) == 1) {
		c_tmp = cscore[maxj];
		for (k = 0, l = maxj; k < r; k ++) {
			if (guess[k * c + l] == TRUE || c_tmp <= 0.0) {
				image_tmp = 0.0;
			} else {
				r_tmp = rscore[k];
				u_tmp = uscore[k + l];
				d_tmp = dscore[k - l + c - 1];
				image_tmp = r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp;
		}
	} else if (get_global_id(1) == 2) {
		u_tmp = uscore[maxi + maxj];
		if (maxi + maxj < c) {
			k = 0;
			l = maxi + maxj;
		} else {
			k = maxi + maxj - c;
			l = c;
		} 
		for (; l >= 0 && k < r; k ++, l --) {
			if (guess[k * c + l] == TRUE || u_tmp <= 0.0) {
				image_tmp = 0.0;
			} else {
				r_tmp = rscore[k];
				c_tmp = cscore[l];
				d_tmp = dscore[k - l + c - 1];
				image_tmp = r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp;
		}
	} else if (get_global_id(1) == 3) {
		d_tmp = dscore[maxi - maxj + c - 1];
		if(maxi < maxj) {
			k = 0;
			l = maxj - maxi;
		} else {
			k = maxi - maxj;
			l = 0;
		}
		for (; k < r && l < c; k ++, l ++) {
			if (guess[k * c + l] == TRUE || d_tmp <= 0.0) {
				image_tmp = 0.0;
			} else {
				r_tmp = rscore[k];
				c_tmp = cscore[l];
				u_tmp = uscore[k + l];
				image_tmp = r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp;
		}
	}
}

__kernel void 
dev_backprojection_simd4 (
						  int r,
						  int c,
						  __global F_TYPE *image,
						  __global unsigned char *guess,
						  __global F_TYPE *rscore,
						  __global F_TYPE *cscore,
						  __global F_TYPE *uscore,
						  __global F_TYPE *dscore,
						  __global int *maxId
						  )
{
	int k, l, maxi, maxj;
	F4_TYPE r_tmp, c_tmp, u_tmp, d_tmp;
	F4_TYPE image_tmp;

	maxi = maxId[0] / c;
	maxj = maxId[0] % c;

	if (get_global_id(1) == 0) {
		r_tmp = (F4_TYPE)rscore[maxi];
		for (k = maxi, l = 0; l < c - 4; l += 4) {
			if (r_tmp.x <= 0.0) {
				image_tmp = (F4_TYPE)0.0;
			} else {
				c_tmp = (F4_TYPE)(cscore[l], cscore[l + 1], cscore[l + 2], cscore[l + 3]);
				u_tmp = (F4_TYPE)(uscore[k + l], uscore[k + l + 1], uscore[k + l + 2], uscore[k + l + 3]);
				d_tmp = (F4_TYPE)(dscore[k - l + c - 1], dscore[k - l + c - 2], dscore[k - l + c - 3], dscore[k - l + c - 4]);
				image_tmp = (F4_TYPE)(guess[k * c + l], guess[k * c + l + 1], guess[k * c + l + 2], guess[k * c + l + 3]) == (F4_TYPE)TRUE
					?  (F4_TYPE)0.0 : r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp.x;
			image[k * c + l + 1] = image_tmp.y;
			image[k * c + l + 2] = image_tmp.z;
			image[k * c + l + 3] = image_tmp.w;
		}
		for (; l < c; l ++) {
			c_tmp.x = cscore[l];
			u_tmp.x = uscore[k + l];
			d_tmp.x = dscore[k - l + c - 1];
			if (guess[k * c + l] == TRUE) {
				image_tmp.x = 0.0;
			} else {
				image_tmp.x = r_tmp.x * c_tmp.x * u_tmp.x * d_tmp.x;
			}
			image[k * c + l] = image_tmp.x;
		}
	} else if (get_global_id(1) == 1) {
		c_tmp = (F4_TYPE)cscore[maxj];
		for (k = 0, l = maxj; k < r; k += 4) {
			if (c_tmp.x <= 0.0) {
				image_tmp = (F4_TYPE)0.0;
			} else {
				r_tmp = (F4_TYPE)(rscore[k], rscore[k + 1], rscore[k + 2], rscore[k + 3]);
				u_tmp = (F4_TYPE)(uscore[k + l], uscore[k + l + 1], uscore[k + l + 2], uscore[k + l + 3]);
				d_tmp = (F4_TYPE)(dscore[k - l + c - 1], dscore[k - l + c], dscore[k - l + c + 1], dscore[k - l + c + 2]);
				image_tmp = (F4_TYPE)(guess[k * c + l], guess[k * c + l + 1], guess[k * c + l + 2], guess[k * c + l + 3]) == (F4_TYPE)TRUE
					?  (F4_TYPE)0.0 : r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp.x;
			image[(k + 1) * c + l] = image_tmp.y;
			image[(k + 2) * c + l] = image_tmp.z;
			image[(k + 3) * c + l] = image_tmp.w;
		}
		for (; k < r; k ++) {
			r_tmp.x = rscore[k];
			u_tmp.x = uscore[k + l];
			d_tmp.x = dscore[k - l + c - 1];
			if (guess[k * c + l] == TRUE) {
				image_tmp.x = 0.0;
			} else {
				image_tmp.x = r_tmp.x * c_tmp.x * u_tmp.x * d_tmp.x;
			}
			image[k * c + l] = image_tmp.x;
		}
	} else if (get_global_id(1) == 2) {
		u_tmp = (F4_TYPE)uscore[maxi + maxj];
		if (maxi + maxj < c) {
			k = 0;
			l = maxi + maxj;
		} else {
			k = maxi + maxj - c;
			l = c;
		} 
		for (; l >= 4 && k < r - 4; k += 4, l -= 4) {
			if (u_tmp.x <= 0.0) {
				image_tmp = (F4_TYPE)0.0;
			} else {
				r_tmp = (F4_TYPE)(rscore[k], rscore[k + 1], rscore[k + 2], rscore[k + 3]);
				c_tmp = (F4_TYPE)(cscore[l], cscore[l - 1], cscore[l - 2], cscore[l - 3]);
				d_tmp = (F4_TYPE)(dscore[k - l + c - 1], dscore[k - l + c + 1], dscore[k - l + c + 3], dscore[k - l + c + 5]);
				image_tmp = (F4_TYPE)(guess[k * c + l], guess[k * c + l + 1], guess[k * c + l + 2], guess[k * c + l + 3]) == (F4_TYPE)TRUE
					?  (F4_TYPE)0.0 : r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp.x;
			image[(k + 1) * c + l - 1] = image_tmp.y;
			image[(k + 2) * c + l - 2] = image_tmp.z;
			image[(k + 3) * c + l - 3] = image_tmp.w;
		}
		for (; l >= 0 && k < r; k ++, l --) {
			r_tmp.x = rscore[k];
			c_tmp.x = cscore[l];
			d_tmp.x = dscore[k - l + c - 1];
			if (guess[k * c + l] == TRUE) {
				image_tmp.x = 0.0;
			} else {
				image_tmp.x = r_tmp.x * c_tmp.x * u_tmp.x * d_tmp.x;
			}
			image[k * c + l] = image_tmp.x;
		}
	} else if (get_global_id(1) == 3) {
		d_tmp = (F4_TYPE)dscore[maxi - maxj + c - 1];
		if(maxi < maxj) {
			k = 0;
			l = maxj - maxi;
		} else {
			k = maxi - maxj;
			l = 0;
		}
		for (; k < r - 4 && l < c - 4; k += 4, l += 4) {
			if (d_tmp.x <= 0.0) {
				image_tmp = (F4_TYPE)0.0;
			} else {
				r_tmp = (F4_TYPE)(rscore[k], rscore[k + 1], rscore[k + 2], rscore[k + 3]);
				c_tmp = (F4_TYPE)(cscore[l], cscore[l + 1], cscore[l + 2], cscore[l + 3]);
				u_tmp = (F4_TYPE)(uscore[k + l], uscore[k + l + 2], uscore[k + l + 4], uscore[k + l + 6]);
				image_tmp = (F4_TYPE)(guess[k * c + l], guess[k * c + l + 1], guess[k * c + l + 2], guess[k * c + l + 3]) == (F4_TYPE)TRUE
					?  (F4_TYPE)0.0 : r_tmp * c_tmp * u_tmp * d_tmp;
			}
			image[k * c + l] = image_tmp.x;
			image[(k + 1) * c + l + 1] = image_tmp.y;
			image[(k + 2) * c + l + 2] = image_tmp.z;
			image[(k + 3) * c + l + 3] = image_tmp.w;
		}
		for (; k < r && l < c; k ++, l ++) {
			r_tmp.x = rscore[k];
			c_tmp.x = cscore[l];
			u_tmp.x = uscore[k + l];
			if (guess[k * c + l] == TRUE) {
				image_tmp.x = 0.0;
			} else {
				image_tmp.x = r_tmp.x * c_tmp.x * u_tmp.x * d_tmp.x;
			}
			image[k * c + l] = image_tmp.x;
		}
	}
}

__kernel 
void dev_findmax (
				  __global F_TYPE *image,
				  __global int *index,
				  __global F_TYPE *image2,
				  __global int *index2,
				  __local F_TYPE *l_image,
				  __local int *l_index
				  )
{
	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int i;
  
	l_image[ lid ] = image[ tid ];
	l_index[ lid ] = index[ tid ];
	barrier(CLK_LOCAL_MEM_FENCE);
 
	for ( i = get_local_size(0) / 2; i > 0; i /= 2 ) {
		if ( lid < i ) {
			if (l_image[ lid ] < l_image[ lid + i ]) {
				l_image[ lid ] = l_image[ lid + i ];
				l_index[ lid ] = l_index[ lid + i ];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if ( lid == 0 ) {
		image2[ get_group_id(0) ] = l_image[0];
		if (l_image[0] <= 0.0)
			index2[ get_group_id(0) ] = -1;
		else
			index2[ get_group_id(0) ] = l_index[0];
	}
}
__kernel 
void dev_findmax_simd (
					   __global F_TYPE *image,
					   __global int *index,
					   __global F_TYPE *image2,
					   __global int *index2,
					   int num,
					   __global F_TYPE *rscore
					   )
{
	int tid;
	int i;
	int maxId;
	F_TYPE max, cmp;
  
	if (get_num_groups(0) != 1) {
		if (rscore[get_global_id(0)] <= 0.0) {
			image2[get_group_id(0)] = 0.0;
			index2[get_group_id(0)] = -1;
			return;
		}
	}

	tid = get_global_id(0)*num;
	max = 0.0;
	maxId = tid;

	for (i = 0; i < num; i++) {
		cmp = image[tid + i];
		if(cmp > max) {
			max = cmp;
			maxId = index[tid + i];
		}
	}

	image2[ get_group_id(0) ] = max;
	if(max <= 0.0)
		index2[ get_group_id(0) ] = -1;
	else
		index2[ get_group_id(0) ] = maxId;
}


__kernel 
void dev_findmax_simd4 (
						__global F4_TYPE *image,
						__global int4 *index,
						__global F_TYPE *image2,
						__global int *index2,
						int num,
						__global F_TYPE *rscore
						)
{
	int tid;
	int i, maxId;
	F_TYPE max;
	int4 vmaxId;
	F4_TYPE vmax, vcmp;
  
	if (get_num_groups(0) != 1) {
		if (rscore[get_global_id(0)] <= 0.0) {
			image2[get_group_id(0)] = 0.0;
			index2[get_group_id(0)] = -1;
			return;
		}
	}

	tid = get_global_id(0)*num/4;
	vmax = (F4_TYPE)0.0;
	vmaxId = (int4)0;

	for (i = 0; i < num/4; i ++) {
		vcmp = image[tid+i];
		vmaxId = (vcmp > vmax) ? index[tid+i] : vmaxId;
		vmax = (vcmp > vmax) ? vcmp : vmax;
	}

	max = vmax.x;
	maxId = vmaxId.x;
	if(vmax.y > max) {
		max = vmax.y;
		maxId = vmaxId.y;
	}
	if(vmax.z > max) {
		max = vmax.z;
		maxId = vmaxId.z;
	}
	if(vmax.w > max) {
		max = vmax.w;
		maxId = vmaxId.w;
	}

	image2[ get_group_id(0) ] = max;
	if(max <= 0.0)
		index2[ get_group_id(0) ] = -1;
	else
		index2[ get_group_id(0) ] = maxId;
}


__kernel 
void dev_decreaseproj (
					   __global int *index,
					   int c,
					   __global unsigned char *guess,
					   __global F_TYPE *rscore,
					   __global F_TYPE *cscore,
					   __global F_TYPE *uscore,
					   __global F_TYPE *dscore,
					   __global int *rproj,
					   __global int *rband,
					   __global int *cproj,
					   __global int *cband,
					   __global int *uproj,
					   __global int *uband,
					   __global int *dproj,
					   __global int *dband,
					   __global int *maxId
					   )
{
	int idx = index[0];
	int i = idx / c;
	int j = idx % c;
	int proj, band;

	maxId[0] = idx;
	if(idx<0) return;

	guess[idx] = TRUE;

	proj = rproj[ i ] - 1;
	rproj[i] = proj;
	band = rband[ i ] - 1;
	rband[i] = band;
	rscore[i] = (F_TYPE) proj / (F_TYPE) band;

	proj = cproj[ j ] - 1;
	cproj[j] = proj;
	band = cband[ j ] - 1;
	cband[j] = band;
	cscore[j] = (F_TYPE) proj / (F_TYPE) band;

	proj = uproj[ i+j ] - 1;
	uproj[i+j] = proj;
	band = uband[ i+j ] - 1;
	uband[i+j] = band;
	uscore[i+j] = (F_TYPE) proj / (F_TYPE) band;

	proj = dproj[ i-j+c-1 ] - 1;
	dproj[i-j+c-1] = proj;
	band = dband[ i-j+c-1 ] - 1;
	dband[i-j+c-1] = band;
	dscore[i-j+c-1] = (F_TYPE) proj / (F_TYPE) band;

}

__kernel 
void dev_decreaseproj_gpu (
						   __global int *index,
						   int c,
						   __global unsigned char *guess,
						   __global F_TYPE *rscore,
						   __global F_TYPE *cscore,
						   __global F_TYPE *uscore,
						   __global F_TYPE *dscore,
						   __global int *rproj,
						   __global int *rband,
						   __global int *cproj,
						   __global int *cband,
						   __global int *uproj,
						   __global int *uband,
						   __global int *dproj,
						   __global int *dband,
						   __global int *maxId,
						   __global F_TYPE *image,
						   __local F_TYPE *l_image,
						   __local int *l_index
						   )
{
	int tid = get_global_id(0);
	int lid = get_local_id(0);
	int i, j, idx, proj, band;
  
	l_image[ lid ] = image[ tid ];
	l_index[ lid ] = index[ tid ];
	barrier(CLK_LOCAL_MEM_FENCE);
 
	for ( i = get_local_size(0) / 2; i > 0; i /= 2 ) {
		if ( lid < i ) {
			if (l_image[ lid ] < l_image[ lid + i ]) {
				l_image[ lid ] = l_image[ lid + i ];
				l_index[ lid ] = l_index[ lid + i ];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if ( lid == 0 ) {  
		maxId[ 0 ] = l_index[0];
		image[ 0 ] = l_image[0];

		if(l_image[0] > 0.0) {
			idx = l_index[0];
			i = idx / c;
			j = idx % c;

			guess[idx] = TRUE;

			proj = rproj[ i ] - 1;
			rproj[i] = proj;
			band = rband[ i ] - 1;
			rband[i] = band;
			rscore[i] = (F_TYPE) proj / (F_TYPE) band;

			proj = cproj[ j ] - 1;
			cproj[j] = proj;
			band = cband[ j ] - 1;
			cband[j] = band;
			cscore[j] = (F_TYPE) proj / (F_TYPE) band;

			proj = uproj[ i+j ] - 1;
			uproj[i+j] = proj;
			band = uband[ i+j ] - 1;
			uband[i+j] = band;
			uscore[i+j] = (F_TYPE) proj / (F_TYPE) band;

			proj = dproj[ i-j+c-1 ] - 1;
			dproj[i-j+c-1] = proj;
			band = dband[ i-j+c-1 ] - 1;
			dband[i-j+c-1] = band;
			dscore[i-j+c-1] = (F_TYPE) proj / (F_TYPE) band;
		}
	}
}


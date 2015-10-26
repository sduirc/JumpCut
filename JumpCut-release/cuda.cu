// includes CUDA Runtime
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>

#include <algorithm>
#include <limits>
#include "BITMAP3.h"
#include "cuda.h"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE; }} while (0)

#define MAX_MEXTURE_WIDTH 65536

int BLOCKDIM = 8;
texture<int1, 2, cudaReadModeElementType> texImageA, texImageB;
texture<float1, 1, cudaReadModeElementType> sourceEdgeTex, targetEdgeTex;
cudaArray *a_Src, *b_Src;

__device__ float sqrt7(float x)
{
	unsigned int i = *(unsigned int*)&x;
	// adjust bias
	i += 127 << 23;
	// approximation of square root
	i >>= 1;
	return *(float*)&i;
}

__device__ float sqrt3(const float x)
{
	union
	{
		int i;
		float x;
	} u;

	u.x = x;
	u.i = (1 << 29) + (u.i >> 1) - (1 << 22);
	return u.x;
}

int iDivUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ void simpleImproveGuess(int ax, int ay, int xp, int yp, int &xbest, int &ybest, int &dbest, int patch_w)
{
	int ans = 0;
	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			int1 ac1 = tex2D(texImageA, ax + dx, ay + dy);
			int1 bc1 = tex2D(texImageB, xp + dx, yp + dy);
			int ac = ac1.x, bc = bc1.x;
			int dr = (ac & 255) - (bc & 255);
			int dg = ((ac >> 8) & 255) - ((bc >> 8) & 255);
			int db = (ac >> 16) - (bc >> 16);
			ans += dr*dr + dg*dg + db*db;
		}
	}

	if (ans < dbest) {
		dbest = ans;
		xbest = xp;
		ybest = yp;
	}
}

__device__ void improve_guess(int ax, int ay, int xp, int yp, int &xbest, int &ybest, int &dbest, int diagonalDis, bool leapPropagate, int patch_w)
{
	float alpha = 10, beta = 3;
	if (leapPropagate)
		beta = 1.5;
	double ansColor = 0, ansDis = 0, ans = 0;
	double dxx = ax - xp, dyy = ay - yp;
	double dis = sqrt7(dxx*dxx + dyy*dyy) / diagonalDis*100.0;
	ansDis = dis*patch_w*patch_w;

	for (int dy = 0; dy < patch_w; dy++) {
		for (int dx = 0; dx < patch_w; dx++) {
			int1 ac1 = tex2D(texImageA, ax + dx, ay + dy);
			int1 bc1 = tex2D(texImageB, xp + dx, yp + dy);
			int ac = ac1.x, bc = bc1.x;
			int dr = (ac & 255) - (bc & 255);
			int dg = ((ac >> 8) & 255) - ((bc >> 8) & 255);
			int db = (ac >> 16) - (bc >> 16);
			ansColor += sqrt7((double)(dr*dr + dg*dg + db*db));
		}
	}
	ansColor = ansColor / 255.0*100.0;
	ans = ansColor*alpha + ansDis*beta;

	if (ans < dbest) {
		dbest = ans;
		xbest = xp;
		ybest = yp;
	}
}

__device__ int RNG(int idx, int idy)
{
	unsigned int m_w = idx;
	unsigned int m_z = idy;

	m_z = 36969 * (m_z & 65535) + (m_z >> 16);
	m_w = 18000 * (m_w & 65535) + (m_w >> 16);

	return abs((int)((m_z << 16) + m_w));
}

__device__ unsigned int g_seed = 0;
__device__ inline int fastrand()
{
	g_seed = (214013 * g_seed + 2531011);
	return (g_seed >> 16) & 0x7FFF;
}

__device__ static unsigned int z1 = 12345, z2 = 12345, z3 = 12345, z4 = 12345;
__device__ unsigned int lfsr113_Bits(void)
{
	unsigned int b;
	b = ((z1 << 6) ^ z1) >> 13;
	z1 = ((z1 & 4294967294U) << 18) ^ b;
	b = ((z2 << 2) ^ z2) >> 27;
	z2 = ((z2 & 4294967288U) << 2) ^ b;
	b = ((z3 << 13) ^ z3) >> 21;
	z3 = ((z3 & 4294967280U) << 7) ^ b;
	b = ((z4 << 3) ^ z4) >> 12;
	z4 = ((z4 & 4294967168U) << 13) ^ b;
	return (z1 ^ z2 ^ z3 ^ z4);
}

__global__ void setup_kernel(curandState *state, int aew, int aeh)
{
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;
	int  idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < aew && idy < aeh)
	{
		int index = idx + idy * aew;
		curand_init(index, 0, 0, &state[index]);
	}
}

__global__ void kernel_propagation2(curandState *state, int *ann, int *annd, int height, int width, int min_wh, int diagonalDis, bool leapPropagate, int patch_w, int patch_iter)
{
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;
	int  idy = blockIdx.y * blockDim.y + threadIdx.y;

	int aew = width - patch_w + 1, aeh = height - patch_w + 1;
	for (int iter = 0; iter < patch_iter; iter++)
	{
		int temp_iter = iter;
		if (iter>patch_iter / 2)
			temp_iter = patch_iter - iter;
		int change = pow((double)2.0, (double)temp_iter);
		if (idx < aew && idy < aeh)
		{
			//printf("a\n");
			int xchange = change, ychange = change;

			int index = idx + idy * aew;
			curandState localState = state[index];

			//curandState localState;
			//curand_init(index, 0, 0, &localState);

			/* Current (best) guess. */
			int v = ann[idy * width + idx];
			int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			int dbest = annd[idy * width + idx];

			/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
			if ((unsigned)(idx - xchange) < (unsigned)aew) {
				int vp = ann[idy * width + idx - xchange];
				int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
				if ((unsigned)xp < (unsigned)aew) {
					improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
				}
			}

			if ((unsigned)(idx + xchange) < (unsigned)aew) {
				int vp = ann[idy * width + idx + xchange];
				int xp = INT_TO_X(vp) - xchange, yp = INT_TO_Y(vp);
				if ((unsigned)xp < (unsigned)aew) {
					improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
				}
			}

			if ((unsigned)(idy - ychange) < (unsigned)aeh) {
				int vp = ann[(idy - ychange) * width + idx];
				int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
				if ((unsigned)yp < (unsigned)aeh) {
					improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
				}
			}

			if ((unsigned)(idy + ychange) < (unsigned)aeh) {
				int vp = ann[(idy + ychange) * width + idx];
				int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - ychange;
				if ((unsigned)yp < (unsigned)aeh) {
					improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
				}
			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = min_wh;
			if (rs_start > MAX(width, height)) { rs_start = MAX(width, height); }
			for (int mag = rs_start; mag >= 1; mag /= 2)
			{
				/* Sampling window */
				int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, aew);
				int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, aeh);
				//int xp = xmin + fastrand() % (xmax - xmin);
				//int yp = ymin + fastrand() % (ymax - ymin);

				//int xp = xmin + lfsr113_Bits() % (xmax - xmin);
				//int yp = ymin + lfsr113_Bits() % (ymax - ymin);

				//int xp = xmin + RNG(idx, idy) % (xmax - xmin);
				//int yp = ymin + RNG(idx, idy) % (ymax - ymin);

				int xp = xmin + curand(&localState) % (xmax - xmin);
				int yp = ymin + curand(&localState) % (ymax - ymin);
				improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
			}

			ann[idy * width + idx] = XY_TO_INT(xbest, ybest);
			annd[idy *width + idx] = dbest;
			state[index] = localState;
		}
	}
}

__global__ void kernel_propagation(curandState *state, int *ann, int *annd, int height, int width, int min_wh, int diagonalDis, bool leapPropagate, int patch_w, int change)
{
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;
	int  idy = blockIdx.y * blockDim.y + threadIdx.y;

	int aew = width - patch_w + 1, aeh = height - patch_w + 1;
	if (idx < aew && idy < aeh)
	{
		//printf("a\n");
		int xchange = change, ychange = change;

		int index = idx + idy * aew;
		curandState localState = state[index];

		//curandState localState;
		//curand_init(index, 0, 0, &localState);

		/* Current (best) guess. */
		int v = ann[idy * width + idx];
		int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		int dbest = annd[idy * width + idx];

		/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
		if ((unsigned)(idx - xchange) < (unsigned)aew) {
			int vp = ann[idy * width + idx - xchange];
			int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
			if ((unsigned)xp < (unsigned)aew) {
				improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
			}
		}

		if ((unsigned)(idx + xchange) < (unsigned)aew) {
			int vp = ann[idy * width + idx + xchange];
			int xp = INT_TO_X(vp) - xchange, yp = INT_TO_Y(vp);
			if ((unsigned)xp < (unsigned)aew) {
				improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
			}
		}

		if ((unsigned)(idy - ychange) < (unsigned)aeh) {
			int vp = ann[(idy - ychange) * width + idx];
			int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
			if ((unsigned)yp < (unsigned)aeh) {
				improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
			}
		}

		if ((unsigned)(idy + ychange) < (unsigned)aeh) {
			int vp = ann[(idy + ychange) * width + idx];
			int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - ychange;
			if ((unsigned)yp < (unsigned)aeh) {
				improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
			}
		}

		/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
		int rs_start = min_wh;
		if (rs_start > MAX(width, height)) { rs_start = MAX(width, height); }
		for (int mag = rs_start; mag >= 1; mag /= 2)
		{
			/* Sampling window */
			int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, aew);
			int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, aeh);

			int xp = xmin + curand(&localState) % (xmax - xmin);
			int yp = ymin + curand(&localState) % (ymax - ymin);
			improve_guess(idx, idy, xp, yp, xbest, ybest, dbest, diagonalDis, leapPropagate, patch_w);
		}

		ann[idy * width + idx] = XY_TO_INT(xbest, ybest);
		annd[idy *width + idx] = dbest;
		state[index] = localState;
	}
}

__global__ void simpleKernelProp(curandState *state, int *ann, int *annd, int height, int width, int patch_w, int change)
{
	int  idx = blockIdx.x * blockDim.x + threadIdx.x;
	int  idy = blockIdx.y * blockDim.y + threadIdx.y;

	int aew = width - patch_w + 1, aeh = height - patch_w + 1;
	if (idx < aew && idy < aeh)
	{
		//printf("a\n");
		int xchange = change, ychange = change;

		int index = idx + idy * aew;
		curandState localState = state[index];

		//curandState localState;
		//curand_init(index, 0, 0, &localState);

		/* Current (best) guess. */
		int v = ann[idy * width + idx];
		int xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		int dbest = annd[idy * width + idx];

		/* Propagation: Improve current guess by trying instead correspondences from left and above (below and right on odd iterations). */
		if ((unsigned)(idx - xchange) < (unsigned)aew) {
			int vp = ann[idy * width + idx - xchange];
			int xp = INT_TO_X(vp) + xchange, yp = INT_TO_Y(vp);
			if ((unsigned)xp < (unsigned)aew) {
				simpleImproveGuess(idx, idy, xp, yp, xbest, ybest, dbest, patch_w);
			}
		}

		if ((unsigned)(idx + xchange) < (unsigned)aew) {
			int vp = ann[idy * width + idx + xchange];
			int xp = INT_TO_X(vp) - xchange, yp = INT_TO_Y(vp);
			if ((unsigned)xp < (unsigned)aew) {
				simpleImproveGuess(idx, idy, xp, yp, xbest, ybest, dbest, patch_w);
			}
		}

		if ((unsigned)(idy - ychange) < (unsigned)aeh) {
			int vp = ann[(idy - ychange) * width + idx];
			int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + ychange;
			if ((unsigned)yp < (unsigned)aeh) {
				simpleImproveGuess(idx, idy, xp, yp, xbest, ybest, dbest, patch_w);
			}
		}

		if ((unsigned)(idy + ychange) < (unsigned)aeh) {
			int vp = ann[(idy + ychange) * width + idx];
			int xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - ychange;
			if ((unsigned)yp < (unsigned)aeh) {
				simpleImproveGuess(idx, idy, xp, yp, xbest, ybest, dbest, patch_w);
			}
		}

		/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
		int rs_start = INT_MAX;
		if (rs_start > MAX(width, height)) { rs_start = MAX(width, height); }
		for (int mag = rs_start; mag >= 1; mag /= 2)
		{
			/* Sampling window */
			int xmin = MAX(xbest - mag, 0), xmax = MIN(xbest + mag + 1, aew);
			int ymin = MAX(ybest - mag, 0), ymax = MIN(ybest + mag + 1, aeh);

			int xp = xmin + curand(&localState) % (xmax - xmin);
			int yp = ymin + curand(&localState) % (ymax - ymin);
			simpleImproveGuess(idx, idy, xp, yp, xbest, ybest, dbest, patch_w);
		}

		ann[idy * width + idx] = XY_TO_INT(xbest, ybest);
		annd[idy *width + idx] = dbest;
		state[index] = localState;
	}
}

extern "C"
void propagation(BITMAP3 *a, BITMAP3 *b, BITMAP3 *&ann, BITMAP3 *&annd, int min_wh, int diagonalDis, bool leapPropagate)
{
	FILE *file = fopen(".\propagation.txt", "w");
	int width = a->w, height = a->h;
	int aew = width - global_patch_w + 1, aeh = height - global_patch_w + 1;
	int *gpu_ann, *gpu_annd;	

	int sz = sizeof(int)*width*height;
	checkCudaErrors(cudaMalloc((void**)&gpu_ann, sz));
	checkCudaErrors(cudaMalloc((void**)&gpu_annd, sz));
	checkCudaErrors(cudaMemcpy(gpu_ann, ann->data, sz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(gpu_annd, annd->data, sz, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMallocArray(&a_Src, &texImageA.channelDesc, width, height));
	checkCudaErrors(cudaMallocArray(&b_Src, &texImageB.channelDesc, width, height));
	checkCudaErrors(cudaMemcpyToArray(a_Src, 0, 0, a->data, sz, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToArray(b_Src, 0, 0, b->data, sz, cudaMemcpyHostToDevice));

	texImageA.normalized = false;
	texImageA.addressMode[0] = cudaAddressModeClamp;
	texImageA.addressMode[1] = cudaAddressModeClamp;
	texImageA.filterMode = cudaFilterModePoint;
	texImageB.normalized = false;
	texImageB.addressMode[0] = cudaAddressModeClamp;
	texImageB.addressMode[1] = cudaAddressModeClamp;
	texImageB.filterMode = cudaFilterModePoint;

	checkCudaErrors(cudaBindTextureToArray(texImageA, a_Src));
	checkCudaErrors(cudaBindTextureToArray(texImageB, b_Src));

	dim3 gridDim(ceil((float)aew / BLOCKDIM), ceil((float)aeh / BLOCKDIM));
	dim3 blockDim(BLOCKDIM, BLOCKDIM);

	curandState *state;
	checkCudaErrors(cudaMalloc((void**)&state, aew*aeh*sizeof(curandState)));

	setup_kernel << <gridDim, blockDim >> >(state, aew, aeh);
	cudaDeviceSynchronize();

	for (int iter = 0; iter < global_pm_iters; iter++)
	{
		int temp_iter = iter;
		int change = pow((double)2.0, (double)temp_iter);
		if (iter == global_pm_iters - 2)
			change = 2;
		if (iter == global_pm_iters - 1)
			change = 1;
		cudaDeviceSynchronize();
		
		kernel_propagation << <gridDim, blockDim >> >(state, gpu_ann, gpu_annd, height, width, min_wh, diagonalDis, leapPropagate, global_patch_w, change);
		cudaDeviceSynchronize();
	}
	
	checkCudaErrors(cudaMemcpy(ann->data, gpu_ann, sz, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(annd->data, gpu_annd, sz, cudaMemcpyDeviceToHost));

	//for (int y = 0; y < aeh; ++y)
	//{
	//	for (int x = 0; x < aew; ++x)
	//	{
	//		int v = ann->data[y*width + x];
	//		int xx = INT_TO_X(v), yy = INT_TO_Y(v);
	//		if (xx >= aew || yy >= aeh || xx < 0 || yy < 0)
	//		{
	//			fprintf(file, "original pos: %d, %d. new pos: %d, %d.\n", x, y, xx, yy);
	//		}
	//	}
	//}

	checkCudaErrors(cudaFree(state));
	checkCudaErrors(cudaFree(gpu_ann));
	checkCudaErrors(cudaFree(gpu_annd));
	checkCudaErrors(cudaUnbindTexture(texImageA));
	checkCudaErrors(cudaUnbindTexture(texImageB));
	checkCudaErrors(cudaFreeArray(a_Src));
	checkCudaErrors(cudaFreeArray(b_Src));
	fclose(file);
}

__device__ void quicksort(float *dists, int *idx, int first, int last)
{
	int pivot, j, i, temp_idx;
	float temp_dist;

	if (first < last)
	{
		pivot = first;
		i = first;
		j = last;

		while (i < j){
			while (dists[i] <= dists[pivot] && i<last)
				i++;
			while (dists[j]>dists[pivot])
				j--;
			if (i < j){
				temp_dist = dists[i];
				dists[i] = dists[j];
				dists[j] = temp_dist;

				temp_idx = idx[i];
				idx[i] = idx[j];
				idx[j] = temp_idx;
			}
		}

		temp_dist = dists[pivot];
		dists[pivot] = dists[j];
		dists[j] = temp_dist;

		temp_idx = idx[pivot];
		idx[pivot] = idx[j];
		idx[j] = temp_idx;

		quicksort(dists, idx, first, j - 1);
		quicksort(dists, idx, j + 1, last);
	}
}

void quicksort_host(float *dists, int *idx, int first, int last)
{
	int pivot, j, i, temp_idx;
	float temp_dist;

	if (first < last)
	{
		pivot = first;
		i = first;
		j = last;

		while (i < j){
			while (dists[i] <= dists[pivot] && i<last)
				i++;
			while (dists[j]>dists[pivot])
				j--;
			if (i < j){
				temp_dist = dists[i];
				dists[i] = dists[j];
				dists[j] = temp_dist;

				temp_idx = idx[i];
				idx[i] = idx[j];
				idx[j] = temp_idx;
			}
		}

		temp_dist = dists[pivot];
		dists[pivot] = dists[j];
		dists[j] = temp_dist;

		temp_idx = idx[pivot];
		idx[pivot] = idx[j];
		idx[j] = temp_idx;

		quicksort_host(dists, idx, first, j - 1);
		quicksort_host(dists, idx, j + 1, last);
	}
}

__global__ void computeAllDists(int *&idxs, float *&dists, int sourceHeight, int targetHeight, int elementSize, int searchSize)
{
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

	if (xIndex < targetHeight)
	{
		//float *vecDist = (float *)malloc(sourceHeight*sizeof(float));
		//int *vecIdx = (int *)malloc(sourceHeight*sizeof(int));
		for (int y = 0; y < sourceHeight; y++)
		{
			float sum = 0;
			for (int x = 0; x < elementSize; x++)
			{
				sum += fabs(tex1Dfetch(targetEdgeTex, xIndex*elementSize + x).x - tex1Dfetch(sourceEdgeTex, y*elementSize + x).x);
			}

			//vecDist[y] = sum;
			//vecIdx[y] = y;
		}

		//quicksort(vecDist, vecIdx, 0, sourceHeight - 1);
		//memcpy(dists + sizeof(float)*searchSize*xIndex, vecDist, sizeof(float)*searchSize);
		//memcpy(idxs + sizeof(int)*searchSize*xIndex, vecIdx, sizeof(int)*searchSize);
		//free(vecDist); vecDist = NULL;
		//free(vecIdx); vecIdx = NULL;
	}
}

void computeAllDists_host(float *source, float *target, int *&idxs, float *&dists, int sourceHeight, int targetHeight, int elementSize, int searchSize)
{
	for (int xIndex = 0; xIndex < targetHeight; xIndex++)
	{
		float *vecDist = (float *)malloc(sourceHeight*sizeof(float));
		int *vecIdx = (int *)malloc(sourceHeight*sizeof(int));
		for (int y = 0; y < sourceHeight; y++)
		{
			float sum = 0;
			for (int x = 0; x < elementSize; x++)
			{
				sum += fabs(target[xIndex*elementSize + x] - source[y*elementSize + x]);
			}

			vecDist[y] = sum;
			vecIdx[y] = y;
		}

		quicksort_host(vecDist, vecIdx, 0, sourceHeight - 1);
		memcpy(dists + sizeof(float)*searchSize*xIndex, vecDist, sizeof(float)*searchSize);
		memcpy(idxs + sizeof(int)*searchSize*xIndex, vecIdx, sizeof(int)*searchSize);
		free(vecDist);
		free(vecIdx);
	}
}

extern "C" void knnSearch(float *sourceSet, float *targetSet, int *m_indices, float *m_dists, int sourceHeight, int targetHeight, int elementNum, int searchSize)
{
	float *source, *target, *dists;
	int *idxs;

	size_t sourceSize = sourceHeight * elementNum * sizeof(float);
	size_t targetSize = targetHeight * elementNum * sizeof(float);
	size_t idxSize = targetHeight * searchSize * sizeof(int);
	size_t distSize = targetHeight * searchSize * sizeof(float);

	checkCudaErrors(cudaMalloc((void **)&source, sourceSize));
	checkCudaErrors(cudaMalloc((void **)&target, targetSize));
	checkCudaErrors(cudaMalloc((void **)&idxs, idxSize));
	checkCudaErrors(cudaMalloc((void **)&dists, distSize));

	checkCudaErrors(cudaMemcpy(idxs, m_indices, idxSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dists, m_dists, distSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(source, sourceSet, sourceSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(target, targetSet, targetSize, cudaMemcpyHostToDevice));

	sourceEdgeTex.normalized = false;
	sourceEdgeTex.addressMode[0] = cudaAddressModeClamp;
	sourceEdgeTex.filterMode = cudaFilterModePoint;
	targetEdgeTex.normalized = false;
	targetEdgeTex.addressMode[0] = cudaAddressModeClamp;
	targetEdgeTex.filterMode = cudaFilterModePoint;

	checkCudaErrors(cudaBindTexture(0, sourceEdgeTex, source, sourceEdgeTex.channelDesc));
	checkCudaErrors(cudaBindTexture(0, targetEdgeTex, target, targetEdgeTex.channelDesc));

	dim3 blockDim(256);
	dim3 gridDim(iDivUp(targetHeight, blockDim.x));

	computeAllDists << <gridDim, blockDim >> >(idxs, dists, sourceHeight, targetHeight, elementNum, searchSize);

	//checkCudaErrors(cudaMemcpy(m_indices, idxs, idxSize, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(m_dists, dists, distSize, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(sourceSet, source, sourceSize, cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(targetSet, target, targetSize, cudaMemcpyDeviceToHost));
	//computeAllDists_host(sourceSet, targetSet, m_indices, m_dists, sourceHeight, targetHeight, elementNum, searchSize);

	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaMemcpy(m_indices, idxs, idxSize, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(m_dists, dists, distSize, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaUnbindTexture(sourceEdgeTex));
	checkCudaErrors(cudaUnbindTexture(targetEdgeTex));
	checkCudaErrors(cudaFree(idxs));
	checkCudaErrors(cudaFree(dists));
	checkCudaErrors(cudaFree(source));
	checkCudaErrors(cudaFree(target));
}
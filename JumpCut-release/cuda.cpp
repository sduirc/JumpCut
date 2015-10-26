// CUDA Runtime
#include <cuda_runtime.h>

// Utility and system includes
#include <helper_cuda.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <float.h>

#include <cv.h>

#include "BITMAP3.h"
#include "cuda.h"
using namespace cv;

// CUDA Kernel function
extern "C" void propagation(BITMAP3 *a, BITMAP3 *b, BITMAP3 *&ann, BITMAP3 *&annd, int min_wh, int diagonalDis, bool leapPropagate);

int global_patch_w = 5;
int global_pm_iters = 9;

int dist(BITMAP3 *a, BITMAP3 *b, int ax, int ay, int bx, int by, int diagonalDis, bool leapPropagate, int cutoff = INT_MAX) 
{
	float alpha = 10, beta = 3;
	if (leapPropagate)
		beta = 2;
	double ansColor = 0, ansDis = 0, ans = 0;
	double dxx = ax - bx, dyy = ay - by;
	double dis = sqrtNani(dxx*dxx + dyy*dyy) / diagonalDis*100.0;
	ansDis = dis*global_patch_w*global_patch_w;

	for (int dy = 0; dy < global_patch_w; dy++) {
		int *arow = &(*a)[ay + dy][ax];
		int *brow = &(*b)[by + dy][bx];
		for (int dx = 0; dx < global_patch_w; dx++) {
			int ac = arow[dx];
			int bc = brow[dx];
			int dr = (ac & 255) - (bc & 255);
			int dg = ((ac >> 8) & 255) - ((bc >> 8) & 255);
			int db = (ac >> 16) - (bc >> 16);
			ansColor += sqrtNani((double)(dr*dr + dg*dg + db*db));
		}
	}
	ansColor = ansColor / 255.0*100.0;
	ans = ansColor*alpha + ansDis*beta;
	return ans;
}

extern "C" void patchmatchGPU(BITMAP3 *a, BITMAP3 *b, BITMAP3 *&ann, BITMAP3 *&annd, bool leapPropagate)
{
	int width = a->w, height = a->h;
	int diagonalDis = sqrtNani((double)(width*width + height*height));

	ann = new BITMAP3(width, height);
	annd = new BITMAP3(width, height);
	memset(ann->data, 0, sizeof(int)*width*height);
	memset(annd->data, 0, sizeof(int)*width*height);

	/* Initialize with random nearest neighbor field (NNF). */
	int aew = width - global_patch_w + 1, aeh = height - global_patch_w + 1;
	int min_wh = diagonalDis / 3;
	if (leapPropagate)
		min_wh = diagonalDis / 2;

	for (int ay = 0; ay < aeh; ay++) {
		for (int ax = 0; ax < aew; ax++) {
			int bx = ax, by = ay;
			(*ann)[ay][ax] = XY_TO_INT(bx, by);
			(*annd)[ay][ax] = dist(a, b, ax, ay, bx, by, diagonalDis, leapPropagate);
		}
	}

	propagation(a, b, ann, annd, min_wh, diagonalDis, leapPropagate);
}
#include "videoCutout.h"

void JumpCutOnce();
int main()
{
	JumpCutOnce();
	return 0;
}

void JumpCutOnce()
{
	clock_t time = clock();
	Mat sourceImg, sourceMask, targetImg, sourceEdge, targetEdge, result;
	char dirName[] = "";
	char filename[256];
	memset(filename, 0, sizeof(filename)); sprintf(filename, "%skongfu_ori_000_001.png", dirName);
	sourceImg = imread(filename);
	memset(filename, 0, sizeof(filename)); sprintf(filename, "%skongfu_tar_000_001.png", dirName);
	targetImg = imread(filename);
	memset(filename, 0, sizeof(filename)); sprintf(filename, "%skongfu_ori_mask_000_001.png", dirName);
	sourceMask = imread(filename, 0);
	memset(filename, 0, sizeof(filename)); sprintf(filename, "%skongfu_ori_edge_000_001.png", dirName);
	sourceEdge = imread(filename, 0);
	memset(filename, 0, sizeof(filename)); sprintf(filename, "%skongfu_tar_edge_000_001.png", dirName);
	targetEdge = imread(filename, 0);

	int intervalNum = 4;
	Mat tempMat;
	bool useEdge = true;
	videoCutout vc;
	vc.PatchMatch(sourceImg, sourceMask, sourceEdge, tempMat, tempMat, tempMat, targetImg, targetEdge, intervalNum);
	vc.EdgeClassifier(tempMat, tempMat, tempMat);
	string videoPath = dirName;
	vc.levelset(result, useEdge);

	time = clock() - time;
	printf("Total time is %f seconds.\n", (float)time / CLOCKS_PER_SEC);
	getchar();
}
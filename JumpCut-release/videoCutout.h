#include <cv.h>
#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <numeric> 
#include <stdio.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <limits>
#include <string>

#include <omp.h>

#include "imgwarp_piecewiseaffine.h"
#include "imgwarp_mls_rigid.h"
#include "imgwarp_mls_similarity.h"
#include "imgwarp_mls.h"
//#include <engine.h>
#include "BITMAP3.h"
#include "cuda.h"
#include "fpm.h"

#define PI 3.1415926535897932384626433832

using namespace std;
using namespace cv;

// #pragma comment(lib, "libmx.lib")
// #pragma comment(lib, "libmat.lib")
// #pragma comment(lib, "libeng.lib")

class videoCutout
{
public:
	videoCutout(void);
	~videoCutout(void);

	struct edgeEle{
		float sinValue, cosValue;
		Point pos;
		Vec3b color;
	};

	enum FB { Foreground, Background, Contour };
	void pipeline(Mat &sourceImgFront, Mat &sourceMaskFront, Mat &sourceEdgeFront, Mat &sourceImgBack, Mat &sourceMaskBack, Mat &sourceEdgeBack, Mat &targetImg, Mat &result, Mat &targetEdge, int intervalNum);

	void levelset(Mat &result, bool useEdge);
	bool PatchMatch(Mat &sourceImgFront, Mat &sourceMaskFront, Mat& sourceEdgeFront, Mat &sourceImgBack, Mat &sourceMaskBack, Mat& sourceEdgeBack, Mat &targetImg, Mat& targetEdge, int intervalNum);
	void EdgeClassifier(Mat &sourceImgBack, Mat &sourceMaskBack, Mat& sourceEdgeBack);

	//preprocessing
	template<typename T> void bilinearInterpolate(Mat source, Mat &target, int width, int height, float scaleX, float scaleY, T nothing);
	void imageDeform(Mat &ori_mask, Mat &tar_mask, vector<Point2f> &srcPoints, vector<Point2f> &tarPoints);
	vector<int> featureMatching(Mat &img_object, Mat &img_scene, Mat &obj_mask, vector<Point2f> &srcPoints, vector<Point2f> &tarPoints, bool isFG);
	Rect templateMatchFG(Mat &ori_img, Mat &tar_img, Mat &ori_mask, Mat &ori_edge, Mat &sourcePatch, Mat &sourceMaskPatch, Mat &sourceEdgePatch, Mat &targetPatch, double scale_fac, bool leapPropagate);
	Rect gen_ssd(Mat &patch, Mat &patch_mask, Mat &tar_image, Mat &tar, FPM_FFT& fpm);
	vector<int> getBoundRect(Mat img_mask);
	void checkRectBorder(int width, int height, Rect &temp);
	void checkPointBorder(int width, int height, Point &temp);
	void checkRectBorderClip(int width, int height, Rect &temp);

	void gradient(Mat &inputImg, Mat &outputImg, int depth, int dx, int dy, int size);

	void removeNoise(Mat &tar_mask, Mat &ori_mask, int maxNoise);
	void removeNoiseSimple(Mat &tar_mask, int maxNoise);

	//Edge classifier
	void classifyTargetEdges(Mat &tar_edge, Mat &tar_edge_result, Mat &tar_img, Mat &tar_dst_dx, Mat &tar_dst_dy, Mat &fg_mat, Mat &bg_mat, Mat &contour_mat, Mat &offsetFGx, Mat &offsetFGy, Mat &offsetBGx, Mat &offsetBGy, Mat &offsetFGBackx, Mat &offsetFGBacky, Mat &offsetBGBackx, Mat &offsetBGBacky, Mat &fgConstraint, Mat &bgConstraint, Mat &tar_weight, int height, int width, int point_dis, int searchSize, int maxDifference, int minNum, float diagonalDis, int sourceFGColorNum, vector<bool> &isForegroundFG, vector<edgeEle> &sourceFGColors, int sourceBGColorNum, vector<bool> &isForegroundBG, vector<edgeEle> &sourceBGColors, int sourceFGColorNumBack, vector<bool> &isForegroundFGBack, vector<edgeEle> &sourceFGColorsBack, int sourceBGColorNumBack, int chosen_num, vector<bool> &isForegroundBGBack, vector<edgeEle> &sourceBGColorsBack, bool isSecond);
	void ComputeEgFeature(float &angle, float x_off, float y_off, int x, int y, int point_dis, int ori_height, int ori_width, Mat &img, Point &pt_a, Point &pt_b, Vec3b &ScalarA, Vec3b &ScalarB);
	void computeSourceMinDis(Mat &m_distsFGA, Mat &m_distsBGA, Mat &m_indicesFGA, Mat &m_indicesBGA, Mat &m_distsFGBackA, Mat &m_distsBGBackA, Mat &m_indicesFGBackA, Mat &m_indicesBGBackA, Mat &distanceA, Mat &labelsA, int count, int searchSize, vector<bool> &isForegroundFG, vector<bool> &isForegroundBG, vector<bool> &isForegroundFGBack, vector<bool> &isForegroundBGBack, bool isSecond);
	void removeEdgeErrors(Mat &tar_edge, Mat &tar_dst_dx, Mat &tar_dst_dy, Mat &tar_edge_result2, Mat &fgConstraint, Mat &bgConstraint, Mat &contour_mat, Mat &bg_mat, Mat &fg_mat, Mat &test_duplicate, int width, int height, int maxNoiseSize, int point_dis);
	void removeEdgeErrors4Diffusion(Mat &tar_edge, Mat &fgConstraint, Mat &bgConstraint, Mat &tar_constraint, Mat &tar_weight, int width, int height, int maxNoiseSize, int disSize);
	void computeSourceColorSet(Mat &ori_edge, Mat &ori_edge_result, Mat &ori_img, Mat & ori_mask, Mat &ori_dst_dx, Mat &ori_dst_dy, int sourceBoundarySize, int sideDis, int &sourceColorNum, int chosen_num, vector<bool> &isForeground, vector<edgeEle> &sourceColors);
};
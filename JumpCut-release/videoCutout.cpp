#include "videoCutout.h"

videoCutout::videoCutout(void)
{
}

videoCutout::~videoCutout(void)
{
}

Point point_mouse4;
bool mouse_downL4 = false;
bool mouse_downR4 = false;

void mouseCall5(int event, int x, int y, int flags, void *param){
	switch (event){
	case CV_EVENT_LBUTTONDOWN:
		point_mouse4.x = x, point_mouse4.y = y;
		mouse_downL4 = true;
		break;
	case CV_EVENT_RBUTTONDOWN:
		point_mouse4.x = x, point_mouse4.y = y;
		mouse_downR4 = true;
		break;
	}
}

void drawArrow2(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, cv::Scalar& color, int thickness, int lineType)
{
	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	circle(img, pStart, 3, color, -1);
	circle(img, pEnd, 3, color, -1);
}

void videoCutout::checkPointBorder(int width, int height, Point &temp)
{
	if (temp.x < 0)
		temp.x = 0;
	if (temp.y < 0)
		temp.y = 0;
	if (temp.x >= width)
		temp.x = width - 1;
	if (temp.y >= height)
		temp.y = height - 1;
}

void videoCutout::checkRectBorderClip(int width, int height, Rect &temp)
{
	if (temp.x < 0)
	{
		temp.width += temp.x;
		temp.x = 0;
	}
	if (temp.y < 0)
	{
		temp.height += temp.y;
		temp.y = 0;
	}
	if (temp.x + temp.width >= width)
		temp.width = width - temp.x;
	if (temp.y + temp.height >= height)
		temp.height = height - temp.y;
}

void videoCutout::checkRectBorder(int width, int height, Rect &temp)
{
	temp.width = temp.width < width ? temp.width : width;
	temp.height = temp.height < height ? temp.height : height;

	if (temp.x < 0)
		temp.x = 0;
	if (temp.y < 0)
		temp.y = 0;
	if (temp.x + temp.width >= width)
		temp.x = width - temp.width;
	if (temp.y + temp.height >= height)
		temp.y = height - temp.height;
}

void videoCutout::removeEdgeErrors(Mat &tar_edge, Mat &targetDx, Mat &targetDy, Mat &tar_edge_result2, Mat &fgConstraint, Mat &bgConstraint, Mat &contour_mat, Mat &bg_mat, Mat &fg_mat, Mat &test_duplicate, int width, int height, int maxNoiseSize, int point_dis)
{
	Mat contour_matOri = contour_mat.clone();

	//remove errors
	vector<vector<Point>> contours;
	Mat contour_mat_clone = contour_mat.clone();
	findContours(contour_mat_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++)
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		Scalar color(255);
		if (area < maxNoiseSize)
		{
			drawContours(contour_mat, contours, i, color, CV_FILLED, 8);
		}
	}
	bg_mat = contour_mat - bg_mat;
	bg_mat = 255 - bg_mat;
	fg_mat = contour_mat - fg_mat;
	fg_mat = 255 - fg_mat;

	Mat bg_mat_clone = bg_mat.clone();
	contours.clear();
	findContours(bg_mat_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++)
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		Scalar color(255);
		if (area < maxNoiseSize)
		{
			drawContours(bg_mat, contours, i, color, CV_FILLED, 8);
		}
	}
	fg_mat = bg_mat - fg_mat;
	fg_mat = 255 - fg_mat;

	Mat fg_mat_clone = fg_mat.clone();
	contours.clear();
	findContours(fg_mat_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++)
	{
		const vector<Point>& c = contours[i];
		double area = fabs(contourArea(Mat(c)));
		Scalar color(255);
		if (area < maxNoiseSize)
		{
			drawContours(fg_mat, contours, i, color, CV_FILLED, 8);
		}
	}

	// 	printf("targetDx.rows: %i, targetDx.cols:%i \n", targetDx.rows, targetDx.cols);
	Mat contourDiff = contour_mat - contour_matOri;
	for (int y = 0; y < height; y++)
	{
		float *targetDxRow = targetDx.ptr<float>(y);
		float *targetDyRow = targetDy.ptr<float>(y);
		for (int x = 0; x < width; x++)
		{
			if (contourDiff.at<uchar>(y, x) == 255)
			{
				float angle;
				float x_off = targetDxRow[x], y_off = targetDyRow[x];
				if (x_off == 0){
					if (y_off > 0)
						angle = PI / 2;
					else if (y_off <= 0)
						angle = PI*1.5;
				}
				else if (y_off == 0){
					if (x_off > 0)
						angle = 0;
					else if (x_off <= 0)
						angle = PI;
				}
				else
					angle = atan2(y_off, x_off);
				if (angle < 0)
					angle += PI;
				Point pt_a, pt_b;
				pt_a.x = x + point_dis*cos(angle); pt_a.y = y + point_dis*sin(angle);
				pt_b.x = x + point_dis*cos(angle + PI); pt_b.y = y + point_dis*sin(angle + PI);
				checkPointBorder(width, height, pt_a); checkPointBorder(width, height, pt_b);
				fgConstraint.at<uchar>(pt_a.y, pt_a.x) = 0; fgConstraint.at<uchar>(pt_b.y, pt_b.x) = 0;
				bgConstraint.at<uchar>(pt_a.y, pt_a.x) = 0; bgConstraint.at<uchar>(pt_b.y, pt_b.x) = 0;
			}
		}
	}

	// color the edge map with R,G,B
	for (int y = 0; y < height; y++)
	{
		Vec3b *tar_result_row = tar_edge_result2.ptr<Vec3b>(y);
		uchar* fg_mat_row = fg_mat.ptr<uchar>(y);
		uchar* bg_mat_row = bg_mat.ptr<uchar>(y);
		uchar* contour_mat_row = contour_mat.ptr<uchar>(y);
		uchar* dupRow = test_duplicate.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (fg_mat_row[x] == 0)
			{
				tar_result_row[x] = Vec3b(0, 255, 255);
				dupRow[x] = 250;
			}
			if (bg_mat_row[x] == 0)
			{
				tar_result_row[x] = Vec3b(255, 0, 0);
				dupRow[x] = 150;
			}
			if (contour_mat_row[x] == 0)
			{
				tar_result_row[x] = Vec3b(0, 0, 255);
				dupRow[x] = 50;
			}
		}
	}

	Mat test_duplicate2 = test_duplicate.clone();
	for (int y = 0; y < height; y++)
	{
		uchar* tar_edge_row = tar_edge.ptr<uchar>(y);
		Vec3b *tar_result_row = tar_edge_result2.ptr<Vec3b>(y);
		uchar* fg_mat_row = fg_mat.ptr<uchar>(y);
		uchar* bg_mat_row = bg_mat.ptr<uchar>(y);
		uchar* contour_mat_row = contour_mat.ptr<uchar>(y);
		uchar* dupRow = test_duplicate.ptr<uchar>(y);
		uchar* dupRow2 = test_duplicate2.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (tar_edge_row[x] == 0 && dupRow[x] == 0)
			{
				int tempRectSize = 3;
				int fg_num, bg_num, con_num;
				while (true)
				{
					fg_num = 0, bg_num = 0, con_num = 0;
					Rect temp_rect = Rect(x - tempRectSize, y - tempRectSize, tempRectSize * 2 + 1, tempRectSize * 2 + 1);
					checkRectBorderClip(width, height, temp_rect);
					Mat temp_mat = test_duplicate(temp_rect).clone();
					for (int yy = 0; yy < temp_mat.rows; yy++)
					{
						for (int xx = 0; xx < temp_mat.cols; xx++)
						{
							uchar temp_vec = temp_mat.at<uchar>(yy, xx);
							if (temp_vec == 250)
								fg_num++;
							if (temp_vec == 150)
								bg_num++;
							if (temp_vec == 50)
								con_num++;
						}
					}
					if (fg_num == 0 && bg_num == 0 && con_num == 0)
						tempRectSize += 2;
					else
						break;
				}
				FB temp_fb;
				temp_fb = fg_num > bg_num ? Foreground : Background;
				if (temp_fb == Foreground)
					temp_fb = con_num > fg_num ? Contour : Foreground;
				if (temp_fb == Background)
					temp_fb = con_num > bg_num ? Contour : Background;
				int temp_num = 0;
				if (temp_fb == Foreground)
				{
					tar_result_row[x] = Vec3b(0, 255, 255);
					dupRow2[x] = 250;
					fg_mat_row[x] = 0;
				}
				else if (temp_fb == Background)
				{
					tar_result_row[x] = Vec3b(255, 0, 0);
					dupRow2[x] = 150;
					bg_mat_row[x] = 0;
				}
				else
				{
					tar_result_row[x] = Vec3b(0, 0, 255);
					dupRow2[x] = 50;
					contour_mat_row[x] = 0;
				}
			}
		}
	}
	test_duplicate = test_duplicate2.clone();
}

void videoCutout::removeEdgeErrors4Diffusion(Mat &contour_mat, Mat &fgConstraint, Mat &bgConstraint, Mat &tar_constraint, Mat &tar_weight, int width, int height, int maxNoiseSize, int edgeMaxDis)
{
	Mat tar_weight_clone = tar_weight.clone();
	tar_weight.setTo(0); tar_constraint.setTo(125);
	for (int y = 0; y < height; y++)
	{
		uchar *fgContRow = fgConstraint.ptr<uchar>(y);
		uchar *bgContRow = bgConstraint.ptr<uchar>(y);
		uchar *tarContRow = tar_constraint.ptr<uchar>(y);
		uchar *tarWeiCloneRow = tar_weight_clone.ptr<uchar>(y);
		uchar *tarWeiRow = tar_weight.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (fgContRow[x] == 255)
			{
				tarContRow[x] = 255;
				tarWeiRow[x] = tarWeiCloneRow[x];
			}
			if (bgContRow[x] == 255)
			{
				tarContRow[x] = 0;
				tarWeiRow[x] = tarWeiCloneRow[x];
			}
		}
	}

	Mat distance; Mat distanceEdges(Size(width, height), CV_8U, Scalar(255));
	distanceTransform(contour_mat, distance, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	for (int y = 0; y < height; y++)
	{
		uchar *disEdgeRow = distanceEdges.ptr<uchar>(y);
		float *disRow = distance.ptr<float>(y);
		for (int x = 0; x < width; x++)
		{
			if (disRow[x] <= edgeMaxDis)
				disEdgeRow[x] = 0;
		}
	}

	Mat tar_constraint_clone(Size(width, height), CV_8U, Scalar(125));
	tar_weight_clone.setTo(0);
	for (int y = 0; y < height; y++)
	{
		uchar *tarConsCloneRow = tar_constraint_clone.ptr<uchar>(y);
		uchar *tarWeightCloneRow = tar_weight_clone.ptr<uchar>(y);
		uchar *tarConsRow = tar_constraint.ptr<uchar>(y);
		uchar *tarWeightRow = tar_weight.ptr<uchar>(y);
		uchar *disEdgeRow = distanceEdges.ptr<uchar>(y);
		uchar *conRow = contour_mat.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (disEdgeRow[x] == 0 && tarConsRow[x] != 125 && conRow[x] != 0)
			{
				tarConsCloneRow[x] = tarConsRow[x];
				tarWeightCloneRow[x] = tarWeightRow[x];
			}
		}
	}

	Mat tar_constraint_clone2 = tar_constraint_clone.clone();
	Mat tar_weight_clone2 = tar_weight_clone.clone();
	for (int y = 0; y < height; y++)
	{
		uchar *tarConsCloneRow = tar_constraint_clone2.ptr<uchar>(y);
		uchar *tarWeightCloneRow = tar_weight_clone2.ptr<uchar>(y);
		uchar *tarConsRow = tar_constraint.ptr<uchar>(y);
		uchar *disEdgeRow = distanceEdges.ptr<uchar>(y);
		uchar *conRow = contour_mat.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (disEdgeRow[x] == 0 && tarConsRow[x] == 125 && conRow[x] != 0)
			{
				int tempRectSize = 3;
				int fg_num, bg_num;
				int fg_value, bg_value;
				while (true)
				{
					fg_num = 0, bg_num = 0;
					fg_value = 0, bg_value = 0;
					Rect temp_rect = Rect(x - tempRectSize, y - tempRectSize, tempRectSize * 2 + 1, tempRectSize * 2 + 1);
					checkRectBorderClip(width, height, temp_rect);
					Mat temp_constraint = tar_constraint_clone(temp_rect).clone();
					Mat temp_weight = tar_weight_clone(temp_rect).clone();
					for (int yy = 0; yy < temp_constraint.rows; yy++)
					{
						for (int xx = 0; xx < temp_constraint.cols; xx++)
						{
							int cons_value = temp_constraint.at<uchar>(yy, xx);
							int weight_value = temp_weight.at<uchar>(yy, xx);
							if (cons_value == 255)
							{
								fg_num++;
								fg_value += weight_value;
							}
							if (cons_value == 0)
							{
								bg_num++;
								bg_value += weight_value;
							}
						}
					}
					if (bg_num == 0 && fg_num == 0)
						tempRectSize += 2;
					else
						break;
				}
				FB temp_fb;
				temp_fb = fg_num > bg_num ? Foreground : Background;
				if (temp_fb == Foreground)
				{
					tarConsCloneRow[x] = 255;
					tarWeightCloneRow[x] = fg_value / fg_num;
				}
				else if (temp_fb == Background)
				{
					tarConsCloneRow[x] = 0;
					tarWeightCloneRow[x] = bg_value / bg_num;
				}
			}
		}
	}
	tar_weight = tar_weight_clone2.clone();
	tar_constraint = tar_constraint_clone2.clone();
}

void videoCutout::ComputeEgFeature(float &angle, float x_off, float y_off, int x, int y, int point_dis, int height, int width, Mat &img, Point &pt_a, Point &pt_b, Vec3b &ScalarA, Vec3b &ScalarB)
{
	if (x_off == 0){
		if (y_off > 0)
			angle = PI / 2;
		else if (y_off <= 0)
			angle = PI*1.5;
	}
	else if (y_off == 0){
		if (x_off > 0)
			angle = 0;
		else if (x_off <= 0)
			angle = PI;
	}
	else
		angle = atan2(y_off, x_off);

	if (angle < 0)
		angle += PI;

	pt_a.x = x + point_dis*cos(angle); pt_a.y = y + point_dis*sin(angle);
	pt_b.x = x + point_dis*cos(angle + PI); pt_b.y = y + point_dis*sin(angle + PI);
	checkPointBorder(width, height, pt_a); checkPointBorder(width, height, pt_b);

	// calculate the average color of that half circle as the color of pt_a and pt_b.
	// 	int circleRadius=point_dis;
	// 	Rect rectA= Rect(pt_a.x-circleRadius, pt_a.y-circleRadius, circleRadius*2+1, circleRadius*2+1);
	// 	Rect rectB= Rect(pt_b.x-circleRadius, pt_b.y-circleRadius, circleRadius*2+1, circleRadius*2+1);
	// 
	//  	Mat circleMaskA(Size(circleRadius*2+1, circleRadius*2+1), CV_8U, Scalar(0)), circleMaskB(Size(circleRadius*2+1, circleRadius*2+1), CV_8U, Scalar(0));
	//  	Point pt1=Point(circleRadius+circleRadius*2*cos(angle+PI/2), circleRadius+circleRadius*2*sin(angle+PI/2));
	//  	Point pt2=Point(circleRadius+circleRadius*2*cos(angle-PI/2), circleRadius+circleRadius*2*sin(angle-PI/2));
	//  	clipLine(Size(circleRadius*2+1, circleRadius*2+1), pt1, pt2);
	//  	line(circleMaskA, pt1, Point(circleRadius,circleRadius), Scalar(50)); line(circleMaskA, pt2, Point(circleRadius,circleRadius), Scalar(50)); 
	//  	line(circleMaskB, pt1, Point(circleRadius,circleRadius), Scalar(50)); line(circleMaskB, pt2, Point(circleRadius,circleRadius), Scalar(50)); 
	//  	circle(circleMaskA,Point(circleRadius,circleRadius),circleRadius,Scalar(100)); circle(circleMaskB,Point(circleRadius,circleRadius),circleRadius,Scalar(100));
	//  	Point ptA=Point(circleRadius+circleRadius*cos(angle)/2, circleRadius+circleRadius*sin(angle)/2);
	//  	Point ptB=Point(circleRadius+circleRadius*cos(angle+PI)/2, circleRadius+circleRadius*sin(angle+PI)/2);
	//  	checkPointBorder(width,height,ptA); checkPointBorder(width,height,ptB);
	//  	floodFill(circleMaskA, ptA, Scalar(255)); floodFill(circleMaskB, ptB, Scalar(255));
	//  	floodFill(circleMaskA, Point(circleRadius,circleRadius), Scalar(255)); floodFill(circleMaskB, Point(circleRadius,circleRadius), Scalar(255));
	//  	Mat maskA(Size(width+circleRadius*2,height+circleRadius*2), CV_8U, Scalar(0)), maskB(Size(width+circleRadius*2,height+circleRadius*2), CV_8U, Scalar(0));
	//  	Mat maskA_rect=maskA(Rect(rectA.x+circleRadius, rectA.y+circleRadius, circleRadius*2+1, circleRadius*2+1));
	//  	Mat maskB_rect=maskB(Rect(rectB.x+circleRadius, rectB.y+circleRadius, circleRadius*2+1, circleRadius*2+1));
	//  	circleMaskA.copyTo(maskA_rect); circleMaskB.copyTo(maskB_rect);
	//  	Mat maskA1=maskA(Rect(circleRadius, circleRadius, width, height)).clone();
	//  	Mat maskB1=maskB(Rect(circleRadius, circleRadius, width, height)).clone();
	//  	checkRectBorderClip(width, height, rectA); checkRectBorderClip(width, height, rectB);
	//  	Mat circleA= img(rectA).clone(); Mat circleB= img(rectB).clone();
	//  	Mat circleMaskA1=maskA1(rectA).clone(); Mat circleMaskB1=maskB1(rectB).clone();
	//  	//check the circle masks, then you know if the codes above is correct or not.
	//  	// 	imshow("circleMaskA1",circleMaskA1);
	//  	// 	imshow("circleMaskB1",circleMaskB1);
	//  	// 	waitKey();
	//  
	//  	Vec3i ScalarIA(0,0,0), ScalarIB(0,0,0);
	//  	int numA=0, numB=0;
	//  	for (int y=0; y<rectA.height; y++)
	//  	{
	//  		for (int x=0; x<rectA.width; x++)
	//  		{
	//  			if(circleMaskA1.at<uchar>(y,x)==255)
	//  			{
	//  				ScalarIA[0]+=circleA.at<Vec3b>(y,x)[0];
	//  				ScalarIA[1]+=circleA.at<Vec3b>(y,x)[1];
	//  				ScalarIA[2]+=circleA.at<Vec3b>(y,x)[2];
	//  				numA++;
	//  			}
	//  		}
	//  	}
	//  	for (int y=0; y<rectB.height; y++)
	//  	{
	//  		for (int x=0; x<rectB.width; x++)
	//  		{
	//  			if(circleMaskB1.at<uchar>(y,x)==255)
	//  			{
	//  				ScalarIB[0]+=circleB.at<Vec3b>(y,x)[0];
	//  				ScalarIB[1]+=circleB.at<Vec3b>(y,x)[1];
	//  				ScalarIB[2]+=circleB.at<Vec3b>(y,x)[2];
	//  				numB++;
	//  			}
	//  		}
	//  	}
	//  	if(numA==0)
	//  	{
	//  		ScalarA=img.at<Vec3b>(pt_a.y,pt_a.x);
	//  		cout<<"num = 0!"<<endl;
	//  	}else
	//  	{
	//  		ScalarA[0]=ScalarIA[0]/(numA);
	//  		ScalarA[1]=ScalarIA[1]/(numA);
	//  		ScalarA[2]=ScalarIA[2]/(numA);
	//  	}
	//  	if(numB==0)
	//  	{
	//  		ScalarB=img.at<Vec3b>(pt_b.y,pt_b.x);
	//  		cout<<"num = 0!"<<endl;
	//  	}else{
	//  		ScalarB[0]=ScalarIB[0]/(numB);
	//  		ScalarB[1]=ScalarIB[1]/(numB);
	//  		ScalarB[2]=ScalarIB[2]/(numB);
	//  	}
	ScalarA = img.at<Vec3b>(pt_a.y, pt_a.x);
	ScalarB = img.at<Vec3b>(pt_b.y, pt_b.x);
}

#ifndef MAX
#define MAX(a, b) ((a)>(b)?(a):(b))
#define MIN(a, b) ((a)<(b)?(a):(b))
#endif

BITMAP3* toBITMAP(const uchar *img, int width, int height, int istep)
{
	BITMAP3 *bmp = new BITMAP3(width, height);
	int bi = 0;
	for (int yi = 0; yi < height; ++yi, img += istep)
	{
		const uchar *px = img;
		for (int xi = 0; xi < width; ++xi, px += 3, ++bi)
		{
			bmp->data[bi] = (px[0] << 16) | (px[1] << 8) | px[2];
		}
	}

	return bmp;
}

BITMAP3 *load_bitmap3(const char *filename) {
	cv::Mat img = imread(filename);
	return toBITMAP(img.data, img.cols, img.rows, img.step);
}

bool UDgreater3(vector<Point> elem1, vector<Point> elem2)
{
	return elem1.size() > elem2.size();
}

bool UDgreaterArea(vector<Point> elem1, vector<Point> elem2)
{
	double area1 = fabs(contourArea(Mat(elem1))), area2 = fabs(contourArea(Mat(elem2)));
	return area1 > area2;
}

void videoCutout::removeNoise(Mat &tar_mask, Mat &ori_mask, int maxNoise)
{
	int width = ori_mask.cols, height = ori_mask.rows;
	int diagonalDis = sqrtNani(width*width + height*height);
	vector<vector<Point>> contours, fgContours, bgContours;

	// count the fg and bg holes in source image
	int sourceFgNum = 0, sourceBgNum = 0;
	contours.clear();
	Mat ori_mask_clone = ori_mask.clone();
	findContours(ori_mask_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
#pragma omp parallel for
	for (int idx = 0; idx < contours.size(); idx++)
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		area += c.size();

		Mat tempMask(Size(width, height), CV_8U, Scalar(0));
		drawContours(tempMask, contours, idx, Scalar(255), CV_FILLED, 8);
		for (int m = 0; m < c.size(); m++)
			tempMask.at<uchar>(c[m].y, c[m].x) = 0;
		vector<vector<Point>> tempContours;
		findContours(tempMask, tempContours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		int boundaryNum = 0, blackNum = 0;
		if (tempContours.size() == 0)
			boundaryNum = 0;
		else
			boundaryNum = tempContours[0].size();
		for (int m = 0; m < boundaryNum; m++)
		{
			if (ori_mask.at<uchar>(tempContours[0][m].y, tempContours[0][m].x) == 0)
				blackNum++;
		}
		if ((double)blackNum / boundaryNum > 0.8 && boundaryNum != 0)
			sourceBgNum++;
		else
			sourceFgNum++;
	}

	for (int m = 0; m < 3; m++)
	{
		Mat tar_mask_clone = tar_mask.clone();
		contours.clear(); fgContours.clear(); bgContours.clear();
		findContours(tar_mask_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		// #pragma omp parallel for
		for (int idx = 0; idx < contours.size(); idx++)
		{
			const vector<Point>& c = contours[idx];
			Mat tempMask(Size(width, height), CV_8U, Scalar(0));
			drawContours(tempMask, contours, idx, Scalar(255), CV_FILLED, 8);
			for (int m = 0; m < c.size(); m++)
				tempMask.at<uchar>(c[m].y, c[m].x) = 0;
			vector<vector<Point>> tempContours;
			findContours(tempMask, tempContours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
			sort(tempContours.begin(), tempContours.end(), UDgreaterArea);
			int boundaryNum = 0, blackNum = 0;
			if (tempContours.size() == 0)
				boundaryNum = 0;
			else
				boundaryNum = tempContours[0].size();
			for (int m = 0; m < boundaryNum; m++)
			{
				if (tar_mask.at<uchar>(tempContours[0][m].y, tempContours[0][m].x) == 0)
					blackNum++;
			}
			//judge the inner loop is a white or black area.
			if ((double)blackNum / boundaryNum > 0.8 && boundaryNum != 0)
				bgContours.push_back(contours[idx]);
			else
				fgContours.push_back(contours[idx]);
		}

		int holenum = 0;
		if (!fgContours.empty())
		{
			sort(fgContours.begin(), fgContours.end(), UDgreaterArea);
			Mat biggestArea = Mat(tar_mask.size(), CV_8U, Scalar(255)), distance;
			drawContours(biggestArea, fgContours, -1, Scalar(0), -1);
			distanceTransform(biggestArea, distance, CV_DIST_L2, CV_DIST_MASK_PRECISE);

			for (int idx = 0; idx < fgContours.size(); idx++)
			{
				double area = fabs(contourArea(Mat(fgContours[idx])));
				if (idx >= sourceFgNum + holenum || area < maxNoise)
					drawContours(tar_mask, fgContours, idx, Scalar(0), CV_FILLED, 8);
				else if (idx < sourceFgNum + holenum)
				{
					float minDis = numeric_limits<float>::max();
					for (int m = 0; m < fgContours[idx].size(); m++)
					{
						minDis = MIN(distance.at<float>(fgContours[idx][m].y, fgContours[idx][m].x), minDis);
					}
					if (minDis>diagonalDis / 10)
						drawContours(tar_mask, fgContours, idx, Scalar(0), CV_FILLED, 8);
				}
			}
		}
		if (!bgContours.empty())
		{
			sort(bgContours.begin(), bgContours.end(), UDgreaterArea);
			for (int idx = 0; idx < bgContours.size(); idx++)
			{
				double area = fabs(contourArea(Mat(bgContours[idx])));
				area -= bgContours[idx].size();
				if (idx >= sourceBgNum + holenum || area < maxNoise)
				{
					drawContours(tar_mask, bgContours, idx, Scalar(255), CV_FILLED, 8);
				}
			}
		}
	}

	int borderDis = global_patch_w / 2 + global_patch_w % 2;
#pragma omp parallel for
	for (int x = 0; x < width; x++)
	{
		if (tar_mask.at<uchar>(borderDis, x) == 255)
			for (int y = 0; y < borderDis; y++)
				tar_mask.at<uchar>(y, x) = 255;
		if (tar_mask.at<uchar>(height - 1 - borderDis, x) == 255)
			for (int y = height - borderDis; y < height; y++)
				tar_mask.at<uchar>(y, x) = 255;
	}
#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		if (tar_mask.at<uchar>(y, borderDis) == 255)
			for (int x = 0; x < borderDis; x++)
				tar_mask.at<uchar>(y, x) = 255;
		if (tar_mask.at<uchar>(y, width - 1 - borderDis) == 255)
			for (int x = width - borderDis; x < width; x++)
				tar_mask.at<uchar>(y, x) = 255;
	}
}

void videoCutout::imageDeform(Mat &ori_mask, Mat &tar_mask, vector<Point2f> &srcPoints, vector<Point2f> &tarPoints)
{
	ImgWarp_MLS *imgTrans;
	// 	imgTrans = new ImgWarp_MLS_Similarity();
	imgTrans = new ImgWarp_MLS_Rigid();
	// 	imgTrans = new ImgWarp_PieceWiseAffine();
	// 	((ImgWarp_PieceWiseAffine *)imgTrans)->backGroundFillAlg = ImgWarp_PieceWiseAffine::BGML	

	imgTrans->alpha = 1;
	imgTrans->gridSize = 10;
	vector<Point> source, target;
	source.resize(srcPoints.size()); target.resize(tarPoints.size());
#pragma omp parallel for
	for (int m = 0; m < srcPoints.size(); m++)
	{
		source[m] = Point((int)srcPoints[m].x, (int)srcPoints[m].y);
		target[m] = Point((int)tarPoints[m].x, (int)tarPoints[m].y);
	}

	tar_mask = imgTrans->setAllAndGenerate(
		ori_mask,
		source,
		target,
		ori_mask.cols, ori_mask.rows, 1);

	delete imgTrans;
}

void videoCutout::gradient(Mat &inputImg, Mat &outputImg, int depth, int dx, int dy, int size)
{
	int height = inputImg.rows, width = inputImg.cols;
	outputImg = Mat(inputImg.size(), depth, Scalar(0));
	if (dx == 1)
	{
		for (int y = 0; y < height; y++)
		{
			outputImg.at<float>(y, 0) = inputImg.at<float>(y, 1) - inputImg.at<float>(y, 0);
			outputImg.at<float>(y, width - 1) = inputImg.at<float>(y, width - 1) - inputImg.at<float>(y, width - 2);
			for (int x = 1; x < width - 1; x++)
			{
				outputImg.at<float>(y, x) = (inputImg.at<float>(y, x + 1) - inputImg.at<float>(y, x - 1)) / 2;
			}
		}
	}
	if (dy == 1)
	{
		for (int x = 0; x < width; x++)
		{
			outputImg.at<float>(0, x) = inputImg.at<float>(1, x) - inputImg.at<float>(0, x);
			outputImg.at<float>(height - 1, x) = inputImg.at<float>(height - 1, x) - inputImg.at<float>(height - 2, x);
			for (int y = 1; y < height - 1; y++)
			{
				outputImg.at<float>(y, x) = (inputImg.at<float>(y + 1, x) - inputImg.at<float>(y - 1, x)) / 2;
			}
		}
	}
}

extern "C" void kdtree_preprocess(Mat &source, Mat &target, Mat &idx, Mat &dists);
extern "C" void knnSearch(float *sourceSet, float *targetSet, int *m_indices, float *m_dists, int sourceHeight, int targetHeight, int elementNum, int searchSize);

void videoCutout::classifyTargetEdges(Mat &tar_edge, Mat &tar_edge_result, Mat &tar_img, Mat &targetDx, Mat &targetDy, Mat &fg_mat, Mat &bg_mat, Mat &contour_mat, Mat &offsetFGx, Mat &offsetFGy, Mat &offsetBGx, Mat &offsetBGy, Mat &offsetFGBackx, Mat &offsetFGBacky, Mat &offsetBGBackx, Mat &offsetBGBacky, Mat &fgConstraint, Mat &bgConstraint, Mat &tar_weight, int height, int width, int point_dis, int searchSize, int maxDifference, int minNum, float diagonalDis, int sourceFGColorNum, vector<bool> &isForegroundFG, vector<edgeEle> &sourceFGColors, int sourceBGColorNum, vector<bool> &isForegroundBG, vector<edgeEle> &sourceBGColors, int sourceFGColorNumBack, vector<bool> &isForegroundFGBack, vector<edgeEle> &sourceFGColorsBack, int sourceBGColorNumBack, int chosen_num, vector<bool> &isForegroundBGBack, vector<edgeEle> &sourceBGColorsBack, bool isSecond)
{
	float alpha = 10.0, beta = 2.0, gamma = 5.0;
	int elementNum = 7;
	int targetBlackNum = width*height - countNonZero(tar_edge);
	int count = 0;

	Mat targetSetFGA = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetFGB = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetBGA = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetBGB = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetFGBackA = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetFGBackB = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetBGBackA = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSetBGBackB = Mat::zeros(targetBlackNum, elementNum, CV_32F);
	Mat targetSidePts = Mat::zeros(targetBlackNum, 4, CV_32S);
	for (int y = 0; y < height; y++)
	{
		uchar *tar_edge_row = tar_edge.ptr<uchar>(y);
		float *targetDxRow = targetDx.ptr<float>(y);
		float *targetDyRow = targetDy.ptr<float>(y);
		for (int x = 0; x < width; x++)
		{
			if (tar_edge_row[x] == 0)
			{
				float x_off = targetDxRow[x], y_off = targetDyRow[x];
				Vec3b ScalarA(0, 0, 0), ScalarB(0, 0, 0);
				Point pt_a, pt_b;
				float angle;
				ComputeEgFeature(angle, x_off, y_off, x, y, point_dis, height, width, tar_img, pt_a, pt_b, ScalarA, ScalarB);
				int offFGxA = offsetFGx.at<int>(pt_a.y, pt_a.x), offFGyA = offsetFGy.at<int>(pt_a.y, pt_a.x), offFGxB = offsetFGx.at<int>(pt_b.y, pt_b.x), offFGyB = offsetFGy.at<int>(pt_b.y, pt_b.x);
				int offBGxA = offsetBGx.at<int>(pt_a.y, pt_a.x), offBGyA = offsetBGy.at<int>(pt_a.y, pt_a.x), offBGxB = offsetBGx.at<int>(pt_b.y, pt_b.x), offBGyB = offsetBGy.at<int>(pt_b.y, pt_b.x);
				int offFGBackxA, offFGBackyA, offFGBackxB, offFGBackyB, offBGBackxA, offBGBackyA, offBGBackxB, offBGBackyB;
				if (isSecond)
				{
					offFGBackxA = offsetFGBackx.at<int>(pt_a.y, pt_a.x), offFGBackyA = offsetFGBacky.at<int>(pt_a.y, pt_a.x), offFGBackxB = offsetFGBackx.at<int>(pt_b.y, pt_b.x), offFGBackyB = offsetFGBacky.at<int>(pt_b.y, pt_b.x);
					offBGBackxA = offsetBGBackx.at<int>(pt_a.y, pt_a.x), offBGBackyA = offsetBGBacky.at<int>(pt_a.y, pt_a.x), offBGBackxB = offsetBGBackx.at<int>(pt_b.y, pt_b.x), offBGBackyB = offsetBGBacky.at<int>(pt_b.y, pt_b.x);
				}

				targetSidePts.at<int>(count, 0) = pt_a.x; targetSidePts.at<int>(count, 1) = pt_a.y; targetSidePts.at<int>(count, 2) = pt_b.x; targetSidePts.at<int>(count, 3) = pt_b.y;

				float *targetFGARow = targetSetFGA.ptr<float>(count);
				float *targetFGBRow = targetSetFGB.ptr<float>(count);
				float *targetBGARow = targetSetBGA.ptr<float>(count);
				float *targetBGBRow = targetSetBGB.ptr<float>(count);

				targetFGARow[0] = ScalarA[0] / 255.0*100.0*alpha;
				targetFGARow[1] = ScalarA[1] / 255.0*100.0*alpha;
				targetFGARow[2] = ScalarA[2] / 255.0*100.0*alpha;
				targetFGARow[3] = offFGxA / diagonalDis*100.0*gamma;
				targetFGARow[4] = offFGyA / diagonalDis*100.0*gamma;
				targetFGARow[5] = sin(angle)*100.0*beta;
				targetFGARow[6] = cos(angle)*100.0*beta;

				targetFGBRow[0] = ScalarB[0] / 255.0*100.0*alpha;
				targetFGBRow[1] = ScalarB[1] / 255.0*100.0*alpha;
				targetFGBRow[2] = ScalarB[2] / 255.0*100.0*alpha;
				targetFGBRow[3] = offFGxB / diagonalDis*100.0*gamma;
				targetFGBRow[4] = offFGyB / diagonalDis*100.0*gamma;
				targetFGBRow[5] = sin(angle + PI)*100.0*beta;
				targetFGBRow[6] = cos(angle + PI)*100.0*beta;

				targetBGARow[0] = ScalarA[0] / 255.0*100.0*alpha;
				targetBGARow[1] = ScalarA[1] / 255.0*100.0*alpha;
				targetBGARow[2] = ScalarA[2] / 255.0*100.0*alpha;
				targetBGARow[3] = offBGxA / diagonalDis*100.0*gamma;
				targetBGARow[4] = offBGyA / diagonalDis*100.0*gamma;
				targetBGARow[5] = sin(angle)*100.0*beta;
				targetBGARow[6] = cos(angle)*100.0*beta;

				targetBGBRow[0] = ScalarB[0] / 255.0*100.0*alpha;
				targetBGBRow[1] = ScalarB[1] / 255.0*100.0*alpha;
				targetBGBRow[2] = ScalarB[2] / 255.0*100.0*alpha;
				targetBGBRow[3] = offBGxB / diagonalDis*100.0*gamma;
				targetBGBRow[4] = offBGyB / diagonalDis*100.0*gamma;
				targetBGBRow[5] = sin(angle + PI)*100.0*beta;
				targetBGBRow[6] = cos(angle + PI)*100.0*beta;

				if (isSecond)
				{
					float *targetFGBackARow = targetSetFGBackA.ptr<float>(count);
					float *targetFGBackBRow = targetSetFGBackB.ptr<float>(count);
					float *targetBGBackARow = targetSetBGBackA.ptr<float>(count);
					float *targetBGBackBRow = targetSetBGBackB.ptr<float>(count);

					targetFGBackARow[0] = ScalarA[0] / 255.0*100.0*alpha;
					targetFGBackARow[1] = ScalarA[1] / 255.0*100.0*alpha;
					targetFGBackARow[2] = ScalarA[2] / 255.0*100.0*alpha;
					targetFGBackARow[3] = offFGBackxA / diagonalDis*100.0*gamma;
					targetFGBackARow[4] = offFGBackyA / diagonalDis*100.0*gamma;
					targetFGBackARow[5] = sin(angle)*100.0*beta;
					targetFGBackARow[6] = cos(angle)*100.0*beta;

					targetFGBackBRow[0] = ScalarB[0] / 255.0*100.0*alpha;
					targetFGBackBRow[1] = ScalarB[1] / 255.0*100.0*alpha;
					targetFGBackBRow[2] = ScalarB[2] / 255.0*100.0*alpha;
					targetFGBackBRow[3] = offFGBackxB / diagonalDis*100.0*gamma;
					targetFGBackBRow[4] = offFGBackyB / diagonalDis*100.0*gamma;
					targetFGBackBRow[5] = sin(angle + PI)*100.0*beta;
					targetFGBackBRow[6] = cos(angle + PI)*100.0*beta;

					targetBGBackARow[0] = ScalarA[0] / 255.0*100.0*alpha;
					targetBGBackARow[1] = ScalarA[1] / 255.0*100.0*alpha;
					targetBGBackARow[2] = ScalarA[2] / 255.0*100.0*alpha;
					targetBGBackARow[3] = offBGBackxA / diagonalDis*100.0*gamma;
					targetBGBackARow[4] = offBGBackyA / diagonalDis*100.0*gamma;
					targetBGBackARow[5] = sin(angle)*100.0*beta;
					targetBGBackARow[6] = cos(angle)*100.0*beta;

					targetBGBackBRow[0] = ScalarB[0] / 255.0*100.0*alpha;
					targetBGBackBRow[1] = ScalarB[1] / 255.0*100.0*alpha;
					targetBGBackBRow[2] = ScalarB[2] / 255.0*100.0*alpha;
					targetBGBackBRow[3] = offBGBackxB / diagonalDis*100.0*gamma;
					targetBGBackBRow[4] = offBGBackyB / diagonalDis*100.0*gamma;
					targetBGBackBRow[5] = sin(angle + PI)*100.0*beta;
					targetBGBackBRow[6] = cos(angle + PI)*100.0*beta;
				}
				count++;
			}
		}
	}

	// 	clock_t knnSearch_time = clock();
	Mat m_indicesFGA(Size(searchSize, count), CV_32S, Scalar(0)), m_indicesFGB(Size(searchSize, count), CV_32S, Scalar(0)), m_distsFGA(Size(searchSize, count), CV_32F, Scalar(0)), m_distsFGB(Size(searchSize, count), CV_32F, Scalar(0));
	Mat m_indicesBGA(Size(searchSize, count), CV_32S, Scalar(0)), m_indicesBGB(Size(searchSize, count), CV_32S, Scalar(0)), m_distsBGA(Size(searchSize, count), CV_32F, Scalar(0)), m_distsBGB(Size(searchSize, count), CV_32F, Scalar(0));
	Mat m_indicesFGBackA(Size(searchSize, count), CV_32S, Scalar(0)), m_indicesFGBackB(Size(searchSize, count), CV_32S, Scalar(0)), m_distsFGBackA(Size(searchSize, count), CV_32F, Scalar(0)), m_distsFGBackB(Size(searchSize, count), CV_32F, Scalar(0));
	Mat m_indicesBGBackA(Size(searchSize, count), CV_32S, Scalar(0)), m_indicesBGBackB(Size(searchSize, count), CV_32S, Scalar(0)), m_distsBGBackA(Size(searchSize, count), CV_32F, Scalar(0)), m_distsBGBackB(Size(searchSize, count), CV_32F, Scalar(0));
	Mat sourceFGSet = Mat::zeros(sourceFGColorNum, elementNum, CV_32F);
	for (int i = 0; i < sourceFGColorNum; i++)
	{
		float *sourceRow = sourceFGSet.ptr<float>(i);
		sourceRow[0] = sourceFGColors[i].color[0] / 255.0*100.0*alpha;
		sourceRow[1] = sourceFGColors[i].color[1] / 255.0*100.0*alpha;
		sourceRow[2] = sourceFGColors[i].color[2] / 255.0*100.0*alpha;
		sourceRow[3] = sourceFGColors[i].pos.x / diagonalDis*100.0*gamma;
		sourceRow[4] = sourceFGColors[i].pos.y / diagonalDis*100.0*gamma;
		sourceRow[5] = sourceFGColors[i].sinValue*100.0*beta;
		sourceRow[6] = sourceFGColors[i].cosValue*100.0*beta;
	}
	Mat sourceBGSet = Mat::zeros(sourceBGColorNum, elementNum, CV_32F);
	for (int i = 0; i < sourceBGColorNum; i++)
	{
		float *sourceRow = sourceBGSet.ptr<float>(i);
		sourceRow[0] = sourceBGColors[i].color[0] / 255.0*100.0*alpha;
		sourceRow[1] = sourceBGColors[i].color[1] / 255.0*100.0*alpha;
		sourceRow[2] = sourceBGColors[i].color[2] / 255.0*100.0*alpha;
		sourceRow[3] = sourceBGColors[i].pos.x / diagonalDis*100.0*gamma;
		sourceRow[4] = sourceBGColors[i].pos.y / diagonalDis*100.0*gamma;
		sourceRow[5] = sourceBGColors[i].sinValue*100.0*beta;
		sourceRow[6] = sourceBGColors[i].cosValue*100.0*beta;
	}

	flann::Index flann_index(sourceFGSet, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_MANHATTAN);
	flann::Index flann_index2(sourceBGSet, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_MANHATTAN);

#pragma omp parallel sections
	{
#pragma omp section
		{
			flann_index.knnSearch(targetSetFGA, m_indicesFGA, m_distsFGA, searchSize, flann::SearchParams(32));
		}
#pragma omp section
		{
		flann_index.knnSearch(targetSetFGB, m_indicesFGB, m_distsFGB, searchSize, flann::SearchParams(32));
	}
#pragma omp section
		{
			flann_index2.knnSearch(targetSetBGA, m_indicesBGA, m_distsBGA, searchSize, flann::SearchParams(32));
		}
#pragma omp section
		{
			flann_index2.knnSearch(targetSetBGB, m_indicesBGB, m_distsBGB, searchSize, flann::SearchParams(32));
		}
	}
	// 	float maxFloat= numeric_limits<float>::max();
	// 	m_distsFGA.setTo(maxFloat); m_distsFGB.setTo(maxFloat); 

	// 	m_distsBGA.setTo(maxFloat); m_distsBGB.setTo(maxFloat); 

	if (isSecond)
	{
		Mat sourceFGSet = Mat::zeros(sourceFGColorNumBack, elementNum, CV_32F);
		for (int i = 0; i < sourceFGColorNumBack; i++)
		{
			float *sourceRow = sourceFGSet.ptr<float>(i);
			sourceRow[0] = sourceFGColorsBack[i].color[0] / 255.0*100.0*alpha;
			sourceRow[1] = sourceFGColorsBack[i].color[1] / 255.0*100.0*alpha;
			sourceRow[2] = sourceFGColorsBack[i].color[2] / 255.0*100.0*alpha;
			sourceRow[3] = sourceFGColorsBack[i].pos.x / diagonalDis*100.0*gamma;
			sourceRow[4] = sourceFGColorsBack[i].pos.y / diagonalDis*100.0*gamma;
			sourceRow[5] = sourceFGColorsBack[i].sinValue*100.0*beta;
			sourceRow[6] = sourceFGColorsBack[i].cosValue*100.0*beta;
		}
		Mat sourceBGSet = Mat::zeros(sourceBGColorNumBack, elementNum, CV_32F);
		for (int i = 0; i < sourceBGColorNumBack; i++)
		{
			float *sourceRow = sourceBGSet.ptr<float>(i);
			sourceRow[0] = sourceBGColorsBack[i].color[0] / 255.0*100.0*alpha;
			sourceRow[1] = sourceBGColorsBack[i].color[1] / 255.0*100.0*alpha;
			sourceRow[2] = sourceBGColorsBack[i].color[2] / 255.0*100.0*alpha;
			sourceRow[3] = sourceBGColorsBack[i].pos.x / diagonalDis*100.0*gamma;
			sourceRow[4] = sourceBGColorsBack[i].pos.y / diagonalDis*100.0*gamma;
			sourceRow[5] = sourceBGColorsBack[i].sinValue*100.0*beta;
			sourceRow[6] = sourceBGColorsBack[i].cosValue*100.0*beta;
		}
		flann::Index flann_index(sourceFGSet, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_MANHATTAN);
		flann::Index flann_index2(sourceBGSet, flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_MANHATTAN);

#pragma omp parallel sections
		{
#pragma omp section
			{
				flann_index.knnSearch(targetSetFGBackA, m_indicesFGBackA, m_distsFGBackA, searchSize, flann::SearchParams(32));
			}
#pragma omp section
			{
			flann_index.knnSearch(targetSetFGBackB, m_indicesFGBackB, m_distsFGBackB, searchSize, flann::SearchParams(32));
		}
#pragma omp section
			{
				flann_index2.knnSearch(targetSetBGBackA, m_indicesBGBackA, m_distsBGBackA, searchSize, flann::SearchParams(32));
			}
#pragma omp section
			{
				flann_index2.knnSearch(targetSetBGBackB, m_indicesBGBackB, m_distsBGBackB, searchSize, flann::SearchParams(32));
			}
		}
	}
	// 	knnSearch_time = clock() - knnSearch_time;
	// 	printf("knnSearch time elapsed is %f seconds.\n", (float)knnSearch_time / CLOCKS_PER_SEC);

	Mat distanceA(Size(searchSize, count), CV_32F, Scalar(0)), distanceB(Size(searchSize, count), CV_32F, Scalar(0));
	Mat labelsA(Size(searchSize, count), CV_32S, Scalar(-1)), labelsB(Size(searchSize, count), CV_32S, Scalar(-1));
#pragma omp parallel sections
	{
#pragma omp section
		{
			computeSourceMinDis(m_distsFGA, m_distsBGA, m_indicesFGA, m_indicesBGA, m_distsFGBackA, m_distsBGBackA, m_indicesFGBackA, m_indicesBGBackA, distanceA, labelsA, count, searchSize, isForegroundFG, isForegroundBG, isForegroundFGBack, isForegroundBGBack, isSecond);
		}
#pragma omp section
		{
		computeSourceMinDis(m_distsFGB, m_distsBGB, m_indicesFGB, m_indicesBGB, m_distsFGBackB, m_distsBGBackB, m_indicesFGBackB, m_indicesBGBackB, distanceB, labelsB, count, searchSize, isForegroundFG, isForegroundBG, isForegroundFGBack, isForegroundBGBack, isSecond);
	}
	}

	int numA = 0, numB = 0;
	count = 0;
	for (int y = 0; y < height; y++)
	{
		uchar *tar_edge_row = tar_edge.ptr<uchar>(y);
		Vec3b *tar_result_row = tar_edge_result.ptr<Vec3b>(y);
		uchar* fg_mat_row = fg_mat.ptr<uchar>(y);
		uchar* bg_mat_row = bg_mat.ptr<uchar>(y);
		uchar* contour_mat_row = contour_mat.ptr<uchar>(y);
		for (int x = 0; x < width; x++)
		{
			if (tar_edge_row[x] == 0)
			{
				int *sidePtRow = targetSidePts.ptr<int>(count);
				Point pt_a = Point(sidePtRow[0], sidePtRow[1]);
				Point pt_b = Point(sidePtRow[2], sidePtRow[3]);

				float *disARow = distanceA.ptr<float>(count);
				int *labelARow = labelsA.ptr<int>(count);
				float *disBRow = distanceB.ptr<float>(count);
				int *labelBRow = labelsB.ptr<int>(count);

				double fgA = 0, fgB = 0, bgA = 0, bgB = 0;
				numA = 0, numB = 0;
				for (int m = 0; m<searchSize; m++)
				{
					numA++;
					if (disARow[m]>maxDifference&&numA > minNum)
						break;
					if (labelARow[m] == 1)
						fgA++;
					else
						bgA++;
				}
				for (int m = 0; m<searchSize; m++)
				{
					numB++;
					if (disBRow[m]>maxDifference&&numB > minNum)
						break;
					if (labelBRow[m] == 1)
						fgB++;
					else
						bgB++;
				}

				bool isAFg, isBFg;
				isAFg = fgA > bgA ? true : false;
				isBFg = fgB > bgB ? true : false;

				if (isAFg&&isBFg)
				{
					tar_result_row[x][2] = 255;
					tar_result_row[x][1] = 255;
					tar_result_row[x][0] = 0;
					fg_mat_row[x] = 0;
				}
				else if ((isAFg&&!isBFg) || (!isAFg&&isBFg))
				{
					if (isAFg)
					{
						fgConstraint.at<uchar>(pt_a.y, pt_a.x) = 255;
						tar_weight.at<uchar>(pt_a.y, pt_a.x) = fgA / numA * 255;
					}
					else{
						bgConstraint.at<uchar>(pt_a.y, pt_a.x) = 255;
						tar_weight.at<uchar>(pt_a.y, pt_a.x) = bgA / numA * 255;
					}
					if (isBFg)
					{
						fgConstraint.at<uchar>(pt_b.y, pt_b.x) = 255;
						tar_weight.at<uchar>(pt_b.y, pt_b.x) = fgB / numB * 255;
					}
					else{
						bgConstraint.at<uchar>(pt_b.y, pt_b.x) = 255;
						tar_weight.at<uchar>(pt_b.y, pt_b.x) = bgB / numB * 255;
					}

					tar_result_row[x][2] = 255;
					tar_result_row[x][1] = 0;
					tar_result_row[x][0] = 0;
					contour_mat_row[x] = 0;
				}
				else
				{
					tar_result_row[x][2] = 0;
					tar_result_row[x][1] = 0;
					tar_result_row[x][0] = 255;
					bg_mat_row[x] = 0;
				}
				count++;
			}
		}
	}
}

Rect videoCutout::templateMatchFG(Mat &ori_img, Mat &tar_img, Mat &ori_mask, Mat &ori_edge, Mat &sourcePatch, Mat &sourceMaskPatch, Mat &sourceEdgePatch, Mat &targetPatch, double scale_fac, bool leapPropagate)
{
	int img_height = tar_img.rows, img_width = tar_img.cols;

	//get the bounding box of the patch image
	vector<int> temp;
	temp = getBoundRect(ori_mask);
	int minX = temp[0]; int minY = temp[1]; int maxX = temp[2]; int maxY = temp[3];

	sourcePatch = Mat(ori_img, Rect(minX, minY, maxX - minX, maxY - minY)).clone();
	sourceMaskPatch = Mat(ori_mask, Rect(minX, minY, maxX - minX, maxY - minY)).clone();
	sourceEdgePatch = Mat(ori_edge, Rect(minX, minY, maxX - minX, maxY - minY)).clone();
	targetPatch = Mat(sourcePatch.size(), CV_8UC3, Scalar(0)).clone();
	resize(targetPatch, targetPatch, Size(), scale_fac, scale_fac);
	int final_width = targetPatch.cols > img_width ? img_width : targetPatch.cols;
	int final_height = targetPatch.rows > img_height ? img_height : targetPatch.rows;
	resize(targetPatch, targetPatch, Size(final_width, final_height));

	// 	imshow("sourcePatch",sourcePatch);
	// 	imshow("sourceMaskPatch",sourceMaskPatch);
	// 	waitKey();
	Rect rect;

	if (leapPropagate)
	{
		FPM_FFT fpm;
		fpm.Plan(tar_img.cols, tar_img.rows, 3, FPMF_WEIGHTED, 0);
		fpm.SetImage(tar_img.data, tar_img.step, FPMT_8U, NULL, 0);
		rect = gen_ssd(sourcePatch, sourceMaskPatch, tar_img, targetPatch, fpm);
	}

	int posx = minX - (final_width - sourcePatch.cols) / 2, posy = minY - (final_height - sourcePatch.rows) / 2;
	Rect tempRect = Rect(posx, posy, final_width, final_height);
	checkRectBorder(img_width, img_height, tempRect);
	ori_img(tempRect).copyTo(sourcePatch);
	ori_mask(tempRect).copyTo(sourceMaskPatch);
	ori_edge(tempRect).copyTo(sourceEdgePatch);

	if (!leapPropagate)
	{
		rect = tempRect;
		tar_img(tempRect).copyTo(targetPatch);
	}

	// 	imshow("targetPatch", targetPatch);
	// 	imshow("sourcePatch", sourcePatch);
	// 	imshow("sourceMaskPatch", sourceMaskPatch);
	// 	waitKey();

	return rect;
}

Rect videoCutout::gen_ssd(Mat &patch, Mat &patch_mask, Mat &tar_image, Mat &tar_patch, FPM_FFT& fpm)
{
	vector<double> weight;
	int height = patch.rows, width = patch.cols;
	int img_height = tar_image.rows, img_width = tar_image.cols;
	for (int i = 0; i < height; i++)
	{
		uchar *maskRow = patch_mask.ptr<uchar>(i);
		for (int j = 0; j < width; j++)
		{
			double temp = (double)maskRow[j];
			weight.push_back(temp);
		}
	}

	int pos; double pssdx;
	fpm.Match(patch.data, patch.cols, patch.rows, patch.step, FPMT_8U, &weight[0], &pos, &pssdx, 1, FPMF_TRUE_SSD);
	int pos_x = pos%img_width, pos_y = pos / img_width;

	// 	Mat tarPatchBackup=patch.clone();
	// 	Rect tarRect=Rect(pos_x, pos_y, patch.cols, patch.rows);
	// 	checkRectBorder(img_width, img_height, tarRect);
	// 	tar_image(tarRect).copyTo(tarPatchBackup);
	// 	imshow("tar_patch", tarPatchBackup);
	// 	waitKey();

	pos_x -= (tar_patch.cols - width) / 2; pos_y -= (tar_patch.rows - height) / 2;
	Rect temRect = Rect(pos_x, pos_y, tar_patch.cols, tar_patch.rows);
	checkRectBorder(img_width, img_height, temRect);
	tar_image(temRect).copyTo(tar_patch);

	// 	imshow("patch", patch);
	// 	imshow("tar_patch", tar_patch);
	// 	waitKey();

	return temRect;
}

vector<int> videoCutout::getBoundRect(Mat img_mask)
{
	Mat threshold_output;
	int thresh = 100, max_thresh = 255;
	threshold(img_mask, threshold_output, thresh, max_thresh, THRESH_BINARY);
	img_mask = threshold_output.clone();

	vector<vector<Point>> contours;
	findContours(threshold_output, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

	// 	Mat result(threshold_output.size(), CV_8U, Scalar(255));
	// 	drawContours(result, contours, -1, Scalar(0), 2);

	int maxX = 0, minX = threshold_output.cols, maxY = 0, minY = threshold_output.rows;

	for (int i = 0; i < contours.size(); i++)
	for (int j = 0; j < contours[i].size(); j++)
	{
		Point p = contours[i][j];

		maxX = max(maxX, p.x);
		minX = min(minX, p.x);

		maxY = max(maxY, p.y);
		minY = min(minY, p.y);
	}

	vector<int> temp;
	temp.push_back(minX);
	temp.push_back(minY);
	temp.push_back(maxX);
	temp.push_back(maxY);

	return temp;
}

vector<int> videoCutout::featureMatching(Mat &img_object, Mat &img_scene, Mat &obj_mask, vector<Point2f> &srcPoints, vector<Point2f> &tarPoints, bool isFG)
{
	srcPoints.clear(); tarPoints.clear();
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 1000;
	if (isFG)
		minHessian = 100;
	else
		minHessian = 500;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_object, keypoints_scene;
	detector.detect(img_object, keypoints_object);
	detector.detect(img_scene, keypoints_scene);

	vector<KeyPoint>::iterator iter;
	for (iter = keypoints_object.begin(); iter != keypoints_object.end();)
	{
		if (isFG)
		{
			if (obj_mask.at<uchar>(iter->pt.y, iter->pt.x) == 0)
				iter = keypoints_object.erase(iter);
			else
				iter++;
		}
		else{
			if (obj_mask.at<uchar>(iter->pt.y, iter->pt.x) == 255)
				iter = keypoints_object.erase(iter);
			else
				iter++;
		}
	}

	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	Mat descriptors_object, descriptors_scene;
	extractor.compute(img_object, keypoints_object, descriptors_object);
	extractor.compute(img_scene, keypoints_scene, descriptors_scene);

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_object, descriptors_scene, matches);

	//-- Quick calculation of max and min distances between keypoints
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		double distance = matches[i].distance;
		if (distance < min_dist) min_dist = distance;
		if (distance > max_dist) max_dist = distance;
	}
	// 	printf("-- Max dist : %f \n", max_dist );
	// 	printf("-- Min dist : %f \n", min_dist );
	// 	cout<<"The size of matches: "<<matches.size()<<endl;

	std::vector< DMatch > good_matches;
	int diagonalDis = sqrtNani((double)(img_object.rows*img_object.rows + img_object.cols*img_object.cols));
	int dxTotal = 0, dyTotal = 0;
	for (int i = 0; i < descriptors_object.rows; i++)
	{
		int dx = keypoints_object[matches[i].queryIdx].pt.x - keypoints_scene[matches[i].trainIdx].pt.x, dy = keypoints_object[matches[i].queryIdx].pt.y - keypoints_scene[matches[i].trainIdx].pt.y;
		int distance = sqrtNani((double)(dx*dx + dy*dy));
		if (isFG)
		{
			if (matches[i].distance < (max_dist - min_dist) / 4 + min_dist && distance < diagonalDis / 4)
			{
				srcPoints.push_back(Point(keypoints_object[matches[i].queryIdx].pt.x, keypoints_object[matches[i].queryIdx].pt.y));
				tarPoints.push_back(Point(keypoints_scene[matches[i].trainIdx].pt.x, keypoints_scene[matches[i].trainIdx].pt.y));
				good_matches.push_back(matches[i]);
				dxTotal += dx, dyTotal += dy;
			}
		}
		else
		{
			if (matches[i].distance < (max_dist - min_dist) / 4 + min_dist && distance < diagonalDis / 4)
			{
				srcPoints.push_back(Point(keypoints_object[matches[i].queryIdx].pt.x, keypoints_object[matches[i].queryIdx].pt.y));
				tarPoints.push_back(Point(keypoints_scene[matches[i].trainIdx].pt.x, keypoints_scene[matches[i].trainIdx].pt.y));
				good_matches.push_back(matches[i]);
				dxTotal += dx, dyTotal += dy;
			}
		}
	}

	vector<Point2f> srcPointsClone, tarPointsClone;
	srcPointsClone.resize(srcPoints.size());
	tarPointsClone.resize(srcPoints.size());
	copy(srcPoints.begin(), srcPoints.end(), srcPointsClone.begin());
	copy(tarPoints.begin(), tarPoints.end(), tarPointsClone.begin());
	if (srcPoints.size() >= 10)
	{
		Mat H = findHomography(srcPoints, tarPoints, RANSAC);
		std::vector<Point2f> target_features(good_matches.size());
		perspectiveTransform(srcPoints, target_features, H);
		vector<Point2f>::iterator tarItr, tarFeaItr, souItr;
		int num = 0;
		for (tarItr = tarPoints.begin(), tarFeaItr = target_features.begin(), souItr = srcPoints.begin(); tarItr != tarPoints.end();)
		{
			float dx = tarItr->x - tarFeaItr->x, dy = tarItr->y - tarFeaItr->y;
			float dis = sqrtNani(dx*dx + dy*dy);
			if (dis > 5)
			{
				dxTotal -= (int)(souItr->x - tarItr->x), dyTotal -= (int)(souItr->y - tarItr->y);
				tarItr = tarPoints.erase(tarItr); tarFeaItr = target_features.erase(tarFeaItr); souItr = srcPoints.erase(souItr);
				num++;
			}
			else{
				tarItr++; tarFeaItr++; souItr++;
			}
		}
		// 	printf("%i bad points are deleted..\n", num);
	}
	if (srcPoints.size() < 6)
	{
		copy(srcPointsClone.begin(), srcPointsClone.end(), srcPoints.begin());
		copy(tarPointsClone.begin(), tarPointsClone.end(), tarPoints.begin());
	}

	int matchSize = srcPoints.size();
	if (matchSize != 0)
	{
		dxTotal /= matchSize;
		dyTotal /= matchSize;
	}
	vector<int> offset;
	offset.push_back(dxTotal); offset.push_back(dyTotal);

	return offset;
}

void videoCutout::computeSourceColorSet(Mat &ori_edge, Mat &ori_edge_result, Mat &ori_img, Mat & ori_mask, Mat &sourceDx, Mat &sourceDy, int sourceBoundarySize, int sideDis, int &sourceColorNum, int chosen_num, vector<bool> &isLabeled, vector<edgeEle> &sourceColors)
{
	int width = ori_edge.cols, height = ori_edge.rows;
	for (int y = 0; y < height; y++)
	{
		Vec3b *ori_result_row = ori_edge_result.ptr<Vec3b>(y);
		uchar *ori_edge_row = ori_edge.ptr<uchar>(y);
		float *sourceDxRow = sourceDx.ptr<float>(y);
		float *sourceDyRow = sourceDy.ptr<float>(y);
		for (int x = 0; x < width; x++)
		{
			if (ori_edge_row[x] == 0)
			{
				Rect tempRect = Rect(x - sourceBoundarySize, y - sourceBoundarySize, sourceBoundarySize * 2 + 1, sourceBoundarySize * 2 + 1);
				checkRectBorderClip(width, height, tempRect);
				Mat temp_ori_mask = ori_mask(tempRect).clone();
				int white_size = countNonZero(temp_ori_mask);

				if (x%chosen_num == 0 || y%chosen_num == 0)
					// 				if (true)
				{
					float x_off = sourceDxRow[x], y_off = sourceDyRow[x];
					Vec3b ScalarA(0, 0, 0), ScalarB(0, 0, 0);
					Point pt_a, pt_b;
					float angle;
					ComputeEgFeature(angle, x_off, y_off, x, y, sideDis, height, width, ori_img, pt_a, pt_b, ScalarA, ScalarB);

					if (ori_mask.at<uchar>(pt_a.y, pt_a.x) == 255)
					{
						isLabeled[sourceColorNum] = true;
					}
					else{
						isLabeled[sourceColorNum] = false;
					}
					sourceColors[sourceColorNum].color = ScalarA;
					sourceColors[sourceColorNum].pos = pt_a;
					sourceColors[sourceColorNum].sinValue = sin(angle);
					sourceColors[sourceColorNum].cosValue = cos(angle);
					sourceColorNum++;
					if (ori_mask.at<uchar>(pt_b.y, pt_b.x) == 255)
					{
						isLabeled[sourceColorNum] = true;
					}
					else{
						isLabeled[sourceColorNum] = false;
					}
					sourceColors[sourceColorNum].color = ScalarB;
					sourceColors[sourceColorNum].pos = pt_b;
					sourceColors[sourceColorNum].sinValue = sin(angle + PI);
					sourceColors[sourceColorNum].cosValue = cos(angle + PI);
					sourceColorNum++;
				}

				if (white_size == 0)
				{
					//BG
					ori_result_row[x] = Vec3b(255, 0, 0);
				}
				else if (white_size == tempRect.width*tempRect.height)
				{
					//FG
					ori_result_row[x] = Vec3b(0, 255, 255);
				}
				else
				{
					//contour
					ori_result_row[x] = Vec3b(0, 0, 255);
				}
			}
		}
	}
	// 	cout << "source color number: " << sourceColorNum << endl;
	isLabeled.erase(isLabeled.begin() + sourceColorNum, isLabeled.end());
	sourceColors.erase(sourceColors.begin() + sourceColorNum, sourceColors.end());
}

void videoCutout::levelset(Mat &result, bool useEdge)
{
	clock_t time = clock();

	/****levelset****/
	cout << "start levelset.." << endl;
	float kGAC = 0.1;
	int maxIter = 15;
	int iterCircle = 10;
	float stopCon = 0.9999;
	int maxNoise = 50;
	bool isDebug = false;
	int dataTermMaxDis = 9;
	int dataTermMinDis = 2;

	char imageName[256]; string outputDir = "", frameNum = "000_001";
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchMask_%s.png", outputDir.c_str(), frameNum.c_str());
	Mat sourceFGMaskPatchFront = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%starPmMask_%s.png", outputDir.c_str(), frameNum.c_str());
	Mat tarPMMask = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatchEdge_%s.png", outputDir.c_str(), frameNum.c_str());
	Mat targetPatchEdge = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatch_%s.png", outputDir.c_str(), frameNum.c_str());
	Mat targetPatch = imread(imageName);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%sdataTermEdge_%s.xml", outputDir.c_str(), frameNum.c_str());
	int width = tarPMMask.cols, height = tarPMMask.rows;
	Mat dataTermEdge, dataTermPM;
	std::ifstream is(imageName, std::ios::in | std::ios::binary);
	int rows, cols, type;
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	dataTermEdge.create(rows, cols, type);
	is.read((char*)dataTermEdge.data, dataTermEdge.step.p[0] * dataTermEdge.rows); is.close();
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%sdataTermPM_%s.xml", outputDir.c_str(), frameNum.c_str());
	is.open(imageName, std::ios::in | std::ios::binary);
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	dataTermPM.create(rows, cols, type);
	is.read((char*)dataTermPM.data, dataTermPM.step.p[0] * dataTermPM.rows); is.close();

	Mat gGAC(Size(width, height), CV_32FC1, Scalar(0));
	for (int y = 0; y < height; y++)
	{
		uchar *tarEdgeRow = targetPatchEdge.ptr<uchar>(y);
		float *gGACRow = gGAC.ptr<float>(y);
		for (int x = 0; x < width; x++)
			gGACRow[x] = 1 / (1 + ((float)(255 - tarEdgeRow[x]) / 255)*((float)(255 - tarEdgeRow[x]) / 255) / (kGAC*kGAC));
	}

	Mat phi, tar_img_clone, preMask = tarPMMask.clone();
	int count = 0;
	int iterNum = 1;
	while (true)
	{
		Mat distanceFG, distanceBG, distance, distance2, phiDx, phiDy, tar_mask_clone = preMask.clone();
		distanceTransform(tar_mask_clone, distanceFG, CV_DIST_L2, CV_DIST_MASK_PRECISE);
		tar_mask_clone = 255 - preMask;
		distanceTransform(tar_mask_clone, distanceBG, CV_DIST_L2, CV_DIST_MASK_PRECISE);
		distance = (distanceFG - distanceBG); distance2 = (distanceFG + distanceBG);

		Mat dataTermPMUpdate(Size(width, height), CV_32FC1, Scalar(0));
		for (int y = 0; y < height; y++)
		{
			float *distanceRow = distance2.ptr<float>(y);
			float *PMRow = dataTermPMUpdate.ptr<float>(y);
			for (int x = 0; x<width; x++)
			{
				if (distanceRow[x]>dataTermMaxDis || distanceRow[x] <= dataTermMinDis)
					PMRow[x] = 0;
				else{
					if (preMask.at<uchar>(y, x) == 255)
						PMRow[x] = dataTermPM.at<float>(y, x) * ((distanceRow[x] - dataTermMinDis) / (dataTermMaxDis - dataTermMinDis));
					else
						PMRow[x] = -dataTermPM.at<float>(y, x) * ((distanceRow[x] - dataTermMinDis) / (dataTermMaxDis - dataTermMinDis));
				}
			}
		}
		if (count%iterCircle == 0)
			phi = distance.clone();

		gradient(distance, phiDx, CV_32F, 1, 0, 1);
		gradient(distance, phiDy, CV_32F, 0, 1, 1);

		Mat dPhi(Size(width, height), CV_32FC1, Scalar(0));
		Mat gacTermX(Size(width, height), CV_32FC1, Scalar(0));
		Mat gacTermY(Size(width, height), CV_32FC1, Scalar(0));
		Mat gacTermDx, gacTermDy;
		Mat gacTerm;
		for (int y = 0; y < height; y++)
		{
			float *gGACRow = gGAC.ptr<float>(y);
			float *phiDxRow = phiDx.ptr<float>(y);
			float *phiDyRow = phiDy.ptr<float>(y);
			float *gacTermXRow = gacTermX.ptr<float>(y);
			float *gacTermYRow = gacTermY.ptr<float>(y);
			for (int x = 0; x < width; x++)
			{
				float phiNorm = sqrtNani(phiDxRow[x] * phiDxRow[x] + phiDyRow[x] * phiDyRow[x]) + 0.000001;
				phiDxRow[x] /= phiNorm; phiDyRow[x] /= phiNorm;
				gacTermXRow[x] = phiDxRow[x] * gGACRow[x]; gacTermYRow[x] = phiDyRow[x] * gGACRow[x];
			}
		}
		gradient(gacTermX, gacTermDx, CV_32F, 1, 0, 1);
		gradient(gacTermY, gacTermDy, CV_32F, 0, 1, 1);
		gacTerm = gacTermDx.mul(abs(phiDx)) + gacTermDy.mul(abs(phiDy));
		// 		gacTerm = gacTermDx + gacTermDy;
		float alpha = 1, beta = 0.5, gamma = 0;
		if (!useEdge)
			beta = 0;
		dPhi = alpha*gacTerm + beta*dataTermEdge + gamma*dataTermPMUpdate;
		GaussianBlur(dPhi, dPhi, Size(3, 3), 0.5);
		phi += dPhi;
		GaussianBlur(phi, phi, Size(3, 3), 0.5);

		// 		Mat colorMap(Size(width, height), CV_8UC3, Scalar(0, 0, 0));
		// 		double maxNum = numeric_limits<double>::min(), minNum = numeric_limits<double>::max();
		// 		minMaxLoc(dataTermEdge, &minNum, &maxNum);
		// 		maxNum = max(maxNum, -minNum);
		// 		for (int y = 0; y < height; y++)
		// 		{
		// 			for (int x = 0; x < width; x++)
		// 			{
		// 				float tempDphi = dataTermEdge.at<float>(y, x) / maxNum;
		// 
		// 				if (tempDphi>0)
		// 					colorMap.at<Vec3b>(y, x) = Vec3b(0, 0, tempDphi * 255);
		// 				else
		// 					colorMap.at<Vec3b>(y, x) = Vec3b(abs(tempDphi) * 255, 0, 0);
		// 			}
		// 		}
		// 		imshow("colorMap", colorMap);
		// 		waitKey();

		Mat tar_mask2(Size(width, height), CV_8U, Scalar(125));
		for (int y = 0; y < height; y++)
		{
			float *phiRow = phi.ptr<float>(y);
			uchar *tarMaskRow = tar_mask2.ptr<uchar>(y);
			for (int x = 0; x < width; x++)
			{
				if (phiRow[x]>0)
					tarMaskRow[x] = 255;
				else if (phiRow[x] < 0)
					tarMaskRow[x] = 0;
			}
		}
		removeNoiseSimple(tar_mask2, maxNoise);

		// 		tar_img_clone = targetPatch.clone();
		// 		for (int y = 0; y < height; y++)
		// 		{
		// 			uchar* tar_mask_row = tar_mask2.ptr<uchar>(y);
		// 			Vec3b* tar_img_row = tar_img_clone.ptr<Vec3b>(y);
		// 			for (int x = 0; x < width; x++)
		// 			{
		// 				if (tar_mask_row[x] == 255)
		// 				{
		// 					tar_img_row[x][2] = tar_img_row[x][2] * 0.7;
		// 					tar_img_row[x][1] = tar_img_row[x][1] * 0.7;
		// 					tar_img_row[x][0] = tar_img_row[x][0] * 0.7 + 255 * 0.3;
		// 				}
		// 			}
		// 		}
		// 		tar_mask_clone = tar_mask2.clone();
		// 		vector<vector<Point>> contours;
		// 		findContours(tar_mask_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		// 		drawContours(tar_img_clone, contours, -1, Scalar(0, 0, 255), 2);
		// // 		string dirPath = videoPath + "levelset_noEdge";
		// // 		if (!exists(path(dirPath)))
		// // 			create_directory(path(dirPath));
		// // 		char iterString[256], filePath[256]; sprintf(iterString, "%i", iterNum);
		// // 		sprintf(filePath, "%slevelset_noEdge\\targetMask_LS_%s.png", videoPath.c_str(), iterString);
		// // 		imwrite(filePath, tar_img_clone);
		// 		imshow("tarImg", tar_img_clone);
		// 		waitKey();

		if (count%iterCircle == 1)
		{
			if (count != 1)
			{
				Mat conMat;
				bitwise_and(preMask, tar_mask2, conMat);
				int nonZero = countNonZero(tar_mask2) + countNonZero(preMask);
				int commonArea = countNonZero(conMat);
				if ((double)(2 * commonArea) / nonZero > stopCon || iterNum >= maxIter)
				{
					removeNoiseSimple(preMask, maxNoise);
					// 					removeNoise2(preMask, sourceFGMaskPatchFront, maxNoise);
					//Only leave the biggest region
					// 					vector<vector<Point>> contours2;
					// 					Mat tar_mask_clone2 = preMask.clone();
					// 					findContours(tar_mask_clone2, contours2, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
					// 					sort(contours2.begin(), contours2.end(), UDgreater3);
					// 					preMask.setTo(0);
					// 					drawContours(preMask, contours2, 0, Scalar(255), -1);

					tar_img_clone = targetPatch.clone();
					for (int y = 0; y < height; y++)
					{
						uchar* tar_mask_row = preMask.ptr<uchar>(y);
						Vec3b* tar_img_row = tar_img_clone.ptr<Vec3b>(y);
						for (int x = 0; x < width; x++)
						{
							if (tar_mask_row[x] == 255)
							{
								tar_img_row[x][2] = tar_img_row[x][2] * 0.7;
								tar_img_row[x][1] = tar_img_row[x][1] * 0.7;
								tar_img_row[x][0] = tar_img_row[x][0] * 0.7 + 255 * 0.3;
							}
						}
					}

					Mat tar_mask_clone = preMask.clone();
					vector<vector<Point>> contours;
					findContours(tar_mask_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
					drawContours(tar_img_clone, contours, -1, Scalar(0, 0, 255), 2);

					memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%star_mask_levelset_%s.png", outputDir.c_str(), frameNum.c_str());
					imwrite(imageName, preMask);
					memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%star_result_levelset_%s.png", outputDir.c_str(), frameNum.c_str());
					imwrite(imageName, tar_img_clone);

					std::ifstream inRect(".\\rect.xml", std::ios::in | std::ios::binary);
					int xx, yy, width, height;
					inRect >> xx; inRect.ignore(1); inRect >> yy; inRect.ignore(1); inRect >> width; inRect.ignore(1); inRect >> height;
					Mat tarMask(Size(width, height), CV_8U, Scalar(0));
					Mat tarRect = tarMask(Rect(xx, yy, preMask.cols, preMask.rows));
					preMask.copyTo(tarRect);
					memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%star_final_mask_%s.png", outputDir.c_str(), frameNum.c_str());
					imwrite(imageName, tarMask);

					//tarMask.copyTo(result);
					result = tarMask.clone();
					break;
				}
			}
		}
		preMask = tar_mask2.clone();
		count++;
		iterNum++;
	}
	/****levelset****/

	time = clock() - time;
	printf("the time elapsed is %f seconds.\n", (float)time / CLOCKS_PER_SEC);
	// 	imshow("tarImg", tar_img_clone);
	// 	waitKey();
}

#define UNKNOWN_FLOW_THRESH2 1e9  
// Color encoding of flow vectors from:  
// http://members.shaw.ca/quadibloc/other/colint.htm  
// This code is modified from:  
// http://vision.middlebury.edu/flow/data/  
void makecolorwheel2(vector<Scalar> &colorwheel)
{
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;

	int i;

	for (i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

void motionToColor2(Mat flow, Mat &color)
{
	if (color.empty())
		color.create(flow.rows, flow.cols, CV_8UC3);

	static vector<Scalar> colorwheel; //Scalar r,g,b
	if (colorwheel.empty())
		makecolorwheel2(colorwheel);

	// determine motion range:  
	float maxrad = -1;

	// Find max flow to normalize fx and fy  
#pragma omp parallel for
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);
			float fx = flow_at_point[0];
			float fy = flow_at_point[1];
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH2) || (fabs(fy) > UNKNOWN_FLOW_THRESH2))
				continue;
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = maxrad > rad ? maxrad : rad;
		}
	}

#pragma omp parallel for
	for (int i = 0; i < flow.rows; ++i)
	{
		for (int j = 0; j < flow.cols; ++j)
		{
			uchar *data = color.data + color.step[0] * i + color.step[1] * j;
			Vec2f flow_at_point = flow.at<Vec2f>(i, j);

			float fx = flow_at_point[0] / maxrad;
			float fy = flow_at_point[1] / maxrad;
			if ((fabs(fx) >  UNKNOWN_FLOW_THRESH2) || (fabs(fy) > UNKNOWN_FLOW_THRESH2))
			{
				data[0] = data[1] = data[2] = 0;
				continue;
			}
			float rad = sqrt(fx * fx + fy * fy);

			float angle = atan2(-fy, -fx) / CV_PI;
			float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);
			int k0 = (int)fk;
			int k1 = (k0 + 1) % colorwheel.size();
			float f = fk - k0;
			//f = 0; // uncomment to see original color wheel  

			for (int b = 0; b < 3; b++)
			{
				float col0 = colorwheel[k0][b] / 255.0;
				float col1 = colorwheel[k1][b] / 255.0;
				float col = (1 - f) * col0 + f * col1;
				if (rad <= 1)
					col = 1 - rad * (1 - col); // increase saturation with radius  
				else
					col *= .75; // out of range  
				data[2 - b] = (int)(255.0 * col);
			}
		}
	}
}

void videoCutout::computeSourceMinDis(Mat &m_distsFG, Mat &m_distsBG, Mat &m_indicesFG, Mat &m_indicesBG, Mat &m_distsFGBack, Mat &m_distsBGBack, Mat &m_indicesFGBack, Mat &m_indicesBGBack, Mat &distance, Mat &labels, int count, int searchSize, vector<bool> &isForegroundFG, vector<bool> &isForegroundBG, vector<bool> &isForegroundFGBack, vector<bool> &isForegroundBGBack, bool isSecond)
{
	for (int y = 0; y < count; y++)
	{
		vector<float> localDis; localDis.resize(searchSize * 2);
		vector<int> localLabel; localLabel.resize(searchSize * 2);
		if (isSecond)
		{
			localDis.resize(searchSize * 4);
			localLabel.resize(searchSize * 4);
		}

		float *disFGRow = m_distsFG.ptr<float>(y);
		float *disBGRow = m_distsBG.ptr<float>(y);
		float *disFGBackRow = m_distsFGBack.ptr<float>(y);
		float *disBGBackRow = m_distsBGBack.ptr<float>(y);
		int *idxFGRow = m_indicesFG.ptr<int>(y);
		int *idxBGRow = m_indicesBG.ptr<int>(y);
		int *idxFGBackRow = m_indicesFGBack.ptr<int>(y);
		int *idxBGBackRow = m_indicesBGBack.ptr<int>(y);
		for (int x = 0; x < searchSize; x++)
		{
			localDis[x] = disFGRow[x];
			if (isForegroundFG[idxFGRow[x]])
				localLabel[x] = 1;
			else
				localLabel[x] = 0;
			localDis[x + searchSize] = disBGRow[x];
			if (isForegroundBG[idxBGRow[x]])
				localLabel[x + searchSize] = 1;
			else
				localLabel[x + searchSize] = 0;

			if (isSecond)
			{
				localDis[x + searchSize * 2] = disFGBackRow[x];
				if (isForegroundFGBack[idxFGBackRow[x]])
					localLabel[x + searchSize * 2] = 1;
				else
					localLabel[x + searchSize * 2] = 0;
				localDis[x + searchSize * 3] = disBGBackRow[x];
				if (isForegroundBGBack[idxBGBackRow[x]])
					localLabel[x + searchSize * 3] = 1;
				else
					localLabel[x + searchSize * 3] = 0;
			}
		}

		vector<double> localDisClone;
		localDisClone.resize(localDis.size());
		copy(localDis.begin(), localDis.end(), localDisClone.begin());
		sort(localDis.begin(), localDis.end());
		float *disRow = distance.ptr<float>(y);
		int *labelRow = labels.ptr<int>(y);
		for (int x = 0; x < searchSize; x++)
		{
			int idx = find(localDisClone.begin(), localDisClone.end(), localDis[x]) - localDisClone.begin();
			disRow[x] = localDisClone[idx];
			labelRow[x] = localLabel[idx];
		}
	}
}

void videoCutout::removeNoiseSimple(Mat &tar_mask, int maxNoise)
{
	int height = tar_mask.rows, width = tar_mask.cols;
	Mat tar_mask_clone = tar_mask.clone();
	vector<vector<Point>> contours;
	findContours(tar_mask_clone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	for (int idx = 0; idx < contours.size(); idx++)
	{
		const vector<Point>& c = contours[idx];
		Mat tempMask(Size(width, height), CV_8U, Scalar(0));
		drawContours(tempMask, contours, idx, Scalar(255), CV_FILLED, 8);
		for (int m = 0; m < c.size(); m++)
			tempMask.at<uchar>(c[m].y, c[m].x) = 0;
		vector<vector<Point>> tempContours;
		findContours(tempMask, tempContours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		int boundaryNum = 0, blackNum = 0;
		if (tempContours.size() == 0)
			boundaryNum = 0;
		else
			boundaryNum = tempContours[0].size();
		for (int m = 0; m < boundaryNum; m++)
		{
			if (tar_mask.at<uchar>(tempContours[0][m].y, tempContours[0][m].x) == 0)
				blackNum++;
		}
		//judge the inner loop is a white or black area.
		if (boundaryNum != 0 && (double)blackNum / boundaryNum > 0.5)
		{
			Scalar color(255);
			double area = fabs(contourArea(Mat(c)));
			if (area < maxNoise)
				drawContours(tar_mask, contours, idx, color, CV_FILLED, 8);
		}
		else
		{
			Scalar color(0);
			double area = fabs(contourArea(Mat(c)));
			area += c.size();
			if (area < maxNoise)
				drawContours(tar_mask, contours, idx, color, CV_FILLED, 8);
		}
	}
}

extern "C" void patchmatchGPU(BITMAP3 *a, BITMAP3 *b, BITMAP3 *&ann, BITMAP3 *&annd, bool leapPropagate);

bool videoCutout::PatchMatch(Mat &sourceImgFront, Mat &sourceMaskFront, Mat& sourceEdgeFront, Mat &sourceImgBack, Mat &sourceMaskBack, Mat& sourceEdgeBack, Mat &targetImg, Mat& targetEdge, int intervalNum)
{
	clock_t time = clock();

	Mat sourceFGPatch, sourceFGMaskPatch, targetPatch, targetPatchMask;
	Mat sourceBGPatchDeform, sourceBGMaskPatchDeform, sourceBGPatchEdgeDeform;
	Mat sourceBGPatch, sourceBGMaskPatch;
	Mat sourceFGPatchBack, sourceFGMaskPatchBack, sourceBGPatchBack, sourceBGMaskPatchBack;
	Mat sourceFGPatchEdge, sourceBGPatchEdge, targetPatchEdge;
	Mat sourceFGPatchEdgeBack, sourceBGPatchEdgeBack;
	Mat sourceFGEdge_intial, sourceBGEdge_intial, targetEdge_intial;
	BITMAP3 *sF, *sB, *sBackF, *sBackB, *t, *annFG, *anndFG, *annBG, *anndBG, *annFGBack, *anndFGBack, *annBGBack, *anndBGBack;
	Mat offsetFGx, offsetFGy, offsetBGx, offsetBGy;
	Mat offsetBackFGx, offsetBackFGy, offsetBackBGx, offsetBackBGy;
	Mat tarPMMask, matchedPositionFG, matchedPositionBG, matchedPosition;
	Mat offsetFG, offsetBG;

	int ori_width = sourceImgFront.cols, ori_height = sourceImgFront.rows;
	int width, height, diagonalDis;
	vector<Point2f> srcPoints, tarPoints; bool isFG = false;
	char frameNum[] = "000_001", imageName[256], edgeName[256], videoPath[] = "";

	bool isSecond = false;
	if (!sourceImgBack.empty() && !sourceMaskBack.empty() && !sourceEdgeBack.empty())
		isSecond = true;

	vector<vector<Point>> contours;
	double scale_fac = 0;
	bool leapPropagate = false;
	if (intervalNum >= 8)
		leapPropagate = true;

	if (leapPropagate)
		scale_fac = 2;

	Mat tar_show;
	bool isTooSmallMask = false;
	int iterNum = 0;
	bool isRepeated = false; Rect repeatRect;
	do
	{
		if (!isRepeated)
			scale_fac += 2.0;
		Rect targetRect;
		if (!isSecond)
		{
			Mat diffMat(sourceImgFront.size(), CV_8UC3, Scalar(0, 0, 0));
			absdiff(sourceImgFront, targetImg, diffMat);
			cvtColor(diffMat, diffMat, CV_BGR2GRAY);
			int num = countNonZero(diffMat);
			if (num == 0)
			{
				sourceFGPatch = sourceImgFront.clone();
				sourceFGMaskPatch = sourceMaskFront.clone();
				sourceBGPatchDeform = sourceImgFront.clone();
				sourceBGMaskPatchDeform = sourceMaskFront.clone();
				sourceFGPatchEdge = sourceEdgeFront.clone();
				sourceBGPatchEdgeDeform = sourceEdgeFront.clone();
				targetPatch = targetImg.clone();
				targetPatchEdge = targetEdge.clone();
				targetRect = Rect(0, 0, sourceImgFront.cols, sourceImgFront.rows);
			}
			else
			{
				targetRect = templateMatchFG(sourceImgFront, targetImg, sourceMaskFront, sourceEdgeFront, sourceFGPatch, sourceFGMaskPatch, sourceFGPatchEdge, targetPatch, scale_fac, leapPropagate);

				if (isRepeated)
				{
					targetRect = repeatRect;
					targetPatch = targetImg(targetRect).clone();
					isRepeated = false;
				}
				targetPatchEdge = targetEdge(targetRect).clone();

				// the second time consuming part is featureMatching.
				vector<int> offset = featureMatching(sourceImgFront, targetImg, sourceMaskFront, srcPoints, tarPoints, isFG);
				Rect sourceRect = Rect(targetRect.x + offset[0], targetRect.y + offset[1], targetRect.width, targetRect.height); checkRectBorder(ori_width, ori_height, sourceRect);
				sourceBGPatch = sourceImgFront(sourceRect).clone(); sourceBGMaskPatch = sourceMaskFront(sourceRect).clone();
				sourceBGPatchEdge = sourceEdgeFront(sourceRect).clone();
				int localDiagDis = sqrtNani(sourceBGPatch.cols*sourceBGPatch.cols + sourceBGPatch.rows*sourceBGPatch.rows);
				if (localDiagDis > 300)
					featureMatching(sourceBGPatch, targetPatch, sourceBGMaskPatch, srcPoints, tarPoints, isFG);

#pragma omp parallel sections
				{
#pragma omp section
					{
						imageDeform(sourceBGPatch, sourceBGPatchDeform, srcPoints, tarPoints);
					}
#pragma omp section
					{
					imageDeform(sourceBGMaskPatch, sourceBGMaskPatchDeform, srcPoints, tarPoints);
				}
#pragma omp section
					{
						imageDeform(sourceBGPatchEdge, sourceBGPatchEdgeDeform, srcPoints, tarPoints);
					}
				}
			}
		}
		else
		{
			FILE *fp = fopen("param.txt", "r");
			if (!fp || fscanf(fp, "%d %d %d %d", &targetRect.x, &targetRect.y, &targetRect.width, &targetRect.height) != 4)
			{
				printf("\nfailed to read targetRect");
			}
			fclose(fp);
			if (intervalNum < 1000)
			{
				targetRect.x *= 0.6; targetRect.y *= 0.6; targetRect.width *= 0.6; targetRect.height *= 0.6;
			}
			Size tarSize = targetRect.size();

			sourceFGPatch = imread("fgL_img.jpg");
			sourceFGMaskPatch = imread("fgL_mask.png", 0);
			sourceBGPatchDeform = imread("bgL_img.jpg");
			sourceBGMaskPatchDeform = imread("bgL_mask.png", 0);
			sourceFGPatchBack = imread("fgR_img.jpg");
			sourceFGMaskPatchBack = imread("fgR_mask.png", 0);
			sourceBGPatchBack = imread("bgR_img.jpg");
			sourceBGMaskPatchBack = imread("bgR_mask.png", 0);
			targetPatch = imread("imgM.jpg");
			sourceFGPatchEdge = imread("fgL_edge.png");
			sourceBGPatchEdgeDeform = imread("bgL_edge.png");
			sourceFGPatchEdgeBack = imread("fgR_edge.png");
			sourceBGPatchEdgeBack = imread("bgR_edge.png");

			if (intervalNum < 1000)
			{
				resize(sourceFGPatch, sourceFGPatch, tarSize);
				resize(sourceFGMaskPatch, sourceFGMaskPatch, tarSize);
				resize(sourceBGPatchDeform, sourceBGPatchDeform, tarSize);
				resize(sourceBGMaskPatchDeform, sourceBGMaskPatchDeform, tarSize);
				resize(sourceFGPatchBack, sourceFGPatchBack, tarSize);
				resize(sourceFGMaskPatchBack, sourceFGMaskPatchBack, tarSize);
				resize(sourceBGPatchBack, sourceBGPatchBack, tarSize);
				resize(sourceBGMaskPatchBack, sourceBGMaskPatchBack, tarSize);
				resize(targetPatch, targetPatch, tarSize);
				resize(sourceFGPatchEdge, sourceFGPatchEdge, tarSize);
				resize(sourceBGPatchEdgeDeform, sourceBGPatchEdgeDeform, tarSize);
				resize(sourceFGPatchEdgeBack, sourceFGPatchEdgeBack, tarSize);
				resize(sourceBGPatchEdgeBack, sourceBGPatchEdgeBack, tarSize);
			}

			threshold(sourceFGMaskPatchBack, sourceFGMaskPatchBack, 100, 255, THRESH_BINARY);
			threshold(sourceBGMaskPatchBack, sourceBGMaskPatchBack, 100, 255, THRESH_BINARY);
			targetPatchEdge = targetEdge(targetRect).clone();
		}
		threshold(sourceFGMaskPatch, sourceFGMaskPatch, 100, 255, THRESH_BINARY);
		threshold(sourceBGMaskPatchDeform, sourceBGMaskPatchDeform, 100, 255, THRESH_BINARY);
		std::ofstream osRect(".\\rect.xml", std::ios::out | std::ios::trunc | std::ios::binary);
		osRect << (int)targetRect.x << " " << (int)targetRect.y << " " << (int)sourceImgFront.cols << " " << (int)sourceImgFront.rows;
		width = targetRect.width, height = targetRect.height;

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatch_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceFGPatch);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchMask_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceFGMaskPatch);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatch_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceBGPatchDeform);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchMask_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceBGMaskPatchDeform);
		if (isSecond)
		{
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceFGPatchBack);
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchMaskBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceFGMaskPatchBack);
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceBGPatchBack);
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchMaskBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceBGMaskPatchBack);
		}
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatch_%s.png", videoPath, frameNum);
		imwrite(imageName, targetPatch);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchEdge_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceFGPatchEdge);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchEdge_%s.png", videoPath, frameNum);
		imwrite(imageName, sourceBGPatchEdgeDeform);
		if (isSecond)
		{
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchEdgeBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceFGPatchEdgeBack);
			memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchEdgeBack_%s.png", videoPath, frameNum);
			imwrite(imageName, sourceBGPatchEdgeBack);
		}
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatchEdge_%s.png", videoPath, frameNum);
		imwrite(imageName, targetPatchEdge);
		/****preprocessing****/

		/****PatchMatch****/
		printf("Running PatchMatch\n");
		int maxNoise = 50;
		int limitedDis = 3;
		if (leapPropagate)
			limitedDis = 2;
		bool keepBiggestLoop = false;
		diagonalDis = sqrtNani((double)(width*width + height*height));

		sF = toBITMAP(sourceFGPatch.data, sourceFGPatch.cols, sourceFGPatch.rows, sourceFGPatch.step);
		sB = toBITMAP(sourceBGPatchDeform.data, sourceBGPatchDeform.cols, sourceBGPatchDeform.rows, sourceBGPatchDeform.step);
		sBackF = NULL, sBackB = NULL;
		if (isSecond)
		{
			sBackF = toBITMAP(sourceFGPatchBack.data, sourceFGPatchBack.cols, sourceFGPatchBack.rows, sourceFGPatchBack.step);
			sBackB = toBITMAP(sourceBGPatchBack.data, sourceBGPatchBack.cols, sourceBGPatchBack.rows, sourceBGPatchBack.step);
		}
		t = toBITMAP(targetPatch.data, targetPatch.cols, targetPatch.rows, targetPatch.step);

		annFG = NULL, anndFG = NULL, annBG = NULL, anndBG = NULL;
		patchmatchGPU(t, sF, annFG, anndFG, leapPropagate);
		patchmatchGPU(t, sB, annBG, anndBG, leapPropagate);
		annFGBack = NULL, anndFGBack = NULL, annBGBack = NULL, anndBGBack = NULL;
		if (isSecond)
		{
			patchmatchGPU(t, sBackF, annFGBack, anndFGBack, leapPropagate);
			patchmatchGPU(t, sBackB, annBGBack, anndBGBack, leapPropagate);
		}

		//  draw colored edge map and initial mask
		float maxColor = 0, maxDistance = 0;
		Mat dataColor(Size(width, height), CV_32F, Scalar(0)), dataDistance(Size(width, height), CV_32F, Scalar(0));
		Mat tarContour(Size(width, height), CV_8U, Scalar(0));
		tarPMMask = Mat(Size(width, height), CV_8U, Scalar(0));
		offsetFGx = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetFGy = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBGx = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBGy = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBackFGx = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBackFGy = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBackBGx = Mat(Size(width, height), CV_32S, Scalar(0));
		offsetBackBGy = Mat(Size(width, height), CV_32S, Scalar(0));
		matchedPositionFG = Mat(Size(width, height), CV_8UC3, Scalar(0, 0, 0));
		matchedPositionBG = Mat(Size(width, height), CV_8UC3, Scalar(0, 0, 0));
		matchedPosition = Mat(Size(width, height), CV_8UC3, Scalar(0, 0, 0));
		offsetFG = Mat(Size(width, height), CV_32FC2, Scalar(0, 0));
		offsetBG = Mat(Size(width, height), CV_32FC2, Scalar(0, 0));
#pragma omp parallel for
		for (int y = 0; y < height - global_patch_w + 1; y++)
		{
			Vec3b *posFGRow = matchedPositionFG.ptr<Vec3b>(y + global_patch_w / 2);
			Vec3b *posBGRow = matchedPositionBG.ptr<Vec3b>(y + global_patch_w / 2);
			Vec3b *posRow = matchedPosition.ptr<Vec3b>(y + global_patch_w / 2);
			uchar *tarMaskRow = tarPMMask.ptr<uchar>(y + global_patch_w / 2);
			uchar *tarContourRow = tarContour.ptr<uchar>(y + global_patch_w / 2);
			float *disRow = dataDistance.ptr<float>(y + global_patch_w / 2);
			float *colorRow = dataColor.ptr<float>(y + global_patch_w / 2);
			int *offFGxRow = offsetFGx.ptr<int>(y + global_patch_w / 2);
			int *offFGyRow = offsetFGy.ptr<int>(y + global_patch_w / 2);
			int *offBGxRow = offsetBGx.ptr<int>(y + global_patch_w / 2);
			int *offBGyRow = offsetBGy.ptr<int>(y + global_patch_w / 2);
			int *offBackFGxRow = offsetBackFGx.ptr<int>(y + global_patch_w / 2);
			int *offBackFGyRow = offsetBackFGy.ptr<int>(y + global_patch_w / 2);
			int *offBackBGxRow = offsetBackBGx.ptr<int>(y + global_patch_w / 2);
			int *offBackBGyRow = offsetBackBGy.ptr<int>(y + global_patch_w / 2);
#pragma omp parallel for
			for (int x = 0; x < width - global_patch_w + 1; x++)
			{
				int px = x + global_patch_w / 2, py = y + global_patch_w / 2;

				int non_zero, vFG = (*annFG)[y][x], vBG = (*annBG)[y][x];
				int xxFG = INT_TO_X(vFG), yyFG = INT_TO_Y(vFG);
				int xxBG = INT_TO_X(vBG), yyBG = INT_TO_Y(vBG);
				int xxBackFG, xxBackBG, yyBackFG, yyBackBG;
				if (isSecond)
				{
					int vFG = (*annFGBack)[y][x], vBG = (*annBGBack)[y][x];
					xxBackFG = INT_TO_X(vFG), yyBackFG = INT_TO_Y(vFG);
					xxBackBG = INT_TO_X(vBG), yyBackBG = INT_TO_Y(vBG);
				}

				int tempDisFG = sqrtNani((double)((xxFG - x)*(xxFG - x) + (yyFG - y)*(yyFG - y)));
				if (tempDisFG > diagonalDis / limitedDis)
				{
					tempDisFG = 0;
					xxFG = x, yyFG = y;
				}
				int tempDisBG = sqrtNani((double)((xxBG - x)*(xxBG - x) + (yyBG - y)*(yyBG - y)));
				if (tempDisBG > diagonalDis / limitedDis)
				{
					tempDisBG = 0;
					xxBG = x, yyBG = y;
				}
				int tempDisBackFG, tempDisBackBG;
				if (isSecond){
					tempDisBackFG = sqrtNani((double)((xxBackFG - x)*(xxBackFG - x) + (yyBackFG - y)*(yyBackFG - y)));
					if (tempDisBackFG > diagonalDis / limitedDis)
					{
						tempDisBackFG = 0;
						xxBackFG = x, yyBackFG = y;
					}
					tempDisBackBG = sqrtNani((double)((xxBackBG - x)*(xxBackBG - x) + (yyBackBG - y)*(yyBackBG - y)));
					if (tempDisBackBG > diagonalDis / limitedDis)
					{
						tempDisBackBG = 0;
						xxBackBG = x, yyBackBG = y;
					}
				}
				posFGRow[px][2] = (double)(xxFG + global_patch_w / 2) / width * 255;
				posFGRow[px][1] = (double)(yyFG + global_patch_w / 2) / height * 255;
				posBGRow[px][2] = (double)(xxBG + global_patch_w / 2) / width * 255;
				posBGRow[px][1] = (double)(yyBG + global_patch_w / 2) / height * 255;
				offsetFG.at<Vec2f>(py, px) = Vec2f(xxFG - x, yyFG - y);
				offsetBG.at<Vec2f>(py, px) = Vec2f(xxBG - x, yyBG - y);

				Mat maskMat;
				Mat minMat; float minColor, minDis;
				if ((*anndFG)[y][x] < (*anndBG)[y][x])
				{
					minMat = sourceFGMaskPatch(Rect(xxFG, yyFG, global_patch_w, global_patch_w)).clone();
					minColor = (*anndFG)[y][x]; minDis = tempDisFG;
					posRow[px][2] = (double)(xxFG + global_patch_w / 2) / width * 255;
					posRow[px][1] = (double)(yyFG + global_patch_w / 2) / height * 255;
				}
				else
				{
					minMat = sourceBGMaskPatchDeform(Rect(xxBG, yyBG, global_patch_w, global_patch_w)).clone();
					minColor = (*anndBG)[y][x]; minDis = tempDisBG;
					posRow[px][2] = (double)(xxBG + global_patch_w / 2) / width * 255;
					posRow[px][1] = (double)(yyBG + global_patch_w / 2) / height * 255;
				}
				offFGxRow[px] = xxFG + global_patch_w / 2;
				offFGyRow[px] = yyFG + global_patch_w / 2;
				offBGxRow[px] = xxBG + global_patch_w / 2;
				offBGyRow[px] = yyBG + global_patch_w / 2;
				if (isSecond)
				{
					if ((*anndFGBack)[y][x] < minColor)
					{
						minMat = sourceFGMaskPatchBack(Rect(xxBackFG, yyBackFG, global_patch_w, global_patch_w)).clone();
						minColor = (*anndFGBack)[y][x]; minDis = tempDisBackFG;
					}
					if ((*anndBGBack)[y][x] < minColor)
					{
						minMat = sourceBGMaskPatchBack(Rect(xxBackBG, yyBackBG, global_patch_w, global_patch_w)).clone();
						minColor = (*anndBGBack)[y][x]; minDis = tempDisBackBG;
					}
					offBackFGxRow[px] = xxBackFG + global_patch_w / 2;
					offBackFGyRow[px] = yyBackFG + global_patch_w / 2;
					offBackBGxRow[px] = xxBackBG + global_patch_w / 2;
					offBackBGyRow[px] = yyBackBG + global_patch_w / 2;
				}

				maskMat = minMat.clone();
				colorRow[px] = minColor;
				disRow[px] = minDis;

				maxColor = colorRow[px] > maxColor ? colorRow[px] : maxColor;
				maxDistance = disRow[px] > maxDistance ? disRow[px] : maxDistance;
				non_zero = countNonZero(maskMat);

				if (non_zero <= global_patch_w*global_patch_w / 2)
				{
					//BG
					tarMaskRow[px] = 0;
				}
				else if (non_zero > global_patch_w*global_patch_w / 2)
				{
					//FG
					tarMaskRow[px] = 255;
				}
				if (non_zero != 0 && non_zero != global_patch_w*global_patch_w)
					tarContourRow[px] = 255;
			}
		}
		removeNoise(tarPMMask, sourceFGMaskPatch, maxNoise);
		Mat tarMaskClone = tarPMMask.clone(), tarPatchClone = targetPatch.clone();
#pragma omp parallel for
		for (int y = 0; y < height; y++)
		{
			uchar *tarMaskRow = tarPMMask.ptr<uchar>(y);
			Vec3b *tarPatchRow = tarPatchClone.ptr<Vec3b>(y);
#pragma omp parallel for
			for (int x = 0; x < width; x++)
			{
				if (tarMaskRow[x] == 255)
				{
					tarPatchRow[x][2] = tarPatchRow[x][2] * 0.7;
					tarPatchRow[x][1] = tarPatchRow[x][1] * 0.7;
					tarPatchRow[x][0] = tarPatchRow[x][0] * 0.7 + 255 * 0.3;
				}
			}
		}
		findContours(tarMaskClone, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		drawContours(tarPatchClone, contours, -1, Scalar(0, 0, 255), 2);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%starPmMask_%s.png", videoPath, frameNum);
		imwrite(imageName, tarPMMask);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%starPmResult_%s.png", videoPath, frameNum);
		imwrite(imageName, tarPatchClone);
		tar_show = tarPatchClone.clone();
		// 		imshow("tarPatchClone", tarPatchClone);
		// 		waitKey();

		for (int y = 0; y < height; y++)
		{
			float *colorRow = dataColor.ptr<float>(y);
			float *disRow = dataDistance.ptr<float>(y);
			for (int x = 0; x < width; x++)
			{
				colorRow[x] = 1 - colorRow[x] / maxColor;
				disRow[x] = 1 - disRow[x] / maxDistance;
			}
		}
		Mat dataTermPM = /*(dataColor + dataDistance) / 2*/dataColor;
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%sdataTermPM_%s.xml", videoPath, frameNum);
		std::ofstream os(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
		os << (int)dataTermPM.rows << " " << (int)dataTermPM.cols << " " << (int)dataTermPM.type() << " ";
		os.write((char*)dataTermPM.data, dataTermPM.step.p[0] * dataTermPM.rows);
		os.close();

		vector<int> temp, temp2;
		temp = getBoundRect(tarPMMask);
		temp2 = getBoundRect(sourceFGMaskPatch);
		int minX = temp[0]; int minY = temp[1]; int maxX = temp[2]; int maxY = temp[3];
		int minX2 = temp[0]; int minY2 = temp[1]; int maxX2 = temp[2]; int maxY2 = temp[3];
		if ((minX <= 2 && targetRect.x != 0) || (minY <= 2 && targetRect.y != 0) || (maxX >= tarPMMask.cols - 2 && targetRect.x + targetRect.width != width) || (maxY >= tarPMMask.rows - 2 && targetRect.y + targetRect.height != height) || abs((maxX + minX) / 2 - (maxX2 + minX2) / 2) >= targetRect.width / 6 || abs((maxY + minY) / 2 - (maxY2 + minY2) / 2) >= targetRect.height / 6)
		{
			repeatRect = Rect((maxX + minX) / 2 - targetRect.width / 2 + targetRect.x, (maxY + minY) / 2 - targetRect.height / 2 + targetRect.y, targetRect.width, targetRect.height);
			checkRectBorder(ori_width, ori_height, repeatRect);
			iterNum++;
			isRepeated = true;

			if (iterNum >= 5)
				break;
			continue;
		}

		float maskNum = countNonZero(tarPMMask);
		float sourceMaskNum = countNonZero(sourceFGMaskPatch);
		if (!leapPropagate)
			isTooSmallMask = (maskNum / sourceMaskNum < 0.75) ? true : false;
		else
			isTooSmallMask = (maskNum / sourceMaskNum < 0.3) ? true : false;
		if (iterNum >= 5)
			break;
		iterNum++;
	} while (isTooSmallMask || isRepeated == true);
	// 	Mat flowFG, flowBG;
	// 	motionToColor2(offsetFG, flowFG);
	// 	motionToColor2(offsetBG, flowBG);

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGx.xml", videoPath);
	std::ofstream os(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
	os << (int)offsetFGx.rows << " " << (int)offsetFGx.cols << " " << (int)offsetFGx.type() << " ";
	os.write((char*)offsetFGx.data, offsetFGx.step.p[0] * offsetFGx.rows);
	os.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGy.xml", videoPath);
	os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
	os << (int)offsetFGy.rows << " " << (int)offsetFGy.cols << " " << (int)offsetFGy.type() << " ";
	os.write((char*)offsetFGy.data, offsetFGy.step.p[0] * offsetFGy.rows);
	os.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGx.xml", videoPath);
	os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
	os << (int)offsetBGx.rows << " " << (int)offsetBGx.cols << " " << (int)offsetBGx.type() << " ";
	os.write((char*)offsetBGx.data, offsetBGx.step.p[0] * offsetBGx.rows);
	os.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGy.xml", videoPath);
	os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
	os << (int)offsetBGy.rows << " " << (int)offsetBGy.cols << " " << (int)offsetBGy.type() << " ";
	os.write((char*)offsetBGy.data, offsetBGy.step.p[0] * offsetBGy.rows);
	os.close();

	if (isSecond)
	{
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGxBack.xml", videoPath);
		std::ofstream os(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
		os << (int)offsetBackFGx.rows << " " << (int)offsetBackFGx.cols << " " << (int)offsetBackFGx.type() << " ";
		os.write((char*)offsetBackFGx.data, offsetBackFGx.step.p[0] * offsetBackFGx.rows);
		os.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGyBack.xml", videoPath);
		os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
		os << (int)offsetBackFGy.rows << " " << (int)offsetBackFGy.cols << " " << (int)offsetBackFGy.type() << " ";
		os.write((char*)offsetBackFGy.data, offsetBackFGy.step.p[0] * offsetBackFGy.rows);
		os.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGxBack.xml", videoPath);
		os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
		os << (int)offsetBackBGx.rows << " " << (int)offsetBackBGx.cols << " " << (int)offsetBackBGx.type() << " ";
		os.write((char*)offsetBackBGx.data, offsetBackBGx.step.p[0] * offsetBackBGx.rows);
		os.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGyBack.xml", videoPath);
		os.open(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
		os << (int)offsetBackBGy.rows << " " << (int)offsetBackBGy.cols << " " << (int)offsetBackBGy.type() << " ";
		os.write((char*)offsetBackBGy.data, offsetBackBGy.step.p[0] * offsetBackBGy.rows);
		os.close();
	}

	delete sF; delete sB; delete sBackF; delete sBackB; delete t;
	delete annFG; delete anndFG; delete annBG; delete anndBG;
	delete annFGBack; delete anndFGBack; delete annBGBack; delete anndBGBack;

	time = clock() - time;
	printf("the time elapsed is %f seconds.\n", (float)time / CLOCKS_PER_SEC);
	// 	imshow("tar_show", tar_show);
	// 	waitKey();

	return isSecond;
}

void videoCutout::EdgeClassifier(Mat &sourceImgBack, Mat &sourceMaskBack, Mat& sourceEdgeBack)
{
	clock_t time = clock();

	/****edge classifier****/
	printf("start edge classifier..\n");
	char frameNum[] = "000_001", imageName[256], videoPath[] = "";
	int maxNoiseSize, contourDis, sourceBoundarySize = 3, sideDis = 3, searchSize = 30, edgeMaxDis = 7, maxDifference = 100, minNum = 5, chosen_num = 3, isResized = false;
	bool isLargeImage = false, isSecond = false;
	if (!sourceImgBack.empty() && !sourceMaskBack.empty() && !sourceEdgeBack.empty())
		isSecond = true;

	Mat sourceFGPatchFront, sourceFGMaskPatchFront, sourceFGPatchEdgeFront, sourceBGPatchFront, sourceBGMaskPatchFront, sourceBGPatchEdgeFront, targetPatch, targetPatchEdge, targetMask;
	Mat sourceFGPatchBack, sourceFGMaskPatchBack, sourceBGPatchBack, sourceBGMaskPatchBack, sourceFGPatchEdgeBack, sourceBGPatchEdgeBack;

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatch_%s.png", videoPath, frameNum);
	sourceFGPatchFront = imread(imageName);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchMask_%s.png", videoPath, frameNum);
	sourceFGMaskPatchFront = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchEdge_%s.png", videoPath, frameNum);
	sourceFGPatchEdgeFront = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatch_%s.png", videoPath, frameNum);
	sourceBGPatchFront = imread(imageName);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchMask_%s.png", videoPath, frameNum);
	sourceBGMaskPatchFront = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchEdge_%s.png", videoPath, frameNum);
	sourceBGPatchEdgeFront = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatch_%s.png", videoPath, frameNum);
	targetPatch = imread(imageName);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%stargetPatchEdge_%s.png", videoPath, frameNum);
	targetPatchEdge = imread(imageName, 0);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%starPmMask_%s.png", videoPath, frameNum);
	targetMask = imread(imageName, 0);
	if (isSecond)
	{
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchBack_%s.png", videoPath, frameNum);
		sourceFGPatchBack = imread(imageName);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchMaskBack_%s.png", videoPath, frameNum);
		sourceFGMaskPatchBack = imread(imageName, 0);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceFGPatchEdgeBack_%s.png", videoPath, frameNum);
		sourceFGPatchEdgeBack = imread(imageName, 0);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchBack_%s.png", videoPath, frameNum);
		sourceBGPatchBack = imread(imageName);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchMaskBack_%s.png", videoPath, frameNum);
		sourceBGMaskPatchBack = imread(imageName, 0);
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%ssourceBGPatchEdgeBack_%s.png", videoPath, frameNum);
		sourceBGPatchEdgeBack = imread(imageName, 0);

		// 		sourceFGPatchFront = imread("fgL_img.jpg");
		// 		sourceFGMaskPatchFront = imread("fgL_mask.png", 0);
		// 		sourceBGPatchFront = imread("bgL_img.jpg");
		// 		sourceBGMaskPatchFront = imread("bgL_mask.png", 0);
		// 		sourceFGPatchEdgeFront = imread("fgL_edge.png");
		// 		sourceBGPatchEdgeFront = imread("bgL_edge.png");
		// 		targetPatch = imread("imgM.jpg");
		// 
		// 		sourceFGPatchBack = imread("fgR_img.jpg");
		// 		sourceFGMaskPatchBack = imread("fgR_mask.png", 0);
		// 		sourceFGPatchEdgeBack = imread("fgR_edge.png", 0);
		// 		sourceBGPatchBack = imread("bgR_img.jpg");
		// 		sourceBGMaskPatchBack = imread("bgR_mask.png", 0);
		// 		sourceBGPatchEdgeBack = imread("bgR_edge.png", 0);
	}

	int beginWidth = sourceFGPatchFront.cols, beginHeight = sourceFGPatchFront.rows;
	float scaleFac = 0.9;
	if (isResized)
	{
		resize(sourceFGPatchFront, sourceFGPatchFront, Size(), scaleFac, scaleFac);
		resize(sourceFGMaskPatchFront, sourceFGMaskPatchFront, Size(), scaleFac, scaleFac);
		resize(sourceFGPatchEdgeFront, sourceFGPatchEdgeFront, Size(), scaleFac, scaleFac);
		resize(sourceBGPatchFront, sourceBGPatchFront, Size(), scaleFac, scaleFac);
		resize(sourceBGMaskPatchFront, sourceBGMaskPatchFront, Size(), scaleFac, scaleFac);
		resize(sourceBGPatchEdgeFront, sourceBGPatchEdgeFront, Size(), scaleFac, scaleFac);
		if (isSecond)
		{
			resize(sourceFGPatchBack, sourceFGPatchBack, Size(), scaleFac, scaleFac);
			resize(sourceFGMaskPatchBack, sourceFGMaskPatchBack, Size(), scaleFac, scaleFac);
			resize(sourceFGPatchEdgeBack, sourceFGPatchEdgeBack, Size(), scaleFac, scaleFac);
			resize(sourceBGPatchBack, sourceBGPatchBack, Size(), scaleFac, scaleFac);
			resize(sourceBGMaskPatchBack, sourceBGMaskPatchBack, Size(), scaleFac, scaleFac);
			resize(sourceBGPatchEdgeBack, sourceBGPatchEdgeBack, Size(), scaleFac, scaleFac);
		}
		resize(targetPatch, targetPatch, Size(), scaleFac, scaleFac);
		resize(targetPatchEdge, targetPatchEdge, Size(), scaleFac, scaleFac);
		resize(targetMask, targetMask, Size(), scaleFac, scaleFac);
	}

	Mat offsetFGx, offsetFGy, offsetBGx, offsetBGy;
	Mat offsetBackFGx, offsetBackFGy, offsetBackBGx, offsetBackBGy;
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGx.xml", videoPath);
	std::ifstream is(imageName, std::ios::in | std::ios::binary);
	int rows, cols, type;
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	offsetFGx.create(rows, cols, type);
	is.read((char*)offsetFGx.data, offsetFGx.step.p[0] * offsetFGx.rows); is.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGy.xml", videoPath);
	is.open(imageName, std::ios::in | std::ios::binary);
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	offsetFGy.create(rows, cols, type);
	is.read((char*)offsetFGy.data, offsetFGy.step.p[0] * offsetFGy.rows); is.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGx.xml", videoPath);
	is.open(imageName, std::ios::in | std::ios::binary);
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	offsetBGx.create(rows, cols, type);
	is.read((char*)offsetBGx.data, offsetBGx.step.p[0] * offsetBGx.rows); is.close();

	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGy.xml", videoPath);
	is.open(imageName, std::ios::in | std::ios::binary);
	is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
	offsetBGy.create(rows, cols, type);
	is.read((char*)offsetBGy.data, offsetBGy.step.p[0] * offsetBGy.rows); is.close();

	if (isSecond)
	{
		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGxBack.xml", videoPath);
		std::ifstream is(imageName, std::ios::in | std::ios::binary);
		int rows, cols, type;
		is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
		offsetBackFGx.create(rows, cols, type);
		is.read((char*)offsetBackFGx.data, offsetBackFGx.step.p[0] * offsetBackFGx.rows); is.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetFGyBack.xml", videoPath);
		is.open(imageName, std::ios::in | std::ios::binary);
		is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
		offsetBackFGy.create(rows, cols, type);
		is.read((char*)offsetBackFGy.data, offsetBackFGy.step.p[0] * offsetBackFGy.rows); is.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGxBack.xml", videoPath);
		is.open(imageName, std::ios::in | std::ios::binary);
		is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
		offsetBackBGx.create(rows, cols, type);
		is.read((char*)offsetBackBGx.data, offsetBackBGx.step.p[0] * offsetBackBGx.rows); is.close();

		memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%soffsetBGyBack.xml", videoPath);
		is.open(imageName, std::ios::in | std::ios::binary);
		is >> rows; is.ignore(1); is >> cols; is.ignore(1); is >> type; is.ignore(1);
		offsetBackBGy.create(rows, cols, type);
		is.read((char*)offsetBackBGy.data, offsetBackBGy.step.p[0] * offsetBackBGy.rows); is.close();
	}

	if (isResized)
	{
		int nothing = 1;
		bilinearInterpolate(offsetFGx, offsetFGx, 0, 0, scaleFac, scaleFac, nothing);
		bilinearInterpolate(offsetFGy, offsetFGy, 0, 0, scaleFac, scaleFac, nothing);
		bilinearInterpolate(offsetBGx, offsetBGx, 0, 0, scaleFac, scaleFac, nothing);
		bilinearInterpolate(offsetBGy, offsetBGy, 0, 0, scaleFac, scaleFac, nothing);
	}

	threshold(sourceFGMaskPatchFront, sourceFGMaskPatchFront, 150, 255, THRESH_BINARY);
	threshold(sourceBGMaskPatchFront, sourceBGMaskPatchFront, 150, 255, THRESH_BINARY);
	threshold(sourceFGPatchEdgeFront, sourceFGPatchEdgeFront, 230, 255, THRESH_BINARY);
	threshold(sourceBGPatchEdgeFront, sourceBGPatchEdgeFront, 230, 255, THRESH_BINARY);
	threshold(targetPatchEdge, targetPatchEdge, 230, 255, THRESH_BINARY);
	if (isSecond)
	{
		threshold(sourceFGMaskPatchBack, sourceFGMaskPatchBack, 150, 255, THRESH_BINARY);
		threshold(sourceBGMaskPatchBack, sourceBGMaskPatchBack, 150, 255, THRESH_BINARY);
		threshold(sourceFGPatchEdgeBack, sourceFGPatchEdgeBack, 230, 255, THRESH_BINARY);
		threshold(sourceBGPatchEdgeBack, sourceBGPatchEdgeBack, 230, 255, THRESH_BINARY);
	}

	int height = sourceFGPatchFront.rows, width = sourceFGPatchFront.cols;
	int diagonalDis = sqrtNani((double)(width*width + height*height));

	if (diagonalDis > 900)
		isLargeImage = true;
	if (isLargeImage)
	{
		maxNoiseSize = 30; // the noise size you wanna remove in colored target edge map
		contourDis = 50;
	}
	else
	{
		maxNoiseSize = 20;
		contourDis = 30;
	}

	if (isResized)
	{
		maxNoiseSize = 10;
		contourDis = 15;
		edgeMaxDis = 5;
	}

	Mat tar_dx, tar_dy, tar_dst_gray;
	Mat sourceFG_dx, sourceFG_dy, sourceFG_gray;
	Mat sourceBG_dx, sourceBG_dy, sourceBG_gray;
	Mat sourceFG_clone = sourceFGPatchFront.clone(), sourceBG_clone = sourceBGPatchFront.clone(), target_clone = targetPatch.clone();
	GaussianBlur(sourceFG_clone, sourceFG_clone, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(sourceBG_clone, sourceBG_clone, Size(3, 3), 0, 0, BORDER_DEFAULT);
	GaussianBlur(target_clone, target_clone, Size(3, 3), 0, 0, BORDER_DEFAULT);
	cvtColor(sourceFG_clone, sourceFG_gray, CV_BGR2GRAY);
	cvtColor(sourceBG_clone, sourceBG_gray, CV_BGR2GRAY);
	cvtColor(target_clone, tar_dst_gray, CV_BGR2GRAY);
	Sobel(sourceFG_gray, sourceFG_dx, CV_32F, 1, 0, CV_SCHARR);
	Sobel(sourceFG_gray, sourceFG_dy, CV_32F, 0, 1, CV_SCHARR);
	Sobel(sourceBG_gray, sourceBG_dx, CV_32F, 1, 0, CV_SCHARR);
	Sobel(sourceBG_gray, sourceBG_dy, CV_32F, 0, 1, CV_SCHARR);
	Sobel(tar_dst_gray, tar_dx, CV_32F, 1, 0, CV_SCHARR);
	Sobel(tar_dst_gray, tar_dy, CV_32F, 0, 1, CV_SCHARR);
	Mat sourceFGBack_dx, sourceFGBack_dy;
	Mat sourceBGBack_dx, sourceBGBack_dy;
	if (isSecond)
	{
		Mat sourceFG_clone = sourceFGPatchBack.clone(), sourceBG_clone = sourceBGPatchBack.clone(), sourceFGBack_gray, sourceBGBack_gray;
		GaussianBlur(sourceFG_clone, sourceFG_clone, Size(3, 3), 0, 0, BORDER_DEFAULT);
		GaussianBlur(sourceBG_clone, sourceBG_clone, Size(3, 3), 0, 0, BORDER_DEFAULT);
		cvtColor(sourceFG_clone, sourceFGBack_gray, CV_BGR2GRAY);
		cvtColor(sourceBG_clone, sourceBGBack_gray, CV_BGR2GRAY);
		Sobel(sourceFGBack_gray, sourceFGBack_dx, CV_32F, 1, 0, CV_SCHARR);
		Sobel(sourceFGBack_gray, sourceFGBack_dy, CV_32F, 0, 1, CV_SCHARR);
		Sobel(sourceBGBack_gray, sourceBGBack_dx, CV_32F, 1, 0, CV_SCHARR);
		Sobel(sourceBGBack_gray, sourceBGBack_dy, CV_32F, 0, 1, CV_SCHARR);
	}

	Mat sourceMaskContour = sourceFGMaskPatchFront.clone();
	vector<vector<Point>> contours;
	contours.clear(); findContours(sourceMaskContour, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	drawContours(sourceFGPatchEdgeFront, contours, -1, Scalar(0), 2);
	sourceMaskContour = sourceBGMaskPatchFront.clone();
	contours.clear(); findContours(sourceMaskContour, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
	drawContours(sourceBGPatchEdgeFront, contours, -1, Scalar(0), 2);

	int sourceFGColorNum = 0;
	vector<edgeEle> sourceFGColors; sourceFGColors.resize(width*height);
	vector<bool> isForegroundFG; isForegroundFG.resize(width*height);
	Mat sourceFGEdgeResult = sourceFGPatchEdgeFront.clone();
	cvtColor(sourceFGEdgeResult, sourceFGEdgeResult, CV_GRAY2BGR);

	int sourceBGColorNum = 0;
	vector<edgeEle> sourceBGColors; sourceBGColors.resize(width*height);
	vector<bool> isForegroundBG; isForegroundBG.resize(width*height);
	Mat sourceBGEdgeResult = sourceBGPatchEdgeFront.clone();
	cvtColor(sourceBGEdgeResult, sourceBGEdgeResult, CV_GRAY2BGR);

#pragma omp parallel sections
	{
#pragma omp section
		computeSourceColorSet(sourceFGPatchEdgeFront, sourceFGEdgeResult, sourceFGPatchFront, sourceFGMaskPatchFront, sourceFG_dx, sourceFG_dy, sourceBoundarySize, sideDis, sourceFGColorNum, chosen_num, isForegroundFG, sourceFGColors);
#pragma omp section
		computeSourceColorSet(sourceBGPatchEdgeFront, sourceBGEdgeResult, sourceBGPatchFront, sourceBGMaskPatchFront, sourceBG_dx, sourceBG_dy, sourceBoundarySize, sideDis, sourceBGColorNum, chosen_num, isForegroundBG, sourceBGColors);
	}

	int sourceFGColorNumBack = 0;
	vector<edgeEle> sourceFGColorsBack; sourceFGColorsBack.resize(width*height);
	vector<bool> isForegroundFGBack; isForegroundFGBack.resize(width*height);
	int sourceBGColorNumBack = 0;
	vector<edgeEle> sourceBGColorsBack; sourceBGColorsBack.resize(width*height);
	vector<bool> isForegroundBGBack; isForegroundBGBack.resize(width*height);
	if (isSecond)
	{
		sourceMaskContour = sourceFGMaskPatchBack.clone();
		contours.clear(); findContours(sourceMaskContour, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		drawContours(sourceFGPatchEdgeBack, contours, -1, Scalar(0), 2);
		sourceMaskContour = sourceBGMaskPatchBack.clone();
		contours.clear(); findContours(sourceMaskContour, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		drawContours(sourceBGPatchEdgeBack, contours, -1, Scalar(0), 2);

		Mat sourceFGEdgeResult = sourceFGPatchEdgeBack.clone();
		cvtColor(sourceFGEdgeResult, sourceFGEdgeResult, CV_GRAY2BGR);
		Mat sourceBGEdgeResult = sourceBGPatchEdgeBack.clone();
		cvtColor(sourceBGEdgeResult, sourceBGEdgeResult, CV_GRAY2BGR);

#pragma omp parallel sections
		{
#pragma omp section
			computeSourceColorSet(sourceFGPatchEdgeBack, sourceFGEdgeResult, sourceFGPatchBack, sourceFGMaskPatchBack, sourceFGBack_dx, sourceFGBack_dy, sourceBoundarySize, sideDis, sourceFGColorNumBack, chosen_num, isForegroundFGBack, sourceFGColorsBack);
#pragma omp section
			computeSourceColorSet(sourceBGPatchEdgeBack, sourceBGEdgeResult, sourceBGPatchBack, sourceBGMaskPatchBack, sourceBGBack_dx, sourceBGBack_dy, sourceBoundarySize, sideDis, sourceBGColorNumBack, chosen_num, isForegroundBGBack, sourceBGColorsBack);
		}
	}

	Mat distanceFG, distanceBG, distance, tarMaskClone = targetMask.clone();
	distanceTransform(tarMaskClone, distanceFG, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	tarMaskClone = 255 - targetMask;
	distanceTransform(tarMaskClone, distanceBG, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	distance = (distanceFG + distanceBG);
	Mat targetContourEdge(height, width, CV_8U, Scalar(255));
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (distance.at<float>(y, x) < contourDis)
				targetContourEdge.at<uchar>(y, x) = targetPatchEdge.at<uchar>(y, x);
		}
	}

	Mat fgConstraint(Size(width, height), CV_8U, Scalar(0)), bgConstraint(Size(width, height), CV_8U, Scalar(0));
	Mat tar_weight(Size(width, height), CV_8U, Scalar(0));
	Mat tarEdgeResult = targetContourEdge.clone();
	cvtColor(tarEdgeResult, tarEdgeResult, CV_GRAY2BGR);
	Mat fg_mat(Size(width, height), CV_8U, Scalar(255)), bg_mat(Size(width, height), CV_8U, Scalar(255)), contour_mat(Size(width, height), CV_8U, Scalar(255));

	// 	clock_t local_time = clock();
	classifyTargetEdges(targetContourEdge, tarEdgeResult, targetPatch, tar_dx, tar_dy, fg_mat, bg_mat, contour_mat, offsetFGx, offsetFGy, offsetBGx, offsetBGy, offsetBackFGx, offsetBackFGy, offsetBackBGx, offsetBackBGy, fgConstraint, bgConstraint, tar_weight, height, width, sideDis, searchSize, maxDifference, minNum, diagonalDis, sourceFGColorNum, isForegroundFG, sourceFGColors, sourceBGColorNum, isForegroundBG, sourceBGColors, sourceFGColorNumBack, isForegroundFGBack, sourceFGColorsBack, sourceBGColorNumBack, chosen_num, isForegroundBGBack, sourceBGColorsBack, isSecond);
	// 	local_time = clock() - local_time;
	// 	printf("the local time elapsed is %f seconds.\n", (float)local_time / CLOCKS_PER_SEC);

	Mat tar_edge_result2 = targetContourEdge.clone();
	cvtColor(tar_edge_result2, tar_edge_result2, CV_GRAY2BGR);
	Mat test_duplicate(Size(width, height), CV_8U, Scalar(0));
	Mat tar_constraint(Size(width, height), CV_8U, Scalar(125));
	removeEdgeErrors(targetContourEdge, tar_dx, tar_dy, tar_edge_result2, fgConstraint, bgConstraint, contour_mat, bg_mat, fg_mat, test_duplicate, width, height, maxNoiseSize, sideDis);
	removeEdgeErrors4Diffusion(contour_mat, fgConstraint, bgConstraint, tar_constraint, tar_weight, width, height, maxNoiseSize, edgeMaxDis);
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%starClassifiedEdge_%s.png", videoPath, frameNum);
	imwrite(imageName, tar_edge_result2);

	//get the distance transform for silhouette edges!
	Mat dataTermEdge;
	distanceTransform(contour_mat, dataTermEdge, CV_DIST_L2, CV_DIST_MASK_PRECISE);
	for (int y = 0; y < height; y++)
	{
		uchar *tarContRow = tar_constraint.ptr<uchar>(y);
		float *edgeRow = dataTermEdge.ptr<float>(y);
		for (int x = 0; x < width; x++)
		{
			if (tarContRow[x] == 125)
				edgeRow[x] = 0;
			else if (tarContRow[x] == 0)
				edgeRow[x] = -edgeRow[x];
		}
	}
	dataTermEdge /= edgeMaxDis; //to normalize edge data term to [0,1].
	if (isResized)
	{
		float nothingfloat = 1.0;
		bilinearInterpolate(dataTermEdge, dataTermEdge, beginWidth, beginHeight, 0, 0, nothingfloat);
	}

	//   	Mat dataTermEdge = Mat(Size(width,height), CV_32F, Scalar(0));
	memset(imageName, 0, sizeof(imageName)); sprintf(imageName, "%sdataTermEdge_%s.xml", videoPath, frameNum);
	ofstream os2(imageName, std::ios::out | std::ios::trunc | std::ios::binary);
	os2 << (int)dataTermEdge.rows << " " << (int)dataTermEdge.cols << " " << (int)dataTermEdge.type() << " ";
	os2.write((char*)dataTermEdge.data, dataTermEdge.step.p[0] * dataTermEdge.rows);
	os2.close();
	/****edge classifier****/

	time = clock() - time;
	printf("the time elapsed is %f seconds.\n", (float)time / CLOCKS_PER_SEC);
}

template<typename T>
void videoCutout::bilinearInterpolate(Mat source, Mat &target, int width, int height, float scaleX, float scaleY, T nothing)
{
	int srcWidth = source.cols, srcHeight = source.rows;
	int tarWidth = width, tarHeight = height;
	if (tarWidth == 0 || tarHeight == 0)
	{
		tarWidth = srcWidth*scaleX, tarHeight = srcHeight *scaleY;
	}
	else
	{
		scaleX = (float)tarWidth / srcWidth, scaleY = (float)tarHeight / srcHeight;
	}
	target.create(Size(tarWidth, tarHeight), source.type());

	for (int y = 0; y < tarHeight; y++)
	{
		for (int x = 0; x < tarWidth; x++)
		{
			int sx = x / scaleX, sy = y / scaleY;
			sx = min(srcWidth - 1, sx), sy = min(srcHeight - 1, sy);

			T p0 = 0, p1 = 0, p2 = 0, p3 = 0;
			if (sx - 1 >= 0 && sy - 1 >= 0)
				p0 = source.at<T>(sy - 1, sx - 1);
			if (sx - 1 >= 0 && sy + 1 < srcHeight)
				p1 = source.at<T>(sy + 1, sx - 1);
			if (sx + 1 < srcWidth && sy - 1 >= 0)
				p2 = source.at<T>(sy - 1, sx + 1);
			if (sx + 1 < srcWidth && sy + 1 < srcHeight)
				p3 = source.at<T>(sy + 1, sx + 1);
			target.at<T>(y, x) = (p0 + p1 + p2 + p3) / 4;
		}
	}
}